"""Tests for the `vsrata` CLI (`cli/vsrata.py`) that need neither a download nor an engine.

They exercise argument parsing, `--help`, the per-stage completion checks, and the no-op re-run
path: a scratch config whose `provision` block points at the Git-ignored fixture tree on this host
(`models/qwen-Q2_0`, `pack/q2_0`, `mtp`) must be recognized as already provisioned, so a re-run
writes its markers and shells out to nothing.

    python -m unittest tools.test_vsrata_cli
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "cli" / "vsrata.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("vsrata_cli", CLI)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CliSurface(unittest.TestCase):
    """`--help` and argument parsing; no config, no work."""

    def test_top_level_help(self):
        r = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("provision", r.stdout)
        self.assertIn("serve", r.stdout)

    def test_subcommand_help(self):
        for sub in ("serve", "provision"):
            r = subprocess.run([sys.executable, str(CLI), sub, "--help"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("--config", r.stdout)

    def test_config_is_required(self):
        r = subprocess.run([sys.executable, str(CLI), "provision"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--config", r.stderr)

    def test_unknown_repo_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "p.json"
            cfg.write_text(json.dumps({"provision": {
                "model": "nope/nope", "quant": "Q2_0", "dataRoot": d,
                "profileDir": d, "packDir": d, "mtpDir": d}}), encoding="utf-8")
            r = subprocess.run([sys.executable, str(CLI), "provision", "--config", str(cfg)],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("unknown model repository", r.stderr + r.stdout)


class StageChecks(unittest.TestCase):
    """The completion checks the CLI uses instead of trusting the tools' gates."""

    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli()

    def test_pack_check_accepts_the_fixture_and_a_stub(self):
        self.assertTrue(self.cli._pack_ok(ROOT / "pack" / "q2_0"))
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d)
            self.assertFalse(self.cli._pack_ok(pack))                 # empty dir
            (pack / "native_experts.txt").write_text("x", encoding="utf-8")
            (pack / "dense.bin").write_text("x", encoding="utf-8")
            (pack / "index.txt").write_text("x", encoding="utf-8")
            self.assertFalse(self.cli._pack_ok(pack))                 # tokenizer still missing
            (pack / "tokenizer").mkdir()
            (pack / "tokenizer" / "vocab.json").write_text("x", encoding="utf-8")
            self.assertTrue(self.cli._pack_ok(pack))

    def test_mtp_check_accepts_the_fixture(self):
        self.assertTrue(self.cli._mtp_ok(ROOT / "mtp" / "rt"))

    def test_pinned_shard_sha_is_checked_on_adoption(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "shard.gguf"
            data = b"some bytes"
            p.write_bytes(data)
            good = {"sha256": {"shard.gguf": (len(data), hashlib.sha256(data).hexdigest())}}
            self.cli._verify_pinned(good, ["shard.gguf"], Path(d))     # no raise
            bad = {"sha256": {"shard.gguf": (len(data), "0" * 64)}}
            with self.assertRaises(SystemExit):
                self.cli._verify_pinned(bad, ["shard.gguf"], Path(d))  # deleted + refused

    def test_marker_paths_sit_beside_the_config(self):
        m = self.cli.Markers(ROOT / "cli" / "vsrata.py")
        self.assertEqual(m.stage("pack"), ROOT / "cli" / "vsrata.pack.done")
        self.assertEqual(m.final(), ROOT / "cli" / "vsrata.done")


class ServeExec(unittest.TestCase):
    """`serve` builds the right server argv and passes the key through the environment, without
    starting anything: `os.execve` is stubbed and the argument/config fallback is asserted."""

    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli()

    def _call(self, cfg: Path, port, env: dict | None = None):
        captured = {}

        def fake_execve(path, argv, child_env):
            captured.update(path=path, argv=argv, env=child_env)
            raise SystemExit(0)

        real = self.cli.os.execve
        self.cli.os.execve = fake_execve
        try:
            with mock.patch.dict(os.environ, env or {}, clear=False):
                with self.assertRaises(SystemExit):
                    self.cli.serve(cfg, port)
        finally:
            self.cli.os.execve = real
        return captured

    def test_port_falls_back_to_the_config_then_an_override_wins(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "profile-01.json"
            cfg.write_text(json.dumps({"port": 8123}), encoding="utf-8")
            got = self._call(cfg, None)
            self.assertEqual(got["argv"][-2:], ["--port", "8123"])
            self.assertTrue(got["argv"][1].endswith("serve/server.py"))
            self.assertEqual(got["argv"][2:6], ["--engine", "strata", "--config", str(cfg.resolve())])
            got = self._call(cfg, 9000)
            self.assertEqual(got["argv"][-2:], ["--port", "9000"])

    def test_api_key_is_read_from_the_credential_file_into_the_env(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "profile-01.json"
            cfg.write_text(json.dumps({"port": 8095}), encoding="utf-8")
            key = Path(d) / "api-key"
            key.write_text("sekret\n", encoding="utf-8")
            got = self._call(cfg, None, {"STRATA_API_KEY_FILE": str(key)})
            self.assertEqual(got["env"]["STRATA_API_KEY"], "sekret")
            self.assertNotIn("sekret", " ".join(got["argv"]))   # never on the command line


class ProvisionNoOp(unittest.TestCase):
    """The fixture tree is already provisioned: a run must skip every stage and touch no tool.

    These tests point the CLI at this checkout's fixture tree, so they only mean anything where one
    exists — without the guard below, a missing `pack/q2_0` sends provision off to download the
    model for real."""

    FIXTURE = ROOT / "pack" / "q2_0" / "native_experts.txt"

    def setUp(self):
        if not self.FIXTURE.exists():
            self.skipTest(f"no provisioned fixtures under {ROOT/'pack'/'q2_0'} (see docs/research/fixtures-models.md)")

    def _config(self, d: str) -> Path:
        cfg = Path(d) / "profile-01.json"
        cfg.write_text(json.dumps({
            "port": 8095,
            "cwd": d,
            "provision": {
                "model": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "quant": "Q2_0",
                "dataRoot": str(ROOT / "models"),
                "profileDir": str(ROOT / "models" / "qwen-Q2_0"),
                "packDir": str(ROOT / "pack" / "q2_0"),
                "mtpDir": str(ROOT / "mtp"),
                "context": 262144,
                "vision": False,
                "visionAccel": "gpu",
            },
        }), encoding="utf-8")
        return cfg

    def _run(self, cfg: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(CLI), "provision", "--config", str(cfg)],
                              capture_output=True, text=True, timeout=300)

    def test_first_run_skips_all_stages(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._config(d)
            r = self._run(cfg)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = r.stdout
            self.assertIn("shards: present, marked complete", out)
            self.assertIn("pack: ", out)
            self.assertIn("MTP: ", out)
            # no tool was shelled out to (no download, no pack, no MTP chain)
            self.assertNotIn("iq_pack.py", out)
            self.assertNotIn("mtp_fetch.py", out)
            self.assertNotIn("tools/", out)
            for name in ("shards", "pack", "mtp"):
                self.assertTrue((Path(d) / f"profile-01.{name}.done").exists(), name)
            self.assertTrue((Path(d) / "profile-01.done").exists())

    def test_second_run_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._config(d)
            self.assertEqual(self._run(cfg).returncode, 0)
            r = self._run(cfg)
            self.assertEqual(r.returncode, 0, r.stderr)
            # markers now exist and the artifacts still check out: every stage takes the marker path
            self.assertEqual(r.stdout.count("already complete"), 3, r.stdout)
            self.assertNotIn("tools/", r.stdout)
            for name in ("shards", "pack", "mtp"):
                self.assertTrue((Path(d) / f"profile-01.{name}.done").exists(), name)


if __name__ == "__main__":
    unittest.main()
