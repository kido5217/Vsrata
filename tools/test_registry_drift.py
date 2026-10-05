"""Drift check for `nix/registry.nix` (the #27 deliverable).

`nix/registry.nix` is generated from `setup.py`'s tables. This test fails when the committed copy
differs from a fresh generation, so the registry cannot silently lag the tables it mirrors.

    python -m unittest tools.test_registry_drift
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "tools" / "make_registry.py"
COMMITTED = ROOT / "nix" / "registry.nix"


class RegistryDrift(unittest.TestCase):
    def test_committed_registry_matches_setup_tables(self):
        r = subprocess.run([sys.executable, str(GEN), "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_two_generations_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.nix", Path(d) / "b.nix"
            for out in (a, b):
                r = subprocess.run([sys.executable, str(GEN), "--out", str(out)],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_check_fails_on_a_perturbed_copy(self):
        # The check must be able to fail: a file that is not the generation is stale.
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "registry.nix"
            bad.write_text(COMMITTED.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            r = subprocess.run([sys.executable, str(GEN), "--check", "--out", str(bad)],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
