#!/usr/bin/env python3
"""`vsrata` — the runtime CLI the systemd units call (the module's #27 deliverable).

Two subcommands, both functions of one generated config file (spec §5):

    vsrata serve     --config <path> [--port <port>]   exec the server for a profile
    vsrata provision --config <path>                   fetch and prepare a profile, idempotently

`serve` execs `python <package>/share/vsrata/serve/server.py --engine strata --config <path>
--port <port>`. The port is the CLI argument, else the config's `port` key, else 8095 — the server
never reads the key. The API key is read from the runtime file in `STRATA_API_KEY_FILE` (which the
unit's `LoadCredential` populates) and handed to the server through the environment; it never
appears on the command line.

`provision` performs, in order, the five stages of spec §3/§5: the GGUF shards (resumable, plain
HTTPS with `Range`), the mmproj when vision is on, the pack via a self-contained `iq_pack` (never
`strata_pack`/`pack_index`, the AVX-512-only branch) with the registry's `packArgs`, the MTP chain
(`mtp_fetch` → `mtp_pack` → `mtp_rt` plus the draft vocab copied from `<repo>/data/`), and finally
the `<profile>.done` marker the systemd unit keys on. Every stage gets its OWN marker, written only
after the stage's artifacts validate: the tools' own gates are existence-only over non-atomic
writers, so the CLI does not trust them.

What this CLI does NOT do: it does not generate configs or units (the Nix module does), build the
engine, decide GPU/RAM settings (`extraArgs` and the config carry those), start or manage systemd,
or serve models outside `setup.py`'s tables. The config's `provision` block is ignored by the
server, so the same file drives both subcommands.

Config shape (spec §3.2), the keys provision reads:

    {"port": 8095,
     "provision": {"model": "<HF repo>", "quant": "<quant>", "dataRoot": "<dir>",
                   "profileDir": "<dir>", "packDir": "<dir>", "mtpDir": "<dir>",
                   "context": 262144, "vision": false, "visionAccel": "gpu"}}

The final marker is written beside the config, with the config's extension swapped for `.done`:
`$XDG_STATE_HOME/vsrata/<profile>.json` → `…/<profile>.done` (spec §3/§4). A config whose name is
not the profile can carry an explicit `provision.doneFile` instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# --------------------------------------------------------------------------------------- locations
def find_root() -> Path:
    """The package root: where `serve/`, `tools/` and `data/` live.

    Works in the checkout (`cli/vsrata.py` → the repo root) and in the installed layout
    (`bin/vsrata` → `<root>/share/vsrata`), with `VSRATA_ROOT` as the explicit override."""
    env = os.environ.get("VSRATA_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    for cand in (here.parent, here.parent / "share" / "vsrata", here.parent.parent / "share" / "vsrata"):
        if (cand / "serve" / "server.py").is_file():
            return cand
    raise SystemExit("vsrata: cannot find serve/server.py; set VSRATA_ROOT to the package root")


def load_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise SystemExit(f"vsrata: config not found: {path}")
    except (OSError, ValueError) as e:
        raise SystemExit(f"vsrata: cannot read {path}: {e}")


def _read_secret_file(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def hf_token() -> str | None:
    """The Hugging Face token: the runtime file the unit delivers, else the conventional env var.

    The registry's repositories are public, so a token is optional; anonymous access works."""
    token = _read_secret_file(os.environ.get("VSRATA_HF_TOKEN_FILE"))
    return token or os.environ.get("HF_TOKEN") or None


# ----------------------------------------------------------------------------------------- markers
class Markers:
    """Per-stage completion markers, all in the config's directory.

    Written by this CLI after a stage's artifacts validate — never the tools' own gates, which are
    existence checks over writers that can be interrupted part-way (provisioning contract §7)."""

    def __init__(self, config: Path):
        config = config.resolve()
        self.dir = config.parent
        self.stem = config.stem

    def stage(self, name: str) -> Path:
        return self.dir / f"{self.stem}.{name}.done"

    def final(self) -> Path:
        return self.dir / f"{self.stem}.done"

    def has(self, name: str) -> bool:
        return self.stage(name).exists()

    def write(self, name: str) -> None:
        p = self.stage(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")


def _nonempty(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0


def _skip(mark: Markers, name: str, artifacts_ok: bool) -> bool:
    """A stage is done when its marker exists AND its artifacts still validate.

    The marker alone is not trusted: someone may have deleted an artifact after the marker was
    written. The artifacts alone are not trusted either (the tools' writers can be interrupted);
    the marker records that this CLI saw a validated stage through to the end."""
    return mark.has(name) and artifacts_ok


def _adopt(mark: Markers, name: str, artifacts_ok: bool) -> bool:
    """Pre-existing, valid artifacts (a pack built by hand or by setup.py): adopt them and mark."""
    if artifacts_ok:
        mark.write(name)
        return True
    return False


# -------------------------------------------------------------------------------------- downloads
def download(url: str, dst: Path, token: str | None = None, expected: tuple[int, str] | None = None) -> None:
    """A resumable HTTPS download: `Range` from the existing `.part`, atomic `.part` → final.

    Mirrors setup.py's own downloader (HEAD for the size, 206 check, retries), with the optional
    bearer token for gated repositories and the pinned (bytes, sha256) check for Unsloth's shards."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    headers = {"User-Agent": "vsrata"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0) or 0)
            break
        except (urllib.error.URLError, OSError) as e:
            if attempt == 4:
                raise SystemExit(f"vsrata: cannot reach {url.split('/')[2] if '//' in url else url}: {e}")
            time.sleep(5)

    if dst.is_file() and total and dst.stat().st_size == total:
        return                                    # finished by an earlier run (no .part left behind)

    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        try:
            req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and getattr(r, "status", None) != 206:   # the server ignored the range
                    have = 0
                mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    while True:
                        b = r.read(8 << 20)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
            if not total or have >= total:
                break
        except (urllib.error.URLError, OSError) as e:
            print(f"  interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)

    if total and part.stat().st_size != total:
        raise SystemExit(f"vsrata: {dst.name} is {part.stat().st_size:,} bytes, the server says {total:,} "
                         "(run again: the download resumes)")
    if expected:
        size, sha = expected
        if part.stat().st_size != size:
            part.unlink(missing_ok=True)
            raise SystemExit(f"vsrata: {dst.name} is {part.stat().st_size:,} bytes, expected {size:,}")
        _verify_sha256(part, sha, dst.name)
    part.replace(dst)


def _verify_sha256(path: Path, sha: str, name: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    if h.hexdigest() != sha:
        path.unlink(missing_ok=True)
        raise SystemExit(f"vsrata: {name} has the wrong SHA-256; deleted, run again")


# ----------------------------------------------------------------------------------------- stages
def _shard_ok(setup, fam: dict, name: str, path: Path) -> bool:
    """A shard is complete when its GGUF tensor directory fits inside the file, and — where the
    pinned shard table exists — its byte size matches."""
    if not path.is_file():
        return False
    pinned = fam.get("sha256", {}).get(name)
    if pinned and path.stat().st_size != pinned[0]:
        return False
    return setup.whole_shard(path)


def _shards_ok(setup, fam: dict, names: list[str], profile_dir: Path) -> bool:
    return all(_shard_ok(setup, fam, n, profile_dir / n) for n in names)


def stage_shards(mark: Markers, setup, fam: dict, names: list[str], urls: list[str],
                 profile_dir: Path, token: str | None) -> None:
    ok = _shards_ok(setup, fam, names, profile_dir)
    if _skip(mark, "shards", ok):
        print("shards: already complete")
        return
    if _adopt(mark, "shards", ok):
        print("shards: present, marked complete")
        return
    for name, url in zip(names, urls):
        dst = profile_dir / name
        if _shard_ok(setup, fam, name, dst):
            print(f"shards: {name} already present")
            continue
        print(f"shards: fetching {name}")
        download(url, dst, token=token, expected=fam.get("sha256", {}).get(name))
    if not _shards_ok(setup, fam, names, profile_dir):
        raise SystemExit("vsrata: shards are incomplete after download")
    mark.write("shards")


def stage_mmproj(mark: Markers, fam: dict, prov: dict, token: str | None) -> None:
    if not prov.get("vision"):
        return
    dst = Path(prov["dataRoot"]) / "models" / fam["mmproj"]
    ok = _nonempty(dst)
    if _skip(mark, "mmproj", ok):
        print(f"vision encoder: {dst.name} already complete")
        return
    if _adopt(mark, "mmproj", ok):
        print(f"vision encoder: {dst.name} present, marked complete")
        return
    print(f"vision encoder: fetching {fam['mmproj']}")
    download(fam["mmproj_hf"] + fam["mmproj"], dst, token=token)
    if not _nonempty(dst):
        raise SystemExit("vsrata: the vision encoder download left no file")
    mark.write("mmproj")


def _pack_ok(pack_dir: Path) -> bool:
    # native_experts.txt is atomic and published last (iq_pack.py:569-575); the rest guard the
    # non-atomic writers a kill can truncate (provisioning contract §7).
    return (_nonempty(pack_dir / "native_experts.txt")
            and _nonempty(pack_dir / "tokenizer" / "vocab.json")
            and _nonempty(pack_dir / "dense.bin")
            and _nonempty(pack_dir / "index.txt"))


def stage_pack(mark: Markers, setup, root: Path, fam: dict, names: list[str],
               profile_dir: Path, pack_dir: Path) -> None:
    ok = _pack_ok(pack_dir)
    if _skip(mark, "pack", ok):
        print(f"pack: {pack_dir} already complete")
        return
    if _adopt(mark, "pack", ok):
        print(f"pack: {pack_dir} present, marked complete")
        return
    shard1 = profile_dir / names[0]
    if not _shard_ok(setup, fam, names[0], shard1):
        raise SystemExit(f"vsrata: {shard1.name} is not a complete shard; cannot pack")
    cmd = [sys.executable, str(root / "tools" / "iq_pack.py"), "--gguf", str(shard1), "--out", str(pack_dir),
           *fam.get("pack_args", [])]
    print("pack: " + " ".join(cmd[1:]))
    _run(cmd)
    if not _pack_ok(pack_dir):
        raise SystemExit("vsrata: iq_pack did not produce a complete pack")
    mark.write("pack")


def _mtp_ok(rt: Path) -> bool:
    return (_nonempty(rt / "experts.bin") and _nonempty(rt / "dense.bin")
            and _nonempty(rt / "dense.txt") and _nonempty(rt / "draft_vocab.bin"))


def _copy_draft_vocab(root: Path, rt: Path) -> None:
    """The draft token subset ships in the repo's `data/` (setup.py's cjk default): the MTP runtime
    never writes one, so the copy is the only producer."""
    src = root / "data" / "draft_vocab.bin"
    dst = rt / "draft_vocab.bin"
    if not src.is_file():
        print("MTP: no data/draft_vocab.bin to copy (the engine's bundled default is used)")
        return
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)


def stage_mtp(mark: Markers, root: Path, mtp_dir: Path) -> None:
    rt = mtp_dir / "rt"
    ok = _mtp_ok(rt)
    if _skip(mark, "mtp", ok):
        print(f"MTP: {rt} already complete")
        _copy_draft_vocab(root, rt)
        return
    if _adopt(mark, "mtp", ok):
        print(f"MTP: {rt} present, marked complete")
        _copy_draft_vocab(root, rt)
        return
    _run_tool(root, "mtp_fetch.py", "fetch", "--out", str(mtp_dir))
    gguf = mtp_dir / "mtp-q2_0.gguf"
    _run_tool(root, "mtp_pack.py", "--src", str(mtp_dir), "--experts", "q2_0", "--out", str(gguf))
    _run_tool(root, "mtp_rt.py", "--gguf", str(gguf), "--out", str(rt))
    _copy_draft_vocab(root, rt)
    if not _mtp_ok(rt):
        raise SystemExit(f"vsrata: the MTP runtime under {rt} is incomplete")
    mark.write("mtp")


def _run_tool(root: Path, tool: str, *args: str) -> None:
    cmd = [sys.executable, str(root / "tools" / tool), *args]
    print("MTP: " + " ".join(cmd[1:]))
    _run(cmd)


def _run(cmd: list[str]) -> None:
    """Run a tool, turning a non-zero exit into a clean message (the child already printed why)."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"vsrata: {Path(cmd[1]).name} failed (exit {e.returncode})")


# --------------------------------------------------------------------------------------- commands
def serve(config: Path, port: int | None) -> int:
    root = find_root()
    cfg = load_config(config)
    if port is None:
        port = cfg.get("port") or 8095
    env = dict(os.environ)
    key = _read_secret_file(env.get("STRATA_API_KEY_FILE"))
    if key:
        env["STRATA_API_KEY"] = key            # the unit's LoadCredential file, never argv
    server = root / "serve" / "server.py"
    argv = [sys.executable, str(server), "--engine", "strata", "--config", str(config), "--port", str(int(port))]
    os.execve(sys.executable, argv, env)
    return 0                                    # not reached: exec replaces this process


def family_for(setup, repo: str) -> tuple[str, dict]:
    """The family whose Hugging Face repository is `repo`, from setup.py's own FAMILIES table."""
    if repo not in setup.HF_REVISIONS:
        raise SystemExit(f"vsrata: unknown model repository {repo!r} (not in setup.py's HF_REVISIONS)")
    for family, fam in setup.FAMILIES.items():
        if fam["hf"].startswith(setup.hf(repo)):
            return family, fam
    raise SystemExit(f"vsrata: no FAMILIES entry publishes {repo!r}")


def provision(config: Path) -> int:
    cfg = load_config(config)
    prov = cfg.get("provision")
    if not isinstance(prov, dict):
        raise SystemExit("vsrata: config has no `provision` block")
    need = ("model", "quant", "dataRoot", "profileDir", "packDir", "mtpDir")
    missing = [k for k in need if not prov.get(k)]
    if missing:
        raise SystemExit(f"vsrata: the provision block is missing {', '.join(missing)}")

    root = find_root()
    sys.path.insert(0, str(root))
    import setup                                # the same tables the registry was generated from

    family, fam = family_for(setup, prov["model"])
    print(f"vsrata: provisioning {family}/{prov['quant']} into {prov['profileDir']}")
    mark = Markers(config)
    done = Path(prov.get("doneFile") or mark.final())
    profile_dir = Path(prov["profileDir"])
    pack_dir = Path(prov["packDir"])
    mtp_dir = Path(prov["mtpDir"])
    names = [setup.model_file(fam, prov["quant"], i) for i in range(1, setup.model_shards(fam, prov["quant"]) + 1)]
    urls = [fam["hf"].format(q=prov["quant"]) + n for n in names]

    stage_shards(mark, setup, fam, names, urls, profile_dir, hf_token())
    stage_mmproj(mark, fam, prov, hf_token())
    stage_pack(mark, setup, root, fam, names, profile_dir, pack_dir)
    stage_mtp(mark, root, mtp_dir)

    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text("", encoding="utf-8")
    print(f"vsrata: done ({done})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vsrata", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="exec the server for a profile")
    s.add_argument("--config", required=True, help="the profile's generated config (absolute path)")
    s.add_argument("--port", type=int, default=None,
                   help="the port to serve on; defaults to the config's `port`, then 8095")
    p = sub.add_parser("provision", help="fetch and prepare a profile, idempotently")
    p.add_argument("--config", required=True, help="the profile's generated config (absolute path)")
    a = ap.parse_args(argv)
    config = Path(a.config).resolve()
    if a.cmd == "serve":
        return serve(config, a.port)
    return provision(config)


if __name__ == "__main__":
    sys.exit(main())
