#!/usr/bin/env python3
"""Generate `nix/registry.nix` from setup.py's own tables.

The real source of a (model, quant) pair is `setup.py`'s `FAMILIES`, `MODELS` and `HF_REVISIONS`
(and the shard dictionaries they reference). The Nix side cannot import Python, so this script
serializes those tables into `nix/registry.nix` once and the module validates against that file.
The CLI reads the same tables from `setup.py` inside the package, so the drift check keeps the two
honest.

    python3 tools/make_registry.py --out nix/registry.nix      # regenerate (the just recipe)
    python3 tools/make_registry.py --check                     # non-zero when the committed file is stale

`--check` is byte-for-byte: running the generator twice writes the same bytes, so a committed file
that differs from a fresh generation fails. `tools/test_registry_drift.py` wraps `--check` for the
test suite. This file does not edit the repo's `justfile`; if a recipe is wanted, add:

    # Regenerate nix/registry.nix from setup.py's tables (fails if the committed copy is stale)
    registry:
        python3 tools/make_registry.py --out nix/registry.nix
        python3 tools/make_registry.py --check

The output is wider than a (model, quant) pair on purpose: it carries everything the engine's
`args` template needs — `name` for `model_name`, `profile` for `--expert-profile`, `packArgs` for
the packer, `mmproj` for vision — so the module never has to guess a family's quirks. `file` is the
family default with a per-quant override only where `MODELS` carries one (UD-IQ4_XS has three
shards, not four; `setup.py:model_file`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402  (the repo's setup.py; its tables are the source of truth)

# The window the rope rule is measured against (setup.py's trained default, `derived_factor`).
TRAINED_CONTEXT = 262144
# setup.py's own default family set for a quant that does not name its families (`setup.py:3871`).
DEFAULT_FAMILIES = ("qwen", "swift")


def _nix_str(s: str) -> str:
    """A Nix double-quoted string. The values here are repo names and file patterns: basic escaping."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _nix_str_list(items: list[str]) -> str:
    if not items:
        return "[ ]"
    return "[ " + " ".join(_nix_str(x) for x in items) + " ]"


def _repo_of(url: str) -> str:
    """The Hugging Face repository that a family's `hf(...)` URL (or `mmproj_hf`) points at.

    hf(repo) is `<endpoint>/<repo>/resolve/<revision>/`; the family's `hf` appends its quant subdir
    and `mmproj_hf` is the bare folder. Matching the pinned revision beats a regex that would have
    to assume the endpoint (HF_ENDPOINT may be a mirror)."""
    for repo in setup.HF_REVISIONS:
        if url.startswith(setup.hf(repo)):
            return repo
    raise SystemExit(f"make_registry: no HF_REVISIONS entry matches {url!r}")


def _subdir(fam: dict, repo: str) -> str | None:
    """The quant subdirectory inside a repository, or None when the shards sit at its root.

    qwen/coder/unsloth append `{q}/`; Swift's files are at the root (`setup.py:186-215`)."""
    suffix = fam["hf"][len(setup.hf(repo)):]
    return suffix or None


def _quant_entry(fam: dict, model: str) -> dict:
    """One quant's row. `file` is added only where the quant overrides the family's pattern."""
    meta = setup.MODELS[model]
    # MODELS[m]["vision"] wins; else the family's. `False` forces vision off (setup.py:3996-3999),
    # so "vision allowed" is every value except an explicit False (None = setup asks).
    vision = meta.get("vision", fam.get("vision"))
    entry = {
        "shards": setup.model_shards(fam, model),
        "downloadGb": meta["download_gb"],
        "vision": vision is not False,
        "experimental": bool(meta.get("experimental", False)),
    }
    if "file" in meta:                       # UD-IQ4_XS: three shards, not four (#621)
        entry["file"] = meta["file"]
    return entry


def build_registry() -> dict:
    """The whole registry, keyed by Hugging Face repository. Deterministic (no clocks, no sets)."""
    registry: dict[str, dict] = {}
    for family, fam in setup.FAMILIES.items():
        repo = _repo_of(fam["hf"])
        quants = {model: _quant_entry(fam, model)
                  for model, meta in setup.MODELS.items()
                  if family in meta.get("families", DEFAULT_FAMILIES)}
        if not quants:
            continue                          # a family with no quant in this checkout: skip it
        registry[repo] = {
            "family": family,
            "name": fam["name"],
            "revision": setup.HF_REVISIONS[repo],
            "subdir": _subdir(fam, repo),
            "file": fam["file"],
            "profile": fam.get("profile"),
            "packArgs": list(fam.get("pack_args", [])),
            "trainedContext": TRAINED_CONTEXT,
            "mmproj": {"repo": _repo_of(fam["mmproj_hf"]), "file": fam["mmproj"]},
            "quants": quants,
        }
    return registry


def render(registry: dict) -> str:
    """The Nix text. Sorted keys everywhere so two runs are byte-identical."""
    L: list[str] = []
    L.append("# Generated by tools/make_registry.py from setup.py's FAMILIES, MODELS and HF_REVISIONS.")
    L.append("# Do not edit by hand: run `python3 tools/make_registry.py` (the drift check fails otherwise).")
    L.append("{")
    for repo in sorted(registry):
        m = registry[repo]
        L.append(f"  {_nix_str(repo)} = {{")
        L.append(f"    family = {_nix_str(m['family'])};")
        L.append(f"    name = {_nix_str(m['name'])};")
        L.append(f"    revision = {_nix_str(m['revision'])};")
        L.append(f"    subdir = {_nix_str(m['subdir']) if m['subdir'] else 'null'};")
        L.append(f"    file = {_nix_str(m['file'])};")
        L.append(f"    profile = {_nix_str(m['profile']) if m['profile'] else 'null'};")
        L.append(f"    packArgs = {_nix_str_list(m['packArgs'])};")
        L.append(f"    trainedContext = {m['trainedContext']};")
        L.append("    mmproj = {")
        L.append(f"      repo = {_nix_str(m['mmproj']['repo'])};")
        L.append(f"      file = {_nix_str(m['mmproj']['file'])};")
        L.append("    };")
        L.append("    quants = {")
        for quant in sorted(m["quants"]):
            e = m["quants"][quant]
            L.append(f"      {_nix_str(quant)} = {{")
            L.append(f"        shards = {e['shards']};")
            L.append(f"        downloadGb = {float(e['downloadGb'])!r};")
            L.append(f"        vision = {'true' if e['vision'] else 'false'};")
            L.append(f"        experimental = {'true' if e['experimental'] else 'false'};")
            if "file" in e:
                L.append(f"        file = {_nix_str(e['file'])};")
            L.append("      };")
        L.append("    };")
        L.append("  };")
    L.append("}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="where to write (default: <repo>/nix/registry.nix)")
    ap.add_argument("--check", action="store_true",
                    help="compare --out with a fresh generation; exit non-zero when it differs")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else ROOT / "nix" / "registry.nix"
    text = render(build_registry())
    if a.check:
        if not out.is_file():
            print(f"make_registry: {out} is missing", file=sys.stderr)
            return 1
        if out.read_text(encoding="utf-8") != text:
            print(f"make_registry: {out} is stale; run `python3 tools/make_registry.py --out {out}`",
                  file=sys.stderr)
            return 1
        print(f"make_registry: {out} is up to date")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"make_registry: wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
