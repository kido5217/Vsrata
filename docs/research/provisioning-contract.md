# The provisioning contract the `vsrata` CLI must implement (ticket #25)

Research date: 2026-10-06. Repo: `kido5217/Vsrata`, branch `main` @ `a8cc3d1` (a copy of `Niko1221/Strata`
v0.1.39, `CMakeLists.txt:11`). Method: reading `setup.py`, `tools/*.py`, `serve/server.py`, `docs/DETAILS.md`
and the gitignored fixture tree already on disk; the only executed commands read GGUF headers and fixture
directories. **Nothing was downloaded, built, or started**, per this ticket's constraints.

This generalizes ticket #18's `qwen/Q2_0` chain to every family and quant and adds the vision case. Where a
prior findings doc disagrees with the code, the code wins and this doc says so.

## TL;DR

- **Five loadable artifacts, not one GGUF.** For a text model the engine needs: (1) a **pack** directory
  (`--pack`) holding at least `index.txt`; either `native_experts.txt` + `dense.bin` (a *native* pack) or
  `dense.bin`/`embd.bin`/`experts.bin` (a *canonical* pack); plus `tokenizer/`; (2) the GGUF shard 1 as
  `--native`; (3) the shard carrying `per_layer_token_embd.weight` as `--ple-gguf` (2-shard files only);
  (4) the MTP draft runtime `mtp/rt/` as `--mtp`; (5) `data/expert-profile.bin` (or `-coder.bin`) as
  `--expert-profile`. Vision adds a sixth: an `mmproj-*.gguf` plus a `strata-vision` encoder binary in the
  config's `vision` block.
- **The pack branch is not one chain.** `setup.py` runs **either** the AVX-512 Q2_0 canonical chain
  (`strata_pack build` → `pack_index`, `setup.py:4175-4184`) **or** a plain `iq_pack` (`setup.py:4185-4189`) —
  they are an `if`/`elif`, mutually exclusive. `iq_pack` is only ever run *after* the canonical chain in the
  low-RAM `--experts-bin` step, and only when the pack has no `experts.bin` (`setup.py:4190-4193`). The
  ticket's premise "`strata_pack build` → `pack_index` → `iq_pack`" describes the fixture's *manual*
  `--base` optimization (`docs/research/fixtures-models.md` §3), not what setup does.
- **The canonical branch requires AVX-512**, because a canonical pack has no `native_experts.txt` and the
  engine then reads the pack's experts and hard-requires AVX-512-VNNI/VBMI (`expert_layout.cpp`; recorded in
  `fixtures-models.md` §5.1). On a non-AVX-512 CPU the branch is plain `iq_pack`, which writes the native
  pack the engine actually loads.
- **Plain `iq_pack` is self-contained.** With no `--base` it builds `dense.bin` from the model itself
  (`iq_pack.py:274-360`). `--base` only hard-links a canonical pack's `dense.bin` and copies its `tokenizer/`
  (`iq_pack.py:418-425`); it is an optimization, never a requirement. I verified against the on-disk Q2_0
  shard 1 that `iq_pack` standalone has **0 tensors needing `--compat-bf16`** (all 303 quantized non-expert
  tensors are served natively by the engine).
- **`iq_pack` also does the tokenizer.** It calls `strata_tokenizer.py` itself when `tokenizer/vocab.json`
  or `tokenizer/chat_template.jinja` is missing (`iq_pack.py:565-567`), so the tokenizer step is not a
  separate branch — except in the canonical Q2_0 AVX-512 path, where `setup.py` calls it explicitly
  (`setup.py:4182-4184`).
- **Completion gates are file-existence checks**, and most pack writers are non-atomic, so a killed process
  can leave a file that *passes* the gate (`index.txt`, `experts.bin`, `rt/experts.bin`). The native pack's
  `native_experts.txt` is the one marker written atomically and published last (`iq_pack.py:569-575`).
- **The engine explicitly refuses types it has no GPU expert kernels for** (`generate.cpp:2003-2013`,
  `native_expert_supported`, `iq_kernels.cu:1863`). Issue #22's `iq_parity` failure is a `native_mmvq` defect
  on IQ2_S/IQ3_S; it affects **serving**, not provisioning (see §9).

---

## 1. The download set, per family and quant

Every download URL is `hf(repo) + [subdir/] filename`, where `hf()` pins a revision
(`setup.py:76-84`, `HF_REVISIONS` at `setup.py:65-70`). `HF_ENDPOINT` (e.g. an HF mirror) replaces the host
but not the pin or the checks (`setup.py:76-79`).

| Family (`--family`) | HF repo (pinned revision) | Quant subdir | File name pattern | Shards |
|---|---|---|---|---|
| `qwen` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09` | `{q}/` | `Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf` | 2 |
| `swift` | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `b22d729eae29b5796f76fb70f91aef549b9fc52c` | — | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf` | 2 |
| `coder` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` @ `5348543e0147355ac9cbcb031184a3546350988e` | `{q}/` | `Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf` | 2 |
| `unsloth` | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `38bb39ee97821de2c9009abb7e93950eec396e66` | `{q}/` | `Qwen3.8-Flash-Next-{q}-0000{i}-of-0000N.gguf` | 4 (UD-Q4_K_XL) / 3 (UD-IQ4_XS) |

Sources: `FAMILIES` (`setup.py:183-217`), `HF_REVISIONS` (`setup.py:65-70`), `model_file`/`model_shards`
(`setup.py:817-823`). The qwen/coder/unsloth repos store files in per-quant subdirectories (`hf(...) + "{q}/"`,
lines 186, 201, 211); Swift's files are at the repository root (line 192).

| `--model` (quant) | Families it exists for | Shards | Download size | Notes |
|---|---|---:|---:|---|
| `Q2_0` | `qwen` | 2 | 66.4 GB | canonical AVX-512 branch eligible |
| `IQ2_XS` | `qwen`, `swift` | 2 | 68.0 GB | |
| `IQ3_XXS` | `qwen`, `swift` | 2 | 75.8 GB | setup's default when RAM ≥ 60 GB |
| `IQ3_S` | `qwen` | 2 | 83.6 GB | Swift 1.5 has no IQ3_S |
| `IQ1_M` | `coder` | 2 | 58.4 GB | 256 of 512 experts; coder profile |
| `UD-Q4_K_XL` | `unsloth` | 4 | 111.3 GB | **experimental**, `budget`, `nvidia_only`; **no vision** |
| `UD-IQ4_XS` | `unsloth` | 3 | 93.7 GB | `budget`; **vision on**; engine ≥ 0.1.38 |

Sources: `MODELS` (`setup.py:122-152`); the family filter is `MODELS[m].get("families", ("qwen","swift"))`
(`setup.py:3871`), which is why `Q2_0`/`IQ3_S` are qwen-only and `IQ1_M` is coder-only (comments at lines
123-124, 130-133). `UD-IQ4_XS` overrides the family's 4-shard pattern with its own 3-shard `file`
(`setup.py:150`) and overrides `vision: False` with `vision: True` (`setup.py:151`, `setup.py:3996`).

**Exact per-shard bytes** are only pinned in code for the Unsloth files (`UNSLOTH_SHARDS` bytes+SHA-256,
`setup.py:155-173`; `UNSLOTH_IQ4_XS_SHARDS`, `setup.py:166-173`) and were measured on this host for Q2_0
(37,623,740,192 + 28,800,138,432 bytes; `fixtures-models.md` §1). The GSQ-RCO repo has no per-shard size
table in-tree — provisioning must rely on the Hub's own `Content-Length` and the whole-file check
(`check_shards`, `setup.py:1233-1251`), which reads each shard's tensor directory and refuses a truncated
file. Unsloth additionally checks size **and** SHA-256 after download (`verify_sha256`, `setup.py:1186-1208`).

**Shard reuse.** Setup's own comment says the original's shard 2 (the 28.8 GB n-gram table) is the same file
for all its sizes and the Coder; when a sibling shard 2 is already downloaded it hard-links it instead of
re-downloading (`setup.py:4145-4154`), for `family in ("qwen","coder")` only. I verified the bytes only for
Q2_0 (the on-disk pair).

**Vision mmproj.** The mmproj is downloaded only when `vision != "none"` (`setup.py:4164-4169`):

| Family | `mmproj_hf` repo (same pin as the model) | File |
|---|---|---|
| `qwen` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` |
| `swift` | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf` |
| `coder` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (the original's file) |
| `unsloth` (UD-IQ4_XS only) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (the original's file) |

Sources: `setup.py:186-215`; the mmproj is ~0.9 GB by setup's own on-screen text (`setup.py:4006`) and was
never downloaded in the fixture tree (`fixtures-models.md` §1). It lands at `<models-dir>/<mmproj>` where
`<models-dir>` = `<data>/models` (not the per-model subdir) — `setup.py:4161`.

**Model directories.** `tag = fam["tag"] + model` (`setup.py:3936`): the model GGUFs go in
`<data>/models/<tag>` (`setup.py:4061`, `<data>` defaults to `Strata-data/` beside the repo,
`setup.py:2635`). The tag is case-preserving (`Q2_0`, `swift-IQ3_XXS`, `coder-IQ1_M`,
`unsloth-UD-IQ4_XS`), while pack dirs are lowercased (`<data>/packs/<tag.lower()>`, `setup.py:4173`). The
dev fixture tree uses a different, hand-made layout (`models/qwen-Q2_0/`), so don't copy its paths.

**MTP source repo.** `mtp_fetch.py` pins `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540`
(`mtp_fetch.py:32`); `STRATA_MTP_REVISION` overrides it (`mtp_fetch.py:33`). The MTP head is absent from the
GGUF release and is range-fetched from the BF16 checkpoint (31 `mtp.*` tensors, ~5 GB; `mtp_fetch.py:1-17`).

---

## 2. The pack build branch and its gates

`setup.py` step 6 (`setup.py:4171-4214`). Two mutually exclusive branches on the same `<data>/packs/<tag>`
directory:

### Branch A — canonical pack (only `Q2_0` + `qwen` + AVX-512)

Condition: `model == "Q2_0" and avx512 and family == "qwen"` (`setup.py:4175`; `avx512` is the full
F+BW+VL+VNNI+VBMI test, `setup.py:403-422`).

```
python tools/strata_pack.py build --gguf <shard1> --out <pack> --skip-hash
python tools/pack_index.py --pack <pack>                     # writes <pack>/index.txt
python tools/strata_tokenizer.py --gguf <shard1> --out <pack>  # only if <pack>/tokenizer/vocab.json missing
```
(`setup.py:4179-4184`)

- **Gate:** `not (pack/"index.txt").exists() or not (pack/"experts.bin").exists()` — if either is missing,
  both commands rerun (`setup.py:4177`). The comment "index.txt is written last" refers to `pack_index` being
  the last command, not to atomicity.
- Output: `experts.bin` (34 GB Q2_0 arena), `dense.bin`, `embd.bin`, `experts.json`, `manifest.json`
  (`strata_pack.py:206-279`, `pack_layer.py:68-115`), then `index.txt` (`pack_index.py:99-256`). No
  `native_experts.txt` is written, so the engine uses the canonical expert layout and hard-requires AVX-512.
- `strata_pack build` reads a sibling `…-00002-of-00002.gguf` derived from the shard-1 name
  (`strata_pack.py:227`); it does not copy it.

### Branch B — native pack (everything else)

Condition: `not (pack/"native_experts.txt").exists() or not (pack/"tokenizer"/"vocab.json").exists()`
(`setup.py:4185`).

```
python tools/iq_pack.py --gguf <shard1> --out <pack> [<family pack_args>]
```
(`setup.py:4188-4189`). `unsloth` passes `--compat-bf16` (`setup.py:215`).

- **Gate:** `native_experts.txt` **and** `tokenizer/vocab.json` (`setup.py:4185`). `native_experts.txt` is
  written to a temporary name and renamed only when every layer is in, and it is published *last*
  (`iq_pack.py:569-575`), so it is the reliable completion marker. `tokenizer/vocab.json` is the tokenizer
  gate; `iq_pack` also rewrites the tokenizer when `chat_template.jinja` alone is missing (`iq_pack.py:565`).
- Output: `native_experts.txt`, `index.txt`, `dense.bin`, `conversions.json`, and `tokenizer/`
  (via the embedded `strata_tokenizer` call). `experts.bin` is **not** written unless `--experts-bin`. This is
  **not a small step**: the standalone `dense.bin` on Q2_0 is ~1.5 GB (the float tensors, as stored), so a CLI
  must budget disk and time for it.
- `iq_pack --base <canonical>` is *optional*: it hard-links the canonical `dense.bin`, copies its
  `tokenizer/`, and writes `extra.bin` for tensors float in the model but quantized in the base
  (`iq_pack.py:379-426`). Setup never passes `--base`.
- **Verified on this host (reviewer's run):** standalone `iq_pack` on the Q2_0 shard 1 printed
  `index.txt: 1079 tensors, 303 served natively, 0 in extra.bin, arena 1.38 GiB`, wrote `native_experts.txt`
  (3,286 B), `index.txt`, `conversions.json`, the embedded `tokenizer/`, and a 1.49 GB `dense.bin` — no
  `--base`, no `--experts-bin`. It is genuinely self-contained, which is what makes it the right branch for a
  non-AVX-512 host.

### The low-RAM step (`--experts-bin`)

```
python tools/iq_pack.py --gguf <shard1> --out <pack> --experts-bin
```
when `low_ram and not (pack/"experts.bin").exists()` (`setup.py:4190-4193`). **Gate:** `pack/experts.bin`.
`--experts-bin` writes `experts.bin` (per-layer raw GGUF expert slices, with an `experts.bin.src.json`
sidecar) atomically and a matching sidecar; a same-size `experts.bin` is reused only when the sidecar names
this model and six sampled blobs match (`iq_pack.py:525-547`, `583-597`). This is the file the engine's
low-RAM `--resident-experts`/`--mmap-experts` modes read (`setup.py:4236-4237`).

Note the collision: in Branch A `pack/experts.bin` is the *canonical arena*; in Branch B with low RAM it is
the *native blob file*. Same name, different content; the CLI must know which branch created it.

### Canonical vs native, one line each

| | Canonical (Branch A) | Native (Branch B) |
|---|---|---|
| When | `qwen/Q2_0` on an AVX-512 CPU | every other family/quant, and Q2_0 on any non-AVX-512 CPU |
| Experts read from | pack `experts.bin` | the `--native` GGUF, offsets in `native_experts.txt` |
| Marker | `index.txt` (+ `experts.bin`) | `native_experts.txt` (+ `tokenizer/vocab.json`) |
| Extra engine flags | none beyond setup's base | `--native SHARD1`, `--spec T≥2`, `--prefill CHUNK` (mandatory, `generate.cpp:2164-2170`) |
| Disk cost | ~40 GB one-time conversion (`setup.py:4176-4178`) | ~1.5 GB `dense.bin` + tokenizer, seconds (`iq_pack.py:302-332`) |

Sources: `setup.py:4175-4193`; `iq_pack.py` docstring lines 1-38; `generate.cpp:2164-2170`;
`fixtures-models.md` §5.1-5.2.

---

## 3. The MTP chain

`setup.py:4195-4214`. The MTP root is found by locating `mtp/rt/experts.bin` under the data roots and taking
its grandparent (`setup.py:4195-4196`).

```
python tools/mtp_fetch.py fetch --out <data>/mtp        # 31 tensors + mtp-manifest.json, resumable
python tools/mtp_pack.py  --src <data>/mtp --experts q2_0 --out <data>/mtp/mtp-q2_0.gguf
python tools/mtp_rt.py    --gguf <data>/mtp/mtp-q2_0.gguf --out <data>/mtp/rt
```
(`setup.py:4201-4208`). **Gate:** `mtp/rt/experts.bin` (`setup.py:4201`). The chain reruns when that file is
absent **or** when `mtp_corrupt()` returns true — `mtp_fetch.py verify` exits 3 if a fetched tensor is
missing/corrupt (`setup.py:3355-3363`, `mtp_fetch.py:259-289, 298-302`).

- `mtp_fetch fetch` is **resumable per tensor** (appends from the existing file's size, retries the whole
  tensor once on SHA-256 mismatch; `mtp_fetch.py:207-256`). `inventory` writes `mtp-inventory.json/.md`.
- `mtp_pack` always uses `--experts q2_0` in setup (`setup.py:4205`), regardless of the model quant;
  `q2_0`/`q4_0`/`q8_0` are the alternatives (`mtp_pack.py:129-131`). Output: `mtp-q2_0.gguf` +
  `<out>.report.json`.
- `mtp_rt` writes `experts.bin`, `dense.bin`, `dense.txt` (`mtp_rt.py:1-13, 61-107`). It writes
  `experts.bin` **first and non-atomically** (`mtp_rt.py:73-75`), so the gate `rt/experts.bin` can pass on a
  truncated file — see §7.
- **Draft vocab.** `refresh_draft_vocab(rt, choice)` copies a committed subset from the checkout's `data/`
  to `rt/draft_vocab.bin` (`setup.py:3366-3385`), choices `cjk` (default, `data/draft_vocab.bin`, 106,299
  ids), `en`, `cyrillic`, `fr` (`DRAFT_VOCABS`, `setup.py:3202-3203`). The files are tracked
  (`data/draft_vocab*.bin`). `mtp_rt` never writes a draft vocab (`fixtures-models.md` §5.4); the copy is the
  only producer. `serve/server.py` never reads the config's `draft_vocab` key (verified: zero occurrences in
  `server.py`); it is a setup-side selector only.

---

## 4. The tokenizer step

`strata_tokenizer.py` writes into `<--out>/tokenizer/` (`strata_tokenizer.py:270-298`):

```
vocab.json  merges.txt  token_type.json  tokenizer.json  [chat_template.jinja]
```

- `--gguf` must be **shard 1** (the metadata shard); `--out` must be the **pack directory**, not the
  tokenizer directory. The tool appends `tokenizer/`, so `--out <pack>/tokenizer` produces the nested
  `tokenizer/tokenizer/` (`fixtures-models.md` §5.8).
- The layout is **identical for every family**; only `chat_template.jinja`'s content differs (read from the
  GGUF's `tokenizer.chat_template`; "fine-tunes change it (Swift 1.5 differs from Qwen3.8-Flash-Next's)",
  `strata_tokenizer.py:292-297`).
- Where it must land for the runtime: the config's `tokenizer` key is `<pack>/tokenizer`
  (`setup.py:4324`). `serve/server.py` reads `cfg["tokenizer"]` and requires `vocab.json` before a
  `strata`-engine start, then reads `merges.txt` and `token_type.json` (`server.py:3946-3960`);
  `chat_template.jinja` is optional (a fallback template is used, `server.py:4014-4015`). The engine's
  `--tokenizer` default is `<repo>/pack/full/tokenizer` (`server.py:3912`) but the config always overrides it.
- `strata_tokenizer.py --check` round-trips a corpus; it is a diagnostic, not a gate, and writes nothing
  additional.

---

## 5. The engine/server runtime contract

### 5.1 The `args` setup writes (`setup.py:4224-4323`)

Base (always):

```
--pack <data>/packs/<tag.lower()>
--native <models-dir>/<shard 1>
[--ple-gguf <shard with per_layer_token_embd.weight>]     # only when len(shards) <= 2
--expert-profile <repo>/data/<expert-profile[.bin] | expert-profile-coder.bin>
--expert-cache auto
--prefill auto
--spec 4
--spec-min-p 0.5
--mtp <data>/mtp/rt
--max-context <ctx>
```
(`setup.py:4224-4227`). `--ple-gguf` is shard 2 for qwen/coder and shard 1 for Swift — the shard is
found by scanning for `per_layer_token_embd.weight` (`setup.py:4219-4220`); for the 3- and 4-shard Unsloth
files it is omitted and the engine finds the table itself
(`setup.py:4223`).

Conditional flags: `--rope-scaling/--rope-scale` past 262144 (`setup.py:4228-4229`); `--kv int8|q4_0|k8v4`
when ctx > 8192 (`4230-4231`); `--resident-experts` or `--mmap-experts` in the low-RAM mode (`4236-4237`);
`--ple-io ram` on a rotational disk when the RAM fits (`4238-4246`); `--kv-resident 32768` from 64K up when
it fits (`4270-4278`); `--resident-budget-gib N` for the Unsloth budget models (`4293-4294`, no layer
split); `--vision --vram-reserve-mib 700` when vision is on (`4295-4296`); an explicit `--vram-reserve-mib`
(`4301-4309`); the experimental control vector (`4320-4323`).

Engine parse sites (all verified in `src/program/generate.cpp`): `--pack` 1263, `--max-context` 1283,
`--ple-gguf` 1311, `--ple-io` 1313, `--kv` 1318, `--native` 1336, `--expert-cache` 1365, `--prefill` 1379,
`--spec` 1395, `--mtp` 1402, `--vision` 1415, `--expert-profile` 1485, `--mmap-experts` 1490,
`--resident-experts` 1493, `--resident-budget-gib` 1503. Unknown flags are a hard error (`generate.cpp`
parse table; recorded in `runtime-contract.md` §1.2).

**Two engine-side constraints a CLI must honor:**

1. A **native pack** requires `--native SHARD1`, `--spec T` with `T >= 2`, and `--prefill CHUNK`
   (`generate.cpp:2164-2170`) — exactly what setup's base args provide.
2. A **layer split** (multi-GPU) requires `--expert-profile` (`generate.cpp:1629-1631`); setup always writes
   one.

### 5.2 The config JSON keys `serve/server.py` reads

The base object setup writes (`setup.py:4324-4326`): `exe`, `args`, `cwd`, `tokenizer`, `model_name`, `log`,
`lib_dirs`, `port`; plus, conditionally, `cuda` (`4328`), `backend`/`env` (AMD, `4330-4338`), `gpu`/
`gpus_asked` (`4340-4341`), `layer_split` (`4344`), `host` (`4348`), `api_key` (`4350`), `draft_vocab`
(`4352`), `open_browser` (`4354`), `parallel` (`4359-4364`), and `vision` (`4369-4372`).

The exhaustive reader table is `docs/research/runtime-contract.md` §1.1-1.3; I re-verified the keys on the
critical load path in this checkout:

| Key | What `server.py` does with it |
|---|---|
| `args` | `engine_args(cfg)` copies it, then appends `--layer-split`, `--split-skip-if-fits`, `--vram-elastic`/`--vram-segment-mib`, `--batch N` from `parallel`, and `--expert-profile-save[-every]` (`server.py:1461-1465, 1466-1541`) |
| `exe` | engine binary; a relative path is resolved against `cwd` (`server.py:3990-3992`) |
| `cwd` | the engine child's working directory and the base for relative `vision`/`expert_profile_save` paths (`server.py:4001, 1536, 3977-3978`) |
| `tokenizer` | overrides `--tokenizer`; `vocab.json` required, `merges.txt`/`token_type.json` read (`server.py:3946-3960`) |
| `log` | engine stderr and the vision encoder's log (`server.py:4001, 3980`) |
| `lib_dirs` | existing dirs prepended to `LD_LIBRARY_PATH`/`PATH` in the engine child's env (`server.py:1571-1574`) |
| `model_name` | the id `/v1/models` reports (`server.py:4016`) |
| `port` | **not read by the server** — setup and the run scripts pass `--port` (`setup.py:3093-3094`); the server's own default is 8095 (`server.py:3909`) |
| `gpu`, `layer_split`, `gpus_asked`, `backend`, `hip_ordinal` | device selection and split (`server.py:1382-1389, 1428-1458, 1544-1568`) |
| `host`, `api_key` | bind address and auth (`server.py:3940, 4031`) |
| `vision` | starts the vision encoder (`server.py:3974-3981`) |
| `draft_vocab` | **never read by the server** (zero occurrences); setup uses it to pick `rt/draft_vocab.bin` |
| user keys: `sampling`, `mcp_servers`/`mcpServers`/`mcp`, `aliases`, `fit_max_tokens`, `idle_unload_s`, `min_free_vram_mib`, `before_load`, `api_monitor`, `cors_origins`, `trusted_origins`, `allowed_hosts`, `lazy_load`, `engine_silence_s`, `repeat_stop_tokens`, `anthropic_thinking`, `reasoning_budget_tokens`, `effort_position`, `split_skip_if_fits`, `vram_elastic`, `vram_segment_mib`, `expert_profile_save`, `expert_profile_save_every`, `open_browser`, `parallel` | read at `main()` start; `runtime-contract.md` §1.3. (`remote_expert_opt` is setup-side only, `setup.py:875`; not read by the server.) |

`SETUP_KEYS` is exactly the set setup claims and re-writes on later runs; user keys survive via `carry_over`
(`setup.py:2694-2726`).

### 5.3 The exact data-dir layout

```
<data>/
  models/<tag>/<shard>-0000i-of-0000N.gguf[.part][.done]   # tag = family tag + quant, case-preserved
  models/mmproj-<...>.gguf                                  # shared at models/ root, vision only
  packs/<tag.lower()>/
    {native_experts.txt,index.txt,dense.bin[,extra.bin,conversions.json]}   # native pack
    | {experts.bin,dense.bin,embd.bin,experts.json,manifest.json,index.txt} # canonical pack
    tokenizer/{vocab.json,merges.txt,token_type.json,tokenizer.json,chat_template.jinja}
    experts.bin                                             # low-RAM copy (native layout)
  mtp/
    tensors/<tensor>.bin  mtp-manifest.json  mtp-inventory.json  mtp-q2_0.gguf[.report.json]
    rt/{experts.bin,dense.bin,dense.txt,draft_vocab.bin}
```
Sources: `setup.py:4061, 4161, 4173, 4190-4196`; `iq_pack.py:1-24`; `strata_pack.py:236-277`;
`mtp_fetch.py`/`mtp_pack.py`/`mtp_rt.py`; confirmed against the on-disk fixture tree
(`pack/q2_0/{index.txt,native_experts.txt,tokenizer/}`, `pack/full/{experts.bin,dense.bin,embd.bin,
manifest.json}`, `mtp/rt/{experts.bin,dense.bin,dense.txt,draft_vocab.bin}`).

---

## 6. The vision case, end to end

1. **Whether vision is offered.** `vision = MODELS[model].get("vision", fam.get("vision"))`; if it is `False`,
   vision is forced off (`setup.py:3996-3999`). That makes `unsloth`/`UD-Q4_K_XL` off (family
   `"vision": False`, `setup.py:215`) and `unsloth`/`UD-IQ4_XS` on (model `"vision": True`, `setup.py:151`).
   `qwen`, `swift`, `coder` have no `vision` key, so images are allowed (asked interactively, or
   `--vision yes|no|gpu|cpu`; `setup.py:4000-4008`).
2. **Download.** `mmproj` is resolved at `<data>/models/<fam["mmproj"]>` and, if missing, downloaded from
   `fam["mmproj_hf"] + fam["mmproj"]` (`setup.py:4161-4169`); see §1's mmproj table. A file already present
   in `--gguf-dir` is used instead (`setup.py:4165-4166`).
3. **Build.** The vision encoder binary `strata-vision`/`.exe` is part of the engine build
   (`VEXE`, `setup.py:227`); it must exist for a prebuilt engine or it is compiled (`setup.py:4112-4116`).
4. **Config block** (`setup.py:4366-4372`):

   ```json
   "vision": {"exe": "<engine>/strata-vision", "mmproj": "<data>/models/mmproj-….gguf",
              "model": "<shard 1>", "gpu": true, "max_tokens": 1024}
   ```
   `gpu` is `vision == "gpu"`; `threads` is added only for the CPU encoder; `max_tokens` defaults to 1024
   (GPU) or 300 (CPU) (`VISION`, `setup.py:224-225`, `vision_tokens`, `setup.py:3216-3238`). Setup also
   appends `--vision --vram-reserve-mib 700` to the engine `args` (`setup.py:4295-4296`).
5. **Runtime.** `server.py` builds a `Vision` from the block, resolving relative `exe`/`mmproj`/`model`
   against `cwd` (`server.py:3974-3981`), and spawns `strata-vision --mmproj … --model … [--gpu]
   [--threads N] [--max-tokens N]` (`server.py:1262-1290`). `lazy_load` is refused with vision
   (`server.py:3972-3973`).

The `vision` block is **not** part of the pack; it points at a separate mmproj file and a separate binary.

---

## 7. Completion gates, resumption, and atomicity

| Stage | Gate (file whose presence means done) | Resumable? | Atomic? |
|---|---|---|---|
| GGUF shard download | `<shard>.done` sidecar (`setup.py:291-293`), plus `check_shards` whole-file test (`setup.py:1233-1251`) | **yes**, HTTP Range + `.part` (`setup.py:1041-1098`) | rename `.part`→final on completion |
| Unsloth shard | `.done` also carries `sha256 <hash>` (`setup.py:1186-1208`) | yes | as above |
| Canonical pack build | `pack/index.txt` **and** `pack/experts.bin` (`setup.py:4177`) | **no** — rewrites | `experts.bin` non-atomic (`pack_layer.py:76`); `manifest.json` last (`strata_pack.py:277`) |
| `pack_index` | `pack/index.txt` (`setup.py:4177`) | no | **non-atomic** write (`pack_index.py:239`) |
| Tokenizer | `pack/tokenizer/vocab.json` (`setup.py:4182, 4185`); `iq_pack` also checks `chat_template.jinja` (`iq_pack.py:565`) | no | non-atomic |
| Native pack (`iq_pack`) | `pack/native_experts.txt` (`setup.py:4185`) | no; `--experts-bin` reuse via sidecar (`iq_pack.py:525-547`) | **yes for the marker**: written `.tmp`→rename last (`iq_pack.py:569-575`); `dense.bin`/`index.txt` published together after it (`iq_pack.py:340-346`) |
| Low-RAM `experts.bin` | `pack/experts.bin` (`setup.py:4190`) | reuse if sidecar+blobs match (`iq_pack.py:528`) | `.tmp`→rename + sidecar (`iq_pack.py:583-596`) |
| MTP overall | `mtp/rt/experts.bin` (`setup.py:4201`), plus `mtp_fetch verify` corruption check (`setup.py:3355-3363`) | fetch per tensor; pack/rt not | fetch yes; `mtp_pack`/`mtp_rt` **non-atomic** (`mtp_rt.py:73-75`) |
| Draft vocab | `rt/draft_vocab.bin` (`setup.py:3372-3385`) | copy overwrites | **no** — direct `shutil.copyfile` (`setup.py:3385`); a killed copy self-heals on the next run |

**Pitfalls a CLI must defend against** (all follow from existence-only gates over non-atomic writers):

- A killed `pack_index` leaves a truncated `index.txt`; Branch A's gate passes because it only checks
  existence. Same for a truncated `embd.bin` or `dense.bin`. A robust CLI should validate (length/parse) or
  use its own sentinel *after* each stage.
- A killed `mtp_rt` leaves a truncated `mtp/rt/experts.bin` that passes the MTP gate; `mtp_corrupt` only
  re-verifies the *fetched tensors*, never the built runtime.
- A killed `refresh_draft_vocab` leaves a truncated `rt/draft_vocab.bin`; it is not itself a gate (the MTP gate
  is `rt/experts.bin`), and the next run overwrites it, so the impact is low.
- `iq_pack` removes `native_experts.txt` before republishing, so a native-pack rerun correctly re-triggers;
  Branch A republishes `index.txt` non-atomically, so it does not.
- The GGUF download's `.done` is the only reason a re-run skips the HEAD. Deleting the mark but keeping a file
  of the right size does **not** re-download: the HEAD yields `Content-Length`, and when the file matches it
  setup re-marks it and returns (`setup.py:1061-1064`). A wrong-size file with no `.part` restarts from zero.

---

## 8. Writes outside `--out` / `--data-dir` (Nix implications)

**The tools are clean.** `strata_pack`, `pack_index`, `iq_pack`, `strata_tokenizer`, `draft_vocab`,
`mtp_fetch`, `mtp_pack`, `mtp_rt` all write only under their `--out`/`--pack`/`--data-dir`. Their temp files
(`*.tmp`, `*.part`, `.report.json`, sidecars) sit beside the target. I found no tool that writes to the repo
root or `/tmp` in its production path (test harnesses do use `/tmp`, e.g. `parking_test.py:55`).

**`setup.py` itself does not.** It writes into the checkout (`ROOT`):

- the model config `strata-<tag.lower()>.json` (`setup.py:4375`, written by `write_setup_config`, atomic with
  `.bak`, `setup.py:2739-2769`);
- the run script `run-<model>.sh`/`.bat` (`write_run_script`, `setup.py:3453-3466`);
- the log path `strata-<tag.lower()>.log` (`setup.py:4325`, written by the server/engine);
- it reads the expert profiles from `<repo>/data/expert-profile[-coder].bin` and the draft vocab from
  `<repo>/data/draft_vocab*.bin` (`setup.py:4225`, `3372`), and `refresh_draft_vocab` **writes** only into the
  data dir.

So for a Nix design where the checkout is read-only:

- the config, the run script and the log must be relocated to the writable data root (the config's `log` and
  `cwd` keys can point anywhere; `cwd` only matters to the engine's relative paths);
- the expert-profile and draft-vocab **inputs** must be provisioned from the checkout's `data/` into the
  writable root (they are small, tracked files: `data/expert-profile.bin` 192 KB,
  `data/expert-profile-coder.bin` 96 KB, `data/draft_vocab*.bin` 158-425 KB);
- the `--gguf`/`--pack`/`--out`/`--data-dir` flags let every tool work entirely under the data root.

Also note `<data>` defaults to `Strata-data/` *next to* the repo, and setup will silently fall back to
`ROOT` when that is not writable (`setup.py:2636-2640`) — a CLI should pass an explicit `--data-dir`
(`setup.py:3641`) or `--models-dir` (`setup.py:3643-3644`) rather than rely on the default.

---

## 9. The MMVQ / IQ-type constraint, and issue #22

Issue #22 ("`iq_parity`: IQ2_S and IQ3_S MMVQ kernels are ~35x over tolerance") reports that the CUDA
`native_mmvq` kernel computes IQ2_S (type 22) and IQ3_S (type 21) with relative error ~0.7 against a 2e-2
threshold, while dequantization is exact. The test is CUDA-only (`src/kernels/iq_parity.cpp` includes
`<cuda_runtime.h>` and calls `native_mmvq` on device; registered only under the CUDA block,
`CMakeLists.txt:590-592, 624-630`).

**It affects serving, not provisioning.**

- **Provisioning is type-agnostic.** `iq_pack`/`strata_pack`/`pack_index` consume whatever GGUF types the
  model has; nothing in the provisioning chain runs or gates on `iq_parity`. The on-disk Q2_0 shard 1 uses
  Q2_0/Q3_K/Q4_K/Q5_K/Q4_0/Q5_0/Q6_K/Q8_0/IQ4_NL/IQ4_XS and needs no `--compat-bf16` (verified this
  session), and the `iq_parity` fixtures pipeline records rather than gates (issue #22; quoted in
  `fixtures-models.md` §7).
- **The engine accepts these types at load.** `native_expert_supported` requires `is_iq(gu) && is_iq(dn)`
  (`iq_kernels.cu:1863-1867`), and `is_iq` includes 16,17,18,20,21,22,23,29,42,11,12,13,7,6,8
  (`iq_kernels.cu:1681-1683`). `generate.cpp:2003-2013` refuses only types outside that set. So IQ3_S
  experts load; #22 is a numeric defect, not a load refusal.
- **Blast radius caveat.** The routed-expert GPU path is `native_expert_grouped`
  (`src/core/verify.cpp:1025-1026`, `iq_kernels.cu:2448`), *not* `native_mmvq`; `native_mmvq` handles the
  **shared expert** (`src/kernels/cuda/shared_expert.cu:199-208`) and dense native projections
  (`src/core/layer.cpp:151`). Issue #22's impact sentence ("gate/up/down rows") therefore names the routed
  experts, but the kernel it tests is the one used for shared/dense projections — the exact per-model blast
  radius depends on which tensors carry IQ2_S/IQ3_S, which I could not enumerate (no IQ3_S or Coder model on
  disk). The Coder's `IQ1_M` gate/up rows are documented to be IQ2_S/IQ3_XXS/IQ3_S
  (`src/kernels/cpu/iq_avx512.cpp:9-10`), so at least the Coder is candidate-affected.

For a provisioning CLI this means: **prepare every quant**, but mark the CUDA serving of any model whose
active tensors include IQ2_S/IQ3_S as unverified until #22 is fixed (`Q2_0`, and every non-IQ type, are
outside it). Provisioning itself never needs to branch on #22.

---

## 10. What I could not verify

- **No non-Q2_0 model was exercised.** The ticket asked to generalize beyond `qwen/Q2_0`; the other
  families/quants were verified by reading code and the `FAMILIES`/`MODELS` tables only, not by building a
  pack or downloading a shard. `fixtures-models.md` §7 already notes the IQ*/swift/coder/unsloth paths were
  never downloaded.
- **No per-shard byte sizes for the GSQ-RCO quants.** They are not pinned in-tree; only Q2_0
  (measured) and the Unsloth shards (pinned) have exact sizes here.
- **The mmproj** was never downloaded; its exact size (~0.9 GB) is setup's on-screen text, not a pinned
  value.
- **Which tensors of the IQ3_S/Coder models use IQ2_S/IQ3_S**, and therefore the true blast radius of
  issue #22, is not determined (those GGUFs are not on disk).
- **The canonical Q2_0 pack on a real AVX-512 CPU** was not run; Branch A's correctness rests on
  `strata_pack`'s own `verify` (documented in `fixtures-models.md` §4) which I did not re-run.
- **Whether `iq_pack --base` and standalone produce byte-identical native packs** was not tested; the
  fixture tree shows `--base` (hard-linked `dense.bin`, empty `extra.bin`, no `conversions.json`), while
  setup's Branch B uses standalone.
- **`serve/server.py`'s full config-read table** is reproduced from `runtime-contract.md` §1 (same
  checkout); I re-verified the load-path keys listed in §5.2 but did not re-audit every user key.
