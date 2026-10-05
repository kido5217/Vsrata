# What model inputs do the tests need? — the minimal test-model manifest (map #15, ticket #16)

Research date: 2026-10-05. Repo: `kido5217/Vsrata`, branch `research/fixtures-manifest` from `main` @ `f7df027`;
a copy of `Niko1221/Strata` v0.1.39 (`CMakeLists.txt` `project(strata VERSION 0.1.39)`; `MIN_ENGINE = (0, 1, 39)`
at `setup.py:118`).

Method: source reading (`setup.py`, `CMakeLists.txt`, `src/`, `tests/`, `tools/`, `serve/`, `bench/`, `data/`,
`.gitignore`, `justfile`, `docs/ORCA.md`), the existing `build/` configure output (`CMakeCache.txt`,
`CTestTestfile.cmake` — read only, no build run), running the mock-only Python suite
(`python3 tools/test_setup_*.py`, per the `justfile` `test` recipe), Hugging Face **metadata** calls
(`/api/models/<repo>`, `/api/models/<repo>/tree/<ref>?recursive=true`), one fetch of
`model.safetensors.index.json` (170 KB weight-name manifest of the Qwen checkpoint — metadata about model
files, no tensor bytes), and `gh` API queries against public `Niko1221/Strata`. Nothing was built,
`setup.py` was not run, and **no model file was downloaded**.

## TL;DR

- **Most tests need no model at all.** 37 of the 40 ctests registered in the current configuration run on
  synthetic data (every one is commented "synthetic, no model" in `CMakeLists.txt` or writes its own
  fixtures). Every `tools/test_*.py`, `serve/test_*.py` unittest is mocked or synthetic — their "golden"
  fixtures contain engine **config output** or rendered prompt text, never model data.
- **The one registered ctest that needs real model artifacts is `ple_parity`** — and it cannot pass in this
  checkout even with a full download: its ggml captures (`bench/micro/ple_in.bin` / `ple_out.bin`) and the
  tools that produce them are not in this checkout and are not in public upstream either (verified against
  `Niko1221/Strata` main, 965 paths). It also needs the real Q2_0 GGUF shard 2 (28.8 GB PLE table) and the
  real pack's `dense.bin`.
- **The minimal real-model download set is ~71.5 GB**: Q2_0 of the original model (2 shards, 66.4 GB) plus
  the 31 MTP draft tensors (~5 GB, all sha256-pinned in `tools/mtp_fetch.py`). From those, the native pack,
  tokenizer, expert profile and MTP runtime are all **generated locally** by repo tools.
- **Both provided HF tokens work** (HTTP 200 on all five repos); all five repos are **public and non-gated**,
  so anonymous access is sufficient. Each repo's HEAD still equals the revision `setup.py` pins.
- **Layout**: reuse the already-gitignored root `models/` (plus `packs/`, `mtp/`) rather than a new
  `fixtures/` tree — one paragraph in §5.

Verdict key used below: **(a)** runs with no model files · **(b)** runs on a locally generated tiny model ·
**(c)** needs real downloaded model files · **(s)** skip / not registered in this checkout.

## 1. Per-test inventory

### 1.1 Engine ctests — as registered in this checkout

Configuration on record: `build/CMakeCache.txt` shows `STRATA_ENABLE_CUDA=ON`, `STRATA_BUILD_TESTS=OFF`,
`STRATA_BUILD_CONVERSATION_TESTS=OFF`, `STRATA_NATIVE_EXPERTS=ON`. `build/CTestTestfile.cmake` lists exactly 40
registered root tests plus the `ggml` subdirectory. That is the authoritative "what `ctest` runs here" for
this ticket.

**Synthetic — no model (37 tests).** All of: `file_expert_source_test` (writes its own "synthetic experts.bin",
`tests/core/file_expert_source_test.cpp:48-72`), `expert_profile_save_test` ("CPU only", byte-for-byte vs
`make_profile.py`'s file format), and: `dequant_s2_parity`, `s2_gemv_parity`, `mmvq_multi_parity`,
`iq_multi_parity`, `gdn_rec_parity`, `gemm_bf16_parity`, `native_grouped_parity`, `shared_expert_parity`,
`gr_parity`, `gdn_parity`, `s2_gemv_q8_parity`, `sampler_parity`, `sampler_parity_one_block`,
`sampler_parity_old`, `decode_cluster_parity`, `rope_parity`, `quantize_act_parity`, `route_window_parity`,
`router_top10_parity`, `s_gemv_parity`, `elementwise_parity`, `bf16_gemv_parity`, `s_gemv_q8k_parity`,
`qsa_parity`, `qsa_topk_parity`, `kv_q8_parity`, `kv_stream_parity`, `kv_hybrid_parity`, `kv_q4_parity`,
`cvec_parity`, `s2_expert_grouped_parity`, `platform_memory_test`, `pinned_shared_test`,
`ple_reader_selftest`, `cuda_device_selftest`, and the `ggml` subdirectory tests. `CMakeLists.txt` comments
them "synthetic, no model" / "synthetic GGUF: no model, no GPU"; where I checked sources, they generate their
own data (e.g. `src/kernels/cpu/pool_test.cpp`-style blob reads are not present in these). **Verdict: (a).**

**Locally generated fixtures (2 tests).** `iq_parity_fixtures` runs `tools/iq_fixture.py` (deterministic,
seeded, synthetic i-quant blocks — no model), then `iq_parity` reads those fixtures from the build dir.
Dependency: python3 + numpy + the **vendored gguf-py** at `third_party/llama.cpp/gguf-py`
(`tools/iq_fixture.py:34` hard-codes that path), which arrives only with setup's llama.cpp code download
(pinned commit `3cf03257…`, `setup.py:93`). It is absent in this checkout, so both tests currently **skip**
(exit code 3 → `SKIP_RETURN_CODE 3`, `CMakeLists.txt:627-630`). **Verdict: (a), currently (s)** until the
llama.cpp source (code, not a model) is present.

**Real model — registered (1 test).** `ple_parity` (`CMakeLists.txt:786`): the PLE n-gram table kernel vs
ggml's captured graph. `ple_parity.cpp:250-300` makes three inputs **hard requirements** (missing → exit 2,
not a skip — "a test that passes by finding nothing to compare is the failure mode this project keeps
re-learning"):

1. the original **Q2_0 GGUF shard 2** (the 28.8 GB PLE table): `$STRATA_PLE_GGUF`, else
   `../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` relative to the source tree; the file is
   size-checked to be exactly the PLE table (`ple_parity.cpp:280-283`);
2. the real pack's **`pack/full/dense.bin`** (5.4 GB; only the PLE key/value/conv regions are read,
   `ple_parity.cpp:596-606`);
3. **`bench/micro/ple_in.bin` + `bench/micro/ple_out.bin`** — ggml graph captures. `bench/micro/` is not in
   this checkout, is gitignored (`bench/micro/*.bin`), and is not in public upstream either: `Niko1221/Strata`
   main's tree (965 paths) has no `bench/micro` and no capture tool; the CMake header comment
   (`CMakeLists.txt:87`) says the test suite and `bench/micro/` are "left out of the published source".

**Verdict: (c) — and blocked by (3) regardless of any download plan** (the capture tooling must come from
the non-public upstream; see §6).

**Not registered — real model (would be (c)).**
- `expert_arena_load` — registered **only** when `pack/full/experts.bin` exists ("the load test needs the
  34 GB artifact", `CMakeLists.txt:276-284`); absent here → not registered. Needs the real Q2_0 expert arena.
- `expert_parity`, `pool_test` — **always** read a real expert blob from `pack/full/experts.bin` (the
  `--selftest` flag switches the *activations* to synthetic, not the weights; `expert_parity.cpp:108-114`,
  `pool_test.cpp:78-84`). Not registered because `STRATA_BUILD_TESTS=OFF` (`CMakeLists.txt:1191`).
- `dequant_bf16_test` — built but deliberately **not** a ctest: "needs the 38 GB model; run it by hand"
  (`CMakeLists.txt:1202-1204`).
- Manual real-model diagnostics (built, no `add_test`): `native_expert_parity <shard.gguf> [layers]`
  (`src/kernels/native_expert_parity.cpp:528-535`), `native_expert_bench`, `ple_q5_parity`,
  `q5_projection_parity`, `ple_fp8_parity` (`CMakeLists.txt:1174-1183`), `hip_ple_iq4` ("pass the GGUF shard
  containing the native PLE key", `CMakeLists.txt:1141`), `strata-load`.
- `canonical_xcheck.py` (`just oracle`): "for every tensor of the real GGUF"; `--gguf` default
  `$STRATA_SHARD1` or the dev-layout `../../Q2_0/…` shard 1 (`tools/canonical_xcheck.py:45-46`). (c).

**Not registered — synthetic anyway (would be (a)).** `expert_layout_test` (reads the committed
`tests/data/native_experts/*.txt` + synthetic GGUF shards "at the real expert dimensions",
`CMakeLists.txt:439-444`), `native_dense_ple_key_test`, `expert_cache_per_layer_test`,
`expert_cache_segmented_test` (write their own tiny synthetic packs, `write_pack`; skip 77 only when no CUDA
device), `pool_stress`,
`expert_multi_test`, `direct_file_async_test`, `pool_affinity_test`, and the `native_expert_parity_*` ctest
variants, which are exactly the model-free modes: `--synthetic <pair>`, `--q5_1-min` ("crafted activations"),
`--bf16-embd` ("random BF16 table", `native_expert_parity.cpp:473-475`), and the `WILL_FAIL`
`--synthetic q6_K/q8_0` refusal check (`CMakeLists.txt:1152-1171`). All gated by `STRATA_BUILD_TESTS=OFF`.
The HIP tests (`tests/hip/*`) are not built on this host (no `STRATA_ENABLE_HIP`). **Verdict: (a)/(s).**

### 1.2 The `tools/` pack / fixture / MTP tooling

| Tool | Reads (path / env) | Verdict |
|---|---|---|
| `strata_pack.py` (build/verify/info) | real GGUF shards (`--gguf`); `verify` re-reads the pack against the source shard, bit-exact | (c) |
| `iq_pack.py` | `--gguf <model>-00001…` + all split shards beside it → native pack (`experts.bin`, `dense.bin`, `index.txt`, `tokenizer/`); optional `--base pack/full` to share `dense.bin` | (c) |
| `pack_index.py` | a written pack's manifest → flat `index.txt` | (a) given a pack |
| `pack_layer.py` | real GGUF → expert arena (`experts.bin`); "48 = the full 34 GB arena" | (c) |
| `strata_tokenizer.py` | GGUF **metadata only** (the tokenizer lives in GGUF metadata, "none of that needs the 35 GB of weights") → pack `tokenizer/` | (a) given a GGUF (metadata only) |
| `canonical_xcheck.py` | real GGUF shard 1 (`$STRATA_SHARD1`, default `../../Q2_0/…`) | (c) |
| `iq_fixture.py` | nothing external — seeded synthetic i-quant blocks; needs numpy + vendored gguf-py (`third_party/llama.cpp/gguf-py`) | (a) |
| `gguf_writer.py` (self-check) | writes a KB-scale synthetic `bench/tiny-selftest.gguf`, reads it back | (a) |
| `gguf_reader.py` | GGUF header parser; no model required | (a) |
| `make_profile.py` | committed `data/expert-profile.bin` base + optional `--dump-routing` traces from one engine run | (a) |
| `mtp_fetch.py` | HF range fetch of the 31 `mtp.*` tensors from `Qwen/Qwen3.8-Flash-Next` @ pinned rev (headers + tensor bytes only, sha256-checked) | (c-lite): ~5 GB fetch |
| `mtp_pack.py` | the fetched raw tensors → `mtp-q2_0.gguf` (docstring: dense ~0.18 GB; experts q2_0 ~0.71 GB) | (a) given the fetch |
| `mtp_rt.py` | the packed MTP GGUF → `rt/experts.bin`, `rt/dense.bin`, `rt/dense.txt` | (a) given the pack |
| `embd_bf16_pack.py` | a full **BF16 checkpoint dir** (the `token_embd` safetensors) → one-tensor BF16 GGUF | (c, large) |
| `ple_fp8_pack.py` | the checkpoint's 128 ngram shards (51.2 GB FP8 PLE table) → one-tensor GGUF | (c, large) |
| `draft_vocab.py` | GGUF shard 1 (tokenizer metadata) → `rt/draft_vocab.bin` variants | (c-lite: needs a shard on disk) |
| `make_tiny_model.py` | **not present in this checkout** — referenced only by the `.gitignore` comment ("~113 MB … regenerate with `python tools/make_tiny_model.py`"); also absent from public upstream (GitHub code search: 0 hits; not in the 965-path main tree). No `bench/tiny-*.gguf` / `bench/wide-*.gguf` consumer exists here either | — |

### 1.3 The `test_setup_*.py` suite (and what the goldens actually are)

Per the `justfile` `test` recipe these "run without a GPU or downloads" — verified by running them
(2026-10-05, this host): `test_setup_golden` is the one known stale fixture; this run also shows
`test_setup_amd` (1 failure) and `test_setup_unsloth` (1 error) failing for host-environment reasons
(§6.3). Everything is mocked:

- `test_setup_golden.py` — runs `setup.main()` fully mocked ("no GPU, no downloads, nothing written outside a
  temp folder") and diffs the written **run config** byte-for-byte against `tools/test_setup_golden.json`.
  The golden is engine **config output** (args like `--pack <T>/data/packs/iq3_s`, `--native <T>/models/IQ3_S/…gguf`,
  `--mtp <T>/data/mtp/rt`), not model data — it tells us exactly what the runtime engine consumes (§1.5).
- `test_iq_pack.py` — writes its own minimal GGUFs in a temp dir (`write_gguf` helper); no model, no GPU.
- `test_shards.py` — "over a minimal GGUF written here (no download, no model, no GPU)".
- `test_make_profile.py` — synthetic profiles/traces. `test_mtp_fetch.py` — "a fake checkpoint behind a mocked
  urlopen — nothing is downloaded". `test_strata_mcp.py` — stand-in `setup.py`/`server.py`. `test_calibrate.py`
  — stand-in engine. `test_draft_vocab.py` — pure counters. `test_setup_amd/choices/config/draft_vocab/hybrid/
  lowram/oldcpu/older_gpus/parallel/pins/prompts/remote_opt/risk/rope/sycl/unsloth/update` — mocked
  host/GPU/ROCm detection and decision logic.

**Verdict: all (a).** (Golden is currently stale: 46 failures — pre-existing, unrelated to model inputs.)

### 1.4 The `serve/` tests

`test_server.py` ("against the mock engine (no GPU, no pack)"), `test_lifecycle.py` / `test_parallel.py`
(`MockEngine`), `test_runconfig.py`, `test_mcp.py`, `test_security.py`, `test_monitor.py`, `test_vram.py`,
`test_winjob.py`, `test_structured.py`, `test_responses.py` — all (a). `chat_golden.json` is a chat-template
**rendering** fixture (messages → rendered prompt string), no model data. The one model-adjacent test:
`test_detok.py` — its parity class is `@unittest.skipIf(find_tokenizer() is None, "no pack tokenizer here")`
and searches `packs/*/tokenizer`, `pack/*/tokenizer` (`test_detok.py:20-23`); the synthetic-tokenizer classes
run without one. **Verdict: (a) overall; the detok parity class is (c-lite: the real pack's `tokenizer/` dir)
and skips otherwise.**

### 1.5 The conversation / batch / cache scripts (and `bench/`)

These spawn the **engine binary** with a real model config (`--exe build/strata --config strata-<model>.json`),
i.e. a running model:

- `batch_test.py`, `batch_interleave_test.py`, `parking_test.py` — token-exact A/B of solo vs batched decode.
- `conversation_cache_parity.py`, `conversation_cache_soak.py`, `conversation_restore_speed.py` — "loads
  private engines sequentially … `--run` requires an available GPU/model-loading window" (dry-run by default).
- `conversation_cache_http_smoke.py`, `conversation_cache_growth.py`, `conversation_cache_isolation.py`,
  `conversation_cache_disabled.py` — engine + HTTP orchestration.
- `early_close_test.py`, `needle_bench.py`, `calibrate.py` — a running server with a loaded model.

What their config must provide (golden config, `tools/test_setup_golden.json`): `--pack <pack dir>`
(dense artifact + `index.txt` + `tokenizer/`), `--native <shard-1.gguf>` (quantized experts served from the
GGUF), `--ple-gguf <shard-2.gguf>` (or a ple pack), `--mtp <mtp/rt>` (+ `rt/draft_vocab.bin`),
`--expert-profile data/expert-profile.bin`. The `pack` is **built locally** from the GGUFs
(`tools/strata_pack.py` / `tools/iq_pack.py`) and the `mtp/rt` from the ~5 GB fetch (§1.2) — so these
scripts are the consumers that justify the minimal download set of §2. **Verdict: (c)** (dry-run modes run
without a model but only exercise orchestration, not the engine).

`bench/`: only committed benchmark **records** (`bench/results/*`, measured numbers and configs). No bench
runner exists in this checkout; nothing under `bench/` consumes model files. **Verdict: (a)** — the
`bench/tiny-*.gguf` / `wide-*.gguf` gitignore entries are upstream vestiges (the generator is absent, §1.2).

## 2. The minimal download set, grouped by model

Pins: `setup.py:65-70` (`HF_REVISIONS`), families `setup.py:183-217`, model sizes `setup.py:122-152`.
Sizes and oids from the HF tree API at the **pinned revisions** (2026-10-05); each repo's HEAD sha equals its
pinned revision, so these are also the current `main` files. "Pinned sha256" = pinned **in this repo**; where
it isn't, the tree-API `lfs.oid` is given (it equals the file's sha256 for LFS files).

**All four qwen sizes and the Coder share one byte-identical shard 2** — the 28,800,138,432 B PLE n-gram
table, oid `316b46f3…` in every case (also `setup.py:197-199`: the Coder's "shard 2 … and vision encoder are
the original's files, shared with it"). One download serves all of them.

### `models/qwen-Q2_0/` — the minimal set for every real-model consumer (~66.4 GB)

| HF repo @ rev | file path in repo | size (bytes) | sha256 | consumers |
|---|---|---:|---|---|
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f920…` | `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 37,623,740,192 | not pinned; lfs.oid `69820c02ec7d…b199b720` | pack build (`strata_pack`/`iq_pack` → `expert_parity`, `pool_test`, `expert_arena_load`, engine `--native`), `canonical_xcheck.py` |
| same | `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` | 28,800,138,432 | not pinned; lfs.oid `316b46f3a2db…c161e113` | `ple_parity` (default `--gguf`), engine `--ple-gguf`, pack `dense.bin` source |
| `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d…` (BF16 checkpoint) | 31 `mtp.*` tensors scattered over 28 of the 131 `model-XXXXX-of-00131.safetensors` shards (shards 00037…00124; exact map from `model.safetensors.index.json`) | ~5 GB total (expert tensors `gate_up_proj [512,1280,2560]` ≈ 3.36 GB + `down_proj [512,2560,640]` ≈ 1.68 GB BF16, per `mtp_pack.py` docstring; dense ~0.18 GB) | **all 31 pinned** in `tools/mtp_fetch.py:44-120` | `mtp_fetch → mtp_pack → mtp_rt` → every engine run's `--mtp` (speculative decoding), i.e. all §1.5 scripts |

`setup.py`'s own mode for the MTP is the range-fetch: "only its ~5 GB of MTP tensors are downloaded"
(`setup.py:4202-4204`), sha256-verified per tensor. A whole-shard fallback (28 shards) is possible but
much larger; the index gives the byte ranges at fetch time.

Total: **~71.5 GB**, matching `MODELS["Q2_0"]["download_gb"] = 66.4` + the ~5 GB MTP (`setup.py:125, 20`).

### Optional per-model sets (only if a test needs that model)

| subfolder | HF repo @ rev | files | sizes (bytes) | sha256 | consumers |
|---|---|---|---:|---|---|
| `models/qwen-IQ2_XS/` | same qwen repo | `IQ2_XS/…-00001-of-00002.gguf` | 39,225,954,592 | lfs.oid `92cee27a…0a49d7` | engine `--native` (68.0 GB total w/ shared shard 2; `setup.py:126`) |
| `models/qwen-IQ3_XXS/` | same | `IQ3_XXS/…-00001-of-00002.gguf` | 47,039,860,096 | lfs.oid `219ea929…856d15` | engine `--native` (75.8 GB; `setup.py:128`) |
| `models/qwen-IQ3_S/` | same | `IQ3_S/…-00001-of-00002.gguf` | 54,817,524,224 | lfs.oid `4c1eb2ce…d2aca3` | engine `--native` (83.6 GB; `setup.py:131`) |
| `models/swift-Q2_0/` | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `b22d729e…` | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`, `…-00002-of-00002.gguf` | 39,799,117,984 + 26,750,834,816 | lfs.oid `79e2a387…b6ef9678`, `ef3bb04f…2ad5ae0` (repo also ships a `SHA256SUMS`) | engine (Swift fine-tune); `setup.py:190-196` |
| `models/swift-IQ2_XS/` | same | `…-IQ2_XS-00001/00002` | 39,788,473,344 + 28,363,693,824 | lfs.oid `ca3b302d…7743b`, `e4996a6d…14c9dbb` | engine |
| `models/swift-IQ3_XXS/` | same | `…-IQ3_XXS-00001/00002` | 39,785,790,560 + 36,180,282,560 | lfs.oid `3bddaa66…3d23`, `b0b15f78…1a0160` | engine |
| `models/coder-IQ1_M/` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` @ `5348543e…` | `IQ1_M/…-00001-of-00002.gguf` (+ shard 2 = the shared qwen shard 2) | 29,608,446,496 (+28,800,138,432) | lfs.oid `e11083ba…087fad` (shard 2: `316b46f3…`) | engine (58.4 GB; `setup.py:135`); profile `data/expert-profile-coder.bin` is committed |
| `models/unsloth-UD-Q4_K_XL/` | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `38bb39ee…` | `UD-Q4_K_XL/…-0000{1..4}-of-00004.gguf` | 10,946,624 + 49,859,583,136 + 49,376,141,504 + 12,087,983,520 (111.3 GB) | **pinned**: `setup.py:155-164` (`44481862…`, `3f342f1c…`, `56758f40…`, `753bda48…`); oids match | engine (experimental; `setup.py:137-142`); `native_expert_parity --synthetic` variants validate its format without it |
| `models/unsloth-UD-IQ4_XS/` | same | `UD-IQ4_XS/…-0000{1..3}-of-00003.gguf` | 10,946,624 + 49,835,229,856 + 43,836,407,744 (93.7 GB) | **pinned**: `setup.py:166-173` (`5ce89370…`, `577a38a2…`, `d4634e6d…`); oids match | engine (`setup.py:143-151`) |
| (vision, optional) | qwen repo / swift repo / coder repo | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 B, oid `b1a82259…`), `mmproj-Swift-…` (907,543,520 B), coder's mmproj copy | | | vision path only; no test requires it |
| (community, optional) | `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` (public) | `Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-0000{1,2}-of-00002.gguf` | 44,637,691,008 + 40,564,977,024 (85.20 GB, as in `docs/ORCA.md`) | repo ships none here | `docs/ORCA.md` compat path via `iq_pack.py --compat-bf16` |

Every qwen-family total above equals `setup.py`'s `download_gb` field (66.4 / 68.0 / 75.8 / 83.6 / 58.4 /
111.3 / 93.7), cross-checking the API sizes against the pins.

## 3. Local generation inventory (what replaces downloads)

| Generator | What it produces | Size / cost | What it replaces |
|---|---|---|---|
| `tools/iq_fixture.py --out <dir>` | deterministic synthetic i-quant blocks + f32 references for `iq_parity` (10 types; default seed 42) | KB–MB; seconds; needs numpy + vendored gguf-py | nothing — it *is* the model input for `iq_parity` |
| `tools/gguf_writer.py` (self-check) | `bench/tiny-selftest.gguf`, a synthetic v3 GGUF with one Q2_0 tensor | KB; instant | nothing (writer round-trip check) |
| `strata_pack.py build` / `iq_pack.py` | the native **pack** (`experts.bin` 34 GB for Q2_0, `dense.bin` 5.4 GB, `index.txt` ~150 KB, `tokenizer/`) from the GGUF shards | CPU minutes–hours; no network | the pack artifacts (the engine's `--pack` input) — built *from* the downloaded GGUFs |
| `pack_index.py`, `strata_tokenizer.py` | `index.txt`; `tokenizer/` (vocab/merges/token_type + chat template, from GGUF metadata) | seconds | nothing extra (pack components) |
| `mtp_fetch.py fetch` + `mtp_pack.py` + `mtp_rt.py` | `mtp/rt/` (`experts.bin` ~0.71 GB q2_0, `dense.bin` ~0.18 GB, `dense.txt`) + `mtp-q2_0.gguf` | from the ~5 GB HF range-fetch; CPU-only after | the draft-layer runtime (engine `--mtp`) — **the one piece of the runtime that must be fetched, not derived**: the GSQ-RCO GGUFs ship no MTP head (`mtp_fetch.py:5`) |
| `make_profile.py` | `expert-profile.bin`-format profiles | seconds; base profile `data/expert-profile.bin` (196,632 B) is committed | the Coder profile is committed too (`expert-profile-coder.bin`, 98,328 B) |
| committed `data/` artifacts | `draft_vocab*.bin` (162–425 KB each), `experimental-speed-projection/` GGUF | already in git | the MTP draft-token subsets; no download |
| `tools/make_tiny_model.py` → `bench/tiny-*.gguf` (~113 MB per the `.gitignore` comment), `bench/wide-*.gguf` | **absent** — the tool is not in this checkout nor in public upstream; no consumer of those files exists here either | — | — |

Net: **there is no locally generated "tiny full model" in this checkout** (no (b)-verdict test exists), and
the only genuinely non-derivable downloads are (1) the GGUF shards of the model you choose to run and
(2) the ~5 GB MTP tensors from the BF16 checkpoint. Everything the engine consumes at runtime (pack,
tokenizer, profile, MTP rt) is generated locally from (1)+(2) plus committed `data/` files.

## 4. Token access verdict (metadata only, 2026-10-05)

- **Both tokens work.** Every call returned HTTP 200 with each of
  `hf_FAUYF…ihyDR` (tok1) and `hf_nnO…njA` (tok2) — 5 repos × metadata + tree. (Tok1 was used for the tree
  fetches; no call needed a token at all, see below.) No invalid/expired token observed.
- **All five required repos are public and non-gated** (`/api/models/<repo>`: `private: false`,
  `gated: false`), so **anonymous access lists them too** — the tokens are not required.
- **Pinned revisions are still the repos' HEADs** (API `sha` == `HF_REVISIONS` pin for all four GGUF repos
  and the Qwen checkpoint), so pinned-ref trees equal `main`:

| repo (pinned rev) | HEAD sha == pin? | files (pinned rev) | repo size | verdict |
|---|---|---:|---:|---|
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f920…` | yes | 28 | 294.82 GB | public, anon-readable |
| `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `b22d729e…` | yes | 39 | 212.20 GB | public, anon-readable |
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` @ `5348543e…` | yes | 13 | 59.32 GB | public, anon-readable |
| `unsloth/Qwen3.8-Flash-Next-GGUF` @ `38bb39ee…` | yes | 60 | 1499.63 GB (all tiers) | public, anon-readable |
| `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d…` | yes | 144 (131 shards) | 360.02 GB | public, anon-readable |
| `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` (optional) | — | 54 | ~1.9 TB (all tiers) | public, anon-readable |

- File sizes in §2 come from `…/tree/<pinned-ref>?recursive=true` responses (the `size` field); sha256s are
  the same responses' `lfs.oid` fields. The unsloth oids match the `setup.py` pins byte-for-byte.

## 5. Layout recommendation

Reuse the existing gitignored root dirs — `models/`, `packs/`, `pack/`, `mtp/` — instead of a new
`fixtures/models/<model>/` tree. The tools already encode these locations as their defaults and fallbacks
(`ple_parity`'s `pack/full` + `STRATA_PLE_GGUF`, `canonical_xcheck`'s `STRATA_SHARD1`, the engine's
`--pack pack/full` at `src/program/generate.cpp:244`, setup's `--models-dir/--gguf-dir` and the golden
config's `<T>/models/<SIZE>/…` layout), and `.gitignore` already covers exactly these plus the generated
pack products — a parallel `fixtures/` tree would duplicate the ignore rules, and every test consumer would
need an extra `--gguf/--pack` or env-var override to find its inputs, for no isolation benefit (the files are
identical whatever the directory name). If the team wants the semantic distinction "test inputs" vs
"user's runtime install" (the `fixtures/` idea), the cheapest form is to keep one on-disk location under
`models/<model-family>-<size>/` and treat "fixture" as a label in the download-plan doc, not a path; if a
separate tree is still wanted, add `fixtures/` to `.gitignore` and point the two hard-coded defaults via the
existing env vars (`STRATA_PLE_GGUF`, `STRATA_SHARD1`) rather than editing tool sources.

## 6. What blocks the download-plan ticket

1. **`ple_parity` (the only real-model ctest) needs artifacts no download can supply**: `bench/micro/
   ple_in.bin` / `ple_out.bin` and the capture tooling that produced them are not in this checkout and not in
   public upstream (`Niko1221/Strata` main, 965 paths: no `bench/micro`, no `make_tiny_model`; GitHub code
   search for `ple_in.bin` finds only the CMake/C++ references). Options for the plan: (a) obtain
   `bench/micro/` from the non-public upstream source and vendor the two `.bin` fixtures; (b) re-capture with
   a local reimplementation (out of scope here); (c) unregister/guard the test. Until then it fails by design
   (hard error, not skip). It additionally needs Q2_0 shard 2 + the pack's `dense.bin` (both covered by the
   minimal set in §2).
2. **`iq_parity` currently skips** until the vendored gguf-py is present (`third_party/llama.cpp/gguf-py`,
   pinned code download `3cf03257…` via `setup.py`'s llama.cpp fetch, or `STRATA_GGUF_PY` — note
   `tools/iq_fixture.py:34` hard-codes the `third_party/llama.cpp` path, so an env override alone does not
   reach `iq_fixture.py`; CMake's `Iq_FIXTURE_PYTHON` only picks the interpreter). Code-only, ~small; no model.
3. **Pre-existing non-model failures on this host** (for the record, so the plan isn't blamed for them):
   `test_setup_golden` — stale config fixture, 46 failures (already noted in the `justfile` `test` recipe);
   `test_setup_amd.test_prebuilt_hip_zip` — mock of the prebuilt HIP zip unpacking;
   `test_setup_unsloth.test_amd_is_not_asked` — setup's `rocm_root()` writes a stamp file into the read-only
   `/nix/store/…` python env (`OSError: Read-only file system`, `setup.py:1801`). None needs a model file.
4. **`expert_arena_load` / `expert_parity` / `pool_test` / `dequant_bf16_test`** are only reachable with the
   real Q2_0 pack (34 GB `experts.bin` arena) and, for the first three, `STRATA_BUILD_TESTS=ON`; they are
   currently not registered, so the download plan should decide whether that is intended.
5. **No (b)-path exists**: `tools/make_tiny_model.py` and its `bench/tiny-*.gguf` products are absent from
   this checkout and from public upstream — any plan assumption of a locally generated tiny model must be
   dropped or the tool restored from non-public upstream.
6. **Size reality check for a 32 GB-VRAM / 128 GB-RAM host**: the minimal set (§2) fits in RAM (66.4 + ~5 GB);
   the pack adds ~39 GB of derived artifacts (arena 34 GB + dense 5.4 GB) *before* runtime; larger sizes or
   Unsloth tiers (77–94 GB of experts) need the budget/RAM-read modes documented in `docs/UNSLOTH_Q4.md`.

## Sources (read this session)

- `setup.py:65-70` (pinned HF revisions), `:122-152` (MODELS sizes), `:155-173` (Unsloth sha256 pins),
  `:183-217` (families: repos, file patterns, mmproj), `:4202-4207` (MTP fetch/pack/rt flow).
- `CMakeLists.txt:87-92, 276-284, 438-459, 577-853, 1132-1234` (test registration and the synthetic/required
  comments); `build/CMakeCache.txt`, `build/CTestTestfile.cmake` (the 38 registered ctests, read only).
- `src/kernels/ple_parity.cpp:250-300` (required inputs), `src/kernels/cpu/expert_parity.cpp:81-114`,
  `src/kernels/cpu/pool_test.cpp:54-84`, `src/kernels/native_expert_parity.cpp:528-535`,
  `src/program/generate.cpp:244` (default `pack/full`), `tests/core/*` (synthetic packs/files).
- `tools/mtp_fetch.py:29-44` (pinned checkpoint rev + sha256 table, 31 tensors/28 shards),
  `tools/mtp_pack.py` (tensor shapes, dense/expert sizes), `tools/iq_pack.py`, `tools/strata_pack.py`,
  `tools/pack_index.py`, `tools/strata_tokenizer.py`, `tools/canonical_xcheck.py:45-46`,
  `tools/iq_fixture.py:16-18,34`, `tools/gguf_writer.py:185-230`, `tools/_paths.py`, `tools/test_*.py`
  docstrings, `serve/test_*.py` docstrings + `serve/chat_golden.json`, `justfile` (`test` recipe),
  `.gitignore`, `data/` (committed profile/draft-vocab artifacts).
- HF API (metadata only): `/api/models/<repo>` (gated/private/HEAD sha) and
  `/api/models/<repo>/tree/<pinned-ref>?recursive=true` (sizes, `lfs.oid`) for the six repos in §4;
  one fetch of `model.safetensors.index.json` (170 KB manifest) for the MTP shard map.
- GitHub: `Niko1221/Strata` main tree (965 paths — no `bench/micro`, no `tools/make_tiny_model.py`),
  `gh search code` (0 hits for `make_tiny_model`; `ple_in.bin` only in CMake/C++).
- `docs/ORCA.md` (community model sizes, MTP-rt steps), `AGENTS.md` (model family statement).
