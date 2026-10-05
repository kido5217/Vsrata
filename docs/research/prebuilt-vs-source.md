# Prebuilt vs. source: the artifact inventory (ticket #5)

Research date: 2026-10-05. Repo: `kido5217/Vsrata`, branch `main` @ `f9d372a`; this is a copy of
`Niko1221/Strata` at v0.1.39 (`CMakeLists.txt:11` `project(strata VERSION 0.1.39)`; `MIN_ENGINE = (0, 1, 39)`
at `setup.py:118`). Method: reading code and docs only (`setup.py`, `CMakeLists.txt`, `Dockerfile`,
`tools/mtp_fetch.py`, `tools/mtp_pack.py`, `data/`, `third_party/`, `docs/INSTALL.md`, `docs/DETAILS.md`,
`docs/AMD_HIP.md`), plus read-only `gh` API queries against `Niko1221/Strata` releases and `nix eval` /
`nix run` against this host's pinned nixpkgs (recorded in §4). Nothing was executed: no `setup.sh`/`setup.py`,
no server or engine start, no model files or release zips downloaded. The only things run were nix
evaluations and one `nix run` of nixpkgs' nvcc.

## TL;DR

- For everything a user runs, the **engine binary** is the only thing that can be a prebuilt blob, and on
  Linux it **never is**: the last 12 upstream releases (v0.1.28–v0.1.39) publish only `strata-windows-x64*.zip`
  assets (re-verified here; ticket #4 swept the last 30 releases and found no Linux asset in any of them).
  Linux therefore always compiles the engine from source (10–20 min) with a CUDA 13 toolkit (12.8+ for older
  cards) or a ROCm 7 wheel set for AMD. Windows gets the prebuilt CUDA and HIP engines; Windows AMD is
  prebuilt-only (setup cannot compile it).
- A release zip is `strata` + `strata-vision` + `BUILD.json` (the HIP zip instead bundles the ROCm runtime:
  `strata.exe`, `strata-device.exe`, `rocm/`). `BUILD.json` is the stamp setup reads to decide
  prebuilt-vs-local, version, and which `sm_` archs the binary covers (`archs`, `ptx` fallback, `cuda`,
  `vision_archs`, `backend`).
- **Model artifacts are always source-side downloads** — pinned Hugging Face revisions per family (original,
  Swift 1.5, Coder, Unsloth), 2-shard GGUFs (shard 2 is the 28.8 GB n-gram table, shared by Coder with the
  original), plus the MTP draft layer: 31 tensors range-fetched from the original's 360 GB BF16 checkpoint at a
  pinned revision, SHA256-checked, and repacked locally by `tools/mtp_pack.py`.
- **Third-party**: the repo carries a 3-file trimmed `third_party/ggml` (a header + the pinned llama.cpp commit
  + license); the full ggml/llama.cpp is fetched as a GitHub source zip at setup time at the
  `LLAMA_CPP_COMMIT` pin; the ready-made engine's cuBLAS/CUDA runtime come from two pinned NVIDIA pip wheels
  (a locally compiled engine uses the toolkit's own libraries instead).
- **nixpkgs**: this host's nixpkgs ships CUDA toolkits 12.6–12.9 and 13.0–13.3 as per-module package sets
  (`cudaPackages_13` = 13.2: nvcc 13.2.51, libcublas 13.3.0.5, `cuda-merged-13.2` toolkit,
  `cuda13.2-cuda_compat-595.45.04`). nixpkgs' nvcc lists `compute_120` natively, so **sm_120 (RTX 50) codegen is
  feasible**. A from-source engine build on NixOS is therefore fully coverable by nixpkgs packages
  (cmake 4.1.6, ninja 1.13.2, gcc 15.3.0, CUDA 13.2, python3 + pip packages); the frictions are
  `allowUnfree` (CUDA EULA), runtime library paths for the CUDA libraries, and the setup flow's own network
  fetches (GitHub zip + Hugging Face).

## 1. Engine artifacts

### 1.1 What the prebuilt blob is: release assets

Setup downloads the ready-made engine from `<PREBUILT_URL><asset>` (`setup.py:99-103`):

```
PREBUILT_URL     = "https://github.com/Niko1221/Strata/releases/latest/download/"
PREBUILT_TAG_URL = "https://github.com/Niko1221/Strata/releases/download/v{version}/"
PREBUILT_ASSET   = "strata-windows-x64.zip" (WIN) | "strata-linux-x64.zip"
CUDA12_ASSET     = "strata-windows-x64-cuda12.zip" | "strata-linux-x64-cuda12.zip"   (setup.py:112)
WIN_HIP_ASSET    = "strata-windows-x64-hip.zip"                                       (setup.py:1461)
```

Asset names per release, from the GitHub API (`gh api repos/Niko1221/Strata/releases`, 2026-10-05):

| Release | Published | Assets |
| --- | --- | --- |
| v0.1.39 | 2026-10-04 | `strata-windows-x64-cuda12.zip`, `strata-windows-x64-hip.zip`, `strata-windows-x64.zip` |
| v0.1.38 | 2026-10-03 | `strata-windows-x64-hip.zip`, `strata-windows-x64.zip` |
| v0.1.37 | 2026-10-02 | `strata-windows-x64-hip.zip`, `strata-windows-x64.zip` |
| v0.1.36 | 2026-10-02 | `strata-windows-x64-hip.zip`, `strata-windows-x64.zip` |
| v0.1.35 | 2026-10-02 | `strata-windows-x64-hip.zip`, `strata-windows-x64.zip` |
| v0.1.34 | 2026-10-02 | `strata-windows-x64-hip.zip`, `strata-windows-x64.zip` |
| v0.1.33 | 2026-10-01 | `strata-windows-x64.zip` |
| v0.1.32 | 2026-10-01 | `strata-windows-x64.zip` |
| v0.1.31 | 2026-10-01 | `strata-windows-x64.zip` |
| v0.1.30 | 2026-09-30 | `strata-windows-x64.zip` |
| v0.1.29 | 2026-09-30 | `strata-windows-x64.zip` |
| v0.1.28 | 2026-09-30 | `strata-windows-x64.zip` |

**No Linux asset exists in any of these 12 releases** — every asset is `strata-windows-x64*`. This
independently confirms ticket #4's 30-release sweep. Consequences, both encoded in setup:

- `strata-linux-x64.zip` / `strata-linux-x64-cuda12.zip` are the names setup *tries* on Linux
  (`setup.py:103,112`); the `get_prebuilt` HEAD check 404s and setup falls through to compiling
  (`setup.py:1975-1981`: "not published (yet), or no internet: compile instead").
- The comment at `setup.py:108-111` is explicit: the ready-made CUDA 12 asset is "Windows; on Linux it is
  compiled here with a CUDA 12.x toolkit".
- The Windows HIP engine is the only AMD prebuilt; on Windows AMD, setup *requires* it (`setup.py:4101-4105`:
  "no ready-made AMD engine for this Strata version" → `fail`).

First appearances: `strata-windows-x64-hip.zip` first shipped in v0.1.34 (the v0.1.34 entry in the
`MIN_ENGINE` notes at `setup.py:118`: "AMD on Windows (a ready-made HIP engine)"; `WIN_HIP_MIN_ENGINE =
max(MIN_ENGINE, (0, 1, 33))` at `setup.py:1462` resolves to 0.1.39, i.e. the floor is the current release, not
v0.1.34 — the v0.1.34 HIP zip is not used by this checkout's setup). `strata-windows-x64-cuda12.zip` first
shipped in v0.1.39.

### 1.2 What a release zip contains

`setup.py:96-97` (the only description in-tree; the zip is built upstream by `tools/make_release.py`,
referenced at `setup.py:97` but **not present in this checkout** — releases are built outside the published
tree):

- `strata-windows-x64.zip` / `strata-linux-x64.zip` (would-be): `strata(.exe)`, `strata-vision(.exe)`,
  `BUILD.json`.
- `strata-windows-x64-cuda12.zip`: the same, built with the CUDA 12.9 toolkit
  (`-DSTRATA_EXPERIMENTAL_SM60=ON`, `setup.py:109`).
- `strata-windows-x64-hip.zip` (`docs/AMD_HIP.md:66-73`): `strata.exe`, `strata-device.exe` and the ROCm
  libraries it loads — `engine/rocm/bin`: the HIP runtime, hipBLAS/rocBLAS/hipBLASLt with their kernels for
  the supported cards, amd_comgr, the Microsoft C++ runtime (ROCm 10.2.0a20260930 from AMD's TheRock builds,
  licenses in `engine/rocm/licenses`); since v0.1.35 `amdhip64_7.dll` and `amd_comgr.dll` are also beside
  `strata.exe` (`docs/AMD_HIP.md:74-78`).

**`BUILD.json` fields.** Setup reads these from the prebuilt stamp (`get_prebuilt`, `setup.py:1951-2030`):

| Field | Meaning (per the reading code) |
| --- | --- |
| `version` | engine version; must be ≥ `MIN_ENGINE` (0.1.39) or setup compiles instead (`setup.py:1995-2000`) |
| `source` | `"prebuilt"` for release zips (anything not `"local"` triggers the pip CUDA wheels, `setup.py:4110`); `"local"` for engines compiled on the PC, `"local-hip"` for AMD (`setup.py:2330,1895`) |
| `backend` | `"hip"` for the AMD engine (`setup.py:1955`) |
| `archs` | the `sm_` archs the binary has cubin code for (`setup.py:1977-1981`); a card outside the set → compile locally |
| `ptx` | boolean: a PTX fallback lets a card *newer* than `max(archs)` JIT (`setup.py:1966,1979,1992`) |
| `cuda` | the CUDA version the engine was built with (printed on install, `setup.py:2023`) |
| `vision` / `vision_archs` | the image encoder's mode and arch coverage; the encoder can cover fewer cards than the engine (0.1.30/0.1.31 had no RTX 20 code, #331, `setup.py:1989-1993`) |

A **locally compiled** engine's `BUILD.json` is written by `build_engine` (`setup.py:2329-2335`) with:
`source: "local"`, `version`, `archs`, `vision`, `toolkit` (12 only), `cuda_dirs` (the toolkit's own library
folders), `src` / `vision_src` (source fingerprints), `isa_floor` (older-CPU builds only). The HIP local build
writes `source: "local-hip"`, `backend: "hip"`, `lib_dirs` instead of `cuda_dirs` (`setup.py:1895-1896`).

The ready-made engine's GPU coverage, from the release notes in-tree: RTX 20 (sm_75) was added to it in
v0.1.27 (`setup.py:118`), and it "runs on RTX 20/30/40/50" (`docs/INSTALL.md:316`, `docs/DETAILS.md:317`).
Its CPU side is built with `STRATA_PORTABLE` (see §1.3). The exact arch list of the release zip lives in the
zip's own `BUILD.json`; this research did not download a release zip, so it is recorded from the setup reader
plus release notes, not from the artifact itself.

### 1.3 When `setup.py` compiles, and with what

**The decision** (`setup.py:4094-4115`, `get_prebuilt` `setup.py:1937-2030`): setup compiles when

1. the HEAD check finds no prebuilt asset at all (Linux always, §1.1), or
2. the prebuilt's `version` < `MIN_ENGINE`, or
3. the prebuilt has no code for a card in use (`archs` miss without `ptx` coverage), or
4. the installed prebuilt's vision encoder is missing and images were chosen (`setup.py:4112-4114`), or
5. `--build` forces a local compile, or
6. a Pascal/Volta card is in the model's GPU set (moves it to the CUDA 12 engine, `ensure_engine_for`
   `setup.py:3387-3413`), or
7. AMD on Linux (no prebuilt AMD engine for Linux exists; `build_engine_hip` `setup.py:1852-1900`), or
8. the engine's source changed since the last local compile: `source_hash` over
   `ENGINE_SOURCES = ("CMakeLists.txt", "src", "include", "third_party/ggml")` mixed with
   `LLAMA_CPP_COMMIT` (and `tools/vision` separately for the encoder) — the hash is stored in `BUILD.json`
   and a `git pull` that touches any of them recompiles "only what changed" (`setup.py:2231-2247,2295-2301`).

**Toolchain selection** (`install_build_tools`, `setup.py:2132-2187`):

- Toolkit version by card: CUDA **13** by default; CUDA **12** when the oldest card is below `sm_75`
  (`CUDA13_MIN_ARCH = 75`, `setup.py:111`) or `--cuda 12` is given (`setup.py:2137-2141`). `sm_120` (RTX 50)
  needs CUDA ≥ 13.0 — engines built with 12.8 crashed on long prompts (#220, #224); CUDA 13 dropped
  sm_60/sm_70 entirely (`setup.py:2138-2146`, `CMakeLists.txt:141-146`).
- Linux needs `g++` (checked via `shutil.which("g++")`) and `nvcc` ≥ the required version. Auto-install is
  Ubuntu 22.04/24.04 only: `sudo apt-get install build-essential` plus `cuda-toolkit-13-0` from NVIDIA's apt
  repo (keyring from `developer.download.nvidia.com`, `setup.py:2169-2185`); every other distro is told to
  install by hand (Arch hint in the error: `pacman -S base-devel cuda`; nvcc is found on PATH, in
  `/usr/local/cuda*` and `/opt/cuda*`, `find_nvcc` `setup.py:2368-2392`, overridable with `STRATA_NVCC`).
- Windows needs Visual Studio 2022 Build Tools (C++) and the CUDA 13.0 toolkit, auto-installed via
  `winget install Microsoft.VisualStudio.2022.BuildTools` / `Nvidia.CUDA --version 13.0` (`setup.py:2153-2166`).

**The CMake invocation** (`cmake_build` `setup.py:2189-2209`, `build_engine` `setup.py:2304-2314`):
Ninja generator, `CMAKE_BUILD_TYPE=Release`, a one-shot retry of the build (CUDA 13's ptxas occasionally fails
to parse a PTX file it just wrote, issue #45), and for the CUDA engine:

```
cmake -G Ninja -DCMAKE_MAKE_PROGRAM=<ninja> -S <repo> -B build
  -DCMAKE_BUILD_TYPE=Release
  -DSTRATA_ENABLE_CUDA=ON
  -DSTRATA_BUILD_TESTS=OFF
  -DCMAKE_CUDA_ARCHITECTURES=<sm archs, e.g. 120>
  -DCMAKE_CUDA_COMPILER=<nvcc>
  -DSTRATA_GGML_DIR=<third_party/llama.cpp>          # setup.py:2310
  [-DSTRATA_EXPERIMENTAL_SM60=ON]                    # cards < sm_75 or any CUDA 12 build
  [-DSTRATA_ISA_FLOOR=avx|none]                      # experimental older-CPU build
→ target `strata`
```

The image encoder is a second, separate build from `tools/vision/` with
`-DLLAMA_DIR=<llama> -DSTRATA_VISION_CUDA=ON|OFF` (+ archs/nvcc for the GPU encoder), target `strata-vision`
(`setup.py:2316-2325`). The HIP engine build uses `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF
-DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=<gfx...> -DCMAKE_HIP_COMPILER=<ROCm clang++>
-DCMAKE_PREFIX_PATH=<ROCm roots> -DSTRATA_GGML_DIR=<llama>` (`setup.py:1887-1893`).

**What the top-level `CMakeLists.txt` adds** (read in full):

- `cmake_minimum_required(VERSION 3.24)`, C++20, default `Release` (`CMakeLists.txt:9,55-62`).
- Backends are mutually exclusive options: `STRATA_ENABLE_CUDA`, `STRATA_ENABLE_HIP` (wave32 RDNA),
  `STRATA_HIP_GFX906` (wave64 compat layer over the CUDA targets), `STRATA_ENABLE_SYCL` (experimental Intel
  Arc port, `sycl/`) (`CMakeLists.txt:14-23,59-73`).
- The CUDA arch guard: any requested arch `< 75` is a fatal error unless `STRATA_EXPERIMENTAL_SM60=ON`;
  `< 60` is fatal even then; `>= 120` with a CUDA < 13.0 compiler prints the #220/#224 warning
  (`CMakeLists.txt:147-166`). The ready-made engine is the "no `STRATA_EXPERIMENTAL_SM60`" build
  (`CMakeLists.txt:150-153`).
- `STRATA_PORTABLE` (`CMakeLists.txt:31-34`): "build for other PCs: AVX2 baseline, static MSVC runtime" — the
  prebuilt release is built this way so its CPU experts run on any AVX2 CPU (ggml-cpu for the AVX2 floor
  instead of the build host's, `CMakeLists.txt:966-971`) and no VC++ redistributable is needed.
- `STRATA_ISA_FLOOR` (`avx`/`none`): experimental engines for pre-AVX2 CPUs, with a matching ggml-cpu feature
  set instead of the host's (`CMakeLists.txt:35-49,972-985`).
- `STRATA_NATIVE_EXPERTS` (default ON): the i-quant CPU experts come from ggml-cpu, built static from a
  llama.cpp checkout — `-DSTRATA_GGML_DIR=<dir>` or, empty, `FetchContent` of `ggml-org/llama.cpp` at the
  **same pinned commit** `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`CMakeLists.txt:947-997`), with
  `GGML_STATIC=ON`, `GGML_CUDA/GGML_OPENMP/GGML_BACKEND_DL=OFF`, and `GGML_NATIVE` (or the
  PORTABLE/floor feature sets) (`CMakeLists.txt:959-985`).
- The main target: `add_executable(strata src/program/generate.cpp)` linking `strata_engine strata_prefill
  strata_spec` + the GPU runtime (`CUDA::cudart` or the HIP runtime); prefill links cuBLAS/hipBLAS
  (`CMakeLists.txt:483-497`). Per-file ISA flags for Strata's own CPU kernels: AVX-512 on `expert.cpp`,
  AVX2 on `q2_avx2.cpp` (`CMakeLists.txt:930-939`).
- The **Dockerfile** is the hermetic recipe for the same build: `nvidia/cuda:13.0.0-devel-ubuntu24.04` base,
  `build-essential` + python venv + `requirements.txt`, then `setup.get_llama_cpp()` +
  `setup.cmake_build(...)` with exactly the same flags, default `CUDA_ARCHITECTURES=75;80;86;89;120`
  (a fat binary with a cubin per arch, `Dockerfile:49-100`). The container never compiles at run time.

**Where it lands**: the local CUDA engine goes to `engine/` (CUDA 13) or `engine-cuda12/` (the experimental
one) with `strata`, `strata-vision` and the local `BUILD.json` (`engine_dir` `setup.py:619-622`); the model
config records `exe`, `cuda`, and `lib_dirs` = the toolkit's own library folders
(`engine_lib_dirs` `setup.py:3432-3435`, `setup.py:2327-2329`).

## 2. Model artifacts

### 2.1 The Hugging Face repos per family

Every HF file is pinned to a fixed commit of its repository (`HF_REVISIONS`, `setup.py:66-70`; "the `sha` of
`https://huggingface.co/api/models/<repo>` when this was pinned"), so any checkout installs the same files on
any day. `hf_endpoint()` honors `HF_ENDPOINT` mirrors (`setup.py:75-79`); `download()` falls back to the repo's
current files with a message when a pinned revision is gone (`setup.py:63-65`).

| Family (`setup.py:183-216`) | HF repo (pinned revision, pin date) | File pattern | Sizes (`MODELS`, `setup.py:122-157`) |
| --- | --- | --- | --- |
| `qwen` — Qwen3.8-Flash-Next (original) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92082…` (2026-09-29) | `Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf`, per-size folder | Q2_0 66.4 GB, IQ2_XS 68.0, IQ3_XXS 75.8, IQ3_S 83.6 (download GB) |
| `swift` — Swift 1.5 (UkisAI fine-tune) | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `b22d729eae…` (2026-09-24) | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf` | Q2_0, IQ2_XS, IQ3_XXS (no IQ3_S) |
| `coder` — Coder (ISTA-DASLab expert-pruned) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` @ `5348543e01…` (2026-09-29) | `Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf` | IQ1_M 58.4 GB only (256 of 512 experts per layer kept) |
| `unsloth` — Unsloth ~4-bit | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `38bb39ee97…` (2026-09-30) | `Qwen3.8-Flash-Next-{q}-0000{i}-of-0000{4,3}.gguf` | UD-Q4_K_XL 111.3 GB (4 shards, experimental), UD-IQ4_XS 93.7 GB (3 shards) |

The vision encoder (`mmproj`) is a per-family file from the same repo: `mmproj-Qwen3.8-Flash-Next-BF16.gguf`
(qwen/coder/unsloth), `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf` (swift) (`setup.py:189,196,204,214`).

### 2.2 The shard layout

The original and Swift families are **2-shard** GGUFs per size. Shard 1 holds the model weights; **shard 2 is
the PLE n-gram table — 28.8 GB** — "a few rows per token, read unbuffered past the OS cache
(`--ple-io direct`, the default, made for SSDs; on a rotational disk `--ple-io ram` keeps the table in RAM,
#605)" (`docs/DETAILS.md:1028-1031`). The Coder's shard 2 and its vision encoder **are the original's
files**: "with the original installed, setup downloads only shard 1" (`docs/DETAILS.md:258-261`,
`setup.py:198-204`). Unsloth's files have 3–4 shards instead (`setup.py:149-157,212`). The setup's shard
integrity check validates each shard as a whole GGUF (header + tensor extents vs. file size,
`setup.py:1241-1250`).

### 2.3 The MTP draft layer

The GSQ-RCO GGUFs ship no MTP head; the model's own MTP block is the draft layer (it "drafts up to 3
tokens; one pass over all 48 layers checks them", `docs/DETAILS.md:1032-1033`). It comes from the **original
BF16 checkpoint `Qwen/Qwen3.8-Flash-Next` — 360 GB in 131 safetensors shards**, with 31 `mtp.*` tensors
scattered over 28 shards (`tools/mtp_fetch.py:1-13`):

- `tools/mtp_fetch.py` reads only the safetensors JSON headers (a few KB per shard) and then HTTP range
  requests for the MTP tensors' byte ranges — it "never downloads anything but the ranges named in the
  headers" (`tools/mtp_fetch.py:14-19`). The checkpoint is pinned to revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540` (its `sha` from the HF API on 2026-09-30; `STRATA_MTP_REVISION`
  overrides) and every tensor is checked against a SHA256 table in the file (guards a mirror that ignores
  `Range`, #327) (`tools/mtp_fetch.py:29-47`).
- `tools/mtp_pack.py` repacks the fetched BF16 tensors into a Strata GGUF: dense tensors (attention,
  indexer, hyper-connections, shared expert, router, fc/norms) stay BF16 (F32 for 1-D norms, ~0.18 GB); the
  routed experts keep the checkpoint's fused layout (`gate_up_proj` [512, 1280, 2560], `down_proj`
  [512, 2560, 640]) quantized to q2_0 (~0.71 GB) or q4_0 (~1.42 GB) — "the same format as the main model's
  experts, so Strata's CPU VNNI kernel and GPU hit kernel serve it unchanged" (`tools/mtp_pack.py:1-23`).
- Setup fetches ~5 GB and writes the packed layer into the data folder's `mtp/` (`setup.py:20`; ticket #4's
  step 7 documents the `mtp/rt/` completion mark).

### 2.4 The `data/` assets in-repo

`data/` (999 KB total) ships four small binary assets; setup copies the chosen one into the model's
`mtp/` folder as the draft head's token subset (`setup.py:3367-3377`):

| File | Role |
| --- | --- |
| `draft_vocab.bin` (425 KB, `cjk`) | the draft head's default token subset: English/code + Chinese, Japanese, Korean (~348 MiB of VRAM, `setup.py:3241`) |
| `draft_vocab_en.bin` (162 KB) | English/code subset (~133 MiB, less VRAM; English answers 1–2% faster) |
| `draft_vocab_cyrillic.bin` (236 KB) | English/code + the whole Cyrillic script (~193 MiB) |
| `draft_vocab_fr.bin` (185 KB) | English/code + French tokens (~151 MiB, #597) |
| `expert-profile.bin` (192 KB) | the shipped expert ranking for the original model, passed as `--expert-profile` so the GPU expert cache starts warm (`setup.py:4225`) |
| `expert-profile-coder.bin` (98 KB) | the same ranking mapped onto the Coder's kept experts through the release's `rco-allocation.txt` — "72% of the expert reads hit the GPU on a 12 GB card" (`docs/DETAILS.md:261-263`, `setup.py:205`) |
| `experimental-speed-projection/` | the control vector for the experimental speed projection, off unless `--speed-projection` (`setup.py:3555,4041-4051`) |

An engine can also *save* a learned profile (`"expert_profile_save"` in the config,
`docs/DETAILS.md:458-462`) — that file is user-side, not shipped.

## 3. Third-party artifacts

### 3.1 The trimmed `third_party/ggml`

`third_party/ggml/` is 58 KB and holds exactly three files:

- `ggml-common.h` — ggml's common header (the `ggml_half`/`ggml_half2` types and the block-layout
  declarations), used for "llama.cpp's block layouts and codebook grids for the i-quant formats (MIT,
  `third_party/ggml`)" — it is on the include path of `strata_kernels` (`CMakeLists.txt:362-363`) and of
  `strata_kernels_cpu` (`CMakeLists.txt:999`).
- `VERSION.txt` — one line: the pinned llama.cpp commit with its commit date:
  `3cf03257f219afbe7334045ff7c6a06ac68c627d Sun Sep 20 16:08:11 2026 +0800`.
- `LICENSE` (MIT).

**Why trimmed**: the full ggml (and gguf-py, mtmd) is not committed — it is fetched at setup time as a source
zip at the pinned commit (§3.2) and again by CMake's `FetchContent` from the same commit when
`STRATA_GGML_DIR` is empty (`CMakeLists.txt:991-997`). The repo keeps only the one header its own code
includes directly plus the pin that makes both fetch paths agree, so the checkout stays small and
reproducible while the "same files on any day" guarantee holds.

### 3.2 The `LLAMA_CPP_COMMIT` pin and when it is fetched

- The pin: `LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"` with
  `LLAMA_CPP_ZIP = "https://github.com/ggml-org/llama.cpp/archive/<commit>.zip"` (`setup.py:93-94`) — the
  same commit `third_party/ggml/VERSION.txt` records.
- The fetch: `get_llama_cpp()` (`setup.py:1254-1283`), called at setup step 4 ("the Strata engine",
  `setup.py:4096-4098`) before anything else in the engine flow — i.e. **on every setup run, not only when
  compiling**: llama.cpp provides `gguf-py` for the pack tools (`tools/iq_pack.py` and friends) and `mtmd`
  for the image encoder, plus the ggml the engine build links. It is idempotent: if
  `third_party/llama.cpp/ggml/CMakeLists.txt` and `third_party/llama.cpp/gguf-py` exist, it returns without
  downloading (`setup.py:1257-1259`).
- The download is a plain GitHub source zip (no git), extracted to `third_party/llama.cpp/` with llama.cpp's
  own web UI (`tools/ui`) skipped — it is unused and its deep paths broke Windows' 260-char limit
  (`setup.py:1262-1266`); the zip is deleted after unpacking (with a PermissionError-retry loop for
  antivirus locks, #63, `setup.py:1272-1282`).
- Offline path: `docs/INSTALL.md:187-189` — put that commit's tree in `third_party/llama.cpp` yourself
  (mirror or another PC) and setup uses it instead (#585).
- It also feeds the compile fingerprint: `source_hash` mixes `LLAMA_CPP_COMMIT` into the engine's `src` hash
  (`setup.py:2237-2239`), so a pin change always recompiles.

### 3.3 The pip CUDA libraries

For the **ready-made engine only**, the NVIDIA cuBLAS + CUDA runtime it links come from pip wheels, not from a
system toolkit (`setup.py:104-105,113`):

| Constant | Wheels | Size (per the install message) |
| --- | --- | --- |
| `CUDA_WHEELS` (CUDA 13 engine) | `nvidia-cublas==13.0.2.14`, `nvidia-cuda-runtime==13.0.96` | ~0.4 GB |
| `CUDA12_WHEELS` (CUDA 12 engine) | `nvidia-cublas-cu12==12.9.1.4`, `nvidia-cuda-runtime-cu12==12.9.79` | ~0.7 GB |

`pip_cuda_libs(toolkit)` installs them into `.venv/` (`setup.py:2124-2129`), called right after a prebuilt
engine is installed (`setup.py:4110,3420`); `cuda_lib_dirs(toolkit)` then locates them under
`site-packages/nvidia/` — `nvidia/cu13/lib` on Linux / `nvidia/cu13/bin/x86_64` on Windows (toolkit 13, by
globbing `libcublas.so.13*`), or the two folders `nvidia/cublas/bin` + `nvidia/cuda_runtime/bin` for the CUDA
12 wheels (`setup.py:1329-1345`). These directories become the config's `lib_dirs` that the engine loads
from (`engine_lib_dirs` `setup.py:3432-3435`: a **compiled** engine's stamp has its own `cuda_dirs` — the
toolkit's `bin`, `bin/x64`, `lib64` — and takes precedence over the pip wheels). The wheels match the CUDA
13.0/12.9 the prebuilt engines are built with; the minimum driver for the CUDA 13 engine is 580
(`MIN_DRIVER = 580`, `setup.py:106`), for the CUDA 12 engine 525 (Linux) / 528 (Windows)
(`setup.py:110,115`).

For **AMD on Linux**, the equivalent "third-party runtime" is ROCm itself: a system ROCm 7 with hipcc and
hipBLAS is used when present, otherwise the AMD "TheRock" wheels `rocm==7.10.0a20251120`
(`ROCM_VERSION`, `setup.py:1357`, overridable via `STRATA_ROCM_VERSION`/`STRATA_ROCM_INDEX`) are pip
installed into `.venv` (~10 GB, no sudo) from the card family's index at `rocm.nightlies.amd.com`
(`setup.py:1352-1356,1795-1801`; `docs/AMD_HIP.md:31-36`).

## 4. nixpkgs: the CUDA toolkit and a from-source build

Environment: this host's `flake:nixpkgs` is an **indirect** reference resolving to a pinned store path —
`/nix/store/3zvg83mg9aavm9bgh26ydljchply7i25-source` (`nix flake metadata nixpkgs --json`). All source
quotes below are from that tree; all versions were confirmed by live `nix eval`/`nix run` (unfree packages
need `NIXPKGS_ALLOW_UNFREE=1` + `--impure`, per this host's convention).

### 4.1 Which CUDA toolkits are available

`pkgs/top-level/cuda-packages.nix` defines per-version package sets, and `pkgs/top-level/all-packages.nix:2013-2025`
exposes them:

| Attribute | CUDA toolkit version (manifest) |
| --- | --- |
| `cudaPackages_12_6` | 12.6.3 |
| `cudaPackages_12_8` | 12.8.1 |
| `cudaPackages_12_9` | 12.9.1 |
| `cudaPackages_13_0` | 13.0.3 |
| `cudaPackages_13_1` | 13.1.1 |
| `cudaPackages_13_2` | 13.2.0 |
| `cudaPackages_13_3` | 13.3.0 |

Aliases (`all-packages.nix:2021-2025`): `cudaPackages_12 = cudaPackages_12_9`, `cudaPackages_13 =
cudaPackages_13_2`, and the **unversioned `cudaPackages` = `recurseIntoAttrs cudaPackages_12`** — i.e. the
default alias is still CUDA 12.9; CUDA 13 is only reached through the versioned attribute.

### 4.2 The package shape

Each set is built by `mkCudaPackages` (`cuda-packages.nix:7-12`) → `pkgs/development/cuda-modules`
(`default.nix` + `packages/` directory): an attrset of per-module derivations fetched from NVIDIA's
manifests — `cuda_nvcc`, `cuda_cudart`, `libcublas`, `libcufft`, `libcusolver`, `nccl`, `cudnn`,
`cudatoolkit` (the merged full toolkit), `cuda_compat` (forward-compat runtime), plus build helpers
(`backendStdenv`, `setupCudaHook`, `markForCudatoolkitRootHook`, …). Live evals:

```
nix eval --raw nixpkgs#cudaPackages_13.cuda_nvcc.version    → 13.2.51
nix eval --raw nixpkgs#cudaPackages_13.cuda_cudart.version  → 13.2.51
nix eval --raw nixpkgs#cudaPackages_13_3.cuda_nvcc.version  → 13.3.33
nix eval --raw nixpkgs#cudaPackages_13.cudatoolkit.name     → cuda-merged-13.2
nix eval --raw nixpkgs#cudaPackages_13.libcublas.version    → 13.3.0.5
nix eval --raw nixpkgs#cudaPackages_13.cuda_compat.name     → cuda13.2-cuda_compat-595.45.04
nix eval --raw nixpkgs#cudaPackages.cudatoolkit.name        → cuda-merged-12.9
```

The CUDA packages carry the unfree `CUDA EULA`: building them requires `NIXPKGS_ALLOW_UNFREE=1` /
`allowUnfree = true` (the eval error names the remedy). `backendStdenv` (the stdenv that compiles against the
toolkit) exposes `config.cudaCapabilities` for picking which compute capabilities to build for
(`pkgs/development/cuda-modules/backendStdenv/default.nix:97-99`), and `cudaForwardCompat` for the compat
runtime.

### 4.3 sm_120 (RTX 50) codegen feasibility — yes

Running nixpkgs' own nvcc:

```
NIXPKGS_ALLOW_UNFREE=1 nix run --impure nixpkgs#cudaPackages_13.cuda_nvcc -- --list-gpu-arch
  → compute_75, compute_80, compute_86, compute_87, compute_88, compute_89,
    compute_90, compute_100, compute_110, compute_103, compute_120, compute_121
```

So nixpkgs' CUDA 13 nvcc (13.2.51) natively supports **`compute_120`** (RTX 50) and `compute_121`, and
`compute_75` (RTX 20) — the exact range Strata's arch guard allows (`CMakeLists.txt:147-166`). Strata's
RTX-50 requirement of "CUDA 13.0 or newer" (§1.3) is met: any of `cudaPackages_13_0` (13.0.3) through
`_13_3` (13.3.0) works, with `cudaPackages_13` (13.2) the default.

### 4.4 What a from-source engine build on NixOS needs

Setup's Linux build path needs, and nixpkgs provides all of it (evals below, same pinned nixpkgs):

| Need (per §1.3) | nixpkgs package | Live eval |
| --- | --- | --- |
| CMake ≥ 3.24 (`CMakeLists.txt:9`) | `cmake` | `cmake-4.1.6` |
| Ninja (`setup.py:2189`) | `ninja` | `ninja-1.13.2` |
| C++ compiler / g++ (`setup.py:2148`) | `stdenv.cc` | `gcc-wrapper-15.3.0` (or `clang`) |
| CUDA 13 toolkit with nvcc + CUDAToolkit + cuBLAS (`CMakeLists.txt:131-137`) | `cudaPackages_13` (nvcc 13.2.51, `cuda-merged-13.2`, `libcublas` 13.3.0.5) | §4.2 |
| Runtime CUDA libs for a local engine (`setup.py:2327-2329`) | `cudaPackages_13.cuda_cudart` / `libcublas` (or `cuda_compat` for driver forward-compat) | §4.2 |
| Python 3.10+ with the `requirements.txt` set (`setup.py:117`, `setup.sh:8`) | `python3` + `python3Packages.{numpy,jinja2,regex,pyyaml,tqdm,requests,cmake,ninja,pillow,psutil}` | packages exist in nixpkgs |
| git (offline llama.cpp fallback, `docs/INSTALL.md:187-189`) | `git` | standard |
| driver ≥ 580 at run time (`setup.py:106`) | host's NVIDIA driver (outside nixpkgs' concern for the build) | — |

What nixpkgs does **not** cover, and what a NixOS home-manager recipe would have to arrange:

1. **The setup flow itself is not Nix-aware**: it looks for nvcc on PATH / `/usr/local/cuda*` / `/opt/cuda*`
   (`find_nvcc`, `setup.py:2368-2392`) or via `STRATA_NVCC` — a nix-built CUDA toolkit is usable through
   `STRATA_NVCC` (the env var is honored first and exclusively, #601) or by exposing the store path on PATH;
   its auto-installer is Ubuntu-only anyway (`setup.py:2169-2175`). A local engine then loads the toolkit's
   own libraries via the `cuda_dirs` recorded in `BUILD.json` (`setup.py:2327-2329`) — a store path that
   survives as long as it is in a user profile / store; `LD_LIBRARY_PATH` or the config's `lib_dirs` carry
   it at run time.
2. **Unfree**: CUDA (and the TheRock ROCm wheels if AMD) need `NIXPKGS_ALLOW_UNFREE` / `allowUnfree`.
3. **Network fetches inside setup** are plain HTTPS, not nix fetchers: the llama.cpp zip from
   `github.com/ggml-org/llama.cpp/archive/<pinned>.zip` (`setup.py:94,1259`) and the model GGUFs / MTP
   ranges from `huggingface.co` at pinned revisions (§2). A hermetic home-manager build would either let
   setup do these at install time or reproduce the pins itself.
4. **AMD**: no nixpkgs ROCm exists here (checked: the CUDA sets above are the only GPU-toolkit families in
   `pkgs/top-level/cuda-packages.nix`); the HIP path needs AMD's TheRock wheels (`setup.py:1352-1357`) or a
   system ROCm 7 — neither is a nixpkgs package on this host.

**Verdict**: a from-source Strata engine build on NixOS is fully feasible from nixpkgs alone for the
NVIDIA/CUDA path (toolkit, arch codegen including sm_120, build tools, python deps all present and pinned
above); the work left to a home-manager module is wiring `STRATA_NVCC`/PATH, `allowUnfree`, and the
run-time library paths, not finding missing toolchain pieces.

## 5. The artifact matrix (answer to the ticket's question)

"Prebuilt blob" = a binary someone else compiled and shipped as a release asset; "source" = fetched/rebuilt
on the user's machine. Everything a user runs, per artifact:

| Artifact | Blob or source | Where it comes from | Where it lands | Nix note |
| --- | --- | --- | --- | --- |
| Engine `strata` (NVIDIA) | **Windows: blob** (`strata-windows-x64.zip`, prebuilt by `tools/make_release.py` upstream, not in-tree); **Linux: source** — no Linux asset has ever been published (§1.1) | GitHub releases (HEAD-check on the version's tag, then `latest`) | `engine/` | nixpkgs covers a source build fully (§4.4): `cudaPackages_13` (nvcc 13.2.51, sm_120 native), cmake/ninja/gcc; wire `STRATA_NVCC`/PATH + `allowUnfree` + runtime lib paths |
| Engine CUDA 12 (Pascal/Volta) | **Windows: blob** (`strata-windows-x64-cuda12.zip`, first in v0.1.39); **Linux: source** | same release mechanism | `engine-cuda12/` | needs a CUDA 12.x toolkit: nixpkgs ships `cudaPackages_12_9` (toolkit 12.9, nvcc 12.9.86); the Windows CUDA-12 zip itself is built with the 12.9.1 toolkit (`setup.py:113`) |
| Engine `strata` (AMD) | **Windows: blob only** (`strata-windows-x64-hip.zip`, since v0.1.34, bundles ROCm runtime); **Linux: source** | same; TheRock wheels for the runtime | `engine/` (+ `engine/rocm/` on Windows) | no nixpkgs ROCm on this host; TheRock wheels / system ROCm 7 are pip-side |
| Image encoder `strata-vision` | blob when the release zip carries one (Windows); otherwise compiled locally (also Linux prebuilt-less path) | the same release zip, or local build from `tools/vision/` | beside the engine in `engine/` | local build needs the same toolchain (+ llama.cpp) |
| CUDA runtime + cuBLAS (for prebuilt engines) | **blob** (pip wheels `nvidia-cublas==13.0.2.14`, `nvidia-cuda-runtime==13.0.96` / cu12 variants) | PyPI (NVIDIA) into `.venv` | `.venv/lib/python*/site-packages/nvidia/...` | nixpkgs' `cudaPackages_13.cuda_cudart`/`libcublas` replace this for local builds |
| ROCm (AMD Linux) | **blob** (AMD TheRock wheels `rocm==7.10.0a20251120`) | `rocm.nightlies.amd.com` pip index into `.venv` | `.venv/...` (+ `lib_dirs` in config) | not in nixpkgs |
| Model GGUF shards (per family/size) | **source-side download** (never a prebuilt blob) — pinned HF revisions, 2-shard files (shard 2 = 28.8 GB n-gram table; Coder shares it) | Hugging Face §2.1 repos | `Strata-data/models/` (next to the repo) | fetches are plain HTTPS; a hermetic recipe would pin the same revisions |
| MTP draft layer | **fetched + repacked locally**: 31 tensors range-read from `Qwen/Qwen3.8-Flash-Next` BF16 (360 GB/131 shards) at pinned revision, SHA256-checked, quantized to `mtp-q2_0.gguf`/`mtp-q4_0.gguf` | HF range requests → `tools/mtp_pack.py` | `Strata-data/mtp/` | ~5 GB fetched, not the 360 GB checkpoint |
| Draft-vocab subsets, expert profiles, speed-projection vector | **in-repo** (shipped with the source) | the checkout's `data/` | copied into the model's `mtp/` folder (vocab); passed via `--expert-profile` | nothing to fetch |
| llama.cpp (ggml, gguf-py, mtmd) | **source** (pinned commit `3cf03257`, GitHub source zip at setup time; CMake `FetchContent` of the same commit as fallback) | `github.com/ggml-org/llama.cpp/archive/<commit>.zip` | `third_party/llama.cpp/` | zip fetch is plain HTTPS; offline: place the tree manually (`docs/INSTALL.md:187-189`) |
| ggml header subset | **in-repo** (3 files: `ggml-common.h`, `VERSION.txt`, `LICENSE`) | the checkout | `third_party/ggml/` (on the kernels' include path) | the trim keeps the repo small while the pin keeps fetches reproducible |
| Python venv + deps | **source-side** (pip from PyPI, pinned `requirements.txt`) | PyPI | `.venv/` | nixpkgs `python3Packages` equivalents exist; setup itself stays pip-based |
| Server / web app | **in-repo source** (Python) | the checkout | `serve/`, run from `.venv` | nothing extra |

## 6. Sources

- In-repo (branch `main` @ `f9d372a`): `setup.py` (constants L66-118, L226-227, L93-94, L104-113, L1461-1462,
  L1852-1900, L1937-2030, L2124-2129, L2132-2187, L2189-2209, L2231-2247, L2274-2336, L2368-2392, L3387-3435,
  L4094-4115), `CMakeLists.txt` (L9-62, L14-23, L31-49, L59-73, L120-166, L258-275, L476-497, L930-999),
  `Dockerfile`, `tools/mtp_fetch.py` (L1-47), `tools/mtp_pack.py` (L1-23), `third_party/ggml/`
  (`VERSION.txt`, `ggml-common.h`), `data/` listing, `requirements.txt`, `docs/INSTALL.md` (L170-189, AMD
  sections), `docs/DETAILS.md` (L255-264, L300-318, L450-465, L1028-1034), `docs/AMD_HIP.md` (L31-36,
  L57-78, L115-121).
- GitHub API (`gh api repos/Niko1221/Strata/releases`, 2026-10-05): asset names for v0.1.28–v0.1.39.
- Ticket #4 findings (`docs/research/setup-flow.md`, branch `research/setup-flow`): the 30-release
  no-Linux-asset sweep and the setup step order this document builds on.
- nix (2026-10-05, this host): `nix flake metadata nixpkgs --json`; source at
  `/nix/store/3zvg83mg9aavm9bgh26ydljchply7i25-source` (`pkgs/top-level/cuda-packages.nix`,
  `pkgs/top-level/all-packages.nix:2013-2025`, `pkgs/development/cuda-modules/{default.nix,backendStdenv/default.nix,packages/}`);
  the `nix eval` / `nix run --list-gpu-arch` outputs quoted in §4.2-4.4.
