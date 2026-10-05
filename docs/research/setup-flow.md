# The setup flow, end to end (ticket #4)

Research date: 2026-10-05. Repo: `kido5217/Vsrata`, branch `main` @ `f9d372a`; this is a copy of
`Niko1221/Strata` at v0.1.39 (`CMakeLists.txt:11` `project(strata VERSION 0.1.39)`; `MIN_ENGINE = (0, 1, 39)`
at `setup.py:118`). Method: reading code and docs only (`setup.sh`, `setup.py`, `serve/server.py`,
`tools/mtp_fetch.py` + the pack tools, `Dockerfile`, `docker-entrypoint.sh`, `docs/INSTALL.md`,
`docs/DETAILS.md`), plus read-only `gh` and Hugging Face API probes (recorded in §5). Nothing was executed:
no `setup.sh`/`setup.py` run, no server or engine start, no model files downloaded.

## TL;DR

- `./setup.sh` bootstraps a Python 3.10+ venv (installing Python via `sudo apt`/`dnf`/`pacman` only if
  missing) and then `exec`s `setup.py` in it (`setup.sh:5-39`).
- First run on Linux/NVIDIA: (1) resolve the shared data folder `Strata-data/` next to the repo, (2) survey the
  PC (nvidia-smi, RAM, CPU, driver), (3) ask four questions (family, size, context, images — plus KV cache and
  experimental speed projection), (4) `pip install` pinned packages into `.venv/`, (5) get the engine:
  download the ready-made `strata-linux-x64.zip` from GitHub releases if a HEAD check succeeds, **else compile
  from source** — as of 2026-10-05 no Linux asset has ever been published, so a bare Linux machine always
  compiles (10-20 min), (6) download the model GGUF shards from Hugging Face at pinned commits (resumable),
  (7) prepare the pack and fetch the ~5 GB MTP draft tensors from the original Qwen checkpoint by HTTP range
  requests, (8) write `strata-<model>.json` + `run-<model>.sh` in the repo root, (9) start `serve/server.py`,
  which spawns the engine with `--serve` and serves the OpenAI/Anthropic API on the port (default 8080).
- Everything is idempotent: each artifact carries a completion mark (`.done` files, `BUILD.json`, pack
  `native_experts.txt`, `mtp/rt/experts.bin`), so later runs skip downloads and preparation and go straight to
  starting the model.
- The unattended path `--setup --yes --no-start` (what `docker-entrypoint.sh:49` drives) is the same steps 1-8
  with every question answered by its recommended default (`setup.py:256-258`), risk confirmations silenced by
  the explicit flags, and step 9 replaced by `return 0` (`setup.py:4412-4413`). The Docker image additionally
  compiles the engine at `docker build` time (so the container never compiles) and persists the model, pack,
  MTP layer and config on the `/data` volume.

## 1. First-run step order (Linux, NVIDIA)

### 1.0 Before `setup.py`: `setup.sh`

`setup.sh` is a POSIX shell bootstrap (`setup.sh:1-39`):

1. `cd` to its own directory (`setup.sh:5`).
2. Find a Python 3.10+ that can create a venv **with pip**: `ok_py()` imports `sys, venv, ensurepip` and checks
   `sys.version_info >= (3, 10)` (`setup.sh:8`). Debian/Ubuntu ship `venv` without `ensurepip`, so that case is
   caught.
3. A half-finished `.venv` (python but no pip) is deleted (`setup.sh:10-12`).
4. If no suitable python3/python exists, install one with `sudo apt-get`/`sudo dnf`/`sudo pacman`
   (`python3`, `python3-venv`, `python3-pip`) (`setup.sh:20-35`), then re-check or fail with a hint
   (`setup.sh:30-34`).
5. Create the private environment: `python3 -m venv .venv` inside the Strata folder (`setup.sh:37`).
6. `exec .venv/bin/python setup.py "$@"` (`setup.sh:39`) — setup.py runs from then on inside `.venv/`.

(`Dockerfile` does the equivalent at build time: `python3 -m venv .venv` + `pip install -r requirements.txt`,
`Dockerfile:61-64`.)

### 1.1 `main()` entry: the data folder and the "already installed" shortcuts

`main()` starts by resolving the **data folder** — where the 70-120 GB of model files live —
(`setup.py:3641`; function at `setup.py:2630-2681`):

- Default: `Strata-data/` **next to** the Strata folder (`ROOT.parent / "Strata-data"`, `setup.py:2635`);
  `--data-dir` overrides; the choice is remembered per user in `~/.config/strata/settings.json`
  (`settings_path()`, `setup.py:2520-2523`).
- Model files from earlier Strata folders on the **same drive** are moved in (merge for `models/`, whole-move
  for `packs/` and `mtp/`, `setup.py:2661-2671`); configs pointing at the old place are repointed
  (`repoint_config`, `setup.py:2602-2627`, called at `setup.py:2672-2673`).
- Files on another drive are used where they are (no 70 GB copy) and listed in `elsewhere`
  (`setup.py:2656-2657`).
- `save_settings` writes `~/.config/strata/settings.json` with `data_dir` + the list of known Strata install
  folders (`setup.py:2679-2680`).

Then `installed_configs()` globs `strata-*.json` **in the repo root** (`setup.py:2849-2851`) and the flow
branches (`setup.py:3646-3713`):

- `--update` → `update_install` and return: pip packages, engine update, per-config upgrades, draft-vocab
  refresh; never starts the model (`setup.py:3648-3649`, `setup.py:3012-3038`).
- No config in this folder **and** no explicit flag → adopt the newest readable config from another Strata
  folder on the PC (`previous_config`, `setup.py:2788-2796`) and set up this copy the same way, reusing the
  model files (`setup.py:3652-3675`).
- Configs exist and none of `--setup/--model/--family/--check/--no-start` was given → this is a **plain
  start**: `update_installed_engine` (replace the engine only if a newer release is required,
  `setup.py:2033-2121`) then `start()` on the installed model(s) (`setup.py:3698-3713`).

A genuine first run has none of `strata-*.json` in the repo root and no `--update`, so it falls through to
step 1.

### 1.2 Step 1 — "checking your PC" (`setup.py:3716-3854`)

- GPUs: `gpus()` parses `nvidia-smi` (`setup.py:529-541`); `amd_gpus()` reads the amdgpu KFD topology from
  `/sys` (`setup.py:1375-1410`). Backend selection: explicit `--backend`, else NVIDIA if any usable NVIDIA
  card, else AMD if a usable Radeon is found; with both kinds present the user is asked (NVIDIA recommended)
  (`setup.py:3727-3737`). For Linux/NVIDIA the flow continues with NVIDIA.
- GPU choice: with several usable cards `choose_gpus` lists the options (the two best together, recommended;
  or one card each) and asks; the main GPU is the first chosen (`setup.py:3776-3784`; function
  `setup.py:763-798`). `--yes` takes the recommended default (`setup.py:797`).
- Driver: NVIDIA driver must be ≥ 580 for CUDA 13 (`MIN_DRIVER`, `setup.py:106`, checked at
  `setup.py:3791-3796`); CUDA toolkit 13 is the default, the experimental CUDA 12 engine (Pascal/Volta)
  when a chosen card is below sm_75 (`setup.py:3788`, `cuda_choice` `setup.py:601-616`).
- RAM: `ram_gb()` from `/proc/meminfo` (`setup.py:315-321`). If RAM is below the smallest model's need
  (32 GB − 4) it stops with a risk confirmation (`confirm_risk`, `setup.py:3803-3814`); the low-RAM mode
  (big GPU, little RAM) is the escape hatch (`setup.py:3802`).
- CPU: model + AVX-512/AVX2 detection via CPUID (`setup.py:402-526`); no AVX2 → setup forces
  `a.build = True` (compile an ISA-floored engine, experimental) (`setup.py:3826-3837`).
- `--check` stops here and prints a per-model "fits / tight / does not fit" verdict
  (`setup.py:3838-3854`).

### 1.3 Step 2 — "your choices" (`setup.py:3857-4088`)

Asked in order (each has an interactive prompt and a `--yes`/flag path):

1. **Family** — `qwen` (original Qwen3.8-Flash-Next) / `swift` (Swift 1.5) / `coder` / `unsloth`
   (`setup.py:3858-3867`; `FAMILIES` `setup.py:183-217`). Default: qwen.
2. **Model size** — from `MODELS` filtered by family: Q2_0 (66.4 GB), IQ2_XS (68.0), IQ3_XXS (75.8), IQ3_S
   (83.6), IQ1_M (Coder, 58.4), UD-IQ4_XS (93.7), UD-Q4_K_XL (111.3, experimental) (`setup.py:122-152`,
   `setup.py:3880-3892`). Recommended default: IQ3_XXS when RAM ≥ 60 GB, else the first listed
   (`setup.py:3891`). Budget-mode models (Unsloth) get a RAM-budget question instead of the low-RAM one
   (`setup.py:3894-3924`); risky picks (too little RAM, experimental) are kept with a confirmation
   (`confirm_risk`/`confirm_paging`, `setup.py:3807-3813`, `setup.py:3933-3934`).
3. **Context** — 8192 … 524288 (`CONTEXTS`, `setup.py:181`); recommended by the (smallest) card's VRAM
   (32K/64K/128K) and capped by the RAM rule (`setup.py:3938-3942`). Past the trained 262144 tokens, RoPE
   scaling is resolved: yarn (default) with factor = final context / 262144, or an interactive question
   (`setup.py:3966-3983`; `resolve_rope` `setup.py:3479-3503`).
4. **KV cache precision** — `int8` (default, "what every published number was measured with") / `q4_0` /
   `k8v4` (`setup.py:3985-3995`). 8K context uses fp16 KV.
5. **Images** — default **no**; with yes the image encoder (`mmproj-Qwen3.8-Flash-Next-BF16.gguf`, 0.9 GB
   download) runs on the GPU (1024 image tokens, 700 MiB VRAM reserved) or `--vision cpu` (300 tokens)
   (`setup.py:4002-4009`; `VISION` `setup.py:224-225`; prompt text `setup.py:4006-4007`).
6. **Low-RAM variant** — for the low-RAM mode only: `resident` (experts copied from the pack's
   `experts.bin` into RAM once) vs `mmap` (read through the OS file cache), chosen by how much fits
   (`setup.py:4015-4040`).
7. **Experimental speed projection** — off by default; when on, the control vector shipped in
   `data/experimental-speed-projection/` is applied to layers 4-44 (`setup.py:4043-4057`, `setup.py:220`).

Finally the **disk-space check**: `to_fetch + 8 GB` plus 40 GB for the Q2_0-on-AVX-512 conversion plus
~1 GB for images plus the low-RAM `experts.bin` arena must fit in the models directory
(`setup.py:4078-4088`).

### 1.4 Step 3 — Python packages (`setup.py:4091-4093`)

`pip_install(requirement_lines(), ...)` (`setup.py:1311-1326`) installs the pinned lines of
`requirements.txt` into `.venv`:

- `numpy` (2.5.3 on py≥3.12, 2.4.6 on 3.11, 2.2.6 older), `jinja2==3.1.6`, `regex==2026.9.10`,
  `pyyaml==6.0.3`, `tqdm==4.70.1`, `requests==2.34.2`, `cmake==4.4.3`, `ninja==1.13.2`, `pillow==12.3.0`,
  `psutil==7.2.2`, plus their pinned deps (`markupsafe`, `certifi`, `charset-normalizer`, `idna`,
  `urllib3`, `colorama` on win32) — `requirements.txt:1-22`.
- Skipped when a stamp file `.venv/.strata-pip.json` (`Path(sys.prefix) / ".strata-pip.json"`,
  `setup.py:1315`) already records the same list (`setup.py:1316-1322`).

### 1.5 Step 4 — the engine (`setup.py:4096-4129`), including `get_prebuilt`'s HEAD/fallback logic

1. **llama.cpp source** — `get_llama_cpp()` (`setup.py:4097`; function `setup.py:1254-1284`): downloads
   `https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.zip`
   (`LLAMA_CPP_COMMIT`/`LLAMA_CPP_ZIP`, `setup.py:93-94`), extracts it (minus `tools/ui`) into
   `third_party/llama.cpp/` — this provides `ggml` (the build), `gguf-py` (the pack/MTP tools) and `mtmd`
   (images) — then deletes the zip (`setup.py:1282`). Skipped when
   `third_party/llama.cpp/ggml/CMakeLists.txt` and `gguf-py/` already exist (`setup.py:1257`).
2. **Ready-made engine** — `get_prebuilt(url_base, gpu, vision, toolkit=13)` (`setup.py:4108`; function
   `setup.py:1946-2030`). The exact logic:

   a. **Installed-engine check** (`setup.py:1952-1969`): if `engine/BUILD.json` + `engine/strata` exist:
      - `source == "local"` (compiled here, or the Docker image's build) → return `None` — the compile path
        verifies it instead (`setup.py:1956-1957`);
      - installed engine lacks code for this GPU's arch(s) → return `None`, "compiling instead"
        (`setup.py:1959-1964`);
      - version ≥ `MIN_ENGINE` (0.1.39) → return `engine/` as-is, "ready-made engine already installed"
        (`setup.py:1965-1967`);
      - older → `BUILD.json` is deleted and the flow falls through to a download
        (`setup.py:1968-1969`).
   b. **URL bases** (`prebuilt_bases`, `setup.py:1937-1943`): with the default URL there are **two** bases,
      tried in order:
      1. `https://github.com/Niko1221/Strata/releases/download/v0.1.39/` — this checkout's own version
         (`PREBUILT_TAG_URL`, `setup.py:102`; version from `source_version()`, `setup.py:2854-2857`);
      2. `https://github.com/Niko1221/Strata/releases/latest/download/` — the latest-release fallback
         (`PREBUILT_URL`, `setup.py:101`; #214: "an older checkout keeps the engine it shipped with").
      An explicit `--prebuilt`/`STRATA_PREBUILT_URL` yields a single base (`setup.py:1941-1942`,
      `setup.py:3592`).
      The asset is `strata-linux-x64.zip` (`PREBUILT_ASSET`, `setup.py:103`; `strata-linux-x64-cuda12.zip`
      for the experimental CUDA 12 engine, `setup.py:112`).
   c. **The HEAD probe** (`setup.py:1975-1987`): for each base, an HTTP **HEAD** (60 s timeout, UA
      `strata-setup`) on `base + asset`. First base that answers 200 wins. If a base fails, the next is tried
      with the message "No ready-made engine for v0.1.39 (…): the latest release instead"
      (`setup.py:1983-1984`); if **all** fail → warn "no ready-made engine at … : compiling instead" and
      return `None` (`setup.py:1986-1987`).
      **Observed 2026-10-05 (verified):** no `strata-linux-x64.zip` exists on either base — all 30 releases
      listed for `Niko1221/Strata` ship Windows assets only (`strata-windows-x64.zip`,
      `strata-windows-x64-hip.zip`, `strata-windows-x64-cuda12.zip`), and HEAD on
      `.../download/v0.1.39/strata-linux-x64.zip` and `.../latest/download/strata-linux-x64.zip` both
      returned **404** (while the Windows zips returned 200). So on a bare Linux machine the default flow
      **always compiles from source**; the prebuilt route is a no-op there today. Windows is the opposite:
      v0.1.39's `strata-windows-x64.zip` (123,629,160 B) downloads.
   d. **Download + install** (when a base answered) (`setup.py:1988-2030`): `download()` the zip to
      `engine/strata-linux-x64.zip` (resumable, §2.1); unpack to `engine/_unpack/`; then validate
      `BUILD.json`: version < `MIN_ENGINE` → drop the archive and compile instead
      (`setup.py:1995-2005`); archs missing the GPU's arch → same (`setup.py:2006-2014`); otherwise move
      `strata`, `strata-vision`, `BUILD.json` into `engine/` (replacing existing files), delete the zip
      (`drop_archive`, `setup.py:2015-2021`), `chmod 0755` the binaries (`setup.py:2024-2027`).
   e. A ready-made **image encoder** that lacks this GPU's arch downgrades images to the CPU encoder
      (`prebuilt_vision`, `setup.py:2260-2271`, applied at `setup.py:4116`).
3. **CUDA libraries for the ready-made engine** — `pip_cuda_libs(13)` (`setup.py:4111`;
   `setup.py:2124-2129`): `nvidia-cublas==13.0.2.14` + `nvidia-cuda-runtime==13.0.96`
   (`CUDA_WHEELS`, `setup.py:105`), "~0.4 GB" by setup's own count (`setup.py:2129`), landing in
   `.venv/.../site-packages/nvidia/cu13/lib` (`cuda_lib_dirs`, `setup.py:1329-1342`). (CUDA 12 wheels for the
   experimental engine: `nvidia-cublas-cu12==12.9.1.4` + `nvidia-cuda-runtime-cu12==12.9.79`, ~0.7 GB,
   `setup.py:113`, `setup.py:2127`.)
4. **Compile fallback** — when `get_prebuilt` returned `None` (always, on Linux today): `build_engine`
   (`setup.py:4118`; function `setup.py:2274-2332`):
   - skips when a local build is current: `BUILD.json` `source == "local"`, the source fingerprint
     (`source_hash` over `CMakeLists.txt`, `src/`, `include/`, `third_party/ggml`, plus the
     `LLAMA_CPP_COMMIT`, `setup.py:2235-2243`) matches, no new GPU archs, and the vision encoder is current
     (`setup.py:2288-2300`) — this is why a `git pull` only recompiles the changed files
     (`setup.py:2307-2310`);
   - otherwise installs build tools first — `install_build_tools` (`setup.py:2132-2205`) finds `nvcc` on PATH
     / `/usr/local/cuda*` / `/opt/cuda*` or offers to install `build-essential` + the CUDA 13 toolkit with
     `sudo` (one question); `find_nvcc` at `setup.py:951-985`;
   - builds `strata` via CMake+Ninja into `build/` and copies it to `engine/strata`
     (`cmake_build`, `setup.py:2208-2228`, `setup.py:2311-2316`), and `tools/vision` → `engine/strata-vision`
     when images are on (`setup.py:2317-2324`);
   - writes `engine/BUILD.json` with `source: "local"`, version, archs, `cuda_dirs` (the toolkit's library
     dirs), and the source fingerprints (`setup.py:2327-2330`).
5. **Engine version gate for the Unsloth budget models** — checked before the 94-111 GB download
   (`setup.py:4124-4128`).

`lib_dirs` for the engine are `BUILD.json`'s `lib_dirs`/`cuda_dirs` or pip's wheel dirs
(`setup.py:4123`; `engine_lib_dirs` `setup.py:3426-3429`).

### 1.6 Step 5 — the model files (`setup.py:4132-4169`)

- Target: `Strata-data/models/<tag>/` (`models_dir`, `setup.py:4061`; `tag = fam["tag"] + model`,
  `setup.py:3936`). Files come from the family's HF base URL **at the pinned revision**
  (`FAMILIES[...]["hf"]`, `setup.py:186-211`; `hf()` builds
  `<endpoint>/<repo>/resolve/<40-hex sha>/`, `setup.py:82-84`; the revisions table `setup.py:65-70`).
- Per shard not already complete (`<file>.done` present, `setup.py:4067`, `setup.py:1030-1031`):
  - **qwen/coder shard 2 reuse**: the n-gram table shard is byte-identical across sizes, so an existing
    `*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf` is **hard-linked** instead of re-downloaded
    (`setup.py:4145-4154`);
  - otherwise `download(fam["hf"].format(q=model) + s.name, s)` (`setup.py:4155`) — the resumable downloader
    (§2.1). Files already copied by hand can be placed in the folder or supplied via `--gguf-dir`
    (`setup.py:4135-4141`, `setup.py:4079-4080`).
- `check_shards` reads each shard's GGUF header and fails if a shard is truncated
  (`setup.py:4156`; `setup.py:1233-1251`).
- Unsloth shards additionally get a SHA-256 check against the pinned hashes in `setup.py`
  (`UNSLOTH_SHARDS`/`UNSLOTH_IQ4_XS_SHARDS`, `setup.py:155-173`, applied at `setup.py:4157-4159`;
  `verify_sha256` `setup.py:1186-1208`).
- **Vision encoder** (only when images are on): `mmproj-*.gguf` from the family's HF repo root
  (`setup.py:4161-4169`; `fam["mmproj"]` `setup.py:188-214`).

### 1.7 Step 6 — the pack and the MTP draft layer (`setup.py:4172-4214`)

**The pack** (into `Strata-data/packs/<tag>/`; `pack`, `setup.py:4173`):

- Q2_0 on an AVX-512 CPU (qwen family): `tools/strata_pack.py build` + `tools/pack_index.py` +
  `tools/strata_tokenizer.py` — a one-time ~40 GB conversion of the experts into the fast AVX-512 layout
  (`setup.py:4175-4184`);
- otherwise `tools/iq_pack.py --gguf <shard 1> --out <pack>`: every tensor as the GGUF stores it, experts
  read natively from the GGUF at start ("seconds to build"); writes `native_experts.txt` **last** (the
  finish marker), `dense.bin`, `conversions.json`, and `tokenizer/` (vocab, merges, token_type, chat
  template) (`iq_pack.py` docstring; `setup.py:4185-4189`); Unsloth gets `--compat-bf16`
  (`FAMILIES["unsloth"]["pack_args"]`, `setup.py:215`);
- low-RAM mode only: `tools/iq_pack.py --experts-bin` writes the whole expert arena into one
  `experts.bin` (`setup.py:4190-4193`).
- Skipped when `pack/native_experts.txt` + `pack/tokenizer/vocab.json` exist (`setup.py:4185`), and
  `experts.bin` when it exists (`setup.py:4190`).

**The MTP draft layer** (into `Strata-data/mtp/`; `setup.py:4195-4214`): the GSQ-RCO GGUFs ship no MTP
head, so the draft layer is built from the **original BF16 checkpoint** `Qwen/Qwen3.8-Flash-Next`
(360 GB in 131 shards) **without downloading it**:

1. `tools/mtp_fetch.py fetch --out <mtp>` (`setup.py:4204`): reads
   `model.safetensors.index.json` and each touched shard's safetensors header with **HTTP range requests**
   at the **pinned revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`** (pin 2026-09-30; `mtp_fetch.py:29-37`;
   overridable with `STRATA_MTP_REVISION`), then fetches only the 31 `mtp.*` tensor byte ranges
   (~5 GB, `setup.py:4202-4203`), writing one raw file per tensor to `mtp/tensors/` plus
   `mtp-manifest.json`; every tensor is checked against SHA-256 (`mtp_fetch.py:44-107`, `mtp_fetch.py:246-253`);
2. `tools/mtp_pack.py --src <mtp> --experts q2_0 --out <mtp>/mtp-q2_0.gguf` (`setup.py:4205`): the MTP block
   as a Strata GGUF — dense tensors BF16 (~0.18 GB) + the 512 routed experts re-quantized Q2_0 (~0.71 GB)
   (`mtp_pack.py` docstring);
3. `tools/mtp_rt.py --gguf <mtp>/mtp-q2_0.gguf --out <mtp>/rt` (`setup.py:4207`): runtime files
   `rt/experts.bin` (engine blob layout), `rt/dense.bin`, `rt/dense.txt` (`mtp_rt.py` docstring).
- The whole fetch is skipped when `mtp/rt/experts.bin` exists and `mtp_corrupt()` passes
  (`setup.py:4197-4201`; `setup.py:3355-3363`); a corrupted fetch (a mirror that ignored Range requests,
  #327) is re-fetched and rebuilt (`setup.py:4198-4200`).
- `refresh_draft_vocab` (`setup.py:4210-4211`; function `setup.py:3366-3385`) copies the draft layer's token
  subset (`cjk` default, or `en`/`cyrillic`/`fr`) from the repo's `data/draft_vocab*.bin` to
  `mtp/rt/draft_vocab.bin`, replacing only shipped subsets.

### 1.8 Step 7 — the run config and the start script (`setup.py:4217-4393`)

- The PLE (per-layer token embedding) table's shard is located by reading the shards' GGUF tensor lists
  (`tools/gguf_reader`, `setup.py:4218-4222`); missing → fail (`setup.py:4223`).
- The engine **`args` list** is assembled (`setup.py:4224-4323`): `--pack <pack> --native <shard 1>
  [--ple-gguf <shard>] --expert-profile <repo>/data/expert-profile.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp <mtp>/rt --max-context <ctx>` plus, conditionally: RoPE flags
  (`setup.py:4228-4229`), `--kv <int8|q4_0|k8v4>` above 8K (`setup.py:4230-4231`),
  `--resident-experts`/`--mmap-experts` (low-RAM, `setup.py:4232-4237`), `--ple-io ram` on rotational disks
  with enough RAM (`setup.py:4238-4250`), `--kv-resident 32768` KV streaming from 64K up when it fits
  (`setup.py:4254-4292`), `--resident-budget-gib` for the Unsloth budget models
  (`setup.py:4293-4294`), `--vision --vram-reserve-mib 700` with images (`setup.py:4295-4300`),
  `--vram-reserve-mib` as given (`setup.py:4301-4311`), and the speed-projection flags
  (`setup.py:4320-4323`).
- The **config dict** (`setup.py:4324-4374`): `exe` (engine path), `args`, `cwd` (repo root), `tokenizer`
  (pack tokenizer dir), `model_name` (`<family name>-<model>`, e.g. `qwen3.8-flash-next-iq2_xs`), `log`
  (`<repo>/strata-<tag>.log`), `lib_dirs`, `port`; plus `cuda: 12` when applicable, `gpu`/`gpus_asked`/
  `layer_split` for multi-card, `host`/`api_key`, `draft_vocab`, `open_browser`, `parallel`, and the
  `vision` block (`exe` = `strata-vision`, `mmproj`, `model`, `gpu`, `max_tokens`).
- Worker-pool defaults: a saved calibration wins, else `recommend_pool_workers` tunes `args`
  (`setup.py:4376-4383`).
- `write_setup_config(cfg_path, cfg, ...)` writes **`<repo>/strata-<tag>.json`** atomically
  (`write_config`, `setup.py:2684-2689`), carrying over keys the user added to an earlier config
  (`carry_over`, `setup.py:2700-2726`) and keeping the earlier file as `strata-<tag>.json.bak` when it
  changes (`setup.py:2739-2769`).
- `write_run_script` writes **`<repo>/run-<tag>.sh`** (Linux) / `run-<tag>.bat` (Windows):
  `cd <repo>; exec <venv python> serve/server.py --engine strata --config <cfg> --port <port> [--open]`
  (`setup.py:3453-3466`, called at `setup.py:4385`).
- Optional calibration: asked only when interactive and not `--yes`/`--no-start`
  (`setup.py:4387-4390`).

### 1.9 `start()` and the server (`setup.py:4414` → `setup.py:3066-3195` → `serve/server.py`)

`start()` (unless `--no-start`):

1. `upgrade_config` migrates old configs in place (e.g. `--prefill 2048` → `auto`, WSL KV-streaming off;
   rewrites the config when changed, `setup.py:3070`, `setup.py:2987-3009`).
2. Fails fast if `cfg["exe"]` or the model GGUFs are missing (`setup.py:3071-3073`).
3. Keep-options from this run's flags are saved into the config and rewritten: `--vram-reserve-mib`,
   `--host`, `--api-key`, `--draft-vocab`, `--no-browser`, and `--gpus` (saved permanently; `--gpu` is
   per-start only) (`setup.py:3074-3089`, `setup.py:3132-3142`).
4. `cfg_path.touch()` marks the most recently used model (`setup.py:3090`); the draft-vocab subset is
   refreshed if the config has `--mtp` (`setup.py:3091-3092`).
5. GPU sanity: re-check the card(s); `ensure_engine_for` compiles the installed engine for any card it
   lacks code for, before starting (`setup.py:3145-3161`; `setup.py:3388-3412`); a newly-added Pascal/Volta
   card moves the model to the experimental CUDA 12 engine (`setup.py:3432-3450`).
6. The command is `[venv python] serve/server.py --engine strata --config <cfg> --port <port>
   [--open]` (`setup.py:3093-3094`, `setup.py:3162-3164`), announced with the 1-3 minute load warning
   (`setup.py:3171-3186`). With `STRATA_EXECV` set (Docker image) the setup process `execv`s the server and
   becomes PID 1; otherwise the server is a child (`setup.py:3190-3195`; `Dockerfile:43-46`).

`serve/server.py` (`main()`, `server.py:3898-4141`): checks the port is free (`server.py:3941-3945`); loads
the pack tokenizer and hard-fails without `vocab.json` (`server.py:3946-3960`); starts the vision encoder
process when the config has a `vision` block (`server.py:3974-3981`); spawns the engine
`subprocess.Popen([exe, "--serve", *args], ...)` with `stderr` appended to `strata-<tag>.log`,
`CUDA_VISIBLE_DEVICES` from the config's `gpu`, and the `lib_dirs` prepended to `LD_LIBRARY_PATH`
(`server.py:454-455`, `child_env` `server.py:1560-1575`); waits for the engine's `READY <max_context>` line
over its stdout control protocol (`server.py:458-467`); then serves the OpenAI- and
Anthropic-compatible API plus the chat page on `host:port` (`server.py:4093-4099`), opens the browser when
`--open`/`open_browser` (`server.py:4115-4117`), and blocks until Ctrl+C/SIGTERM, at which point it stops
the engine and exits (`server.py:4127-4141`). The engine talks to the server over stdin/stdout control
lines (`INFO`/`READY`/`PP`/...; `server.py:458-467`, `server.py:517-532`), not via files.

## 2. Every external download

### 2.1 The downloader itself

`download()` (`setup.py:1026-1098`): skipped when `<file>.done` exists (`setup.py:1030-1031`); a HEAD gives
the total size — **on a 404 of a pinned-revision URL it warns and re-targets the same file at the
repository's `main` revision** (`hf_unpinned`, `setup.py:87-89`, applied at `setup.py:1049-1053`; #214);
then GETs with `Range: bytes=N-` appended to `<file>.part`, up to 30 attempts with 10 s backoff, resuming a
partial file (`setup.py:1065-1092`); verifies the final size and replaces `<file>.part` with `<file>`,
writing the `<file>.done` mark (`setup.py:1093-1097`). `HF_ENDPOINT` retargets everything to a mirror
(`setup.py:76-79`); the pinned revisions and SHA-256 checks apply on any host. `mtp_fetch.py` has its own
range-based fetcher with the same 404-then-`main` fallback for the checkpoint index
(`mtp_fetch.py:153-167`).

### 2.2 Download table (first run, Linux/NVIDIA, defaults: qwen + IQ2_XS + 32K + no images)

| # | What | Source (repo / asset / pinned commit) | Size | Where it lands |
|---|------|----------------------------------------|------|----------------|
| 1 | llama.cpp source (ggml build, gguf-py, mtmd) | `github.com/ggml-org/llama.cpp` archive zip at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (2026-09-20, "CUDA: enable sparse fa for qwen4 (#28770)") — `setup.py:93-94` | 39,564,399 B (~37.7 MiB; HEAD-measured 2026-10-05) | `third_party/llama.cpp-3cf0325.zip` → extracted to `third_party/llama.cpp/` (minus `tools/ui`); zip deleted — `setup.py:1259-1282` |
| 2 | Ready-made engine zip (CUDA 13) | `Niko1221/Strata` release asset `strata-linux-x64.zip`, tag `v0.1.39` first, then `latest` — `setup.py:101-103, 1937-1943` | **Not published as of 2026-10-05** (HEAD 404 at both bases; Windows v0.1.39 assets for reference: `strata-windows-x64.zip` 123,629,160 B, `strata-windows-x64-cuda12.zip` 175,358,734 B, `strata-windows-x64-hip.zip` 598,941,708 B) | `engine/` — `strata`, `strata-vision`, `BUILD.json`; zip deleted — `setup.py:1989-2021`. **On Linux today: never reached → compile from source instead** |
| 3 | (Windows only, same as 2) CUDA 12 experimental engine | release asset `strata-windows-x64-cuda12.zip` — `setup.py:112` | 175,358,734 B (v0.1.39) | `engine-cuda12/` |
| 4 | NVIDIA CUDA 13 libraries (cuBLAS + runtime) | PyPI pip wheels `nvidia-cublas==13.0.2.14`, `nvidia-cuda-runtime==13.0.96` — `setup.py:105`, installed at `setup.py:4111` via `pip_cuda_libs` `setup.py:2124-2129` | ~0.4 GB (setup.py's own count, `setup.py:2129`) | `.venv/lib/python3.x/site-packages/nvidia/cu13/lib` — `cuda_lib_dirs` `setup.py:1329-1342`. (CUDA 12 wheels, experimental: `nvidia-cublas-cu12==12.9.1.4`, `nvidia-cuda-runtime-cu12==12.9.79`, ~0.7 GB — `setup.py:113, 2127`) |
| 5 | Model GGUF shards (the big one) | Hugging Face, family repo at pinned revision: qwen/coder `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (pin 2026-09-29); swift `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `b22d729eae29b5796f76fb70f91aef549b9fc52c` (pin 2026-09-24); coder `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` @ `5348543e0147355ac9cbcb031184a3546350988e` (pin 2026-09-29); unsloth `unsloth/Qwen3.8-Flash-Next-GGUF` @ `38bb39ee97821de2c9009abb7e93950eec396e66` (pin 2026-09-30) — `setup.py:65-70`; URL shape `<endpoint>/<repo>/resolve/<sha>/<file>` `setup.py:82-84` | per size, measured 2026-10-05 (HEAD at `main`; all four repos' current sha still equals the pinned sha, so identical): see 2.3 | `Strata-data/models/<tag>/<name>-0000<i>-of-0000N.gguf` (+ `.part` while downloading, `.done` after) — `setup.py:4061-4063, 4155`. Coder + qwen shard 2 (28,800,138,432 B) is hard-linked across sizes when present — `setup.py:4145-4154` |
| 6 | Vision encoder (only with images) | same family repo, root file: qwen `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, swift `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf` (unsloth reuses qwen's) — `setup.py:188-214` | 907,543,008 B (qwen) / 907,543,520 B (swift); HEAD-measured 2026-10-05 — matches setup's "0.9 GB" (`setup.py:4006`) | `Strata-data/models/mmproj-*.gguf` — `setup.py:4161-4168` |
| 7 | MTP draft tensors (speculative decoding, ~2x faster output) | `Qwen/Qwen3.8-Flash-Next` (the original BF16 checkpoint, 360 GB / 131 shards) at pinned revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` (pin 2026-09-30) — `mtp_fetch.py:29-37`; only the 31 `mtp.*` tensor byte ranges via HTTP 206 requests (shard safetensors headers read by range too), never whole shards — `mtp_fetch.py:170-256` | ~5 GB (setup.py's count, `setup.py:4202-4203`) | `Strata-data/mtp/tensors/<tensor>.bin` (31 raw files) + `Strata-data/mtp/mtp-manifest.json` — `mtp_fetch.py:227, 255`. Skipped when `Strata-data/mtp/rt/experts.bin` is intact — `setup.py:4197-4201` |
| 8 | Python packages (step 3) | PyPI, pinned by `requirements.txt` (see 1.4) | not measured (wheel cache) | `.venv/` |
| 9 | Build tools (only when compiling — always, on bare Linux today) | distro packages via sudo: `build-essential` + CUDA 13 toolkit (`setup.py:2132-2205`); Python itself via apt/dnf/pacman if missing (`setup.sh:20-35`) | not measured | system |
| 10 | (AMD/HIP only, out of scope) ROCm wheels, ~10 GB | `setup.py:1753-1849` (rocm index) | ~10 GB (`docs/INSTALL.md:19`) | `.venv/` |

Not downloads (shipped with the repo, read-only inputs): the expert profile `data/expert-profile.bin`
(qwen) / `expert-profile-coder.bin` (coder) used via `--expert-profile` (`setup.py:4225`; files present at
196,632 B / 98,328 B), the draft-vocab subsets `data/draft_vocab*.bin` (`setup.py:3372`), and the
experimental speed-projection vector `data/experimental-speed-projection/*.gguf` (`setup.py:220`; 483,520 B).

### 2.3 Model shard sizes, measured 2026-10-05 (HF HEAD at `main`; the repos' current sha equals the pinned sha)

| family / model | shard 1 (B) | shard 2 (B) | other shards (B) | total (B) | setup.py `download_gb` |
|----------------|------------|-------------|------------------|-----------|------------------------|
| qwen Q2_0 | 37,623,740,192 | 28,800,138,432 | — | 66,423,878,624 (~66.4 GB) | 66.4 |
| qwen IQ2_XS (Docker default) | 39,225,954,592 | 28,800,138,432 | — | 68,026,093,024 (~68.0 GB) | 68.0 |
| qwen IQ3_XXS | 47,039,860,096 | 28,800,138,432 | — | 75,839,998,528 (~75.8 GB) | 75.8 |
| qwen IQ3_S | 54,817,524,224 | 28,800,138,432 | — | 83,617,662,656 (~83.6 GB) | 83.6 |
| swift IQ2_XS | 39,788,473,344 | 28,363,693,824 | — | 68,152,167,168 (~68.2 GB) | 68.0 |
| swift IQ3_XXS | 39,785,790,560 | 36,180,282,560 | — | 75,966,073,120 (~76.0 GB) | 75.8 |
| coder IQ1_M | 29,608,446,496 | 28,800,138,432 | — | 58,408,584,928 (~58.4 GB) | 58.4 |
| unsloth UD-Q4_K_XL (experimental) | 10,946,624 | 49,859,583,136 | 49,376,141,504 + 12,087,983,520 (4 shards) | 111,334,654,804 (~111.3 GB) | 111.3 |
| unsloth UD-IQ4_XS | 10,946,624 | 49,835,229,856 | 43,836,407,744 (3 shards) | 93,682,584,224 (~93.7 GB) | 93.7 |

(Unsloth byte counts are the pinned SHA-256 table's sizes, `setup.py:155-173`; every other row was
HEAD-measured this session. qwen/coder shard 2 is the identical 28.8 GB n-gram table across sizes —
`setup.py:4145-4154` and the 28.8 GB default at `setup.py:4242`.)

## 3. Every file written, and who rewrites them on later runs

`<tag>` is `fam["tag"] + model` lowercased: `iq2_xs` (qwen), `swift-iq2_xs`, `coder-iq1_m`,
`unsloth-ud-iq4_xs`, ... (`setup.py:3936`, `setup.py:4375`).

### 3.1 In the Strata folder (repo root)

| File | What it is | Written by (first run) | Rewritten on later runs? |
|------|------------|------------------------|--------------------------|
| `.venv/` | the private Python env | `setup.sh:37` (or `Dockerfile:61`) | packages updated only when `requirements.txt` outgrows the stamp `.venv/.strata-pip.json` — `pip_install`, `setup.py:1315-1326` |
| `.venv/.strata-pip.json` | pip stamp (installed list) | `setup.py:1325` | same rule as above |
| `third_party/llama.cpp/` | llama.cpp source at the pinned commit | `setup.py:1263-1281` | **skipped** when `ggml/CMakeLists.txt` + `gguf-py/` exist — `setup.py:1257`. (A `git pull` that changes it is what the engine's source-hash check reacts to, `setup.py:2235-2243`.) |
| `third_party/llama.cpp-3cf0325.zip`, `third_party/_unpack/` | download/unpack temporaries | `setup.py:1259-1261` | **deleted** after unpack — `setup.py:1281-1283`; never kept |
| `engine/strata`, `engine/strata-vision`, `engine/BUILD.json` | the engine | prebuilt: unpack-and-replace `setup.py:2015-2021`; Linux reality: compiled `setup.py:2316, 2324` + `BUILD.json` `setup.py:2327-2330` (`source: "local"`) | plain start: `update_installed_engine` re-downloads the prebuilt only if the installed version < `MIN_ENGINE` — `setup.py:3700`, `setup.py:2033-2121` (a `source: "local"` engine is never re-downloaded, `setup.py:1956-1957`; instead `build_engine` recompiles only when the source hash or GPU archs changed, `setup.py:2295-2300`). `--setup`: same checks, then the `get_prebuilt` path above. Docker: fixed at image build — `Dockerfile:70-99` |
| `engine/strata-linux-x64.zip` (or cuda12) | prebuilt archive | `setup.py:1989` | **deleted** after a successful unpack (`drop_archive`, `setup.py:2021`) — so it is never on disk between runs |
| `engine-cuda12/...` | experimental CUDA 12 engine (Pascal/Volta) | `get_prebuilt(toolkit=12)` / `build_engine(toolkit=12)` — `setup.py:2282, 2305` | same rules as `engine/`, per model config (`use_cuda12`, `setup.py:3432-3450`) |
| `build/`, `build-vision/` (or `-cuda12`) | CMake/Ninja build trees | `build_engine`, `setup.py:2305-2324` | kept; reused for incremental rebuilds after a `git pull` (`setup.py:2299-2310`). The Docker image deletes them — `Dockerfile:103` |
| `build-strata.bat`, `build-vision.bat` | Windows-only build scripts | `cmake_build`, `setup.py:2218-2222` | rewritten on Windows builds |
| `strata-<tag>.json` | the run config (exe, engine args, cwd, tokenizer, log, lib_dirs, port, gpu/layer_split, host, api_key, vision, ...) | `write_setup_config`, `setup.py:4384` (step 7) | **rewritten by every `--setup` run for that model** (user-added keys carried over, `setup.py:2700-2726`); also rewritten in place by `start()` when it saves keep-options (`--host`/`--api-key`/`--vram-reserve-mib`/`--no-browser`/`--gpus`, `setup.py:3074-3089, 3132-3142`), by `upgrade_config` migrations (`setup.py:3007-3008`), and by `ensure_engine_for`/`use_cuda12` when the engine changes (`setup.py:3411, 3448`) |
| `strata-<tag>.json.bak` | the previous config, kept when a `--setup` run changes it | `write_setup_config`, `setup.py:2754-2761` | replaced each time the config changes on a `--setup` run |
| `strata-<tag>.json.tmp` | atomic-write temp | `write_config`, `setup.py:2687-2689` | momentary; `os.replace`d over the config |
| `run-<tag>.sh` (Linux) / `run-<tag>.bat` | one-line start script for this model | `write_run_script`, `setup.py:4385` (step 7) | **rewritten on every `--setup` run** for that model (`setup.py:3453-3466`) — it just re-invokes `serve/server.py` with the config, so it is stable content |
| `strata-<tag>.log` | engine + vision-encoder log | created/appended by the server: `cfg["log"]` `setup.py:4325`; `open(log, "a")` `server.py:443`, `stderr=log` `server.py:454-455`, vision `server.py:3980` | **appended by every start** (the server records where each start begins, `server.py:445-446`) |
| `strata-<tag>.shared-settings.json` | web-page chat settings shared with other API apps | `server.py:4078` (`svc.shared_path`), written by the web app when the user saves settings for other apps | rewritten by the web app; not by setup |
| `data/...` (profiles, draft vocabs, ESP vector) | shipped inputs | — (part of the repo) | read-only inputs; setup never rewrites them |

### 3.2 In the data folder (`Strata-data/` by default; `/data` in Docker)

| File | What it is | Written by (first run) | Rewritten on later runs? |
|------|------------|------------------------|--------------------------|
| `models/<tag>/<shard>.gguf` | the model GGUFs | `download`, `setup.py:4155` | **skipped** when `<file>.done` exists (`setup.py:1030-1031`, `setup.py:4067`); a re-download resumes from `.part` (`setup.py:1065-1073`). A newer copy of Strata finds them via `data_folder` and adopts the previous config (`setup.py:3652-3675`) |
| `models/<tag>/<shard>.gguf.part` / `.done` | download temp / completion mark | `setup.py:1041, 1096-1097` | `.part` removed on success; `.done` persists (that is the skip signal) |
| `models/mmproj-*.gguf` | image encoder weights | `setup.py:4168` | skipped when present (`setup.py:4162, 4164`) |
| `packs/<tag>/native_experts.txt`, `dense.bin`, `conversions.json`, `tokenizer/` (vocab.json, merges.txt, token_type.json, chat_template.jinja) | the prepared pack | `tools/iq_pack.py` — `setup.py:4188` (Q2_0/AVX-512: `tools/strata_pack.py` + `tools/pack_index.py` + `tools/strata_tokenizer.py`, `setup.py:4179-4184`) | **skipped** when `native_experts.txt` + `tokenizer/vocab.json` exist (`setup.py:4185`) — `native_experts.txt` is written last precisely so a half-built pack is not finished |
| `packs/<tag>/experts.bin` (+ `experts.bin.src.json`) | the whole expert arena in one file (low-RAM mode; or the Q2_0 AVX-512 pack) | `setup.py:4192` / `setup.py:4179` | skipped when present (`setup.py:4190`) |
| `mtp/tensors/*.bin` (31 files), `mtp/mtp-manifest.json` | the raw MTP tensors from the original checkpoint | `tools/mtp_fetch.py fetch` — `setup.py:4204` (`mtp_fetch.py:227-256`) | skipped together with `mtp/rt` when `rt/experts.bin` exists and is not corrupt (`setup.py:4197-4201`); a corrupt fetch deletes and re-fetches the bad tensors (`setup.py:4198-4200`, `mtp_fetch.py:249-253`) |
| `mtp/mtp-q2_0.gguf` | the MTP block as a Strata GGUF | `tools/mtp_pack.py` — `setup.py:4205` | rebuilt only in the corrupt/re-fetch case |
| `mtp/rt/experts.bin`, `mtp/rt/dense.bin`, `mtp/rt/dense.txt` | the engine's runtime MTP files | `tools/mtp_rt.py` — `setup.py:4207` | **skipped** when `rt/experts.bin` exists and passes `mtp_corrupt` (`setup.py:4197-4201`); this file's existence is the MTP step's done-signal |
| `mtp/rt/draft_vocab.bin` | the draft layer's token subset (default `cjk`) | `refresh_draft_vocab` — `setup.py:4211` | **refreshed on every `--setup` run, `--update` run, and plain `start()`** of a model with `--mtp` (copied only when a *different shipped* subset is present, `setup.py:3372-3385`; a hand-made subset is kept) — `setup.py:3091-3092`, `setup.py:3029-3030` |
| `mtp/mtp-inventory.json` / `mtp-inventory.md` / `tensors/verified.json` | inventory / verify stamps | only via `tools/mtp_fetch.py inventory|verify` (not part of the setup flow) | n/a |

### 3.3 User config

| File | What it is | Written by | Rewritten on later runs? |
|------|------------|------------|--------------------------|
| `~/.config/strata/settings.json` (or `$XDG_CONFIG_HOME/strata/settings.json`; Windows `%APPDATA%\Strata\settings.json`) | remembers `data_dir` and the list of known Strata install folders (`installs`) | `data_folder` → `save_settings` on **every** `setup.py` run — `setup.py:2679-2680` (path: `setup.py:2520-2523`) | **rewritten on every run** (even a plain start: `main()` calls `data_folder` first, `setup.py:3641`) |

### 3.4 Who rewrites what, summarized by later-run path

- **Plain start** (`./setup.sh` with `strata-*.json` present; `setup.py:3698-3713`): `update_installed_engine`
  (engine re-download only when `MIN_ENGINE` is newer, `setup.py:2033-2121`), then `start()`:
  `upgrade_config` may rewrite the config (`setup.py:3007-3008`), keep-options may rewrite it
  (`setup.py:3074-3089`), `cfg_path.touch()` updates its mtime (`setup.py:3090`), the draft vocab is
  refreshed (`setup.py:3091-3092`); the server appends to `strata-<tag>.log`. No model/pack/MTP work.
- **`--setup` re-run** (another model, or changed settings; `setup.py:3650`): the full steps 1-8 again, each
  download skipped by its `.done`/existence mark; `strata-<tag>.json` rewritten with the earlier file kept
  as `.bak`; `run-<tag>.sh` rewritten; settings.json rewritten. Model files from other installs are merged
  into the data folder by `data_folder` (`setup.py:2661-2671`).
- **`--update`** (`UPDATE.bat`/`update.sh` after a `git pull`; `setup.py:3012-3038`): pip packages,
  `update_installed_engine`, per-config `upgrade_config` + draft-vocab refresh. No start, no model files.
- **A newer copy of Strata unzipped next to the old `Strata-data`** (`setup.py:3652-3675`): adopts the
  previous install's choices, reuses the model files (hard-link/merge via `data_folder`), downloads nothing
  big.

## 4. The unattended path: `--setup --yes --no-start` (as `docker-entrypoint.sh` drives it)

`docker-entrypoint.sh` (first container start, no config on the volume yet):

```sh
.venv/bin/python setup.py --setup --yes \
  --family qwen --model IQ2_XS --context 32768 --vision no \
  --data-dir /data --host 0.0.0.0 --api-key "" \
  --port 8080 --no-start --low-ram auto   # + --kv / --gpus / --gpu / --layer-split only when the env vars are set
```

(`docker-entrypoint.sh:40-49`; env-var defaults at `docker-entrypoint.sh:9-21`; the config filename it
expects is `strata-<family>-<model lowercase>.json`, computed at `docker-entrypoint.sh:27-29`.)

What each flag changes versus the interactive flow:

- **`--setup`** — sets `explicit = True` (`setup.py:3650`), which (a) skips the "adopt a previous install"
  shortcut (`setup.py:3652` requires `not explicit`) and (b) bypasses the plain-start branch
  (`setup.py:3698`), so the run always goes through the setup steps 1-8 even when configs exist. This is
  what "install only, don't start" means here.
- **`--yes`** — `ask()` returns the default answer without printing the question (`setup.py:256-258`).
  Concretely on this path: GPU choice takes the recommended option (`choose_gpus`, `setup.py:797` — the two
  best cards together when they can share, else the biggest card); family/model/context/vision are already
  explicit flags; KV defaults to `int8` (the `ctx > 8192 and not a.kv` question is skipped, `setup.py:3987`);
  images default to no (the flag says `no` anyway); RoPE is none (32K ≤ 262144); the speed projection
  question defaults to off (`setup.py:4052`); the calibration question is skipped entirely
  (`setup.py:4387` requires `not a.yes and not a.no_start`). Additionally, an explicit flag is consent to
  the risks setup would otherwise confirm interactively: the low-RAM confirmation passes because
  `--model` is given (`confirm_risk(..., bool(a.model), a.yes, ...)`, `setup.py:3809`).
- **`--no-start`** — after step 7 (config + run script written), `main()` prints the "All set" summary and
  `return 0`s instead of calling `start()` (`setup.py:4412-4413`). Nothing listens on any port; the engine
  is never spawned. (Also suppresses the calibration question, `setup.py:4387`.)

Everything else in the unattended run is identical to the interactive one: the same hardware survey, the
same downloads with the same pinned revisions and `.done` marks, the same pack/MTP preparation, the same
`strata-iq2_xs.json` + `run-iq2_xs.sh` written into the image's `/opt/strata`.

### 4.1 What the Docker image adds on top

- **The engine is compiled at `docker build` time, not at first start** (`Dockerfile:70-100`): the image
  builds `strata` + `strata-vision` with CMake (CUDA archs 75;80;86;89;120 by default, `Dockerfile:56-59`)
  and writes `engine/BUILD.json` with `source: "local"` and the source fingerprint (`Dockerfile:94-99`).
  At first start, `get_prebuilt` therefore returns `None` for the `source == "local"` case
  (`setup.py:1956-1957`), and `build_engine` finds `engine_ok` true (same source hash, `setup.py:2295-2300`)
  — so the container **reuses the image's engine and never compiles or downloads an engine** (the entry
  point even says so: "the engine is already in the image", `docker-entrypoint.sh:41`).
- **Persistence**: only `models/`, `packs/`, `mtp/` and the install config live on the `/data` volume
  (`--data-dir /data`, `docker-entrypoint.sh:9, 43`; `Dockerfile:105`); the engine is part of the image.
  After the setup pass, the entry point copies the freshly written config from `/opt/strata` back onto the
  volume (`cmp -s`/`cp -f`, `docker-entrypoint.sh:50`) so a recreated container finds it. Later starts
  instead symlink the volume's config into `/opt/strata` (`docker-entrypoint.sh:52`) — `setup.py` starts
  the newest `strata-*.json` it finds, so exactly this model's config must be visible.
- **The actual start is a second, separate invocation**: after the setup pass, the entry point does
  `exec .venv/bin/python setup.py --port $PORT [--gpus ... | --gpu ... | --layer-split ...]`
  (`docker-entrypoint.sh:61-65`) — a plain start: `setup.py` finds the installed config, runs
  `update_installed_engine` (a no-op here), and `start()` spawns the server. Because the image sets
  `STRATA_EXECV=1` (`Dockerfile:46`), `start()` `os.execv`s the server instead of spawning a child, so the
  server is PID 1 and `docker stop`'s SIGTERM reaches it (`setup.py:3190-3194`; `server.py:4118-4126`).
  `--gpus`/`--gpu`/`--layer-split` are repeated on purpose to pin the cards for this model
  (`docker-entrypoint.sh:55-60`).
- **`--low-ram` is always passed** (`docker-entrypoint.sh:21, 37-39, 44`): setup reads RAM from
  `/proc/meminfo`, which in a container is the **host's** total, not the container's limit, so a
  memory-capped container must ask for the low-RAM mode itself (`docker-entrypoint.sh:36-39`;
  `LOW_RAM=on` in the documented run, `docs/INSTALL.md:116-119`).
- **Host binding**: `--host 0.0.0.0` by default (`docker-entrypoint.sh:14`), with `--api-key` taken from
  the `API_KEY` env var — the documented rule "never expose beyond 127.0.0.1 without an API key"
  (`AGENTS.md`; `docs/INSTALL.md:120-122`). The image's `HEALTHCHECK` polls `/health`, which answers
  before the API-key gate (`Dockerfile:108-113`).
- **Skipping the setup pass entirely**: when the config for this family+model already exists on the volume
  and `REINSTALL` is unset, the entry point skips `--setup --yes` and goes straight to the plain start
  (`docker-entrypoint.sh:40, 52`); `REINSTALL=1` forces the setup pass again (settings-only change,
  `docker-entrypoint.sh:32-34`). Switching between models already on the volume needs no setup pass
  (`docker-entrypoint.sh:33-34`, `docs/INSTALL.md:108-111`).

### 4.2 Interactive vs unattended, side by side

| Aspect | Interactive (`./setup.sh`) | Unattended (`--setup --yes --no-start`) |
|--------|----------------------------|------------------------------------------|
| Questions | family, size, context, KV, images, (RoPE, ESP, GPUs, calibration) asked with recommended defaults | every question answered by its recommended default (`setup.py:256-258`); family/size/context/vision pinned by flags |
| Risk confirmations (low RAM, experimental, too little RAM) | y/n prompt; explicit `--model --yes` consents | silenced by the explicit flags (`setup.py:3809`); a bare `--yes` alone would still stop (`setup.py:36-37` docstring) |
| End of run | `start()` → server + browser | `return 0` after config + run script (`setup.py:4412-4413`); serving starts only when something later runs `setup.py --port …` (the entry point does this as a second exec) |
| Engine on Linux | compile from source (no Linux prebuilt asset, §2.2 row 2) | same, except in the Docker image where the engine came from the build step (§4.1) |
| Calibration | offered at the end (default y) | not offered (`setup.py:4387` requires `not a.yes and not a.no_start`) |
| Config location | repo root of the Strata folder | `/opt/strata/strata-<tag>.json` inside the image, copied back to the `/data` volume (`docker-entrypoint.sh:50`) |

## 5. Verified external facts (read-only probes, 2026-10-05)

GitHub (`gh api`, `Niko1221/Strata`):

- Releases v0.1.34 → v0.1.39 (published 2026-10-02/04); **v0.1.39** (published 2026-10-04T12:32Z) assets:
  `strata-windows-x64.zip` 123,629,160 B; `strata-windows-x64-cuda12.zip` 175,358,734 B;
  `strata-windows-x64-hip.zip` 598,941,708 B. No Linux asset in any of the 30 listed releases
  (v0.1.10 → v0.1.39; older ones carried `strata-windows-x64.zip` alone).
- Direct HEAD: `releases/download/v0.1.39/strata-linux-x64.zip` → **404**;
  `releases/latest/download/strata-linux-x64.zip` → **404**; `.../strata-windows-x64.zip` → 200. This is
  what `get_prebuilt`'s HEAD probe hits on a bare Linux machine (§1.5 step c).
- llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` exists in `ggml-org/llama.cpp`, authored
  2026-09-20, "CUDA: enable sparse fa for qwen4 (#28770)"; its source zip is 39,564,399 B (codeload).

Hugging Face (`huggingface.co/api/models/...`, plus HEAD/GET-range probes):

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`: current sha **equals the pinned**
  `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (lastModified 2026-09-29); repo storage 208,415,070,045 B.
- `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`: current sha **equals the pinned**
  `b22d729eae29b5796f76fb70f91aef549b9fc52c` (lastModified 2026-09-24); storage 212,200,792,228 B.
- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`: current sha **equals the pinned**
  `5348543e0147355ac9cbcb031184a3546350988e` (lastModified 2026-09-29); storage 59,316,437,437 B.
- `unsloth/Qwen3.8-Flash-Next-GGUF`: current sha **equals the pinned**
  `38bb39ee97821de2c9009abb7e93950eec396e66` (lastModified 2026-09-02); storage 1,389,468,786,938 B.
- `Qwen/Qwen3.8-Flash-Next`: current sha **equals the pinned**
  `de4b8e4d43b917e7706784d8bb445c9af86a3540` (lastModified 2026-08-27); storage 360,065,103,050 B — the "360 GB in
  131 shards" checkpoint `mtp_fetch.py` reads from (`mtp_fetch.py:3-6`).
- The dates above are the HF API `lastModified` (commit date) of each repo's current sha, which equals the pinned sha. Two pins were captured later than their commit: unsloth was pinned 2026-09-30 (`setup.py:68`) against the 2026-09-02 commit, and the checkpoint was pinned 2026-09-30 (`mtp_fetch.py:30-31`) against the 2026-08-27 commit — the §2 tables report the pin dates.
- Per-file sizes at `main` are in §2.3/§2.2 (all HEAD-measured this session; the repos being unchanged
  since the pins means pinned-revision sizes are identical).
- **Serve-side quirk observed**: `HEAD <repo>/resolve/<pinned sha>/<file>` returned **404** for the ISTA
  (qwen and coder) IQ2_XS/IQ1_M shard-1 files and for unsloth's UD-IQ4_XS shard 1, while `GET` with a
  `Range` header on the *same pinned-sha URL* returned **206** with the correct `Content-Range` and GGUF
  magic bytes (probed 2026-10-05); `swift` and the Qwen checkpoint answered HEAD fine at their pinned
  shas. The revision-scoped API confirms the files exist at those shas (e.g.
  `api/models/ISTA-DASLab/.../revision/ed59f920...` lists the IQ2_XS shards). Practical consequence:
  `download()`'s HEAD-first path may hit the 404 and take the #214 fallback — warn "not at the pinned
  revision any more; downloading the repository's current file" and re-target the `main/` URL
  (`setup.py:1049-1053`). Today that fallback is content-identical (the pinned sha is still the current
  sha), and the Unsloth SHA-256 gate (`setup.py:4157-4159`) still verifies the bytes against the pinned
  revision's hashes. If the repos ever do move on, the fallback is the intended, safe behavior.

Docs cross-check: `docs/INSTALL.md` ("Where things are stored", `docs/INSTALL.md:170-189`; Linux,
`docs/INSTALL.md:49-58`; Docker, `docs/INSTALL.md:89-122`) and `docs/DETAILS.md` ("Before you start",
`docs/DETAILS.md:296-318`; "Linux", `docs/DETAILS.md:402-421`) describe the same flow, paths and defaults
found in the code; no discrepancies found in this pass.

## Caveats / notes

- **Bare Linux always compiles today** (no `strata-linux-x64.zip` published): expect `sudo` for
  `build-essential` + CUDA 13 and 10-20 minutes of compilation on first run, plus the ~0.4 GB pip CUDA
  wheels only when a *prebuilt* engine is actually used (which on Linux is not — the compiled engine links
  the toolkit's own libraries via `cuda_dirs`, `setup.py:2325-2329`). The Docker route and the Windows
  route are the ones that download a ready-made engine.
- Sizes marked "measured 2026-10-05" are byte counts from this session's HEAD probes; `download_gb` values
  are setup.py's own table (`setup.py:122-152`) and match within rounding.
- The MTP total (~5 GB) is setup.py's stated figure (`setup.py:4202-4203`); the exact per-tensor inventory
  is written to `mtp-inventory.json` only by `mtp_fetch.py inventory` (not run here, and not part of
  setup).
- Windows specifics (`START-HERE.bat`, `build-*.bat`, the ready-made Windows engine, the AMD prebuilt HIP
  engine) are out of scope for this ticket but are where the `strata-windows-x64*.zip` downloads apply.
- Nothing in this pass executed `setup.sh`, `setup.py`, the engine, or the server, and no model files were
  downloaded (per the ticket's constraints).
