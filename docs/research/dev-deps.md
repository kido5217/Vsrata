# Dev dependencies: what the hermetic flake needs to provide (ticket #10)

Research date: 2026-10-05. Repo: `kido5217/Vsrata`, branch `main` @ `4c8ed22`; this is a copy of
`Niko1221/Strata` at v0.1.39 (`CMakeLists.txt:11` `project(strata VERSION 0.1.39)`; `MIN_ENGINE =
(0, 1, 39)` at `setup.py:118`). Locked workload set: engine build (CUDA sm_120), the
`tools/test_setup_*.py` suite, server start, the `tools/vision` encoder build, and the Python oracle
tools. No AMD/HIP. `setup.py` stays untouched (the flake orchestrates it).

Method: reading repo code (`requirements.txt`, `CMakeLists.txt`, `tools/vision/CMakeLists.txt`,
`setup.py`, `Dockerfile`, `tools/test_setup_*.py`, `tools/_paths.py`, the oracle tools, `serve/`),
`nix eval` / `nix build` against this host's pinned nixpkgs, configure-only CMake runs of the engine
root (five runs, §4) and of the vision encoder (no builds, no engine or server run, no model
downloads), and a subset of the test suite under a bare nix python. Builds that ran: nixpkgs
cmake/ninja/gcc derivations, the `cuda_nvcc`/`cuda_cudart`/`libcublas` `cudaPackages_13` module
derivations, the llama.cpp `fetchTree` source derivation, and `cudaPackages_13.cudatoolkit` (the
merged toolkit). Nothing in the repo was executed.

The prior research on main (`docs/research/prebuilt-vs-source.md` §4, same day) already established
the CUDA-toolkit availability facts; everything it claimed that this document relies on was
re-verified live (its numbers below are marked ✓ where re-confirmed, and one correction is noted in
§7: the pinned nixpkgs is the **nixos-26.05 stable branch**, not unstable).

## TL;DR

- **The pinned nixpkgs covers the whole locked workload.** Every build tool, both interpreters, the
  CUDA 13.2 toolkit, and all 13 pinned Python packages exist.
- **The one real gap is CMake's CUDA compiler identification with nixpkgs' CUDA module layout.**
  `cuda_nvcc` and `cuda_cudart` are separate store paths, and nvcc resolves its own *real* store
  path (so even inside the merged `cuda-merged` tree its `nvcc.profile` INCLUDES still point at the
  split `cuda_nvcc` include dir) — the compiler-ID probe cannot find `cuda_runtime.h`, and no CMake
  variable (`CUDAToolkit_ROOT`, `CMAKE_PREFIX_PATH`, `CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES`) feeds
  that probe. The working wiring (measured, engine configure exits 0): the **merged toolkit**
  `cudaPackages_13.cudatoolkit` (`cuda-merged-13.2`, one FHS-shaped tree: `bin/nvcc` + `include/` +
  `lib/`) on `PATH` for `setup.find_nvcc`/CMake and on `CMAKE_PREFIX_PATH` for `FindCUDAToolkit`,
  **plus `NVCC_PREPEND_FLAGS="-I<merged>/include"`** to feed nvcc's own header search (the same env
  lever nixpkgs' `setupCudaHook` uses) — `§3`/`§4`.
- **cmake 4.1.6 is accepted** by every build in scope (root needs ≥3.24, vision ≥3.21, the pinned
  llama.cpp `3.14...3.28`). The pip pin 4.4.3 is "latest at pin time"; nothing requires >4.1.6, and
  nixpkgs ships no newer cmake (`§2`).
- **A read-only store path of llama.cpp works** as the ggml source: the vision encoder's configure
  against `add_subdirectory(<store path>)` exited 0 (`§5`), and the engine's `STRATA_GGML_DIR`
  configure-only test exited 0 as well — it reached and passed the store-path `add_subdirectory`
  (ggml 0.24.0 configured; "ggml commit: unknown" is harmless), with the only failure mode along the
  way being the toolkit-wiring gap above, which is independent of the store path (`§4`).
- **The test suite runs under a bare nix python** (3.13.15): `test_setup_choices` 35 OK,
  `test_setup_config` 13 OK. `test_setup_golden` fails (4 tests, 46 sub-failures) for a
  **pre-existing, environment-independent** reason — the golden fixture predates `normalize()`'s
  `<EXE>` rewrite (`§6`). It fails identically under the host's python.
- **nixpkgs revision:** this host's registry pin is a `nixos-26.05` snapshot inside the commit
  window `eb29f4b39cab` (2026-09-24) → `c2d6e79d0716b1` (2026-10-02) — *not* the branch head
  `0d9e9b832d03`, which the first fingerprint already disproved (corrected after review) (`§7`).
- **just 1.51.0, git 2.54.0, gh 2.102.0** (`§8`).

## 1. The pinned Python packages under nixpkgs

`requirements.txt` pins 13 Linux-relevant packages (plus `colorama` which is
`sys_platform == "win32"`-only and out of scope, and `cmake`/`ninja` handled in §2). The pins:
numpy 2.5.3 (`python_version >= "3.12"`), numpy 2.4.6 (`== "3.11"`), jinja2 3.1.6, regex 2026.9.10,
pyyaml 6.0.3, tqdm 4.70.1, requests 2.34.2, pillow 12.3.0, psutil 7.2.2, markupsafe 3.0.3,
certifi 2026.7.22, charset-normalizer 3.5.1, idna 3.20, urllib3 2.8.0 (`requirements.txt:4-22`).

Interpreters in the pinned nixpkgs (live `nix eval`, `legacyPackages.x86_64-linux`):

```
python3    3.13.15   (the default)
python311  3.11.16
python312  3.12.14
python313  3.13.15
```

Note the default `python3` is 3.13 — the requirements.txt pin matrix (2.5.3 for ≥3.12) still applies
to it, but the "python312 vs python311" question in the ticket is answered below for both (and 3.13).

Package versions are identical across `python311Packages`, `python312Packages` and
`python313Packages` in this nixpkgs (every row below was eval'd for all three sets):

| package | pin (requirements.txt) | nixpkgs (3.11 / 3.12 / 3.13, same) | gap |
| --- | --- | --- | --- |
| numpy | 2.5.3 (≥3.12) / 2.4.6 (3.11) | **2.4.4** | older than either pin line |
| jinja2 | 3.1.6 | 3.1.6 | exact |
| regex | 2026.9.10 | **2026.4.4** | older |
| pyyaml | 6.0.3 | 6.0.3 | exact |
| tqdm | 4.70.1 | **4.67.1** | older |
| requests | 2.34.2 | **2.33.1** | older |
| pillow | 12.3.0 | 12.3.0 | exact |
| psutil | 7.2.2 | 7.2.2 | exact |
| markupsafe | 3.0.3 | 3.0.3 | exact |
| certifi | 2026.7.22 | **2026.01.04** | older (stale CA bundle) |
| charset-normalizer | 3.5.1 | **3.4.7** | older |
| idna | 3.20 | **3.15** | older |
| urllib3 | 2.8.0 | **2.7.0** | older |

(Verbatim: `nix eval --raw nixpkgs#legacyPackages.x86_64-linux.python312Packages.numpy.version` →
`2.4.4`, etc. The nixpkgs flake of this generation exposes no `packages`/`defaultPackage` outputs —
the correct attr root is `legacyPackages.x86_64-linux`, see §7.)

**Which deltas matter — the test suite's imports are the arbiter:**

- `tools/test_setup_*.py` (18 files): every import is stdlib (`contextlib, io, json, re, sys,
  tempfile, types, unittest, pathlib, unittest.mock, hashlib, os, contextlib`) plus `import setup`
  (each file, e.g. `tools/test_setup_golden.py:23`) and, in three files,
  `from test_setup_golden import PROFILES, install` (`test_setup_config.py:21`,
  `test_setup_hybrid.py:17`, `test_setup_remote_opt.py:17`).
- `setup.py` itself is stdlib-only: its top-level imports are
  `argparse, ctypes, hashlib, json, math, os, platform, re, shutil, struct, subprocess, sys,
  textwrap, time, urllib.error, urllib.request, zipfile, pathlib` (`setup.py:39-58`), and a grep for
  lazy third-party imports (`numpy|jinja2|regex|yaml|tqdm|requests|psutil|PIL|...`) inside
  `setup.py` returns **zero matches**. The package list it installs (`PY_PACKAGES`,
  `setup.py:119`) is data, not imports.

  ⇒ **None of the nixpkgs-vs-pin deltas can affect the test suite.** It imports no third-party code.

- The **oracle tools** import a little third-party: `numpy` in `canonical_xcheck.py:33`,
  `iq_pack.py:52`, `strata_pack.py:31`, `pack_layer.py:31`, `mtp_rt.py:20`, `mtp_pack.py:30`,
  `gguf_writer.py:28`, `iq_fixture.py:31` and `ref/load.py:24`; `regex` in `strata_tokenizer.py:22`.
  `mtp_rt.py:23-25` / `mtp_pack.py:34-36` additionally `import gguf` from llama.cpp's vendored
  `gguf-py` via `tools/_paths.py` (`gguf_py()`: `$STRATA_GGUF_PY`, else
  `<repo>/third_party/llama.cpp/gguf-py`, else a dev-tree path; `tools/_paths.py:15-23`) — the pinned
  llama.cpp checkout **contains** `gguf-py/` (verified in the store path, §4), so no pip package is
  needed; `iq_fixture.py:33` hard-codes `ROOT/third_party/llama.cpp/gguf-py` and needs
  `$STRATA_GGUF_PY` set to the store path instead.
  For these, numpy 2.4.4 vs 2.5.3 and regex 2026.4.4 vs 2026.9.10 are the only deltas: the tools use
  plain array/struct math and the stable `regex` search/split API — no version-gated feature.
  Low risk; the pins are not load-bearing here.
- The **server** (`serve/`) imports `jinja2` (`serve/frontend.py:25`,
  `from jinja2.sandbox import ImmutableSandboxedEnvironment`), `PIL` (`serve/server.py:1329`) and
  `psutil` (`serve/server.py:3602`, `serve/telemetry.py:282`) — all three **version-exact** in
  nixpkgs — plus `strata_tokenizer` (→ `regex`, `serve/server.py:3953`). `serve/mcp.py` and
  `tools/strata_mcp.py` are stdlib-only. numpy is not imported anywhere under `serve/` or
  `setup.py`; it serves the packer/oracle tools.

**Recommendation:** put `python312` (3.12.14) + `python312Packages.{numpy, jinja2, regex, pyyaml,
tqdm, requests, pillow, psutil}` in the devShell — it matches the pin matrix's ≥3.12 line and keeps
the server/oracle imports covered. `python311` is an equally workable alternative (numpy pin line
2.4.6; nixpkgs still has 2.4.4). The interpreter version is not the arbiter anywhere; the suite is
stdlib-only.

## 2. The cmake gap

Minimums in scope:

- root `CMakeLists.txt:10`: `cmake_minimum_required(VERSION 3.24)`
- `tools/vision/CMakeLists.txt:7`: `cmake_minimum_required(VERSION 3.21)`
- pinned llama.cpp (store path, §4): root and `ggml/CMakeLists.txt` both say
  `cmake_minimum_required(VERSION 3.14...3.28)`

nixpkgs cmake (live `nix eval`): **4.1.6** — a single version; there is no versioned attr
(`cmake_4`/`cmake_3` do not exist — the eval error offers only "cmake or cmakerc"), so "a newer
cmake from nixpkgs" is not an option at all.

Does the build accept 4.1.6? **Yes.**

- 4.1.6 ≥ 3.24 / 3.21 / 3.14, so every `cmake_minimum_required` is satisfied. CMake 4.x's removal
  of compatibility with pre-3.5 projects is not implicated (the oldest minimum is 3.14).
- The `3.14...3.28` policy ranges in llama.cpp emit at most a deprecation *warning* under CMake 4.x
  (cosmetic; confirmed by the vision configure completing with exit 0, §5).
- Empirically: both configure-only runs in this document used nixpkgs' cmake 4.1.6 and got well past
  the root project's `project()`/options into the CUDA language step and, for vision, to
  "Generating done" (§4/§5).
- The pip pin `cmake==4.4.3` (`requirements.txt:12`) exists for the one-click installer's user flow
  (`setup.py`'s `find_tool("cmake")` falls back to the pip-installed cmake, `setup.py:2000-2010`); it
  is not a version requirement of any build in scope. Keeping pip's cmake in the flake would add a
  network fetch and break hermeticity for no build benefit.

**Verdict: use nixpkgs `cmake` (4.1.6); no fix path is needed.** (For completeness, the alternatives
and why they're not chosen: a source-built cmake in the flake — unbounded cost, zero benefit;
keeping pip cmake — non-hermetic, and the engine configure proves 4.1.6 suffices.)

## 3. CUDA devShell wiring from `cudaPackages_13`

Cross-check against `build_engine`'s configure line and the Dockerfile:

- `setup.py:2311-2315` (`build_engine`):
  `cmake_build(ROOT, bdir, "strata", ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
  f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
  f"-DSTRATA_GGML_DIR={llama}", ...])`, where `cmake_build` (`setup.py:2208-2214`) runs
  `cmake -G Ninja -DCMAKE_MAKE_PROGRAM=<ninja> -S <src> -B <bdir> -DCMAKE_BUILD_TYPE=Release <defs>`
  using `find_tool("cmake")`/`find_tool("ninja")` (PATH first, then the pip copies).
- `Dockerfile:79-86`: the same flags (`-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
  -DCMAKE_CUDA_ARCHITECTURES=… -DCMAKE_CUDA_COMPILER=… -DSTRATA_GGML_DIR=…`, vision with
  `-DLLAMA_DIR=… -DSTRATA_VISION_CUDA=ON`), in a container where the toolkit is the
  `nvidia/cuda:13.0.0-devel` image (FHS layout, nvcc on PATH).
- Neither flow passes `CMAKE_PREFIX_PATH` or `CUDAToolkit_ROOT`: CMake is expected to derive the
  toolkit from the nvcc location, and `setup.find_nvcc` (`setup.py:951-985`) finds nvcc via
  `STRATA_NVCC` (exclusive override), `PATH` (`shutil.which("nvcc")`), `CUDA_PATH`/`CUDA_HOME`,
  `/usr/local/cuda*`, `/opt/cuda*`.

So the devShell must present a toolkit whose **nvcc sits inside an FHS-shaped tree** (`bin/`,
`include/`, `lib64/` siblings) — that is what both CMake's CUDA language support and
`find_nvcc`'s "nvcc on PATH" expectation assume.

What `cudaPackages_13` (13.2, ✓ re-verified live) provides:

```
cudaPackages_13.cuda_nvcc.version    13.2.51
cudaPackages_13.cuda_cudart.version  13.2.51
cudaPackages_13.cudatoolkit.name     cuda-merged-13.2
cudaPackages_13.libcublas.version    13.3.0.5
cudaPackages_13.cuda_compat.name     cuda13.2-cuda_compat-595.45.04
cudaPackages_13_0.cuda_nvcc.version  13.0.88
cudaPackages_13_3.cuda_nvcc.version  13.3.33
cudaPackages.cudatoolkit.name        cuda-merged-12.9   (unversioned alias = 12.9)
```

**The split-module problem (measured, not assumed).** The individual modules are separate store
paths with a *reduced* layout:

- `cuda_nvcc` (`/nix/store/qj5xbxrk584cszpb7nvzhxzm4kg63lmd-cuda13.2-cuda_nvcc-13.2.51`): `bin/nvcc`
  (a small C wrapper + `nvcc.profile`), `include/` containing **only** `fatbinary_section.h`, and
  `nvvm/`. At build time nixpkgs patches its `nvcc.profile` (`pkgs/development/cuda-modules/
  packages/cuda_nvcc.nix:102-165`): the include line points at its **own** `include` output, the
  CCCL line at `cuda_cccl`'s include, and it appends
  `compiler-bindir = /nix/store/79mr0jw3…-gcc-wrapper-15.3.0/bin` (a pinned, nvcc-compatible host
  compiler). `cuda_runtime.h` is **not** there — it lives in `cuda_cudart`'s `include` output
  (`/nix/store/xwk6j0c0…-cuda13.2-cuda_cudart-13.2.51/include/cuda_runtime.h`).
- None of the module store paths ship a `lib/cmake` tree (checked `cuda_nvcc`, `cuda_cudart`,
  `libcublas` `out`/`dev`/`lib` outputs): there is no `CUDAToolkitConfig.cmake`, so
  `find_package(CUDAToolkit REQUIRED)` (`CMakeLists.txt:137`) can only run in *module* mode.

Consequence, observed with cmake 4.1.6: with `CMAKE_CUDA_COMPILER=<cuda_nvcc store path>/bin/nvcc`
(and also with a hand-assembled FHS symlink shim + `CUDAToolkit_ROOT`/`CMAKE_PREFIX_PATH` +
`CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES` — none of which reach the probe), `enable_language(CUDA)`
fails at the CUDA compiler-ID step, verbatim:

```
  #$ gcc -D__CUDA_ARCH_LIST__=750 -E -x c++ -D__CUDACC__ -D__NVCC__
  -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=2 … -include "cuda_runtime.h" -m64
  "CMakeCUDACompilerId.cu" -o "tmp/CMakeCUDACompilerId.cpp4.ii"

  <command-line>: fatal error: cuda_runtime.h: No such file or directory
  compilation terminated.

CMake Error at …/Modules/CMakeDetermineCompilerId.cmake:8 (CMAKE_DETERMINE_COMPILER_ID_BUILD)
Call Stack (most recent call first):
  …/CMakeDetermineCUDACompiler.cmake:162 (CMAKE_DETERMINE_COMPILER_ID)
  CMakeLists.txt:134 (enable_language)
-- Configuring incomplete, errors occurred!
```

The probe compiles through nvcc's *own* include search (its real store path's `nvcc.profile`), which
no CMake cache variable can extend. A toolkit where `bin/nvcc`'s sibling `../include` contains
`cuda_runtime.h` is the shape that works — and even that is not enough on its own, because the
nixpkgs `cuda_nvcc` wrapper resolves its **real** store path: inside the merged tree the probe still
used the split `cuda_nvcc` include dir (verbatim in §4, run 4). What makes nvcc's own header search
complete is `NVCC_PREPEND_FLAGS` (nvcc's standard env-var flag prepend; nixpkgs' `setupCudaHook`
appends to it, `setup-cuda-hook.sh:101-103`): `-I<merged>/include` supplies `cuda_runtime.h` and
`crt/host_config.h` (the CUDA 13 CRT headers, a separate `cuda_crt` module — the merged `include/`
carries both, verified: 183 entries including `cublas_v2.h`).

**The merged toolkit is that shape — and it is nixpkgs' own pattern for CMake+CUDA.**
`cudaPackages_13.cudatoolkit` is `cuda-merged-13.2`: a `symlinkJoin` of all module outputs
(`pkgs/development/cuda-modules/packages/cudatoolkit.nix:66-84`), one tree with `bin/nvcc`,
`include/` (cudart+crt+cccl+… headers; 183 entries) and `lib/` (cudart, cublas, …; **no `lib64`** —
nixpkgs drops the target-size dir, `cuda_nvcc.nix:107-113`). nixpkgs' own CMake CUDA packages build
against it: `nvbandwidth` uses the split modules with `backendStdenv.mkDerivation` + the
`setupCudaHook` propagation (`packages/nvbandwidth/package.nix:22-55`), and the CUDALibrarySamples
test sets `buildInputs = [ cudatoolkit ]` (`packages/tests/cuda-library-samples.nix:32`). Note
`cudatoolkit.nix:73` carries `lib.warn "cudaPackages.cudatoolkit is deprecated, … use splayed
packages instead"` — that deprecation targets nixpkgs-*internal* derivation builds (where
`backendStdenv` + `setupCudaHook` supply the wiring inside stdenv; `backendStdenv` itself is only
host-compiler selection, `backendStdenv/default.nix:221-266`, and `setupCudaHook` only fires inside
`stdenv.mkDerivation` builds, `setup-cuda-hook.sh:5`). For an out-of-tree CMake project driven by a
store-path nvcc — which is exactly the devShell situation — the hooks do not run, so the merged tree
**plus `NVCC_PREPEND_FLAGS`** is the measured-working layout (final configure evidence: §4 run 5).

**Recommended devShell wiring (for `setup.py`'s configure line, which passes no prefix paths):**

| item | value | why |
| --- | --- | --- |
| packages | `pkgs.cudaPackages_13.cudatoolkit` | one FHS tree; nvcc + all headers + cudart/cublas libs |
| `PATH` | prepend `<merged>/bin` | `setup.find_nvcc` (`setup.py:967`) and CMake both locate nvcc there; the engine then gets `-DCMAKE_CUDA_COMPILER=<merged>/bin/nvcc` exactly as in `build_engine` (`setup.py:2313`) |
| `NVCC_PREPEND_FLAGS` | `-I<merged>/include` | feeds nvcc's own header search (compiler-ID probe **and** real `.cu` compiles): its store-path-bound `nvcc.profile` does not reference the merged headers (§4 run 4 vs run 5); the nixpkgs-blessed lever (`setup-cuda-hook.sh:101-103`) |
| `CMAKE_PREFIX_PATH` | `<merged>` | `FindCUDAToolkit` (module mode) finds `libcudart.so.13` / `libcublas.so.13` under the prefix → `CUDA::cudart` / `CUDA::cublas` targets used at `CMakeLists.txt:178-179` (worked: §4 run 5) |
| `LD_LIBRARY_PATH` | `<merged>/lib` (no `lib64` exists) | runtime resolution for the compiled engine; `BUILD.json`'s `cuda_dirs` (`setup.py:2325-2326`) = nvcc's neighbours, which in the merged tree contain the libs (the generated build.ninja also bakes `-Wl,-rpath,<merged>/lib`) |
| host compiler | `pkgs.gcc15` (15.3.0) on PATH | matches the `compiler-bindir` baked into `nvcc.profile` (gcc-wrapper-15.3.0); `CMAKE_C(XX)_COMPILER` = its `g++`/`gcc` |
| optional | `pkgs.cudaPackages_13.cuda_compat` | forward-compat `libcuda` only if the host driver is older than the toolkit's; this host runs driver 595.91.07 (≥ the compat 595.45 line), so not needed here |

`allowUnfree = true` (the CUDA EULA) is required to build the CUDA packages.

## 4. llama.cpp as a read-only store path (the configure-only test)

Materializing the pinned commit (nix 2.34.8 — `builtins.fetchFromGitHub` is no longer a builtin;
`builtins.fetchTree` is, and returns the source-info attrset directly):

```
$ nix eval --impure -f llama.nix      # llama.nix = builtins.fetchTree { type = "github";
                                      #   owner = "ggml-org"; repo = "llama.cpp";
                                      #   rev = "3cf03257f219afbe7334045ff7c6a06ac68c627d"; }
{ lastModified = 1789891691; lastModifiedDate = "20260920080811";
  narHash = "sha256-SRGoXa+4ACBCB3eaG9XFYhMN1i0FyPEy9Rrer+dFGYI=";
  outPath = "/nix/store/q9r84kv1dgdllrhcs3d3gy9g5qggwi0f-source";
  rev = "3cf03257f219afbe7334045ff7c6a06ac68c627d"; shortRev = "3cf0325"; }
```

`lastModified` = 2026-09-20T08:08:11Z, matching the repo's `third_party/ggml/VERSION.txt`
(`3cf03257… Sun Sep 20 16:08:11 2026 +0800`) exactly. The store path is what a flake input
(`github:ggml-org/llama.cpp/3cf03257f219afbe7334045ff7c6a06ac68c627d`, `flake = false` — the repo
carries its own flake.nix) would expose as `outPath`; the `narHash` above is the pin to lock it
with.

**Engine, configure-only** (no build), exactly the flags of `build_engine`/`Dockerfile` plus the
store-path ggml:

```
cmake -S <repo> -B /tmp/opencode/cfg-build -G Ninja -DCMAKE_MAKE_PROGRAM=<ninja store path>
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=<nvcc>
  -DSTRATA_GGML_DIR=/nix/store/q9r84kv1dgdllrhcs3d3gy9g5qggwi0f-source
  [-DCUDAToolkit_ROOT=… -DCMAKE_PREFIX_PATH=…]
```

Run 1 (split modules: `CMAKE_PREFIX_PATH` = nvcc/cudart/cublas store paths,
`CMAKE_CUDA_COMPILER` = the nvcc store path) failed at the CUDA compiler-ID probe
(`cuda_runtime.h: No such file or directory` — the §3 split-module problem), **before** reaching the
ggml `add_subdirectory`. Run 2/3 (hand-assembled FHS symlink shim + `CUDAToolkit_ROOT`, and the
preset `CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES`) failed at the same probe.

Run 4 — `CMAKE_CUDA_COMPILER=<merged>/bin/nvcc`, `CMAKE_PREFIX_PATH=<merged>`, no env overrides —
failed the *same* probe: nvcc resolves its own real store path (`/proc/self/exe`-style), so its
patched `nvcc.profile` INCLUDES point at the split `cuda_nvcc`/`cuda_cccl` store paths, never at the
merged tree. The probe line from that run shows it verbatim:

```
  INCLUDES="-I/nix/store/qj5xbxrk584cszpb7nvzhxzm4kg63lmd-cuda13.2-cuda_nvcc-13.2.51/include"
           "-I/nix/store/qj5xbxrk584cszpb7nvzhxzm4kg63lmd-cuda13.2-cuda_nvcc-13.2.51/nvvm/include"
  compiler-bindir=/nix/store/79mr0jw3qccq7hhf1hh62knxd88dwazc-gcc-wrapper-15.3.0/bin
  ...
  <command-line>: fatal error: cuda_runtime.h: No such file or directory
```

Run 5 — the same plus **`NVCC_PREPEND_FLAGS="-I<merged>/include"`** (nvcc's standard env-var flag
prepend — the same mechanism nixpkgs' own `setupCudaHook` uses, `setup-cuda-hook.sh:101-103`) —
**configured and generated successfully, exit 0**:

```
-- The CXX compiler identification is GNU 15.3.0
-- The CUDA compiler identification is NVIDIA 13.2.51 with host compiler GNU 15.3.0
-- Check for working CUDA compiler: /nix/store/afglzv6g…-cuda-merged-13.2/bin/nvcc - skipped
-- Found CUDAToolkit: /nix/store/qj5xbx…-cuda_nvcc-13.2.51/include;/nix/store/k7ygcz…-cuda_cccl-13.2.27/include;
   /nix/store/afglzv6g…-cuda-merged-13.2/include (found version "13.2.51")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Strata: CUDA enabled, arch 120
-- strata: pack/full/experts.bin absent - the expert arena load test is NOT registered
-- The C compiler identification is GNU 15.3.0
-- The ASM compiler identification is GNU
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- GGML_SYSTEM_ARCH: x86
-- Adding CPU backend variant ggml-cpu: -march=native
-- ggml version: 0.24.0
-- ggml commit:  unknown
-- Configuring done (3.0s)
-- Generating done (0.1s)
-- Build files have been written to: /tmp/opencode/cfg-build
(exit 0)
```

The generated `build.ninja` links the real merged libs with a baked rpath, e.g.
`LIBRARIES = -Wl,-rpath,/nix/store/afglzv6g…-cuda-merged-13.2/lib  libstrata_core.a …
/nix/store/afglzv6g…-cuda-merged-13.2/lib/libcudart.so  …` and references
`cuda-merged-13.2/lib/libcublas(.so)` — so `FindCUDAToolkit` (module mode) produced working
`CUDA::cudart`/`CUDA::cublas` from the merged prefix.

The same command with the *vision* project (which takes the **full** llama.cpp, §5) against the very
same store path succeeded end-to-end:

```
-- Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
-- The ASM compiler identification is GNU
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- GGML_SYSTEM_ARCH: x86
-- Found OpenMP: TRUE (found version "4.5")
-- Including CPU backend
-- Adding CPU backend variant ggml-cpu: -march=native
-- ggml version: 0.24.0
-- ggml commit:  unknown
-- Configuring done (1.0s)
-- Generating done (0.0s)
-- Build files have been written to: /tmp/opencode/cfg-vision
(exit 0)
```

**Verdict:** `add_subdirectory(<read-only store path>/ggml … EXCLUDE_FROM_ALL)`
(`CMakeLists.txt:990-991`) and `add_subdirectory(<read-only store path> … EXCLUDE_FROM_ALL)`
(`tools/vision/CMakeLists.txt:39`) work fine via `-DSTRATA_GGML_DIR`/`-DLLAMA_DIR`: CMake writes
nothing to the source tree (all build state goes to `${CMAKE_BINARY_DIR}/ggml`, a writable dir), the
read-only store path is only read, and the missing `.git` in the store path degrades `ggml commit:`
to `unknown` harmlessly. With the §3 toolkit wiring, the engine configure **completes end-to-end
against the store-path ggml** (run 5 above, exit 0, `Generating done`); the only failure modes seen
were the toolkit wiring itself (runs 1-4), which the store path never touched. (One caveat carried
into the full build: `CMakeLists.txt:1051-1055` also reads
`${STRATA_GGML_DIR}/ggml/src/ggml-cuda/quantize.cu` for the `strata_mmq` target — present in the
store path; that file list is static, so it configures the same way.)

## 5. The `tools/vision` encoder build

- Build system: its **own CMake project** — `tools/vision/CMakeLists.txt`:
  `cmake_minimum_required(VERSION 3.21)`, `project(strata_vision C CXX)`,
  `add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)` (line 39) with
  `LLAMA_DIR` defaulting to `third_party/llama.cpp` but overridable as a cache var (line 10) — the
  **full** llama.cpp, not just `ggml/`: `LLAMA_BUILD_MTMD=ON` (line 24) with every other
  `LLAMA_BUILD_*` forced OFF (lines 18-23), plus `LLAMA_OPENSSL=OFF`, `LLAMA_CURL=OFF` (lines 25-26).
  Target `strata-vision` (`strata_vision.cpp`) links `mtmd llama`, C++17 (line 43).
- `mtmd`'s dependencies at the pinned commit (`tools/mtmd/CMakeLists.txt`): `Threads` and the
  **vendored** libraries `vendor::hash`, `vendor::miniaudio`, `vendor::stb`, `vendor::sheredom`
  (line 84 — all under llama.cpp's `vendor/`, no third-party fetch); `MTMD_VIDEO` defaults ON but
  needs the `ffmpeg` binary only at *runtime* and auto-disables when `LLAMA_SUBPROCESS` is OFF
  (lines 3-11).
- So: **the same toolchain as the engine** (cmake ≥3.21 ✓ 4.1.6, the same C++ compiler, and the
  same §3 CUDA wiring when `-DSTRATA_VISION_CUDA=ON` — `setup.py:2317-2324` adds
  `-DCMAKE_CUDA_ARCHITECTURES`/`-DCMAKE_CUDA_COMPILER` for the `vision == "gpu"` case; the Dockerfile
  does the same, `Dockerfile:83-86`), **plus the same pinned llama.cpp checkout in full**. Nothing
  extra is pulled in by the vision build.
- Configure-only evidence: the vision configure against the store path exited 0 (§4 transcript).

## 6. What the test suite and oracle tools import; bare-nix-python subset

Imports are covered in §1 (suite: stdlib + `setup`, which is stdlib-only; oracle tools: numpy,
regex, and llama.cpp's vendored `gguf-py` for `mtp_rt`/`mtp_pack`/`iq_fixture`).

Subset run under bare nix python — `nix run
nixpkgs#legacyPackages.x86_64-linux.python3 -- tools/<name>.py` from the repo root (python 3.13.15;
the tests are documented to run without GPU or downloads, `AGENTS.md`):

```
=== test_setup_choices
Ran 35 tests in 0.063s
OK
=== test_setup_config
Ran 13 tests in 0.053s
OK
=== test_setup_golden
Ran 4 tests in 0.145s
FAILED (failures=46)
```

The golden failure is **pre-existing and environment-independent**, not a nix/python artifact. The
only diff (full comparison via `maxDiff=None`) is the config's `log` field:

```
-  'log': '<T>/<EXE>-iq3_xxs.log',      (what setup.py's run produces now)
+  'log': '<T>/strata-iq3_xxs.log',     (what the golden fixture holds)
```

Mechanism: `setup.py:4325` writes `"log": str(ROOT / f"strata-{tag.lower()}.log")`, and the test's
`normalize()` (`tools/test_setup_golden.py:51-59`) rewrites the exe name —
`setup.EXE = "strata"` on Linux (`setup.py:226`) — to `<EXE>` *inside* the string, yielding
`<EXE>-iq3_xxs.log`. The fixture `tools/test_setup_golden.json` (23 `strata-*.log` entries) was
recorded before that rewrite existed, so every config comparison that includes `log` fails. Running
the same file under the host's python (3.13.15, outside nix) gives the identical result
(`Ran 4 tests … FAILED (failures=46)`). **Fact to report, not fix here**: the hermetic env is not
what breaks `test_setup_golden`; the repo's fixture is stale relative to `setup.py`+`normalize()`.
(`tools/test_setup_golden.py:207` documents the re-record: `GOLDEN.write_text(json.dumps(record(), …))`.)

## 7. Candidate nixpkgs revisions

`nix flake metadata nixpkgs --json` on this host:

```json
{"original":{"id":"nixpkgs","type":"indirect"},
 "originalUrl":"flake:nixpkgs",
 "resolvedUrl":"path:/nix/store/3zvg83mg9aavm9bgh26ydljchply7i25-source?lastModified=0&narHash=sha256-nQyFkMR78WP6PiXKkDW2SjqzLf/spQqjRuxRGSbcV8k%3D",
 "path":"/nix/store/3zvg83mg9aavm9bgh26ydljchply7i25-source", …}
```

The system registry (`/etc/nix/registry.json`) maps `flake:nixpkgs` to that bare store path —
**no git revision is recorded anywhere locally**, and the tree carries no `.git`. So the rev had to
be fingerprinted. Method: `git hash-object` on marker files in the store tree, matched against
GitHub blob SHAs per commit:

- `pkgs/top-level/all-packages.nix` blob `b322bf02…`: introduced on the **`nixos-26.05`** branch by
  commit `eb29f4b39cab` (2026-09-24T21:29:40Z); absent from `master`, `nixos-25.11` heads, and the
  50 newest master commits touching the file (2026-09-27 → 10-05) — and **absent from the
  `nixos-26.05` head `0d9e9b832d03`** (whose blob there is `0ee33736…`).
- `lib/attrsets.nix` blob `909d34c0…` and `flake.nix` blob `6cd80a6d…`: equal to the blobs at the
  head `0d9e9b832d03` — but those two files are unchanged across a wide commit span, so they cannot
  distinguish the snapshot from the head.

**Correction (reviewer re-fingerprinting, 2026-10-05):** the store tree is *not* the head
`0d9e9b832d03` — the first fingerprint above already disproved it (the head's `all-packages.nix` is
a different blob), and the original conclusion missed that contradiction. A per-commit blob match
over a shallow `nixos-26.05` clone (518 commits since 2026-09-23) shows the three fingerprinted
files' state identical across 415 commits, windowed **`eb29f4b39cab`
(2026-09-24T21:29:40Z) → `c2d6e79d0716b129c38c7d04a1896d60a09b30c8` (2026-10-02T23:17:13Z)**:
the measured snapshot is some commit inside that window. Since `all-packages.nix` (the file that
pins every package version) is among the fingerprinted files, the whole package-version surface —
every §1–§6 measurement above — holds exactly at every commit in the window, including its newest,
`c2d6e79d0716…`. Candidates for the flake input, in preference order:

1. `github:NixOS/nixpkgs/c2d6e79d0716b129c38c7d04a1896d60a09b30c8` — the newest commit whose
   fingerprinted files match the measured tree; identical package-version surface. Re-run the §1–§2
   evals there to confirm before locking.
2. `github:NixOS/nixpkgs/nixos-26.05` (branch) — same channel, drifts with backports; re-run the
   evals before relying on it.
3. master head at fetch time, `b61f9760c19537cc3f983c24cd93bdc28842486d` (2026-10-05T16:54:56Z) —
   the "recent" alternative; expect small version drift (e.g. the python package set).

Structural note for the flake author: this nixpkgs flake exposes only
`lib / checks / htmlDocs / devShells / formatter / legacyPackages / nixosModules`
(`flake.nix:24-268`) — **no `packages` and no `defaultPackage` outputs** — so every reference must
go through `nixpkgs.legacyPackages.x86_64-linux`.

## 8. `just`, `git`, `gh`

Live `nix eval` on the pinned nixpkgs:

```
just   1.51.0     (attr `just`)
git    2.54.0     (attr `git`; `gitFull` if LFS is ever needed)
gh     2.102.0    (attr `gh`)
```

All three are ordinary `legacyPackages.x86_64-linux` attrs; names confirmed by successful eval.

## devShell draft inputs

Exact nixpkgs attrs the flake should put in the shell (pinned nixpkgs = `nixos-26.05` @
`0d9e9b832d03…`, `pkgs = nixpkgs.legacyPackages.x86_64-linux`, with `config.allowUnfree = true`
for the CUDA set):

```nix
packages = with pkgs; [
  # build tools (§2): cmake 4.1.6, ninja 1.13.2, gcc 15.3.0 (matches nvcc.profile's compiler-bindir)
  cmake
  ninja
  gcc15

  # CUDA toolkit (§3): the merged FHS tree + NVCC_PREPEND_FLAGS is the measured-working layout
  cudaPackages_13.cudatoolkit          # cuda-merged-13.2 (nvcc 13.2.51 + all headers + cudart/cublas libs)
  # cudaPackages_13.cuda_compat        # only if the host driver is older than the toolkit's

  # python + the pinned package set (§1); python312 = 3.12.14
  python312
  python312Packages.numpy              # 2.4.4
  python312Packages.jinja2             # 3.1.6
  python312Packages.regex              # 2026.4.4
  python312Packages.pyyaml             # 6.0.3
  python312Packages.tqdm               # 4.67.1
  python312Packages.requests           # 2.33.1
  python312Packages.pillow             # 12.3.0
  python312Packages.psutil             # 7.2.2
  # (markupsafe/certifi/charset-normalizer/idna/urllib3 come transitively; pull them explicitly only
  #  if their nixpkgs versions ever need pinning)

  # dev workflow (§8)
  just
  git
  gh
];

let cuda = pkgs.cudaPackages_13.cudatoolkit; in
env = {
  # §3/§4: nvcc on PATH (setup.find_nvcc + CMake), toolkit on the search paths (FindCUDAToolkit),
  # merged headers fed to nvcc's own search (compiler-ID probe and .cu compiles)
  NVCC_PREPEND_FLAGS = "-I${cuda}/include";
  CMAKE_PREFIX_PATH  = "${cuda}";
  LD_LIBRARY_PATH    = "${cuda}/lib";   # the merged tree has lib/, no lib64/
  # optional explicit overrides (setup.py honours STRATA_NVCC exclusively, setup.py:953-966):
  # STRATA_NVCC      = "${cuda}/bin/nvcc";
  # CMAKE_C_COMPILER / CMAKE_CXX_COMPILER = "${pkgs.gcc15}/bin/gcc|g++";
};
# plus: the pinned llama.cpp checkout as a flake input (github:ggml-org/llama.cpp/
# 3cf03257f219afbe7334045ff7c6a06ac68c627d, flake = false, narHash
# sha256-SRGoXa+4ACBCB3eaG9XFYhMN1i0FyPEy9Rrer+dFGYI=, §4) and, for the oracle tools,
# STRATA_GGUF_PY = "<llama.cpp store path>/gguf-py" (tools/_paths.py:16).
```

Open items this document does **not** settle (they need a full build or a run, out of scope here):
that the engine + vision **builds** (not just configure) succeed against the store path, and that the
server starts against the nixpkgs python set.

## Sources

- Repo (branch `research/dev-deps`, base `main` @ `4c8ed22`): `requirements.txt:4-23`,
  `CMakeLists.txt:10-11,126-185,597,944-1003,1042-1060`, `tools/vision/CMakeLists.txt`,
  `setup.py:39-58,93-94,119,226-227,951-985,1254-1284,2000-2010,2124-2130,2208-2232,2274-2332,4325`,
  `Dockerfile:41-100`, `tools/test_setup_golden.py:25,51-59,158-208` (+ the 18 `tools/test_setup_*.py`
  import blocks), `tools/_paths.py`, `tools/canonical_xcheck.py:24-43`, `tools/strata_tokenizer.py:16-22`,
  `tools/{iq_pack,strata_pack,pack_layer,mtp_rt,mtp_pack,gguf_writer,iq_fixture}.py` import blocks,
  `ref/load.py:18-28`, `serve/{server,frontend,telemetry,mcp}.py` import blocks,
  `tools/strata_mcp.py:25-39`, `third_party/ggml/VERSION.txt`, `AGENTS.md`.
- nixpkgs (pinned store path `/nix/store/3zvg83mg9aavm9bgh26ydljchply7i25-source` = `nixos-26.05` @
  `0d9e9b832d03`): `flake.nix`, `pkgs/development/cuda-modules/packages/{cudatoolkit.nix,
  cuda_nvcc.nix,nvbandwidth/package.nix,tests/cuda-library-samples.nix,
  setupCudaHook/setup-cuda-hook.sh}`, `pkgs/development/cuda-modules/backendStdenv/default.nix`,
  `pkgs/development/cuda-modules/README.md`, `doc/languages-frameworks/cuda.section.md`.
- Pinned llama.cpp store path `/nix/store/q9r84kv1dgdllrhcs3d3gy9g5qggwi0f-source`: root and
  `ggml/CMakeLists.txt` (`cmake_minimum_required(VERSION 3.14...3.28)`), `tools/mtmd/CMakeLists.txt`,
  `gguf-py/`.
- CMake 4.1.6 modules from the nixpkgs store: `FindCUDAToolkit.cmake`,
  `CMakeDetermineCUDACompiler.cmake`, `Internal/CMakeCUDAFindToolkit.cmake`.
- GitHub API (`gh api repos/NixOS/nixpkgs/…`): commit lists and blob SHAs for the rev fingerprinting
  (§7); `nix flake metadata nixpkgs --json`; `/etc/nix/registry.json`; `nix --version` (2.34.8).
- All `nix eval`/`nix run`/configure commands and outputs are quoted verbatim in §1-§8; the engine
  configure runs 1-5 (split modules, shim, merged, merged+`NVCC_PREPEND_FLAGS`) and the vision
  configure are quoted in §4.
