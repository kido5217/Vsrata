# The Nix packaging surface: a CUDA package and a home-manager module (ticket #24)

Written 2026-10-06. Repo `kido5217/Vsrata` (a copy of `Niko1221/Strata` v0.1.39). Host: NixOS,
RTX 5090 (32 GB), Ryzen 7 5800X3D, 126 GiB RAM. Tools read here: `nix` 2.34.8, systemd 260.4
(the host's man pages), nixpkgs at the flake's pin `0d9e9b832d03ac387417e16ce1febf73b2e631e1`
(`flake.lock`), home-manager at `master` head `aa23f72` (fetched 2026-10-06).

This is the findings document for ticket #24 on map #23. It supplies the Nix **idioms** for the two
artifacts this repo must expose — the package and the home-manager module — with every claim cited
to a primary source (nixpkgs source/manual, home-manager source/manual, the host's systemd man
pages, the Nix manual) or to a file in this repo. It does **not** design the module's option set;
that is the design ticket's job (the ticket's own "not in scope").

## TL;DR

- **The arch string is a one-liner in nixpkgs, not hand-written.** `pkgs.cudaPackages.flags.cmakeCudaArchitecturesString`
  turns `cudaCapabilities = [ "12.0a" ]` into `"120a"` — verified by evaluating the pinned nixpkgs
  on this host (§1.2). The `a` is part of the capability string, exactly as it is in
  `CMAKE_CUDA_ARCHITECTURES`.
- **cmake+ninja needs no explicit `-GNinja`.** Adding `ninja` to `nativeBuildInputs` makes nixpkgs'
  ninja setup hook set `buildPhase = ninjaBuildPhase`, and nixpkgs' cmake setup hook then appends
  `-GNinja` itself (§1.3). This is the idiomatic path and the repo needs no CMake change.
- **The repo's CMake has no `install()` rules** (checked: `CMakeLists.txt`, `tools/vision/CMakeLists.txt`),
  so the derivation installs the built binaries by hand in `installPhase` (§1.4).
- **`systemd.user.services.<name>` is a free-form attrset of systemd directives**, capitalized as
  systemd spells them; home-manager writes each to `$XDG_CONFIG_HOME/systemd/user/<name>.service`
  and activation reloads (§2.1). Every requested idiom is an ordinary `Service.*`/`Unit.*` key:
  `LoadCredential`, `Environment`, `LimitMEMLOCK = "infinity"`, `Type = "oneshot"` +
  `RemainAfterExit = true`, `Requires`/`After`, `Restart = "on-failure"`, and
  `Install.WantedBy = [ "default.target" ]` (§2.2–§2.7).
- **For user units, `WantedBy=default.target` is correct, and systemd.special(7) says so** — the
  system-level warning against `default.target` does not apply to the per-user manager, whose
  `default.target` *is* the main user target (§2.3).
- **`homeManagerModules.vsrata` is a community convention, not a home-manager-defined output.**
  Home-manager's own flake exposes `nixosModules`, `darwinModules` and `flakeModules`, and its
  flake-parts module defines `flake.homeModules` (not `homeManagerModules`). Recommend exposing
  **both names**, with one an alias of the other (§3.2).
- **What I could not verify:** whether every nvcc from the nixpkgs `cudatoolkit` works with this
  repo's `enable_language(CUDA)` probe without the devShell's `NVCC_PREPEND_FLAGS` workaround
  (§4.1); no build was run (out of scope), so no derivation here was actually built.

## §0 Primary sources read

| Source | Revision / version | What it settled |
|---|---|---|
| The repo's `flake.nix` | working tree | The existing allow-unfree import, `cudaPackages_13.cudatoolkit`, python env, `STRATA_GGML_DIR`, `NVCC_PREPEND_FLAGS` (§1.1) |
| The repo's `CMakeLists.txt`, `tools/vision/CMakeLists.txt`, `justfile` | working tree | Build knobs, default arch 120, no `install()` rules (§1.4) |
| nixpkgs CUDA manual, `doc/languages-frameworks/cuda.section.md` | pin `0d9e9b83` | `cudaCapabilities`/`cudaSupport`/`cudaForwardCompat`; unfree; `autoAddDriverRunpath`; `cudatoolkit` discouraged (§1.1, §1.7) |
| nixpkgs `_cuda` source, `pkgs/development/cuda-modules/` | pin `0d9e9b83` | `cmakeCudaArchitecturesString`; `12.0a`; `cudaPackages.flags`; `cudaPackages_13 = cudaPackages_13_2` = CUDA 13.2 (§1.2) |
| nixpkgs cmake + ninja setup hooks | pin `0d9e9b83` | the `-GNinja` wiring (§1.3) |
| nixpkgs Python manual + `doc/build-helpers/trivial-build-helpers.chapter.md` | pin `0d9e9b83` | `python3.withPackages`; `writeShellScriptBin`/`writeShellApplication`; `lib.getExe` (§1.5–§1.6) |
| Nix manual, `nix flake` "Flake format" | nix 2.34 manual | `packages.<system>.<name>` must be a derivation (§3.1) |
| home-manager `modules/systemd.nix`, `flake.nix`, `flake-module.nix` | `aa23f72` (master) | the `systemd.user.services` shape; the real output names (§2.1, §3.2) |
| home-manager manual flake-parts page; PR #6392 | live / GitHub | `flake.homeModules`; the `homeManagerModules` vs `homeModules` naming history (§3.2) |
| systemd 260.4 man pages (host) | `systemd.exec(5)`, `.service(5)`, `.unit(5)`, `.special(7)` | each directive's exact semantics (§2) |

No engine build, model download or GPU work was done. The one evaluation run was
`nix eval` of the pinned nixpkgs (§1.2).

## §1 The package

### §1.1 Importing nixpkgs with unfree CUDA — reuse the existing idiom

The repo already imports nixpkgs from the `flake = false` input with its own config, precisely
because the flake's `legacyPackages` is evaluated config-free and CUDA is unfree-by-EULA
(`flake.nix:4-17`, the comment at `:6-8`):

```nix
pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
```

The nixpkgs CUDA manual states the same requirement and adds the CUDA-specific config keys
(`doc/languages-frameworks/cuda.section.md` — "Configuring Nixpkgs for CUDA"):

```nix
{ pkgs }: {
  allowUnfreePredicate = pkgs._cuda.lib.allowUnfreeCudaPredicate;  # or allowUnfree
  cudaCapabilities = [ <target-architectures> ];
  cudaForwardCompat = true;
  cudaSupport = true;
}
```

Two points the manual makes explicitly and this repo must honour:

- "The majority of CUDA packages are unfree, so either `allowUnfreePredicate` or `allowUnfree`
  should be set." (cuda.section.md, same section.)
- Non-baseline feature-sets such as `9.0a` "must be explicitly set" in `cudaCapabilities`
  (cuda.section.md, caution box). `12.0a` is the same kind of capability (§1.2), so it must be in
  `cudaCapabilities` — you cannot rely on a default.

The repo's `flake.nix:23` already uses `pkgs.cudaPackages_13.cudatoolkit`. The CUDA manual *discourages*
`cudatoolkit` ("all new projects should use the CUDA redistributables available in `cudaPackages`
instead"), but this repo's CMake needs the merged toolkit's `bin/nvcc` and `include/` for its
compiler-ID probe (`flake.nix:39-40`), so keeping `cudatoolkit` is the consistent choice here. The
manual's caveat is worth recording but not a reason to diverge from the existing convention.

The derivation adds the CUDA config to the *same* import — it does not re-import nixpkgs:

```nix
config = {
  allowUnfree = true;
  cudaSupport = true;
  cudaCapabilities = [ "12.0a" ];   # see §1.2
  cudaForwardCompat = false;        # an `a` capability cannot build forward-compat PTX (manual)
};
```

### §1.2 `CMAKE_CUDA_ARCHITECTURES` and the `a` suffix

The repo's default is `120` (`CMakeLists.txt:137-140`), and the ticket says the package's default is
`120a`. Two different layers both use the `a` suffix, and nixpkgs has an exact bridge between them.

**CMake's side.** `CMAKE_CUDA_ARCHITECTURES` is CMake's variable for the CUDA target property
`CUDA_ARCHITECTURES`; a suffix `a` on an architecture means "architecture-conditional" features
(`120a` = the Blackwell feature set), and the variable accepts a `;`-separated list. The repo passes
this variable straight through (`justfile:12`, `setup.py:2312`), so the value the derivation must
produce is the literal `120a` (or a list containing it).

**nixpkgs' side.** `cudaCapabilities` is nixpkgs' own vocabulary and uses the dotted
`major.minor` form plus the same optional `a` suffix. The capability database lists both `12.0` and
`12.0a` for Blackwell (`pkgs/development/cuda-modules/_cuda/db/bootstrap/cuda.nix:309-317`), each
with `minCudaMajorMinorVersion = "12.8"`. `cudaPackages.flags` is computed from the configured
capabilities (`pkgs/development/cuda-modules/default.nix:157-170`), and `formatCapabilities`
exposes the CMake string directly (`_cuda/lib/strings.nix:187-236`):

```nix
# strings.nix:230-236 (documented example, verbatim)
mkCmakeCudaArchitecturesString [ "8.9" "10.0a" ]
=> "89;100a"
```

So `cudaCapabilities = [ "12.0a" ]` yields `"120a"`. Note that `cmakeCudaArchitecturesString` is itself
already a **String**, not a function (`_cuda/lib/strings.nix:196`:
`cmakeCudaArchitecturesString = cudaLib.mkCmakeCudaArchitecturesString cudaCapabilities;`), so it is
`flags.cmakeCudaArchitecturesString` — used directly, never applied as `f: f { … }`.

**Verified on this host** by evaluating the pinned nixpkgs (2026-10-06). The expression, run with
`nix eval --raw --offline --impure -f arch-eval.nix` (the `nixpkgs` path is the store path of the
`flake.lock` pin, resolved with `builtins.fetchTree { type = "github"; owner = "NixOS"; repo =
"nixpkgs"; rev = "0d9e9b832d03ac387417e16ce1febf73b2e631e1"; narHash =
"sha256-9+n3ReS2y9xS199LoJjKUWHwZrISGHWgkD/LUZTDZyA="; }`):

```nix
# arch-eval.nix
let
  nixpkgs = /nix/store/hwg0y945hq5cr4rpkjsr9ffl2yqrvz3m-source;
  pkgs = import nixpkgs {
    system = "x86_64-linux";
    config = { allowUnfree = true; cudaSupport = true;
               cudaCapabilities = [ "12.0a" ]; cudaForwardCompat = false; };
  };
in
"arch=" + pkgs.cudaPackages.flags.cmakeCudaArchitecturesString + "\n"
+ "cudaVersion=" + pkgs.cudaPackages_13.cudatoolkit.version + "\n"
+ "nvcc=" + (pkgs.cudaPackages_13.cudatoolkit.outPath + "/bin/nvcc") + "\n"
+ "caps=" + builtins.concatStringsSep "," pkgs.cudaPackages.flags.cudaCapabilities + "\n"
+ "cmake=" + pkgs.cmake.version + "\n"
+ "python=" + pkgs.python312.version + "\n"
```

output:

```
arch=120a
cudaVersion=13.2
nvcc=/nix/store/afglzv6gqhsjxvxi6zr3phymdjs89jhr-cuda-merged-13.2/bin/nvcc
caps=12.0a
cmake=4.1.6
python=3.12.14
```

The `cudaVersion=13.2` also confirms the pin: `cudaPackages_13` is `cudaPackages_13_2`
(`pkgs/top-level/all-packages.nix:2024`), whose `cuda` manifest is `"13.2.0"`
(`pkgs/top-level/cuda-packages.nix:155-176`) — the same 13.2 the existing `flake.nix:23` comment
names. No hand-maintained arch map is needed; do not hard-code `"120a"` when the capability string
is already the source of truth.

The derivation therefore sets:

```nix
"-DCMAKE_CUDA_ARCHITECTURES=${pkgs.cudaPackages.flags.cmakeCudaArchitecturesString}"
```

### §1.3 cmake + ninja

The idiomatic pattern is `nativeBuildInputs = [ cmake ninja ... ]` and **no** explicit `-G Ninja`.
It works through the two setup hooks:

- nixpkgs' ninja hook sets `buildPhase = ninjaBuildPhase` when `buildPhase` is unset
  (`pkgs/by-name/ni/ninja/setup-hook.sh`, tail: `if [ -z "${dontUseNinjaBuild-}" ] && [ -z "${buildPhase-}" ]; then buildPhase=ninjaBuildPhase; fi`).
- nixpkgs' cmake hook then reads `buildPhase` at configure time and appends the generator flag:
  `pkgs/by-name/cm/cmake/setup-hook.sh:97-98` — `if [ "${buildPhase-}" = ninjaBuildPhase ]; then prependToVar cmakeFlags "-GNinja"; fi`.

The cmake hook also supplies `-DCMAKE_INSTALL_PREFIX`, `-DCMAKE_BUILD_TYPE=Release` (unless set),
the compiler variables, and `-DBUILD_TESTING=OFF` when `doCheck` is **unset or empty** (the
condition is `if [ -z "${doCheck-}" ]`, `setup-hook.sh:82-85`). Nix serialises `doCheck = false`
to an empty environment variable, so the guard fires for an unset **and** a `false` `doCheck`
alike, and the flag is supplied in both cases (`setup-hook.sh:26-95` overall); the derivation only
adds its own `cmakeFlags`.

### §1.4 The repo's build knobs, and the missing install rules

From the repo (all working-tree citations):

- Engine: `cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
  -DCMAKE_CUDA_ARCHITECTURES=<archs> -DCMAKE_CUDA_COMPILER=<nvcc> -DSTRATA_GGML_DIR=<llama>`
  (`justfile:11-14`, `setup.py:2311-2315`).
- Vision encoder: `cmake -S tools/vision -B build-vision -DLLAMA_DIR=<llama>
  -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<archs>` (`justfile:15-17`,
  `setup.py:2322-2323`). The vision build links llama.cpp's `mtmd` and `llama`
  (`tools/vision/CMakeLists.txt`, `target_link_libraries(strata-vision PRIVATE mtmd llama)`), and
  the engine compiles ggml from the same source via `add_subdirectory(${STRATA_GGML_DIR}/ggml ...)`
  (`CMakeLists.txt:990-1002`).
- The raw binaries the engine target produces include `strata` (`CMakeLists.txt:495`) plus
  `strata-gguf`, `strata-dequant`, `strata-plan`, and (CUDA-gated) `strata-device`, `strata-load`,
  `strata-concurrent`, `strata-overlap` (`CMakeLists.txt:218-271`); the vision target produces
  `strata-vision` under `<build>/bin` (`tools/vision/CMakeLists.txt`, `RUNTIME_OUTPUT_DIRECTORY`).
- **Neither CMakeLists has an `install()` rule.** Checked this session: no `install(` in either
  file. A `ninja install` would install nothing, so the derivation's `installPhase` copies the
  built binaries itself. (Adding install rules to the tracked CMake files is a separate change and
  was not made.)

The server is not run through CMake; it is invoked directly as
`python3 serve/server.py --engine strata --config <cfg>` (`setup.py:3093`, `:3455`), and the pinned
Python deps are in the repo root `requirements.txt` (numpy, jinja2, regex, pyyaml, tqdm, requests,
cmake, ninja, pillow, psutil + their transitive pins).

### §1.5 The Python environment

`python3.withPackages` builds an interpreter with the packages on its path; the function gets the
correct package set for the interpreter version (`doc/languages-frameworks/python.section.md`,
"python.withPackages function", lines 534-576):

```nix
python312.withPackages (ps: with ps; [ numpy jinja2 regex pyyaml tqdm requests pillow psutil ])
```

This is the right idiom for a *script* (`server.py`) that is run, not pip-installed; `buildPythonApplication`
(`python.section.md:334-380`) is for a package with a `pyproject`/`setup.py` that pip installs, which
this server does not have. The env's `bin/` provides `python3`, so the wrapper below runs
`python3 ${serverPy} ...`.

### §1.6 The `vsrata` CLI wrapper and the raw binaries

Both trivial builders are nixpkgs-documented (`doc/build-helpers/trivial-build-helpers.chapter.md`):

- `writeShellScriptBin "vsrata" ''...''` writes `bin/vsrata` and is a combination of
  `writeShellScript` + `writeScriptBin` (lines 639-680).
- `writeShellApplication { name; runtimeInputs; text; ... }` additionally puts `runtimeInputs` on
  the script's `$PATH` and `shellcheck`s it (lines 728-826) — the cleaner choice when the wrapper
  calls other derivations.

The idiomatic split: one derivation installs the binaries; one `writeShellApplication` is the CLI;
`symlinkJoin` merges them so the package exposes *both* `bin/vsrata` and the raw binaries
(`symlinkJoin` doc: trivial-build-helpers lines 828-836). `lib.getExe`/`lib.getExe'` are the
canonical way to spell a store binary (`lib/meta.nix:526-575`).

### §1.7 CUDA runtime and driver runpath

The CUDA manual's troubleshooting list: "First ensure that dependencies are patched with
`autoAddDriverRunpath`" (`cuda.section.md:326`). Adding `pkgs.autoAddDriverRunpath` as a build input
makes the hook add the driver library runpath, so a binary that dlopens `libcuda.so` finds it. The
toolkit libraries the engine links (`CUDA::cudart`, `CUDA::cublas`, from
`find_package(CUDAToolkit REQUIRED)`, `CMakeLists.txt:135-179`) get their runpath from the
`cudatoolkit`'s setup hooks. The existing devShell additionally needs
`NVCC_PREPEND_FLAGS=-I${cuda}/include` for CMake's compiler-ID probe (`flake.nix:39-40`); a
derivation should carry the same workaround unless it is proven unnecessary (see §4.1).

### §1.8 Worked derivation skeleton

Minimal, not production-complete; it shows each idiom above. The crate names (`vsrata-engine`,
`vsrata-server`) are placeholders — naming is the design ticket's job.

```nix
# pkgs/vsrata.nix — called with the flake's pinned inputs and the CUDA-configured pkgs (§1.1).
# It lives at <repo>/pkgs/vsrata.nix, so `../.` is the repo root and `../tools/vision` the encoder.
{ pkgs, lib, llamaCpp }:
let
  cuda = pkgs.cudaPackages_13;              # = cudaPackages_13_2, CUDA 13.2 (§1.2)
  cudaFlags = cuda.flags;                   # cmakeCudaArchitecturesString = "120a" (§1.2)

  pythonEnv = pkgs.python312.withPackages (ps: with ps; [   # §1.5
    numpy jinja2 regex pyyaml tqdm requests pillow psutil
  ]);

  # --- the C++/CUDA engine (repo root CMakeLists.txt) ---
  engine = pkgs.stdenv.mkDerivation (finalAttrs: {
    pname = "vsrata-engine";
    version = "0.1.39";
    src = ../.;

    # ninja in nativeBuildInputs => buildPhase = ninjaBuildPhase => cmake hook adds -GNinja (§1.3)
    nativeBuildInputs = [
      pkgs.cmake
      pkgs.ninja
      cuda.cudatoolkit
      pkgs.pkg-config
    ];
    buildInputs = [ pkgs.autoAddDriverRunpath ];         # §1.7

    cmakeFlags = [
      "-DSTRATA_ENABLE_CUDA=ON"
      "-DSTRATA_BUILD_TESTS=OFF"
      "-DCMAKE_CUDA_ARCHITECTURES=${cudaFlags.cmakeCudaArchitecturesString}"
      "-DCMAKE_CUDA_COMPILER=${cuda.cudatoolkit}/bin/nvcc"
      "-DSTRATA_GGML_DIR=${llamaCpp}"
    ];

    # no install() rules in CMakeLists.txt => install by hand (§1.4)
    installPhase = ''
      runHook preInstall
      install -Dm755 strata           "$out/bin/strata"
      install -Dm755 strata-gguf      "$out/bin/strata-gguf"
      install -Dm755 strata-dequant   "$out/bin/strata-dequant"
      install -Dm755 strata-plan      "$out/bin/strata-plan"
      install -Dm755 strata-device    "$out/bin/strata-device"
      install -Dm755 strata-load      "$out/bin/strata-load"
      install -Dm755 strata-concurrent "$out/bin/strata-concurrent"
      install -Dm755 strata-overlap   "$out/bin/strata-overlap"
      runHook postInstall
    '';

    meta.mainProgram = "strata";
  });

  # --- the image encoder (tools/vision/CMakeLists.txt) ---
  vision = pkgs.stdenv.mkDerivation {
    pname = "vsrata-vision";
    version = "0.1.39";
    src = ../tools/vision;

    nativeBuildInputs = [ pkgs.cmake pkgs.ninja cuda.cudatoolkit pkgs.pkg-config ];
    buildInputs = [ pkgs.autoAddDriverRunpath ];

    cmakeFlags = [
      "-DLLAMA_DIR=${llamaCpp}"
      "-DSTRATA_VISION_CUDA=ON"
      "-DCMAKE_CUDA_ARCHITECTURES=${cudaFlags.cmakeCudaArchitecturesString}"
      "-DCMAKE_CUDA_COMPILER=${cuda.cudatoolkit}/bin/nvcc"
    ];

    installPhase = ''
      runHook preInstall
      install -Dm755 bin/strata-vision "$out/bin/strata-vision"
      runHook postInstall
    '';

    meta.mainProgram = "strata-vision";
  };

  # --- the Python server tree (serve/, tools/) ---
  server = pkgs.stdenv.mkDerivation {
    pname = "vsrata-server";
    version = "0.1.39";
    src = ../.;
    dontBuild = true;
    installPhase = ''
      mkdir -p "$out/share/vsrata"
      cp -r serve tools "$out/share/vsrata/"
    '';
  };

  # --- the `vsrata` CLI: a shell wrapper around server.py (§1.6) ---
  vsrataCli = pkgs.writeShellApplication {
    name = "vsrata";
    runtimeInputs = [ pythonEnv engine vision ];
    text = ''
      exec python3 ${server}/share/vsrata/serve/server.py \
        --engine "$(command -v strata)" "$@"
    '';
  };

  # --- the package exposes BOTH bin/vsrata and the raw binaries (§1.6) ---
in
pkgs.symlinkJoin {
  name = "vsrata-0.1.39";
  paths = [ vsrataCli engine vision server ];
  meta.mainProgram = "vsrata";
}
```

## §2 The home-manager module

The module must be a **home-manager** module, not a NixOS system module (ticket). Nothing here uses
`systemd.services` (the NixOS option); it uses `systemd.user.services` (the home-manager option).

### §2.1 Where systemd user units come from

home-manager defines `systemd.user.services.<name>` as an attrset with sections `Unit`, `Service`,
`Install`, plus `Socket`/`Path`/`Timer` for other unit kinds. Its description is explicit: "Definition
of systemd per-user service units. Attributes are merged recursively. Note that the attributes
follow the capitalization and naming used by systemd"
(home-manager `modules/systemd.nix`, master `aa23f72`). A handful of attributes get typed options of
their own — `Service.Environment` (type `listOf str`, "Environment variables available to executed
processes"), `Service.ExecStart` (a package, a string, or a list of those), and the home-manager-only
`Unit.X-Restart-Triggers` / `X-Reload-Triggers` / `X-SwitchMethod`. Everything else — `LoadCredential`,
`LimitMEMLOCK`, `Type`, `RemainAfterExit`, `Restart`, `Requires`, `After`, `Install` — goes through
the free-form type (`attrsOf (attrsOf (either primitive (listOf primitive)))`), i.e. it is accepted
verbatim and rendered `Key=value` into the unit file.

The module writes each unit to `$XDG_CONFIG_HOME/systemd/user/<name>.service`, and
`Install.WantedBy` / `Install.RequiredBy` become `.wants/` / `.requires/` symlinks
(`buildService` in the same file). Activation reloads services via `sd-switch` (or prints suggested
`systemctl` commands when `systemd.user.startServices = false`). There is also
`systemd.user.packages`, "the Home Manager equivalent of NixOS's `systemd.packages`", for shipping
unit files from a derivation; this ticket's module generates units inline, so it does not need it.

Because the unit *file* is a Nix store path, **nothing in it can be a secret** — see §2.6.

### §2.2 `Type=oneshot` + `RemainAfterExit=`

systemd.service(5): "If set to `oneshot` … the service manager will consider the unit up after the
main process exits"; and "`RemainAfterExit=` … specifies whether the service shall be considered
active even when all its processes exited" (defaults `no`). The two are normally paired: the oneshot
run finishes, and `RemainAfterExit=yes` leaves the unit `active` so dependents' `After=` actually
waits for a completed job (without it the unit goes straight to `dead`, per the same man text). In
the module:

```nix
Service = {
  Type = "oneshot";
  RemainAfterExit = true;            # rendered "RemainAfterExit=true"
  ExecStart = "${vsrata}/bin/vsrata warmup";
};
```

### §2.3 `Requires=` / `After=` and `WantedBy=`

- systemd.unit(5): "`Requires=` … declares a stronger requirement dependency … If this unit gets
  activated, the units listed will be activated as well. If one of the other units fails to
  activate, and an ordering dependency `After=` on the failing unit is set, this unit will not be
  started." So `Requires` (activation) and `After` (ordering) are usually written together.
- `Before=, After=` are "ordering dependencies"; `After=bar.service` means "the listed unit is
  fully started up before the configured unit is started".
- `WantedBy=` goes in the `[Install]` section and creates a `.wants/` symlink when the unit is
  enabled; it is the weak counterpart that makes a target pull the unit in.

For a **user** unit the target is `default.target`. systemd.special(7), under the *user* manager's
entry: "`default.target` … This is the main target of the user service manager … Various services
that compose the normal user session should be pulled into this target." (The system-level entry's
warning to prefer `multi-user.target`/`graphical.target` over `default.target` is about the *system*
manager and does not apply to the per-user one.) So a serve unit and a oneshot that depends on it
look like:

```nix
# the serve unit
Install.WantedBy = [ "default.target" ];

# the oneshot, ordered after the serve unit
Unit = {
  Description = "vsrata warmup";
  Requires = [ "vsrata.service" ];
  After    = [ "vsrata.service" ];
};
Install.WantedBy = [ "default.target" ];
```

### §2.4 `Restart=on-failure`

systemd.service(5): "`Restart=` … Configures whether the service shall be restarted when the
service process exits, is killed, or a timeout is reached … Takes one of `no`, `on-success`,
`on-failure`, …". For a server that should recover from crashes but not be respawned after a clean
stop (systemd's own stop/restart never triggers `Restart=`), the idiomatic value is
`Restart = "on-failure"`. It is permitted on a `Type=oneshot` unit as well: systemd.service(5) only
says "`Type=oneshot` services will never be restarted on a clean exit status, i.e. `always` and
`on-success` are rejected for them" — `on-failure` is not rejected. A oneshot that has already
exited still has nothing to restart, so the setting belongs on the long-running serve unit. (The
"does not have any effect on `Type=oneshot` services" sentence is `RuntimeMaxSec=`, not `Restart=`.)

### §2.5 `ExecStart` from a store path

home-manager accepts either a package or a string. The store-path form is the plain string
interpolation, or `lib.getExe` when the package sets `meta.mainProgram`:

```nix
Service.ExecStart = "${vsrata}/bin/vsrata --host 127.0.0.1 --port 8080";
# or, with meta.mainProgram = "vsrata":
Service.ExecStart = "${lib.getExe vsrata} --host 127.0.0.1 --port 8080";
```

### §2.6 Secrets: `LoadCredential` vs `Environment` / `EnvironmentFile`

The two options differ in exactly the way that matters here.

- `Environment=` (systemd.exec(5)) "Sets environment variables for executed processes." It is a
  literal in the unit file — and home-manager's unit file is a Nix store path. Putting an API key
  in `Environment` therefore writes it into the world-readable store. Acceptable for non-secret
  config (port, host, log level); **not** for a secret.
- `EnvironmentFile=` is "Similar to `Environment=`, but reads the environment variables from a text
  file" — so the file, not the unit, holds the values. It still points at a path that must be
  readable by the service manager at start; keeping that path outside the store avoids the leak.
- `LoadCredential=ID[:PATH]` (systemd.exec(5)) "Pass a credential to the unit … The data is
  accessible from the unit's processes via the file system, at a read-only location … When
  available, the location of credentials is exported as the `$CREDENTIALS_DIRECTORY` environment
  variable." The ID is a short filename-like string; an absolute `PATH` is opened as a regular file.
  The man page explicitly covers the per-user manager ("Note that encrypted credentials targeted
  for services of the per-user service manager must be encrypted with `systemd-creds encrypt
  --user` …"), so this works for home-manager units.

The idiomatic pattern for "a secret read at runtime": leave the secret in a file the user owns
(e.g. under `~/.config/vsrata/api-key`, mode 600), point `LoadCredential` at it, and read it from
`$CREDENTIALS_DIRECTORY`. The ID becomes the filename inside that directory:

```nix
Service = {
  LoadCredential = [ "api-key:${config.home.homeDirectory}/.config/vsrata/api-key" ];
  # the process reads $CREDENTIALS_DIRECTORY/api-key at startup
};
```

If the process wants a path rather than the env var, systemd.unit(5) defines `%d` as "Credentials
directory … the value of the `$CREDENTIALS_DIRECTORY` environment variable if available", so
`Environment = [ "STRATA_API_KEY_FILE=%d/api-key" ]` passes that path without inlining the secret.
The point either way: `LoadCredential`'s *value* in the unit is the path, never the secret itself.

### §2.7 `LimitMEMLOCK=infinity`

systemd.exec(5), the `LimitCPU=, …, LimitMEMLOCK=, …` section: "Set soft and hard limits on various
resources for executed processes … Use the string `infinity` to configure no limit on a specific
resource." `LimitMEMLOCK=` maps to `ulimit -l` (Bytes), i.e. the amount of memory a process may lock into RAM —
the setting a GPU/inference server needs so pages handed to the device are not swapped. In the
module it is a string:

```nix
Service.LimitMEMLOCK = "infinity";
```

### §2.8 Worked module skeleton

Minimal idiom demonstration only (the real option set is the design ticket's). It assumes the
package from §1 is available as `pkgs.vsrata`; §3 exposes it as `packages.x86_64-linux.vsrata` and,
so that `pkgs.vsrata` resolves, as an `overlays.default` the consumer imports.

```nix
# modules/vsrata.nix — a home-manager module (not NixOS)
{ config, lib, pkgs, ... }:
let
  inherit (lib) mkIf mkEnableOption mkOption types;
  cfg = config.services.vsrata;
  vsrata = pkgs.vsrata;                                    # the package from §1
  secretFile = "${config.home.homeDirectory}/.config/vsrata/api-key";
in
{
  options.services.vsrata = {
    enable = mkEnableOption "the vsrata model server";
    port = mkOption { type = types.port; default = 8080; };
    apiKeyFile = mkOption { type = types.path; default = secretFile; };
  };

  config = mkIf cfg.enable {
    systemd.user.services = {
      # the long-running server
      vsrata = {
        Unit = {
          Description = "vsrata model server";
          X-Restart-Triggers = [ "${vsrata}" ];           # restart on package change
        };
        Service = {
          ExecStart = "${lib.getExe vsrata} --host 127.0.0.1 --port ${toString cfg.port}";
          Restart = "on-failure";                          # §2.4
          LimitMEMLOCK = "infinity";                       # §2.7
          LoadCredential = [ "api-key:${cfg.apiKeyFile}" ];# §2.6 — path, not the secret
          Environment = [ "STRATA_API_KEY_FILE=%d/api-key" ];
        };
        Install.WantedBy = [ "default.target" ];           # §2.3
      };

      # a one-shot that runs after the server is up
      vsrata-warmup = {
        Unit = {
          Description = "vsrata warmup";
          Requires = [ "vsrata.service" ];                 # §2.3
          After    = [ "vsrata.service" ];
        };
        Service = {
          Type = "oneshot";                                # §2.2
          RemainAfterExit = true;
          ExecStart = "${lib.getExe vsrata} warmup";
        };
        Install.WantedBy = [ "default.target" ];
      };
    };
  };
}
```

## §3 The flake outputs

### §3.1 `packages.<system>.<name>`

The Nix manual, "Flake format": the `outputs` function "must be an attribute set … however, various
`nix` subcommands require specific attributes to have a specific value (e.g. `packages.x86_64-linux`
must be an attribute set of derivations built for the `x86_64-linux` platform)." Its example is
literally `packages.x86_64-linux.default = stdenv.mkDerivation { … }`. The repo's flake already uses
`packages.x86_64-linux.default` and `devShells.x86_64-linux.default` (`flake.nix:35,37`). The
extension is `packages.x86_64-linux.vsrata = …;` (and optionally
`packages.x86_64-linux.default = self.packages.x86_64-linux.vsrata;`). Because the flake fixes the
system (`flake.nix:22`), the existing one-system shape is kept; a `forAllSystems`-style `genAttrs`
is the multi-system idiom if that ever changes.

### §3.2 The home-module output name: `homeManagerModules` vs `homeModules`

This is the one place the ticket's assumed name needs a correction, so it is worth stating exactly
what home-manager does and does not define.

- home-manager's **own flake does not define `homeManagerModules` or `homeModules` as outputs.** Read
  at `release-22.11`, `release-23.05`, `release-23.11`, `release-24.11`, `release-25.05`, `release-26.05`
  and `master`: each exposes `nixosModules`, `darwinModules`, `flakeModules`, `packages`, `templates`,
  `lib`, etc. There is no `homeManagerModules` key.
- The **official flake-parts option is `flake.homeModules`** (home-manager `flake-module.nix`, master):
  `homeModules = mkOption { type = types.lazyAttrsOf types.deferredModule; … }`, documented in the
  manual's "flake-parts module" page (`flake.homeModules` / `flake.homeConfigurations`).
- `homeManagerModules` is nevertheless a **widely used downstream convention** for third-party
  flakes. The naming history: commit `066ba0c5c` "flake-module: rename `homeModules` to
  `homeManagerModules` (#6392)" was reverted by commit `2c87a6475` "flake-module: rename
  `homeManagerModules` to `homeModules` (#6406)", so the settled upstream name is `homeModules`.
  The #6392 discussion records the counts — roughly 10.5k GitHub usages of `homeManagerModules` vs
  ~4k of `homeModules`, with the experimental `flake-schemas` project using `homeModules`. There is
  no enforced standard.

**Recommendation:** expose **both**, one as an alias of the other, so the ticket's requested
`homeManagerModules.vsrata` works and the upstream-aligned `homeModules.vsrata` also works:

```nix
homeModules.vsrata = import ./modules/vsrata.nix;
homeManagerModules.vsrata = self.homeModules.vsrata;
```

`nix flake check`/`nix flake show` do not treat either as a standard output (neither appears in the
`nix flake show`/`check` manual pages), so this is purely a consumer-convention attribute — which is
why aliasing is cheap.

### §3.3 Worked flake skeleton

Extending the existing `flake.nix`. Two things stay as they are: the existing
`packages.x86_64-linux.default` — today the `strata-dev` `buildEnv` (`flake.nix:35`) — keeps its
value, and the `devShells` entries are untouched. The additions are the `.vsrata` package, an
`overlays.default` (needed so the module's `pkgs.vsrata` resolves — see below), and the two module
outputs. The CUDA-configured `pkgs` is already there and gains the CUDA config from §1.1:

```nix
{
  inputs = {
    nixpkgs  = { url = "github:NixOS/nixpkgs/nixos-26.05"; flake = false; };
    llamaCpp = { url = "github:ggml-org/llama.cpp/3cf03257f219afbe7334045ff7c6a06ac68c627d"; flake = false; };
  };

  outputs = { self, nixpkgs, llamaCpp }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "12.0a" ];
          cudaForwardCompat = false;
        };
      };
      vsrata = import ./pkgs/vsrata.nix { inherit pkgs llamaCpp; lib = pkgs.lib; };
    in
    {
      packages.x86_64-linux.vsrata = vsrata;                  # §3.1 (default unchanged)

      # devShells.x86_64-linux.default — unchanged (the repo's own dev shell, flake.nix:35)

      # so `pkgs.vsrata` resolves for the module below (§2.8)
      overlays.default = final: prev: { vsrata = vsrata; };

      homeModules.vsrata        = import ./modules/vsrata.nix;   # §3.2
      homeManagerModules.vsrata = self.homeModules.vsrata;        # alias
    };
}
```

In a consumer that evaluates home-manager through NixOS (`home-manager.users` and `nixpkgs.overlays`
both live there), the overlay is what makes the module's `pkgs.vsrata` resolve:

```nix
{
  inputs.vsrata.url = "github:kido5217/Vsrata";
  # ...
  nixpkgs.overlays = [ inputs.vsrata.overlays.default ];     # so pkgs.vsrata resolves (§2.8)
  home-manager.users.alice = {
    imports = [ inputs.vsrata.homeManagerModules.vsrata ];   # or .homeModules.vsrata
    services.vsrata = {
      enable = true;
      apiKeyFile = "/home/alice/.config/vsrata/api-key";
    };
  };
}
```

The alternative — the module taking the package as an option (`services.vsrata.package`) instead of
reading `pkgs.vsrata` — avoids the overlay but changes the module's interface; that choice is the
design ticket's. Either way the package must be in scope when the module is evaluated, and the flake
overlay is the flake-native way to put it there.

## §4 What I could not verify

### §4.1 The nvcc compiler-ID probe inside the sandbox

The repo's devShell sets `NVCC_PREPEND_FLAGS=-I${cuda}/include` because "CMake's CUDA compiler-ID
probe needs the merged toolkit's include dir" (`flake.nix:39-40`). I did **not** build the
derivation, so I cannot confirm whether the same workaround is required inside the sandbox (the
cudatoolkit setup hooks may already provide the include path there). The worked skeleton omits it;
add it as `env.NVCC_PREPEND_FLAGS = "-I${cuda.cudatoolkit}/include";` if the configure step fails
the probe. No build, no GPU work, and no model download was done for this ticket.

### §4.2 `cudatoolkit` vs the redistributables

The CUDA manual discourages `cudatoolkit` in favour of the `cudaPackages` redistributables
(`cudart`, `cublas`, `cuda_nvcc`, …). I did not determine which exact redistributable set replaces
the merged toolkit for this repo's `find_package(CUDAToolkit REQUIRED)` + `enable_language(CUDA)`
probe; the skeleton keeps the existing `cudatoolkit` convention to match `flake.nix:23`.

### §4.3 The home-module names in the wild

"`homeManagerModules` is the most common downstream convention" rests on home-manager PR #6392's
GitHub code-search counts, which are a snapshot and not a spec. I did not survey individual
downstream flakes. The recommendation to expose both names sidesteps the ambiguity.

### §4.4 The server's exact launch contract

The worked wrapper runs `python3 .../serve/server.py --engine <strata> --config <cfg>`, which is how
`setup.py` launches it (`setup.py:3093`, `:3455`), but `setup.py` also passes a generated config
file and other flags. The precise `vsrata` sub-command surface (start/stop/warmup/config) is a
design decision and is out of scope here; the skeleton's `vsrata warmup` is a placeholder.

## §5 Not in scope

- The module's option names, defaults and schema (design ticket).
- Any change to the tracked `CMakeLists.txt` / `tools/vision/CMakeLists.txt` (e.g. adding `install()`
  rules) — the skeleton installs by hand instead.
- A `NixOS` system module, Windows, AMD/HIP, or non-`x86_64-linux` output shapes.
