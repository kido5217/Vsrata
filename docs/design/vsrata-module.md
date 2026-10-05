# The `vsrata` module: option schema, generated artifacts, and CLI contract

Ticket [#26](https://github.com/kido5217/Vsrata/issues/26) — "The module's option schema and generated
artifacts", under map [#23](https://github.com/kido5217/Vsrata/issues/23). This is the spec
[#27](https://github.com/kido5217/Vsrata/issues/27) implements; every option, file and unit below is a
decision, not a sketch.

## 0. What this spec encodes

1. The flake exposes `packages.<system>.vsrata`, `overlays.default`, **`homeModules.vsrata` and a
   `homeManagerModules.vsrata` alias** (home-manager defines neither name itself — its flake-parts
   option is `flake.homeModules`). The module takes `programs.vsrata.package`, defaulting to
   `pkgs.vsrata`, so a consumer must either import the overlay or set `package` explicitly: a `pkgs`
   without `vsrata` fails to evaluate (research #24).
2. Namespace `programs.vsrata`. The vocabulary is the repo's: **profile**, **model**, **quant**,
   **context window**, **provisioning** (`GLOSSARY.md`).
3. x86_64-linux and NVIDIA only. The package's `cudaArchitectures` defaults to `[ "120a" ]`; `120a`
   SASS runs only on sm_120 and this is documented.
4. Services are **manual-start**, with `Restart=on-failure`.
5. Secrets: `hfToken` (string) **or** `hfTokenFile` (path). The unit delivers one with
   `LoadCredential`; the CLI reads the runtime file. The string form is convenience only — it lands in
   the Nix store, and the option's documentation says so.
6. Ports are per profile; `null` means `basePort + index`, where the index is the profile's position
   in alphabetical order. The service binds `host` (default `127.0.0.1`), and a non-loopback `host`
   without an API key is a **nix evaluation error**.
7. Provisioning is a per-profile `oneshot` that the serve unit `Requires=` and `After=`; the CLI
   writes a completion marker and the unit carries `ConditionPathExists=!<marker>`, so a finished
   profile costs nothing at boot.
8. `modelDir` **is** the data root. The generated config lives in `$XDG_STATE_HOME/vsrata/`.
9. `LimitMEMLOCK=infinity` on both units, with the system-level prerequisite documented (home-manager
   cannot raise a user's hard limit).
10. Vision: `vision` (bool) and `visionAccel` (`"gpu"` | `"cpu"`); the package always builds
    `strata-vision`.
11. The package ships a `vsrata` CLI; the units call `vsrata provision|serve --config <path>`.
12. The server's Python dependencies come from `python3.withPackages`.
13. A bad profile (unknown model or quant, impossible context window) fails **`nix eval`**, against a
    generated `nix/registry.nix`.
14. **Concurrent profiles on one GPU are out of scope**: the module neither serializes nor caps them.

## 1. The option tree

```nix
programs.vsrata = {
  enable      = true;
  package     = pkgs.vsrata;
  hfToken     = null;                    # convenience only: lands in the store
  hfTokenFile = null;                    # the safe form
  modelDir    = "/home/kido/trash/ai/vsrata/models";   # the DATA ROOT
  basePort    = 8095;
  host        = "127.0.0.1";
  profiles    = { };
};
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `enable` | bool | `false` | Generate the units and the files |
| `package` | package | `pkgs.vsrata` | Where the engine, server and CLI come from |
| `hfToken` | null or str | `null` | A literal token; convenience, stored in the world-readable store |
| `hfTokenFile` | null or path | `null` | A file with the token; 0600, never in the store |
| `modelDir` | null or str | `null` | The data root; **required** when `enable` |
| `basePort` | port | `8095` | The first profile's port |
| `host` | str | `"127.0.0.1"` | The bind address |
| `profiles` | attrs of profile | `{ }` | One service per attribute name |

A profile:

```nix
profiles."profile-01" = {
  enable        = true;
  model         = "unsloth/Qwen3.8-Flash-Next-GGUF";
  quant         = "UD-Q4_K_XL";
  contextWindow = 262144;
  vision        = false;
  visionAccel   = "gpu";
  port          = null;
  apiKeyFile    = null;
  extraArgs     = [ ];
};
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `enable` | bool | `true` | Whether this profile has units |
| `model` | str | — | The Hugging Face repository, validated against the registry |
| `quant` | str | — | The quantization within it, validated against the registry |
| `contextWindow` | int | — | One of the seven sizes setup offers (`8192 … 524288`) |
| `vision` | bool | `false` | Serve images as well as text |
| `visionAccel` | enum `gpu`/`cpu` | `"gpu"` | Which encoder to run |
| `port` | null or port | `null` | `null` → `basePort + index` |
| `apiKeyFile` | null or path | `null` | A file with this profile's API key |
| `extraArgs` | list of str | `[ ]` | Appended to the engine's `args` verbatim |

`extraArgs` is the only escape hatch for engine tuning — VRAM budgeting between profiles is not the
module's job (decision 14).

## 2. The registry — `nix/registry.nix`

The module validates a `(model, quant)` pair at evaluation time, so it needs the registry Nix-side.
The registry's real source is `setup.py`'s `FAMILIES`, `MODELS` and `HF_REVISIONS` tables (the CLI
uses those same tables from the package), so `nix/registry.nix` is **generated** and checked:

- `tools/make_registry.py` writes `nix/registry.nix` from `setup.py`'s tables (a #27 deliverable).
- A `just registry` recipe regenerates it.
- A test fails when a regenerated copy differs from the committed one, so it cannot silently lag.

Shape:

```nix
{
  "unsloth/Qwen3.8-Flash-Next-GGUF" = {
    family     = "unsloth";
    name       = "qwen3.8-flash-next-unsloth";      # the prefix of the config's model_name
    revision   = "<pinned commit>";
    subdir     = "{q}/";                            # or null when the shards sit at the repo root
    file       = "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf";   # the family's default
    profile    = null;                               # "expert-profile-coder.bin" for coder, else null
    packArgs   = [ "--compat-bf16" ];                # handed to the packer (§5)
    trainedContext = 262144;                         # the window the rope rule is measured against
    mmproj     = { repo = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF";
                   file = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"; };
    quants = {
      "UD-Q4_K_XL"  = { shards = 4; downloadGb = 111.3; vision = false; experimental = true; };
      "UD-IQ4_XS"   = { shards = 3; downloadGb =  93.7; vision = true;
                        file = "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf"; };  # per-quant override
    };
  };
  # … one entry per registry model
}
```

The shape is deliberately wider than a model/quant pair: it carries everything the `args` template
needs — `name` for `model_name`, `profile` for `--expert-profile`, `packArgs` for the packer, `mmproj`
for vision — so the module never has to guess a family's quirks. `file` is a family default that a
quant may override: `UD-IQ4_XS` has three shards, not four (`setup.py`'s `model_file`, #621).

The module reads this file; the CLI reads the same information from `setup.py` inside the package.
The drift check keeps the two honest.

## 3. What one profile generates

### Paths

| Path | Contents |
|---|---|
| `<modelDir>/models/<profile>/` | the GGUF shards (and the mmproj when vision is on) |
| `<modelDir>/packs/<profile>/` | the pack: `native_experts.txt`, `index.txt`, `dense.bin`, `tokenizer/` |
| `<modelDir>/mtp/<profile>/` | the fetched tensors, the packed MTP GGUF, and `rt/` |
| `$XDG_STATE_HOME/vsrata/<profile>.json` | the generated config (below) |
| `$XDG_STATE_HOME/vsrata/<profile>.done` | the provisioning marker the CLI writes |
| `$XDG_STATE_HOME/vsrata/<profile>.log` | the engine's log |

### The config JSON

One file per profile. The server has **no top-level key validation** — it reads the keys it knows and
never looks at the rest — so the `provision` block is simply ignored there; only a separate
`shared-settings.json` rejects unknown keys (`serve/server.py:3835`).

```json
{
  "exe":            "<package>/bin/strata",
  "args":           [ … ],                        // §3.3
  "cwd":            "<state dir>",
  "tokenizer":      "<modelDir>/packs/<profile>/tokenizer",
  "model_name":     "<family>-<quant>",
  "log":            "<state dir>/<profile>.log",
  "lib_dirs":       [ ],
  "host":           "127.0.0.1",
  "port":           8095,
  "api_key":        "",                           // never written; the file path is passed in
  "vision":         { … },                        // only when vision is on
  "provision": {
    "model":        "unsloth/Qwen3.8-Flash-Next-GGUF",
    "quant":        "UD-Q4_K_XL",
    "dataRoot":     "<modelDir>",
    "profileDir":   "<modelDir>/models/<profile>",
    "packDir":      "<modelDir>/packs/<profile>",
    "mtpDir":       "<modelDir>/mtp/<profile>",
    "context":      262144,
    "vision":       false,
    "visionAccel":  "gpu"
  }
}
```

`host` is read by the **server** from this file (and can be overridden with `--host`); **`port` is not**
— `serve/server.py` never looks at it, so the CLI (and the unit) pass `--port` on the command line,
which is why the key is written into the file at all. `api_key` is likewise passed on the command
line (or through `STRATA_API_KEY`) so a key never lands in the generated file.

The vision block — every key `serve/server.py` reads, the last two optional:

```json
"vision": { "exe": "<package>/bin/strata-vision", "mmproj": "<…>/mmproj-….gguf",
            "model": "<…>/shard-00001-of-…gguf", "gpu": true,
            "max_tokens": 1024, "threads": 0, "cuda_device": null }
```

### The engine's `args`

Assembled exactly as `setup.py:4224-4227` does:

```
--pack <packDir> --native <models/<profile>/<shard 1>
[--ple-gguf <the shard holding per_layer_token_embd.weight>]   # when the model has ≤ 2 shards
--expert-profile <package>/share/vsrata/data/<registry profile> --expert-cache auto
--prefill auto --spec 4 --spec-min-p 0.5 --mtp <mtpDir>/rt --max-context <contextWindow>
[--rope-scaling yarn --rope-scale <max 1.0 (contextWindow / 262144)>]   # when contextWindow > 262144
[--kv <kv>]               # when contextWindow > 8192
[--vision --vram-reserve-mib 700]              # when vision is on; 700 is VISION[mode].reserve_mib
… plus the profile's extraArgs
```

`--pack` takes the pack directory and `--native` the first shard. `<registry profile>` is
`expert-profile-coder.bin` for the coder family (`setup.py:205`) and `expert-profile.bin` otherwise
(`setup.py:4225`). The
rope pair mirrors `resolve_rope`/`derived_factor` — `yarn` with the final ÷ trained factor, at least 1 —
which is what keeps a 524288 window inside the model's extension rule; #27 should confirm the factor is
exactly that expression before relying on it. The expert profile and draft vocab are **reads** from the
package's `data/`, which is why they can live in the read-only store. A native pack requires `--native`,
`--spec ≥ 2` and `--prefill`, and must not be given `--keep-canonical`
(`generate.cpp:2165-2166`), all of which this template satisfies.

## 4. The unit pair

Two units per profile, both in `systemd.user.services`, both with `LimitMEMLOCK=infinity` and the
same environment:

```
vsrata-provision-<profile>.service          vsrata-<profile>.service
  Type=oneshot                                Type=simple
  RemainAfterExit=yes                         Requires=vsrata-provision-<profile>.service
  ConditionPathExists=!<state>/<profile>.done After=vsrata-provision-<profile>.service
  ExecStart=<package>/bin/vsrata provision --config <state>/<profile>.json
                                              Restart=on-failure
  LoadCredential=hf-token:<hfTokenFile>       ExecStart=<package>/bin/vsrata serve --config <state>/<profile>.json --port <port>
  Environment=VSRATA_HF_TOKEN_FILE=%d/hf-token
                                              LoadCredential=api-key:<apiKeyFile>
                                              Environment=STRATA_API_KEY_FILE=%d/api-key
```

Notes:

- **No `WantedBy`** on either unit: services are manual-start, and the serve unit pulls the provision
  unit in through `Requires=`, so `systemctl --user start vsrata-<profile>` really does download on
  first run and then serve.
- `Restart=on-failure` sits on the **serve** unit, where it belongs; the oneshot would accept it too
  (only `always`/`on-success` are rejected there), but provisioning retries by re-running the unit.
- A finished profile is skipped by systemd itself (`ConditionPathExists=!`), before the CLI runs.
- `hfToken` as a **string** cannot use `LoadCredential`; #27 must materialize it into the state dir
  with 0600 and point the unit at that file, and the option's docs must state the store caveat.

## 5. The CLI — `vsrata`

Two subcommands, both taking an absolute `--config` path, so they are pure functions of one file and
can be run by hand:

- `vsrata serve --config <path> [--port <port>]` — execs the server:
  `python <package>/share/vsrata/serve/server.py --engine strata --config <path> --port <port>`,
  with the API key passed through the environment.
- `vsrata provision --config <path>` — idempotent, in this order, with its **own** marker per stage
  (the tools' gates are existence-only over non-atomic writers, so the CLI cannot trust them):
  1. the GGUF shards (resumable; `.part` → `replace`) into `models/<profile>/`;
  2. the mmproj when vision is on;
  3. the pack — a **self-contained `iq_pack`**, which is the path `setup.py` itself takes on a
     non-AVX-512 CPU (`setup.py:4175-4193`), which embeds the tokenizer, and which must be given the
     registry's `packArgs` — `--compat-bf16` for `unsloth`, without which the packer refuses and
     writes nothing (`docs/UNSLOTH_Q4.md:84-86`);
  4. the MTP chain — `mtp_fetch` (resumable) → `mtp_pack` → `mtp_rt`, plus the draft vocab copied
     from the package's `data/`;
  5. write `<profile>.done`.
  It re-checks each stage and no-ops the ones already complete.

## 6. Validation and failure modes

Assertions, all raised at evaluation time:

1. `modelDir` is set when `enable` is true.
2. `(model, quant)` exists in `nix/registry.nix`.
3. `contextWindow` is one of the seven sizes.
4. `host` is loopback, or **every** profile sets `apiKeyFile` — the host is global, the key is per
   profile.
5. When `profiles` is non-empty, **at most** one of `hfToken` / `hfTokenFile` is set. A token is
   optional at all: the registry's repositories are public and anonymous access works.
6. Profile names are valid unit-name components.

Each failure names the offending profile and lists the allowed values.

## 7. Out of scope

- **Budgeting several profiles on one GPU** — the module neither serializes nor caps them; the user
  ensures one runs at a time (map #23).
- A NixOS system module, an overlay beyond the one the module needs, non-NVIDIA backends, and serving
  models outside the registry.
- **Low-RAM packs.** A pack built for a host that cannot hold the experts in RAM
  (`--resident-experts`, `--resident-budget-gib`, `--ple-io`, and `iq_pack --experts-bin`) is not
  modelled: the registry carries no `budget`/`arenaGb`, and such a host is a later effort. This host
  (126 GB) fits the registry's models without it.

## 8. Risks for #27

1. **`serve/server.py`'s `ROOT`-relative defaults.** The server computes paths from its own location;
   under a store layout those defaults point into the store. Every one the server uses (the tokenizer
   default above all, plus any static asset directory) must be overridden by the config or the layout.
2. **The sandbox's CUDA compiler-ID probe.** The dev shell needs
   `NVCC_PREPEND_FLAGS=-I${cuda}/include`; whether a sandboxed build needs it is open (research #24).
3. **`lib_dirs` vs RPATH.** The engine may need the CUDA runtime libraries on `LD_LIBRARY_PATH`; the
   package should prefer RPATHs, with `lib_dirs` as the fallback.
4. **The string `hfToken`'s materialization** (§4) needs a 0600 write from an activation script, and
   the store caveat must be documented rather than hidden.
5. **`iq_pack`'s disk cost** — the pack's `dense.bin` is model-specific (≈1.4 GB for `UD-Q4_K_XL`,
   ≈5.4 GiB for `Q2_0`), on top of the downloaded shards.
6. **The rope factor.** The module computes `max 1.0 (ctx / 262144)` in Nix to mirror
   `derived_factor`; if the routine ever grows a model-specific table, the module must stop computing
   it and let the CLI add those flags instead.
