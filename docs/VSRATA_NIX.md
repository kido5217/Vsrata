# Vsrata as a NixOS package + home-manager program — usage, models, limits

Everything below is grounded in the repository at `f762d27` (PR #31) and was verified by running
commands when it was written, except where a line says otherwise. Written 2026-10-06.

Pointers:

- Delivery: PR [#31](https://github.com/kido5217/Vsrata/pull/31), squash-merged as `f762d27`; ticket
  [#27](https://github.com/kido5217/Vsrata/issues/27); map
  [#23](https://github.com/kido5217/Vsrata/issues/23).
- Design: `docs/design/vsrata-module.md` — **on the `design/vsrata-module` branch**, not on `main`.
- Research: `docs/research/nix-packaging-surface.md` and `docs/research/provisioning-contract.md` —
  both on their own `research/*` branches.

---

## 1. Status

| | |
|---|---|
| Merged | `f762d27` — `feat: the vsrata Nix package, home-manager module and CLI (#27) (#31)` |
| Package | `vsrata-0.1.39` → `/nix/store/c1zd4igd06ikv2bkccj4py7b7mw3s514-vsrata-0.1.39` |
| Derivation | `/nix/store/fg760gvlmlx0fsh4f9xklbfxxda83zpj-vsrata-0.1.39.drv` — `nix build .#vsrata` at that
  commit is a cache hit, so the built artifact is exactly main's tree |
| Contents | `bin/strata` (45 MB), `bin/strata-vision` (74 MB), `bin/vsrata`; `share/vsrata/{serve,tools,data,cli,ref,setup.py,third_party/llama.cpp/gguf-py}` |
| Tested | 15 tests (`tools.test_vsrata_cli`, `tools.test_registry_drift`); 8/8 installed modules import;
  two-profile home-manager evaluation |
| Confirmed | a second, scoped reviewer re-ran all five post-review fixes at `f762d27` — all confirmed |

**Two low-severity items are open** (both cosmetic, deliberately left; fixing either means a full
engine rebuild because `src = self`):

1. A non-whole rope factor prints `1.500000` rather than `1.5` — only reachable at the 393216-window
   profile. `setup.py` uses `%g`. The engine parses a float, so no runtime difference; only an exact
   string comparison against `setup.py`'s own output would notice.
2. `modules/vsrata.nix`'s header cites `docs/design/vsrata-module.md`, which is not tracked on `main`
   (the spec lives on the `design/vsrata-module` branch and in issue #27).

---

## 2. Consumer configuration

`vsrata` is a flake in this repository exposing a package and a home-manager module. The module's
default package is `pkgs.vsrata`, which only exists once the flake's overlay is imported — a `pkgs`
without it fails to evaluate.

### `flake.nix` (NixOS host, home-manager as a NixOS module)

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    home-manager = {
      url = "github:nix-community/home-manager/release-26.05";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    vsrata.url = "github:kido5217/Vsrata";
  };

  outputs = { self, nixpkgs, home-manager, vsrata }: {
    nixosConfigurations.kido = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        home-manager.nixosModules.home-manager
        {
          # Puts `vsrata` into `pkgs`. Without this the configuration does not evaluate.
          nixpkgs.overlays = [ vsrata.overlays.default ];

          # The engine pins its expert arena and PLE table. A user cannot raise their own hard
          # RLIMIT_MEMLOCK, so it is raised here; user@.service inherits it and passes it to the
          # user's units. NOT exercised in this session (see §7).
          systemd.settings.Manager.DefaultLimitMEMLOCK = "infinity";

          home-manager.users.kido = {
            imports = [ vsrata.homeModules.vsrata ];
            programs.vsrata = { /* §2.1 */ };
          };
        }
      ];
    };
  };
}
```

`allowUnfree` is **not** needed on the consumer side: the flake builds its own CUDA toolkit internally
and the overlay hands over the finished package. Verified by evaluating a consumer without it.

### 2.1 `programs.vsrata` — a full example

```nix
programs.vsrata = {
  enable = true;

  # The data root. `models/<profile>/` (GGUF shards), `packs/<profile>/` (the pack it builds) and
  # `mtp/<profile>/` are created inside it. Budget ~100 GB for the first large model.
  modelDir = "/home/kido/trash/ai/vsrata/models";

  # The token, read by the provisioning unit at runtime. `hfToken = "hf_…"` also works but writes
  # the literal into the world-readable Nix store.
  hfTokenFile = "/run/secrets/vsrata-hf-token";

  basePort = 8095;        # profile N → basePort + its alphabetical index (8095, 8096, …)
  # host = "127.0.0.1";   # a non-loopback host requires apiKeyFile on every profile

  profiles = {
    "profile-01" = {
      model = "unsloth/Qwen3.8-Flash-Next-GGUF";
      quant = "UD-IQ4_XS";
      contextWindow = 262144;    # 8192, 32768, 65536, 131072, 262144, 393216, 524288
      vision = true;
      visionAccel = "gpu";
    };

    "profile-02" = {
      model = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF";
      quant = "IQ1_M";
      contextWindow = 131072;
      port = 9100;                               # optional; else basePort + index
      extraArgs = [ "--pcie-frac" "0.20" ];      # straight to the engine
    };
  };
};
```

### 2.2 Every option

Global:

| Option | Type | Default | Meaning |
|---|---|---|---|
| `enable` | bool | `false` | Turns the module on |
| `package` | path | `pkgs.vsrata` | The package the units run |
| `hfToken` | null or string | `null` | Convenience; **lands in the world-readable store** |
| `hfTokenFile` | null or path | `null` | Preferred: read by the unit at runtime |
| `modelDir` | null or string | `null` | **Required** when enabled; the data root |
| `basePort` | port | `8095` | The first profile's port |
| `host` | string | `"127.0.0.1"` | Bind address; non-loopback demands a key |
| `profiles` | attrset | `{}` | One model-serving instance per name |

Per profile:

| Option | Type | Default | Meaning |
|---|---|---|---|
| `enable` | bool | `true` | Whether this profile has units |
| `model` | string | — | Hugging Face repo, as named in `nix/registry.nix` |
| `quant` | string | — | Quantization within that repo |
| `contextWindow` | positive int | — | One of the seven sizes |
| `vision` | bool | `false` | Also serve images (needs the repo's mmproj) |
| `visionAccel` | `"gpu"` or `"cpu"` | `"gpu"` | Which encoder, when vision is on |
| `port` | null or port | `null` | `null` → `basePort` + alphabetical index |
| `apiKeyFile` | null or path | `null` | Delivered to the serve unit as a credential |
| `extraArgs` | list of strings | `[]` | Appended to the engine's arguments verbatim |

### 2.3 What is refused at evaluation

Three global assertions and two per profile, plus the registry's own refusal:

- `modelDir` must be set when `enable` is true.
- At most one of `hfToken` / `hfTokenFile`.
- A non-loopback `host` requires `apiKeyFile` on **every** profile.
- `contextWindow` must be one of the seven sizes.
- A profile name must be usable in a unit name.
- An unknown `model` or `quant` throws, naming the profile and listing the known values.

### 2.4 Generated per profile

- `$XDG_STATE_HOME/vsrata/<profile>.json` — the serve config **and** a `provision` block, installed
  there at activation (not the store: the CLI writes each stage's marker beside the config).
- `vsrata-provision-<profile>.service` — `oneshot`, `RemainAfterExit`, `ConditionPathExists=!…/<profile>.done`,
  `LoadCredential`, `LimitMEMLOCK=infinity`.
- `vsrata-<profile>.service` — `simple`, `Requires=`/`After=` the provision unit, `Restart=on-failure`,
  `LimitMEMLOCK=infinity`, **no `WantedBy`** (manual start only).

---

## 3. Models and quants a profile may name

The registry is generated from `setup.py`'s installer tables by `tools/make_registry.py`, and
`tools/make_registry.py --check` fails if `nix/registry.nix` drifts. **Every valid value:**

| `model` | valid `quant` | shards | `vision = true` |
|---|---|---|---|
| `unsloth/Qwen3.8-Flash-Next-GGUF` | `UD-IQ4_XS` | 3 | yes |
| | `UD-Q4_K_XL` | 4 | **no** — and flagged experimental |
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `Q2_0` | 2 | yes |
| | `IQ3_S`, `IQ3_XXS`, `IQ2_XS` | 2 | yes |
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | `IQ1_M` | 2 | yes |
| `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `IQ2_XS`, `IQ3_XXS` | 2 | yes |

Anything else fails `nix eval`.

---

## 4. First run and operation

```bash
sudo nixos-rebuild switch --flake .#kido     # writes ~/.local/state/vsrata/<profile>.json

systemctl --user start vsrata-profile-01     # pulls in the provisioning unit first
journalctl --user -u vsrata-provision-profile-01 -f    # the download and pack build
systemctl --user status vsrata-profile-01
systemctl --user list-units 'vsrata*'
```

The first start downloads the shards (tens to ~120 GB depending on the quant), builds the pack and
fetches the MTP draft tensors, then serves. The provision unit's `ConditionPathExists` means later
starts skip straight to serving. Nothing starts at login.

```bash
curl -s localhost:8095/v1/models
curl -s localhost:8095/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}' | head -c 400
```

Routes: the web app at `/`; `/v1/chat/completions`, `/v1/responses` (OpenAI dialects) and
`/v1/messages` (Anthropic); `/v1/models`, `/status`, `/metrics`, `/props`, `/slots`.

The CLI the units call is also usable by hand:

```bash
vsrata provision --config ~/.local/state/vsrata/profile-01.json   # idempotent; per-stage markers
vsrata serve     --config ~/.local/state/vsrata/profile-01.json --port 8095
```

---

## 5. System prerequisites (outside the module)

| Need | Why | Where |
|---|---|---|
| `RLIMIT_MEMLOCK` raised | The engine pins its expert arena and PLE table; a user cannot raise their own hard limit | `systemd.settings.Manager.DefaultLimitMEMLOCK = "infinity";` (nixpkgs exposes `systemd.settings.Manager`, i.e. `system.conf`'s `[Manager]`; the pinned nixpkgs has **no** `systemd.user.settings` option) — **not exercised in this session** |
| Disk | `modelDir`: ~58–120 GB per model for the shards, plus the pack it builds | — |
| RAM | The Orca doc measures a ~49.8 GiB expert arena for one model; this host has 128 GB | `docs/ORCA.md` |
| One GPU, one profile | The module neither serializes nor caps profiles; a 32 GB card fits one of these models | spec §0.14 |

---

## 6. Orca — not covered, and what it would take

`orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` is a **manual** compatibility workflow in this repo:
`docs/ORCA.md` (IQ3_XXS, 2 shards, 85.20 GB) and `docs/ORCA_Q4_K_S.md` (Q4_K_S, 3 shards). Its own
README says: *"you set it up by hand. It is not in the installer's menu."* The module's registry is
generated from that menu, so Orca falls outside by construction; `orcarouter` appears nowhere in
`setup.py`, `tools/make_registry.py` or `nix/`.

Three concrete gaps:

1. **Not in the registry** — `model = "orcarouter/…"` fails `nix eval`, named and refused.
2. **The PLE shard is wrong** — Orca's PLE table is in **shard 1**; `modules/vsrata.nix:43` assumes
   shard 2 for every family but Swift. The `entry.pleShard` hook exists, but the generator emits no
   such field (0 occurrences in `nix/registry.nix`), so it cannot be expressed today.
3. **Q4_K_S needs a different engine** — that quant's Q5_0 expert down-matrices need
   `STRATA_ORCA_Q4KS_MMQ=ON` (`CMakeLists.txt:77`, **OFF** by default), which `pkgs/vsrata.nix` does
   not pass. IQ3_XXS needs no special flag — only `--compat-bf16`, which the packer already has.
   (`STRATA_NATIVE_EXPERTS` is ON by default, `CMakeLists.txt:949`.)

Smaller mismatches: `docs/ORCA.md` recommends `--prefill 512` where the module emits `auto`
(overridable per profile with `extraArgs = [ "--prefill" "512" ]`); and Orca's IQ3_M is unsupported
outright — its Q5_0 expert matrices are not handled by the native GPU expert path.

**What works today:** the packaged tools can do the manual workflow (`tools/iq_pack.py --compat-bf16`
and the MTP chain are both installed), and `vsrata serve --config <your own config>` serves any
hand-written config whose `exe` points at the packaged engine. `vsrata provision` will not — it
re-derives everything from `setup.py`'s installer tables and refuses an unknown repo — so an Orca
profile has to be provisioned by hand.

**To make it a profile:** add an Orca entry to the generator (including a PLE-shard field it currently
never emits), and for Q4_K_S a package variant built with the Orca MMQ flag. That is a design change,
not a config tweak.

---

## 7. Known caveats

1. **`src = self` means any tracked change rebuilds the engine.** A one-line doc or test edit changes
   the package's source hash, so `nix build .#vsrata` recompiles everything (~10–15 min). Scoping the
   source to the engine/server/tools inputs is the fix; recorded as map #23 fog.
2. **The server's `ROOT` is `__file__`-derived, and importing `serve.server` as a *module* from a CWD
   that is another Strata checkout shadows the package.** The units invoke it by script path, so they
   are unaffected. Setting `WorkingDirectory` on the units is a candidate hardening.
3. **The fixture-dependent tests skip without a fixture tree** — they point the CLI at the checkout's
   `pack/q2_0`. An unguarded run once started a real 62 GB download into a worktree; the guard exists
   because of that.
4. **The memlock line in §2 is the documented mechanism, not a tested one.** This session verified the
   module, the package, the CLI and the generated configs; it did not evaluate a full NixOS system
   containing `systemd.settings.Manager.DefaultLimitMEMLOCK`.
5. **No profile has been served end-to-end yet.** The package builds, imports and provisions against
   the fixture tree; a real first run (download → pack → serve → decode) is ticket
   [#28](https://github.com/kido5217/Vsrata/issues/28), still open.
