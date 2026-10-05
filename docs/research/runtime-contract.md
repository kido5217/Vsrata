# The unattended runtime contract (ticket #6)

Research date: 2026-10-05. Repo: `kido5217/Vsrata`, branch `main` @ `f9d372a`; this is a copy of
`Niko1221/Strata` at v0.1.39 (`CMakeLists.txt:11` `project(strata VERSION 0.1.39)`; `MIN_ENGINE = (0, 1, 39)`
at `setup.py:118`). Method: reading code and docs only (`serve/server.py`, `serve/runconfig.py`,
`serve/responses.py`, `serve/telemetry.py`, `serve/mcp.py`, `serve/winjob.py`, `setup.py`,
`src/program/generate.cpp`, `src/core/pinned.cu`, `src/core/expert_source.cpp`, `src/kernels/ngram.cpp`,
`src/platform/memory.cpp`, `Dockerfile`, `docker-entrypoint.sh`, `docs/INSTALL.md`, `docs/DETAILS.md`,
`docs/MULTI_GPU.md`, `bench/results/2026-09-29-rtx3090-epyc-milan/README.md`, `requirements.txt`), building on
tickets #4 (`research/setup-flow`) and #5 (`research/prebuilt-vs-source`). Nothing was executed: no server or
engine start, no model files downloaded.

## TL;DR

- **What runs:** `serve/server.py --engine strata --config strata-<model>.json --port 8080` (what `setup.py`
  `start()` spawns, `setup.py:3093-3094`, and what `run-<model>.sh` execs, `setup.py:3455-3463`). The server
  then spawns the compiled engine once, resident: `engine/strata --serve <config's args>` as a child process
  driven over stdin/stdout (`server.py:454`), and serves the OpenAI/Anthropic/Responses API plus a web app on
  the port.
- **The config is a complete contract:** every key the runtime reads is listed in §1 with its writer and
  reader. Setup writes the "core" keys (`exe`, `args`, `cwd`, `tokenizer`, `model_name`, `log`, `lib_dirs`,
  `port`, and — conditionally — `backend`, `env`, `gpu`, `gpus_asked`, `layer_split`, `host`, `api_key`,
  `cuda`, `draft_vocab`, `vision`, `open_browser`, `parallel`); every other key (`sampling`, `mcp*`,
  `cors_origins`, `aliases`, `idle_unload_s`, …) is user-added and **carried over by `carry_over`** on later
  setup runs (`setup.py:2694-2726`).
- **Environment:** the server copies its own environment, adds `CUDA_VISIBLE_DEVICES` (or `HIP_VISIBLE_DEVICES`
  on AMD) from the config's `gpu`, appends the config's `env` entries, and **prepends the config's `lib_dirs`
  to `LD_LIBRARY_PATH`** (`server.py:1560-1575`) — those dirs point at the CUDA libraries the engine needs
  (pip's `nvidia/*` wheels for a prebuilt engine, the CUDA toolkit's own `lib64` for a locally compiled one,
  `setup.py:4121-4123`). CWD does not matter to the server (it resolves everything from `ROOT` = the checkout
  and absolute config paths); the engine runs with `cwd = config's "cwd"` (`server.py:4001`).
- **Runtime limits:** the expert arena (32-62 GB of RAM by model size, `setup.py:122-152`) is pinned
  (`mlock`/`cudaHostRegister`), so the service needs an **unlimited memlock ulimit** (`--ulimit memlock=-1` in
  Docker, `Dockerfile:20`; `LimitMEMLOCK=infinity` in the one documented systemd precedent,
  `bench/results/2026-09-29-rtx3090-epyc-milan/README.md:21`). Below that, the engine still runs — `mlock`
  failure falls back to pre-touching pages (`src/platform/memory.cpp:115`, `src/kernels/ngram.cpp:271`).
  Port default is **8080** for a setup install (`setup.py:3681`); the server's own argparse default is 8095
  (`server.py:3909`). API key default is **empty (no auth)**; with a key, everything under `/v1/*` and the
  other metadata endpoints require it — **`/health` never does** (`server.py:3030-3033`), which is what the
  Docker `HEALTHCHECK` polls (`Dockerfile:108-113`).
- **Telemetry is local-only:** `serve/telemetry.py` samples GPU/CPU/RAM/disk once a second from NVML (ctypes),
  amdgpu sysfs or ps//proc, keeps 60 samples in memory, and exposes them **only** through `GET /metrics` and
  `GET /v1/status`. It sends nothing anywhere — the file contains no sockets, requests or URLs
  (`telemetry.py:11-19`). It always runs and has no disable flag; that is fine because it is read-only and
  "nothing here can stop the server" (`telemetry.py:9`).
- **Minimal invocation:** one JSON config + one command + one ulimit. See §5 for the exact systemd-ready spec.

## 1. The `strata-<model>.json` config schema

Written by `write_setup_config` (setup step 7, `setup.py:4384`; atomic write + `.bak`, `setup.py:2684-2689`,
`2739-2769`), read by `serve/server.py main()` at start (`server.py:3937`) and by `setup.py start()` on later
launches (`setup.py:3070`). A model's config is `strata-<tag>.json` in the **repo root**, `<tag>` =
family tag + lowercased size (`iq3_xxs`, `swift-iq3_xxs`, `coder-iq1_m`, …, `setup.py:3936`, `setup.py:4375`).

The base object setup writes is `setup.py:4324-4326`:

```python
cfg = {"exe": str(eng / EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
       "model_name": f"{fam['name']}-{model.lower()}", "log": str(ROOT / f"strata-{tag.lower()}.log"),
       "lib_dirs": lib_dirs, "port": port}
```

### 1.1 Core keys (setup writes them; a setup re-run rewrites them, carrying over the rest)

`SETUP_KEYS` is exactly the set setup claims: `{"exe", "args", "cwd", "tokenizer", "model_name", "log",
"lib_dirs", "port", "backend", "env", "gpu", "gpus_asked", "layer_split", "host", "api_key", "draft_vocab",
"vision"}` (`setup.py:2694-2695`); user keys outside it survive re-runs via `carry_over`
(`setup.py:2700-2726`). Hand-added `args` entries are **not** merged — setup rewrites `args` and names the
dropped flags (`setup.py:2704-2705`, `2729-2736`).

| Key | Type | Writer | Reader | Default when absent |
| --- | --- | --- | --- | --- |
| `exe` | str, path to engine | `setup.py:4324` (`<repo>/engine/strata`; `engine-cuda12/strata` for the experimental CUDA 12 engine); `use_cuda12` rewrites it (`setup.py:3443`) | `server.py:3992` (a relative exe is resolved against `cwd`), `server.py:437` (reads `BUILD.json` beside it for the version); `setup.py:3071` (start refuses if the file is missing) | required for `--engine strata` |
| `args` | list of engine CLI flags | `setup.py:4224-4374` (the model's flags, below), + calibration `setup.py:4380` + pool workers `setup.py:4383`; `start()` rewrites it for a kept `--vram-reserve-mib` (`setup.py:3076-3082`) | `server.py:1465` (`engine_args` — the server **appends** `--layer-split` 1466-1467, `--split-skip-if-fits` 1469-1470, `--vram-elastic`/`--vram-segment-mib` 1473-1477, `--batch` from `parallel` 1485-1503, `--expert-profile-save`/`-every` 1521-1541, `--tail-role-token` from `effort_position` 1392-1414); the engine parses them, `src/program/generate.cpp:1228-1531` (unknown flag = error, `generate.cpp:1523-1528`) | written on every setup |
| `cwd` | str, path | `setup.py:4324` (the repo root) | `server.py:4001` (the engine child's cwd), `server.py:3977-3979` (vision block's relative paths), `server.py:3992` (relative `exe`), `server.py:1536` (relative `expert_profile_save`) | `.` (`server.py:3977`, `1536`) |
| `tokenizer` | str, path to the pack's tokenizer dir | `setup.py:4324` (`<data>/packs/<tag>/tokenizer`) | `server.py:3946-3947` (overrides the `--tokenizer` flag default), `3950-3960` (needs `vocab.json`/`merges.txt`/`token_type.json`; `chat_template.jinja` optional, `server.py:4014`) | `--tokenizer` default `<repo>/pack/full/tokenizer` (`server.py:3912`); **without a `vocab.json` a `strata`-engine start aborts** (`server.py:3950-3951`) |
| `model_name` | str | `setup.py:4325` (`<family-name>-<size>`, e.g. `qwen3.8-flash-next-iq3_xxs`; family names `setup.py:189,195,204,214`) | `server.py:4016` (the id `/v1/models` and answers carry) | `qwen3.8-flash-next` (`server.py:4016`) |
| `log` | str, path | `setup.py:4325` (`<repo>/strata-<tag>.log`) | `server.py:443,454` (engine stderr appended there, one handle per start), `server.py:3980` (vision encoder shares it) | none (stderr to `DEVNULL`) |
| `lib_dirs` | list of dirs | `setup.py:4326`, value from `setup.py:4121-4123`: the HIP libs, else `BUILD.json`'s `lib_dirs`/`cuda_dirs` (a locally compiled engine's toolkit folders, `setup.py:2327-2330`), else pip's `cuda_lib_dirs(toolkit)` (`setup.py:1329-1342`); `use_cuda12` rewrites it (`setup.py:3445`) | `server.py:1571-1574` (existing dirs are **prepended** to `LD_LIBRARY_PATH` — `PATH` on Windows — in the engine child's environment) | `[]` (the child inherits the server's environment as-is) |
| `port` | int | `setup.py:4326` (`a.port or 8080`, `setup.py:3681`) | **not the server** — `setup.py start()` passes it as `--port` (`setup.py:3093-3094`) and `write_run_script` bakes it into `run-<model>.sh/.bat` (`setup.py:3455-3456`); `settings_summary` prints it (`setup.py:3054`) | 8080 for a new install; the server's own default when launched without `--port` is 8095 (`server.py:3909`) |
| `backend` | `"hip"` | `setup.py:4330` (AMD only) | `server.py:1564` (`HIP_VISIBLE_DEVICES` instead of CUDA vars), `server.py:1586` (vision), `server.py:4074` (`svc.backend` → AMD sysfs telemetry), `server.py:4005` (desktop VRAM note) | absent (NVIDIA) |
| `env` | str→str map | `setup.py:4336` (`STRATA_HIPBLASLT_TUNING` = this card's hipBLASLt tuning table), `setup.py:4338` (`STRATA_RESIDENT_PIN: "0"`), both AMD-only; `SETUP_ENV` limits what setup writes (`setup.py:2696`) | `server.py:1569-1570` (merged into the engine child's environment) | `{}` |
| `gpu` | int or list (nvidia-smi ordinals) | `setup.py:4339-4344` (one chosen card, or several for a layer split); `start()` saves `--gpus` (`setup.py:3106,3134`) | `server.py:1382-1389` (`gpu_list`), `server.py:1566-1568` (child's `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES`), `server.py:1544-1557` (AMD via `hip_visible`), `server.py:4072-4073` (Monitor's card readings) | absent → no device restriction set (the engine sees every visible GPU) |
| `gpus_asked` | bool | `setup.py:4341` (a card was chosen at setup), `setup.py:3106,3134` (`--gpus` given at a start) | `setup.py:888` (`offer_together`: a one-card config on a two-card PC is offered the split **once**, and only if this is not set) | absent → the offer happens |
| `layer_split` | `"auto"` or `"K[,K2..]"` (first layer of each later GPU) | `setup.py:4344` (multi-GPU only); `start()` sets it with `--gpus` (`setup.py:3107,3135`) | `server.py:1428-1458` (`layer_split_value`, validated before the long start, `server.py:3984-3988`) → `--layer-split`; the engine parses it, `generate.cpp:1410,1549-1597` (`--serve` only, rising K from 2, one distinct GPU per K) | `"auto"` (placed by each card's free VRAM, `server.py:1434-1435`) |
| `host` | str | `setup.py:4348` (only when `--host` was given; the keep-option path `setup.py:3084-3089` also saves it) | `server.py:3940` (`--host` flag, else the config, else `127.0.0.1`) | `127.0.0.1` |
| `api_key` | str | `setup.py:4350` (only when `--api-key` was given; also kept by `start()`, `setup.py:3084-3089`) | `server.py:4031` (`--api-key` flag / `$STRATA_API_KEY` first, `server.py:3918`); enforced constant-time in `_authorized` (`server.py:2951-2959`); an explicitly **empty** key exits 2 (`server.py:4025-4030`) | `""` — no authentication |
| `cuda` | `12` | `setup.py:4328` (experimental CUDA 12 engine only) | `setup.py:626` (`config_toolkit`), `setup.py:2818`; the server never reads it | 13 (absent) |
| `draft_vocab` | str (`cjk`/`en`/`cyrillic`/`fr`) | `setup.py:4352` | `setup.py:3092` (a start refreshes `mtp/rt/draft_vocab.bin` from it), `setup.py:3030` (`--update`), `setup.py:4210` (a setup) — **the server never reads it** | `"cjk"` |
| `vision` | object: `exe`, `mmproj`, `model`, `gpu` (bool), `max_tokens` (1024 GPU / 300 CPU), `threads` (CPU only), + user's `cuda_device` | `setup.py:4369-4372` (`SETUP_VISION` keys `setup.py:2697`); the vision `vram-reserve-mib` default 700 rides in `args` (`setup.py:4296`, `VISION` `setup.py:224-225`) | `server.py:3974-3981` (starts `strata-vision --mmproj … --model … [--gpu] [--threads N] [--max-tokens N]`, `server.py:1268-1274`; relative paths resolved against `cwd`, `server.py:3977-3979`), `server.py:1582-1590` (its own `cuda_device`) | absent (text only) |
| `open_browser` | bool | `setup.py:4354` (only when `--browser` was given); kept by `start()` | `server.py:4115` (browser opens only with `--open` **and** this not false), `setup.py:3162` (setup passes `--open` unless false) | opens when the launcher passes `--open` (the run scripts do) |
| `parallel` | int 2..8 | `setup.py:4359` (only when `--parallel` was given; 1 also written) | `server.py:1491` (`parallel_args` → `--batch N`; the engine may run fewer slots, `server.py:490-494`) | 1 (one request at a time) |

### 1.2 The base `args` setup writes (NVIDIA, the common path)

`setup.py:4224-4227` always:

```
--pack <data>/packs/<tag> --native <data>/models/<tag>/<shard 1>.gguf [--ple-gguf <PLE shard>]
--expert-profile <repo>/data/expert-profile.bin [--expert-profile-coder.bin for coder] --expert-cache auto
--prefill auto --spec 4 --spec-min-p 0.5 --mtp <data>/mtp/rt --max-context <ctx>
```

- `--ple-gguf` is the shard carrying `per_layer_token_embd.weight` — shard 2 for qwen/coder (the shared
  28.8 GB n-gram table), shard 1 for Swift — and only for 2-shard models (`setup.py:4219-4224`).
- Conditionals: `--rope-scaling/--rope-scale` past the trained 262144 (`setup.py:4228-4229`); `--kv
  int8|q4_0|k8v4` when context > 8192 (`setup.py:4230-4231`); low-RAM `--resident-experts` /
  `--mmap-experts` / `--resident-budget-gib` (`setup.py:4232-4237,4293-4294`); `--ple-io ram` on a
  rotational disk when the RAM fits (`setup.py:4238-4246`); KV streaming `--kv-resident 32768` from 64K up
  when it fits (`setup.py:4270-4278`); images `--vision --vram-reserve-mib 700` (`setup.py:4295-4296`);
  an explicit `--vram-reserve-mib` (`setup.py:4301-4309`); the experimental speed projection
  `--control-vector-scaled <vec>:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir
  per-layer` (`setup.py:4320-4323`); calibration may replace `--pcie-frac`/`--spec-min-p`/`--pool-workers`
  values (`setup.py:4380`).

The engine accepts all of these (parse table `generate.cpp:1262-1523`): model files
(`--pack` 1263, `--native` 1336, `--ple-gguf` 1311, `--mtp` 1402, `--expert-profile` 1485,
`--expert-profile-save[-every]` 1486-1488), context (`--max-context` 1283, `--rope-scaling/-scale`
1284-1285, `--kv` 1318, `--kv-resident` 1319), memory tiers (`--expert-cache` 1365, `--vram-reserve-mib`
1378, `--vram-elastic/-segment-mib` 1391-1392, `--mmap-experts` 1490, `--resident-cpu-experts` 1492,
`--resident-experts` 1493, `--resident-budget-gib` 1503, `--ple-io` 1313), speed
(`--prefill` 1379, `--spec` 1395, `--spec-min-p` 1407, `--pcie-frac` 1404, `--pool-workers` 1346),
serving (`--serve` 1414, `--batch`/`--slots` 1396-1397, `--batch-groups` 1399, `--layer-split` 1410,
`--split-skip-if-fits` 1412, `--vision` 1415, `--prompt-cache*` 1416/1431-1432, `--conversation-cache-*`
1417-1430, `--tail-role-token` 1434, `--control-vector*` 1438-1468, `--remote-expert-opt` 1375).

### 1.3 User/server keys (setup never writes them; `carry_over` keeps them)

Read at `main()` start or in the helpers; a bad value refuses to start (a typo'd config "should not quietly
change sampling", `server.py:3845-3846`).

| Key | Type | Reader | Default when absent |
| --- | --- | --- | --- |
| `sampling` | object: `temperature` (≥0), `top_p` (0<x≤1), `top_k` (1..64), `min_p` (0..1), `presence_penalty`/`frequency_penalty` (≥0), `repetition_penalty` (>0), `penalty_last_n` (int ≥0), `seed` (int >0), `experimental_speed_projection` (bool) | `server.py:3840-3895` (`sampling_defaults_from_config`); a request's own fields always win (`server.py:3843-3844`); unknown keys named and ignored (3893-3894) | `{}` — greedy |
| `mcp_servers` / `mcpServers` | object `{name: entry}`; stdio entry `command` (required) + `args` + `env` + `cwd`; http entry `url` + `headers`; `disabled: true` skips | `server.py:3961` → `mcp.py:648-666` (`hub_from_config`; the `--mcp-config` file's entries override same-named config ones); a bad entry stops the start (3961 comment); entries checked `mcp.py:601-617` | none |
| `mcp` | object: `timeout_s`, `start_timeout_s` (>0), `max_result_chars`, `max_rounds` (int ≥1) | `mcp.py:630-645` | `mcp.py:37`: `timeout_s` 60, `start_timeout_s` 120, `max_result_chars` 20000, `max_rounds` 8 |
| `aliases` | list of names | `server.py:4020` (`set_aliases`; listed in `/v1/models`, `server.py:3058-3060`) | `[]` |
| `fit_max_tokens` | bool | `server.py:4018` | false (over-long `max_tokens` → 400, `server.py:3915-3917`) |
| `idle_unload_s` | number | `server.py:4050` → `start_idle_unload` (1898-1910), `unload` (1874-1896) | 0 (never; the model stays loaded) |
| `min_free_vram_mib` | int | `server.py:4051-4052` → `ensure_loaded` (1785-1793; waits 15 s for memory to be given back) | 0 (always load) |
| `before_load` | str or list | `server.py:4053` → `ensure_loaded` (1778-1784; a failing hook loads anyway) | none |
| `api_monitor` | bool | `server.py:4041` (the last 100 requests' prompts/answers in memory at `/api-monitor`) | false |
| `cors_origins` | origin(s) or list (`"*"` allowed) | `server.py:4032` (`origins_of`, 3778-3797) | `[]` (no CORS) |
| `trusted_origins` | origin(s) or list (no wildcard) | `server.py:4033` | `[]` |
| `allowed_hosts` | names, `".domain"` wildcards, or `"*"` | `server.py:4035` (+ `$STRATA_ALLOWED_HOSTS`); with a key the Host check is off anyway (`server.py:4038-4040`, `docs/DETAILS.md:611-619`) | loopback/IP/`localhost`/LAN-name only (anti-DNS-rebinding, `server.py:3717-3747`) |
| `lazy_load` | bool | `server.py:3971` (start unloaded; first request loads; **text only** — errors with a `vision` block, 3972-3973) | false |
| `engine_silence_s` | number ≥ 0 | `server.py:1417-1425`, checked before the long start (`server.py:3994-3996`); 0 = wait forever | 300 s (`ENGINE_SILENCE_S`, `server.py:88`) |
| `repeat_stop_tokens` | int ≥ 0 | `server.py:4058-4061` (a reply repeating one token this many times is ended, `#606`) | 256 (`REPEAT_STOP_TOKENS`, `server.py:76`); 0 = off |
| `anthropic_thinking` | `"model"` \| `"on_request"` | `server.py:4054-4057` | `"model"` (unasked Anthropic requests think) |
| `reasoning_budget_tokens` | int | `server.py:4063-4071` (a default thinking cap for every request; a request's own wins) | 0 (no cap) |
| `effort_position` | `"start"` \| `"end"` | `server.py:3998` → `effort_end_args` (1392-1414; needs an engine containing `--tail-role-token`, checked by scanning the binary) | `"start"` |
| `split_skip_if_fits` | bool | `server.py:1469` (multi-GPU only → `--split-skip-if-fits`) | false |
| `vram_elastic` | bool | `server.py:1473` (→ `--vram-elastic`; enables `POST /v1/vram`, `server.py:1815-1843`) | false |
| `vram_segment_mib` | int | `server.py:1475-1477` (→ `--vram-segment-mib`) | the engine's 512 |
| `expert_profile_save` | str path | `server.py:1521-1541` (→ `--expert-profile-save`; a learned profile of the same shape replaces `--expert-profile` on the next start) | none (nothing counted or written) |
| `expert_profile_save_every` | number | `server.py:1531-1533` (→ `--expert-profile-save-every`, minutes; 0 = at QUIT only) | 10 (the engine's) |
| `remote_expert_opt` | bool | `setup.py:875` (multi-GPU: keeps `--remote-expert-opt` out of `args`) | on for 2+ GPU configs (`setup.py:865-883`) |
| `hip_ordinal` | int | `server.py:1551-1556` (AMD Windows: the HIP ordinal setup recorded for the card) | absent (the config's `gpu`) |

The web page's Settings view can read/write a documented subset of these at runtime: exactly the 16 entries
of `runconfig.EDITABLE` (`serve/runconfig.py:19-39`: the four `sampling.*` defaults,
`reasoning_budget_tokens`, `fit_max_tokens`, `anthropic_thinking`, `effort_position`, `aliases`,
`idle_unload_s`, `lazy_load`, `engine_silence_s`, `api_monitor`, `open_browser`, `vram_reserve_mib` (as an
`args` flag)). `runconfig` never touches the network keys, `mcp*`, `before_load` or other `args`
(`runconfig.py:3-8`); a change is written to the config (`.bak` kept, `runconfig.py:164-172`) and used from
the next start (`runconfig.py:7-8`).


## 2. The environment: exe resolution, library paths, CWD, data dirs, `STRATA_EXECV`

### 2.1 How the engine is resolved and spawned

1. **The exe.** `main()` takes the config's `exe` verbatim; a relative one is made absolute against the
   config's `cwd` (a Windows `CreateProcess` quirk, `server.py:3990-3992`). The server never searches for it —
   `setup.py start()` fails fast when `cfg["exe"]` or any `.gguf` argument is missing
   (`setup.py:3071-3073`), so a config pointing at a gone engine never reaches the server.
2. **The child process.** One resident process: `subprocess.Popen([exe, "--serve", *engine_args(cfg)],
   cwd=cfg.get("cwd"), stdin=PIPE, stdout=PIPE, stderr=<log>, env=child_env(cfg), text, line-buffered)`
   (`server.py:454-455`). `--serve` is prepended by the server, not stored in the config
   (`server.py:392` docstring). The server blocks until the engine prints `READY <ctx> [stop]`
   (`server.py:457-468`); the engine's `INFO key=value` lines (context, kv, expert slots, `arena_mib`,
   `pool_workers`, `vram_elastic`, `batch_slots`, `engine=<version>` …, `generate.cpp:5696-5713`) feed the
   Monitor tab. If the engine exits first, the server raises with a hint and the log's last lines
   (`server.py:469-479`, `174-211`).
3. **The line protocol.** stdin: `GEN <max_new> [temperature=F top_p=F top_k=N min_p=F seed=N penalties…]
   <id,id,…>` and `GENI <max_new> <embeddings-file> <ids…>` (images; needs `--vision`, `generate.cpp:6239-6299`),
   plus `STOP`, `VRAM <reserve>` and `QUIT`; stdout: `T <id>` tokens, `PP` prompt-chunk progress, `DONE …`
   figures (`server.py:392-397`, `generate.cpp:5745-6241`). A dead engine is restarted on the next request
   (`StrataEngine.restart`, `server.py:577-606`; `ensure_loaded`, `server.py:1773-1811`).
4. **The vision encoder** (only with a `vision` block) is a second resident process: `strata-vision
   --mmproj … --model … [--gpu] [--threads N] [--max-tokens N]`, `ENC`/`DONE` lines, same log, its own
   environment (2.2) (`server.py:1262-1290,3974-3981`).

### 2.2 The engine child's environment (`child_env`, `server.py:1560-1575`)

Built per start (also used for `restart()` via the stored `spawn`, `server.py:423`):

1. `dict(os.environ)` — the server's own environment (the service unit's) is inherited whole, so any engine
   env var set on the service applies: `STRATA_RESIDENT_PIN` (generate.cpp:1496-1501),
   `STRATA_RESIDENT_HEADROOM_GIB` (1500-1501), `STRATA_ARENA_LOCK` (pinned.cu:391-392,417-418),
   `STRATA_ARENA_PIN_GIB` (pinned.cu:387), `STRATA_PREFILL_AUTO_MAX` (generate.cpp:1383-1385),
   `STRATA_WATCHDOG_S` (generate.cpp:5720-5721; default 60 s), `STRATA_RING_BYTES` (DETAILS.md:1048),
   `STRATA_SNAPSHOT_FULL_CAPTURE` (DETAILS.md:682), `CUDA_MODULE_LOADING` (set to `EAGER` by the engine itself
   when unset, `generate.cpp:1245-1251`).
2. **Device selection.** NVIDIA: `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=<gpu list>`
   (`server.py:1566-1568`, comment citing issue #51 — CUDA's default "fastest first" numbering differs from
   nvidia-smi's). AMD: `HIP_VISIBLE_DEVICES` from `hip_visible(cfg)` (the config's `gpu`, or the recorded
   `hip_ordinal` on one-card Windows, `server.py:1544-1557`). No `gpu` key: neither is set.
3. **The config's `env` map** is merged (stringified), `server.py:1569-1570` — e.g. the AMD GEMM tuning table.
4. **Library paths.** Every existing dir in `lib_dirs` is **prepended** to `LD_LIBRARY_PATH` (POSIX) or
   `PATH` (Windows), `server.py:1571-1574`. Where those dirs come from (`setup.py:4121-4123`, `3426-3429`):
   - a **locally compiled engine**: `BUILD.json`'s `cuda_dirs` — the CUDA toolkit's own library folders
     (written at `setup.py:2327-2330`; the Docker image writes the toolkit's `lib64`, `Dockerfile:93-99`) —
     so a nix-built engine needs the store-path toolkit dirs (they survive because nix keeps them);
   - a **prebuilt engine**: pip's `nvidia` wheels, found by `cuda_lib_dirs(toolkit)` scanning `site-packages`
     of the **server's own interpreter** for `libcublas.so.13*` / the cu12 pair (`setup.py:1329-1342`) —
     the `nvidia-cublas==13.0.2.14` + `nvidia-cuda-runtime==13.0.96` pair setup installed,
     `setup.py:105,2124-2129` (~0.4 GB, ticket #4 §1.5).
   The vision encoder gets the same environment unless `vision.cuda_device` overrides the device
   selection (`vision_env`, `server.py:1578-1591`).

### 2.3 CWD assumptions

- **The server** is checkout-relative, not CWD-relative: `ROOT = Path(__file__).resolve().parents[1]`
  (`server.py:51`); it puts `<ROOT>/tools` and `<ROOT>` on `sys.path` (`server.py:52-53`) to import
  `strata_tokenizer` and `serve.*`, serves the web app from `<ROOT>/serve/web` (`server.py:2966,2983,3024`)
  and the fallback chat template from `<ROOT>/serve/chat_template.jinja` (`server.py:4015`). It must be run
  **from a complete checkout** (the `serve/` and `tools/` packages); the process CWD itself is irrelevant.
- **The engine** runs with `cwd = config's "cwd"` (`server.py:4001`), which setup writes as the repo root
  (`setup.py:4324`). All of setup's paths in `args` are absolute, so CWD only matters for user-added relative
  values (a relative `expert_profile_save` is resolved against it, `server.py:1535-1538`).
- **`--tokenizer`** defaults to `<ROOT>/pack/full/tokenizer` (`server.py:3912`) but the config's `tokenizer`
  always overrides it (`server.py:3946-3947`); without a `vocab.json` a `strata`-engine start aborts
  (`server.py:3950-3951`).

### 2.4 The data-dir layout (what the paths in a config point at)

Default `<data>` = `Strata-data/` **next to** the repo (`setup.py:2635`), `/data` in Docker
(`docker-entrypoint.sh:9`); the config's `args` carry the absolute paths (ticket #4 §1.6-1.7):

```
<data>/
  models/<tag>/<name>-0000<i>-of-0000N.gguf[.part/.done]   # the GGUF shards (+ .done completion marks)
  models/mmproj-*.gguf                                     # the image encoder (only with images)
  packs/<tag>/
    native_experts.txt dense.bin conversions.json         # the prepared pack (finish marker: native_experts.txt)
    tokenizer/{vocab.json,merges.txt,token_type.json,chat_template.jinja}
    experts.bin                                            # only in low-RAM mode
  mtp/
    tensors/<tensor>.bin  mtp-manifest.json               # the range-fetched draft tensors
    mtp-q2_0.gguf
    rt/{experts.bin,dense.bin,dense.txt,draft_vocab.bin}  # the engine's MTP runtime files
```

What each path means to the engine: `--pack <data>/packs/<tag>` (the dense side + tokenizer + expert blobs),
`--native <shard 1>` (experts read natively from the GGUF), `--ple-gguf <shard with the PLE table>` (the
28.8 GB n-gram table, read unbuffered past the OS cache by default, `--ple-io direct`
`docs/DETAILS.md:1030-1031`), `--mtp <data>/mtp/rt` (the speculative draft layer),
`--expert-profile <repo>/data/expert-profile[-coder].bin` (a repo file, `setup.py:4225`). The server's
`model_path` is parsed from `--native`/`--pack` for its UI only (`server.py:424-425`, default `pack/full`).

### 2.5 `STRATA_EXECV` semantics

`setup.py start()` finishes with ( `setup.py:3190-3194` ):

```python
if not WIN and os.environ.get("STRATA_EXECV"):
    # Replace this process instead of spawning a child. The Docker image sets STRATA_EXECV=1,
    # so there the server is PID 1 and docker stop's SIGTERM reaches the process that can
    # answer the engine with QUIT. Normal Linux starts keep spawning the server as a child.
    os.execv(cmd[0], cmd)
return subprocess.call(cmd)
```

- **Set (any non-empty value, POSIX only):** the setup.py process is *replaced* by the server
  (`os.execv`), so the server inherits the terminal/pipe fds and is the (container's) PID 1; the image sets
  `ENV … STRATA_EXECV=1` for exactly this (`Dockerfile:43-46`). The server installs a `SIGTERM →
  KeyboardInterrupt` handler so `docker stop` reaches the engine with a `QUIT` (`server.py:4118-4126`).
- **Unset (the normal Linux case, and what a systemd unit wants):** setup.py spawns the server as a child and
  `subprocess.call`s it (foreground, its own process to signal). A service that runs `serve/server.py`
  directly does not need `STRATA_EXECV` at all — the exec path is only how *setup* launches the server.

### 2.6 Other server-side env vars

| Var | Effect | Where |
| --- | --- | --- |
| `STRATA_API_KEY` | an alternative to `--api-key`/config `api_key` (flag and config win) | `server.py:3918,4031` |
| `STRATA_ALLOWED_HOSTS` | comma-separated extra Host names, as config `allowed_hosts` | `server.py:4035` |
| `STRATA_REQUEST_LINES=1` | one stdout line per finished request (from the engine's log summary), so a supervisor that only sees the server's stdout has per-request numbers | `server.py:451-453,214-234` |
| `CUDA_MODULE_LOADING` | set to `EAGER` by the engine itself when unset (~30 MB VRAM, fixes a mid-prompt OOM) | `generate.cpp:1242-1251` |
| `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER` | written into the child env from the config's `gpu` (2.2) | `server.py:1564-1568` |

## 3. Runtime limits

### 3.1 memlock / ulimit

The model's 24,576 experts live **pinned in RAM** ("RAM: all 24,576 experts, pinned", `docs/DETAILS.md:1029`);
the pinned arena is what the GPU reads at full speed. The pinning has two layers, and both touch the memlock
ulimit:

1. **`cudaHostRegister`** (pinned host memory) for the arena — tried first, in slices when the driver refuses
   the whole region (`src/core/pinned.cu:380-398`).
2. **`mlock`** over what the register did not cover — `lock_resident` (`src/platform/memory.cpp:112-120`),
   called from `pinned.cu:394,419` and the file-backed resident arena (`src/core/expert_source.cpp:1584`).
   `mlock` failure is **not fatal**: `r.note = "mlock failed (raise ulimit -l)"` (`memory.cpp:115`) and the
   engine continues with the pages merely pre-touched (the A/B arm `STRATA_ARENA_LOCK=0` skips the lock
   entirely, `pinned.cu:417-424`).
3. The **28.8 GB n-gram PLE table** with `--ple-io ram` is `mlock`ed the same way; on failure it prints
   `"strata: PLE table mlock failed (…; raise \`ulimit -l\`): touching its pages instead"` and pre-touches
   (`src/kernels/ngram.cpp:262-276`).

Consequence for unattended runs: with the default 8 MB memlock limit the pin mostly fails and the pages are
vulnerable to reclamation under memory pressure (slow prompts); the documented setups therefore raise it:

- Docker: `docker run … --ulimit memlock=-1` (`Dockerfile:20`, `docs/INSTALL.md:101`,
  `bench/results/2026-10-03-community-2x-mi50/README.md:68`, `docs/AMD_HIP.md:379`);
- the one native-systemd precedent in-tree: "Native runs used a systemd service with
  `LimitMEMLOCK=infinity`" (`bench/results/2026-09-29-rtx3090-epyc-milan/README.md:21`);
- Windows needs `SeLockMemoryPrivilege` for its working-set lock instead (`docs/DETAILS.md:388-389`,
  `include/strata/platform/memory.hpp:6`).

In a systemd unit this is `LimitMEMLOCK=infinity` (or `LimitMEMLOCK=-1`) on the service; in NixOS also
`RLIMIT_MEMLOCK` via the same directive. It is a **performance-correctness** setting (page reclamation
during prompts), not a hard requirement — the engine degrades gracefully.

### 3.2 RAM behavior (32-62 GB loaded)

- Per-model RAM the setup budget is the `MODELS` table's `ram_gb`: IQ1_M (Coder) 32, Q2_0/IQ2_XS/UD-Q4_K_XL/
  UD-IQ4_XS 48, IQ3_XXS 60, IQ3_S 62 (`setup.py:122-152`) — hence "Strata loads 32-62 GB into RAM"
  (`Dockerfile:30-32`, `docs/INSTALL.md:112`). The expert *arena* is smaller (`arena_gb`: 23.4-77.0,
  `setup.py:125-151`); the rest is the dense side, the KV/PLE decisions and headroom.
- At start the server reads the engine's `INFO arena_mib` and warns when **less than 6 GB** of the system's RAM
  remains besides it: `"WARNING: RAM is tight … Linux may stop the engine in the middle of an answer"`
  (`server.py:3595-3612`). The OOM killer ending the engine mid-answer is the known failure mode (issue #27,
  cited at `server.py:423` and `server.py:1774`); the server then restarts the engine on the next
  request.
- **Low-RAM mode** (≤ ~40-50 GB PCs): the experts come from the pack's `experts.bin` — `--resident-experts`
  (copy what the GPU does not hold into RAM, pinned) or `--mmap-experts` (read through the OS file cache,
  nothing preloaded, `setup.py:4236-4237`, `docs/INSTALL.md:116-119`); a container with a memory cap must ask
  for it (`-e LOW_RAM=on`) because setup reads `/proc/meminfo` = the **host's** total
  (`docker-entrypoint.sh:36-39`).
- Loading is slow and system-wide visible: 30-90 s typical (`docs/DETAILS.md:345`), and the server's narrator
  prints the "PC can be slow or stop responding for 1-3 minutes" warning while it happens
  (`server.py:258-307`). Under systemd this means the unit's `TimeoutStartSec` must be generous (the Docker
  image allows 600 s before its first health probe, `Dockerfile:112`).

### 3.3 Port + API key defaults

- **Port.** Setup installs default to **8080** (`setup.py:3557,3681`); the config stores it (`port`,
  `setup.py:4326`) and the launchers pass it (`setup.py:3094,3455`). `serve/server.py`'s own argparse default
  is **8095** (`server.py:3909`) — a hand-launched server without `--port` lands there. Before loading the
  model, `main()` bind-tests the port and exits with a "port already in use" error if taken
  (`server.py:3941-3945`). On non-Windows `SO_REUSEADDR` is set (a second start then fails loudly only in the
  documented Windows case, `server.py:3585-3588`).
- **API key.** Default **empty = no authentication** (`server.py:3918,4031`). When set (flag >
  `$STRATA_API_KEY` > config), every `/v1/*` request needs `Authorization: Bearer <key>` or `x-api-key:
  <key>` (constant-time compare, `server.py:2951-2959`), and so do `/metrics`, `/status`, `/v1/models`,
  `/props`, `/slots`, `/mcp`, `/settings`, `/config` (`server.py:2996-3074`); the web page and `/health` are
  open (below). An explicitly **empty** key (`--api-key ""` or `STRATA_API_KEY=""`) **exits 2** rather than
  silently disabling auth (`server.py:4025-4030`). Binding beyond loopback without a key prints a WARNING
  naming the fix (`server.py:4107-4109`); the documented rule is "never expose beyond `127.0.0.1` without an
  API key" (`AGENTS.md`, `docs/INSTALL.md:120-122`). With a key set, the Host-name (DNS-rebinding) check is
  turned off entirely (`server.py:4038-4040`, `docs/DETAILS.md:611-619`).
- `/health` (also `/api/health`) needs **no key** and answers before the model loads:
  `{"status": "ok", "max_context": …, "model": …, "images": …, "api_key": <bool>, "loaded": <bool>,
  "service": "strata"}` (`server.py:3030-3033`; `max_context` is 0 while unloaded/lazy). The Docker
  `HEALTHCHECK` polls exactly this: `--interval=30s --timeout=5s --start-period=600s --retries=3` on
  `curl -fs http://127.0.0.1:${PORT:-8080}/health` (`Dockerfile:108-113`), with the comment "/health is
  answered before the API key gate, so it works with or without one".
- `GET /v1/status` **does** require the key and reports what the server is: service/model/`loaded`/
  `auto_load` (true when the engine can self-start, i.e. it can restart after death/unload), engine version,
  start time + uptime, context (`cache_max_tokens`, `context.native`/`max_positions`), concurrency
  (`serving`/`requested` = the batch slots, 1 when serial), dialects (`/v1/chat/completions`,
  `/v1/messages`, `/v1/responses`), vision state, request totals + last-request timings (llama.cpp's names),
  VRAM figures, and a machine block (GPU name/used/total/util/temp/power, RAM used/total) from telemetry
  (`server.py:2075-2115`). It is the endpoint the Docker docs point to ("`GET /v1/status` says what it is
  running", `docs/INSTALL.md:122`).
- `GET /metrics` (key-gated) is the superset: engine facts, live request state, last 12 requests,
  conversation-cache card, hardware with 60 s of history (`server.py:2024-2073`,
  `docs/DETAILS.md:481`).

### 3.4 Watchdogs, sharing the GPU, parallelism

- **Engine watchdog:** a request with no engine output for `STRATA_WATCHDOG_S` (default **60 s**) aborts the
  engine with a stall report so the server restarts it (`generate.cpp:5715-5741`, issue #29).
- **Server silence guard:** a request whose engine prints nothing for `engine_silence_s` (default **300 s**,
  0 = forever) is ended and the engine restarted (`server.py:85-94,1417-1425`, issue #481).
- **GPU sharing (all off by default, `docs/DETAILS.md:424-454`):** `idle_unload_s` (auto-unload after idle),
  `min_free_vram_mib` (refuse to load into a busy GPU: 503), `before_load` (run a command first),
  `POST /unload` / `POST /load` (409 while busy; `/health` says `"loaded"` either way), `vram_elastic`
  (`POST /v1/vram` shrinks/grows the expert cache between requests). Unloading ends the engine (and encoder)
  processes; reloading reuses the OS file cache ("loading again takes seconds", `docs/DETAILS.md:440-441`).
- **Parallelism:** `"parallel": N` (2..8, `server.py:1482-1503`) maps to `--batch N`; the engine reports how
  many slots it actually runs (`INFO batch_slots`) and the server reconciles with a note
  (`server.py:488-494`). `PARALLEL_MAX = 8` is the engine's batch-window limit.
- **Requests longer than the context are rejected 400, never truncated** (`server.py:14-15`,
  `docs/DETAILS.md:576-579`); `fit_max_tokens` clamps `max_tokens` instead.


## 4. `serve/telemetry.py`: what, where, when, how to disable

**What it collects** (sampled by one background daemon thread, `telemetry.py:296,349-358`):

- **GPU** — NVIDIA: the driver's own NVML library loaded through `ctypes` (no pip package), per card:
  utilization, VRAM used/total, temperature, power, power limit, PCIe generation/width and RX/TX throughput
  (`_Nvml.read`, `telemetry.py:78-104`). AMD: the amdgpu driver's **sysfs** files (`gpu_busy_percent`,
  `mem_info_vram_*`, `hwmon/temp1_input`, `power1_*`), with the card found through the KFD topology the same
  numbering `HIP_VISIBLE_DEVICES` uses (`amd_device_dir`/`_Amd`, `telemetry.py:111-182`). Several cards:
  totals/means/hottest, plus a per-card list (`Telemetry.sample`, `telemetry.py:314-330`).
- **CPU / RAM** — `psutil` when installed (setup installs it, `requirements.txt`), else the OS fallbacks
  (`/proc/stat`, `/proc/meminfo` on Linux; `GlobalMemoryStatusEx`/`GetSystemTimes` on Windows)
  (`telemetry.py:7-8,217-262`).
- **Disk** — read/write rate via `psutil.disk_io_counters`; absent without psutil (`telemetry.py:298-310`).
- **The server's own speed** — `tok_s`, `tok_s_mean`, `prefill_tok_s_mean` injected as `extra`
  (`server.py:1951-1955`).
- **Static facts** — GPU name/count, CPU name, core/thread counts, whether psutil is present
  (`telemetry.py:287-294`).

**Where it sends it: nowhere.** The module's own header says it is "hardware readings for the web app's
Monitor tab" (`telemetry.py:1-9`); the file imports only `collections, ctypes, os, platform, sys,
threading, time` — no sockets, no `urllib`, no `requests`, no URLs (verified: zero matches). Data stays in
in-memory deques of 60 (`HISTORY = 60`, `telemetry.py:21,276`) and leaves the process only as the JSON of
`GET /metrics` (`server.py:2067-2073`) and `GET /v1/status`'s machine block (`server.py:2082-2115`). Nothing
is persisted; nothing is uploaded; the web page is the only consumer. (The server's *only* outbound network
calls in the whole `serve/` package are: fetching an image the client sent as an http(s) URL,
`server.py:1312-1315`; and talking to the user's own MCP servers, `serve/mcp.py`.)

**When:** the sampler thread starts inside `serve()` (`server.py:3772`), immediately before the HTTP server
  binds the port (`server.py:3773`) — i.e. **after the engine is READY** in
the normal start (it is `main()`'s last step, `server.py:4093`), or at process start in `lazy_load` mode.
It samples every second for the process's whole life; every read is exception-guarded and
"nothing here can stop the server" (`telemetry.py:9,342-346`).

**How to disable it:** there is **no flag, config key or env var that turns it off** — `serve()` calls
`start_telemetry()` unconditionally (`server.py:3772`), and nothing in the config schema gates it
(`runconfig.EDITABLE` has no entry, `serve/runconfig.py:19-39`). The only levers are indirect: it does no
I/O, so it is safe to leave; its readings only reach a client that calls the endpoints; on a machine without
the GPU libs/psutil the samples degrade to `None` instead of failing (`telemetry.py:9,314-346`). If a
deployment truly needs it gone, the change is a one-line patch (`server.py:3772`), not a setting.

## 5. Minimal invocation spec (no `setup.py` involvement)

Everything a running system needs, assuming the install artifacts already exist (engine compiled to
`engine/strata`, model files under `<data>`, as tickets #4/#5 documented setup produces):

**1. The config file** `<repo>/strata-<model>.json`. The minimal core a working install has (keys and their
required-ness per §1):

```json
{
 "exe": "/home/user/Strata/engine/strata",
 "args": [
  "--pack", "/home/user/Strata-data/packs/iq3_xxs",
  "--native", "/home/user/Strata-data/models/iq3_xxs/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
  "--ple-gguf", "/home/user/Strata-data/models/iq3_xxs/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
  "--expert-profile", "/home/user/Strata/data/expert-profile.bin",
  "--expert-cache", "auto",
  "--prefill", "auto",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--mtp", "/home/user/Strata-data/mtp/rt",
  "--max-context", "65536",
  "--kv", "int8"
 ],
 "cwd": "/home/user/Strata",
 "tokenizer": "/home/user/Strata-data/packs/iq3_xxs/tokenizer",
 "model_name": "qwen3.8-flash-next-iq3_xxs",
 "log": "/home/user/Strata/strata-iq3_xxs.log",
 "lib_dirs": ["/nix/store/<…>/cuda-merged-13.x/lib" ],
 "port": 8080
}
```

Required: `exe`, `args` (at least `--pack`, `--native`, `--mtp`, `--max-context`; `--expert-profile` is
mandatory for a layer split, `generate.cpp:1629-1631`, and always written by setup), `tokenizer` (or a
`vocab.json` at the default), `cwd`, `log` (optional but strongly advised), `lib_dirs` (mandatory for a
prebuilt engine whose CUDA libs live in pip wheels; empty is fine when the engine's libraries are already on
the system path — a nix-built engine's `BUILD.json cuda_dirs` are store paths the loader can reach, but
`child_env` only prepends `lib_dirs`, so a service relying on the bare environment must keep the `LD_*`
paths the build recorded). Optional but meaningful for unattended service: `gpu` (else **every** visible GPU
is offered to the engine), `host`/`api_key` (any non-loopback binding wants a key), `port`, `sampling`,
`idle_unload_s`, `min_free_vram_mib`, `engine_silence_s`. Everything in `args` must be a flag the engine
actually parses — the engine rejects unknown flags with exit 2 (`generate.cpp:1523-1528`) and the server
reports that in its log.

**2. The command.** Run with any Python 3.10+ that can import `jinja2` and `regex` (the server's only
third-party imports in the runtime chain: `serve/frontend.py:11`, `tools/strata_tokenizer.py:22`; `psutil`
and `Pillow` are optional — telemetry fallback, `telemetry.py:217-262`, and exotic image formats,
`server.py:1329-1331` — the stock `.venv` created by setup has them all, `requirements.txt`):

```sh
exec /home/user/Strata/.venv/bin/python /home/user/Strata/serve/server.py \
     --engine strata --config /home/user/Strata/strata-iq3_xxs.json --port 8080
```

- `--port` should be given (the config's `port` is not read by the server — it is only what setup and the run
  scripts pass, §1.1); without it the server listens on 8095 (`server.py:3909`).
- Do **not** pass `--host` for a local service: absent `--host`, the config's `host`, else `127.0.0.1`
  (`server.py:3940`). For LAN/internet: `--host 0.0.0.0` (or in the config) **and** a key (`--api-key` /
  `api_key` / `STRATA_API_KEY`).
- `--open` (browser) is what the run scripts add; a service leaves it out (`open_browser: false` in the
  config also suppresses it, `server.py:4115`).
- The server must be able to find its checkout: it resolves `ROOT` from its own file path
  (`server.py:51`) and imports `serve.*`/`tools/strata_tokenizer` and serves `serve/web/` from there, so the
  command line names the checkout's `serve/server.py` — no CWD requirement (the run scripts' `cd` is a
  convenience, `setup.py:3463`).
- The server sends the engine `cwd = config.cwd` and builds its environment per §2.2 — **no other
  environment is needed**. Optionally `STRATA_REQUEST_LINES=1` for per-request stdout lines for the
  supervisor (`server.py:214-234`).
- Lifecycle: SIGTERM/SIGINT are handled — the server shuts down the engine with `QUIT` (a second SIGINT kills
  it at once), `server.py:4118-4140`. `STRATA_EXECV` is **not** needed (it is the setup-side exec, §2.5).

**3. The NixOS systemd unit** (contract elements in bold; the rest is standard service hygiene):

```ini
[Unit]
Description=Strata - Qwen3.8-Flash-Next model server
After=network.target

[Service]
Type=simple
# the checkout, the config and the data dir per the install (paths from the example above)
ExecStart=/home/user/Strata/.venv/bin/python /home/user/Strata/serve/server.py \
          --engine strata --config /home/user/Strata/strata-iq3_xxs.json --port 8080
# the memlock contract (§3.1): pin the expert arena and the PLE table; without it the
# engine falls back to pre-touched, reclaimable pages (slower prompts, OOM-ish stalls)
LimitMEMLOCK=infinity
# loading takes 1-3 minutes with the PC slowed; give the start (and a Restart) room
TimeoutStartSec=900
Restart=on-failure
RestartSec=5
# optional: per-request numbers on stdout for the journal
Environment=STRATA_REQUEST_LINES=1
# the model needs 32-62 GB of RAM depending on size (§3.2); a cgroup cap must stay above it
# or the low-RAM mode (args: --mmap-experts / --resident-experts, §3.2) must be used instead

[Install]
WantedBy=multi-user.target
```

Readiness check for the unit (mirrors the Docker `HEALTHCHECK`, `Dockerfile:112`): poll
`http://127.0.0.1:8080/health` (no key needed, answers `loaded:false` while loading, `max_context>0` and
`loaded:true` when serving — though note `loaded` reflects the engine state, `server.py:3030-3033`).

## 6. The unattended precedents already in the tree

- **Docker (the reference unattended path).** `docker-entrypoint.sh` runs `setup.py --setup --yes --no-start`
  once (every question answered by its recommended default; no start), then a plain `setup.py --port 8080`
  that `start()` turns — via `STRATA_EXECV=1` (`Dockerfile:46`) — into the server as PID 1
  (`docker-entrypoint.sh:40-65`, ticket #4 §4). The documented run adds `--ulimit memlock=-1`
  (`docs/INSTALL.md:101`) and warns that a memory-capped container needs `-e LOW_RAM=on` and
  `-e API_KEY=<secret>` before exposing the port (`docs/INSTALL.md:116-122`).
- **systemd (one measurement).** `bench/results/2026-09-29-rtx3090-epyc-milan/README.md:21`: native Linux,
  "a systemd service with `LimitMEMLOCK=infinity`" — the same ulimit contract, no other special unit
  settings recorded there.
- **Windows Task Scheduler (a caution).** Autostart throttling makes the ~40 GB expert load 24x slower
  (13-14 min) unless the task is Normal priority **and** "Run with highest privileges" (which restores
  `SeLockMemoryPrivilege`) — `docs/DETAILS.md:372-392`. The Linux analogue of that privilege is the memlock
  limit itself.

## 7. Method and source notes

- All file:line references are against this checkout at `main` @ `f9d372a` (v0.1.39). Line numbers in
  `src/` point at the CUDA source tree (`src/program/generate.cpp`); the SYCL mirror
  (`sycl/src/program/generate.cpp`) carries the same flags at different lines and is out of scope.
- Ticket #4's `start()` command line (`setup.py --engine strata --config <cfg> --port 8080`) was re-verified
  against `setup.py:3093-3094`; its config-key table against `setup.py:2694-2695,4324-4374,3066-3195`; its
  `STRATA_EXECV` description against `setup.py:3190-3194` and `Dockerfile:43-46`. Ticket #5's engine layout
  (`engine/strata`, `BUILD.json`, pip CUDA libs) was re-verified against `setup.py:226-227,1329-1342,4121-4123,
  3426-3429`.
- What is *not* in the contract: model downloads (ticket #4), the engine build (ticket #5), the AMD/HIP
  details beyond the keys and env vars named here (`docs/AMD_HIP.md`), and the Responses-API request format
  (`docs/DETAILS.md:703-761`, `serve/responses.py`).
- Nothing in this research was executed: no server, no engine, no downloads; every claim above is a
  file:line read of this checkout.
