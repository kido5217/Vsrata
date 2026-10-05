# Model fixtures: the minimal test-model tree (ticket #18)

Built and verified 2026-10-05. Repo `kido5217/Vsrata` (a copy of `Niko1221/Strata` v0.1.39). Host:
NixOS, RTX 5090 (32 GB), Ryzen 7 5800X3D (Zen 3 — no AVX-512), 126 GiB RAM.

This records the fixture tree built for map #15 ("Model fixtures — minimal test model files in a
gitignored tree") and the verification that closes ticket #18. The download plan is locked in ticket
#17; the input inventory that plan came from is `docs/research/fixtures-manifest.md` (ticket #16,
branch `research/fixtures-manifest`).

## TL;DR

- **~102 GB on disk, every byte inside the repo's existing gitignored trees** (`/models/`, `/mtp/`,
  `pack/`, `/strata-*.json`) — nothing to commit, and every tool finds its inputs at its default path.
- **Downloaded:** the two Q2_0 shards (37.6 + 28.8 GB) and the MTP head's 31 tensors (3.9 GB); the
  canonical pack and the native pack are generated locally from them.
- **Verified end-to-end:** pack bit-exact (1079 tensors / 4,947,698,560 elements, 0 bad; 2304 expert
  checks, 0 bad); oracle on the real shard (1223/1223 tensors); `batch_test` exit 0 with 4/4 slots
  solo-identical; `early_close_test` exit 0; `conversation_cache_http_smoke` PASS.
- **A *native* pack is required on this CPU.** `strata_pack.py build` alone writes no
  `native_experts.txt`, so the engine takes the canonical layout and hard-requires AVX-512 — a Zen 3
  host exits. The working chain is `strata_pack build` → `pack_index` → `iq_pack` (§3), and the
  native pack then needs `--spec >= 2` and `--prefill` (§5).
- **The GPU must be free.** The engine needs a real VRAM budget; the host's own `frinfer` server
  (30.8 GB) starves it (§5.7).

## §1 What was downloaded

| File | Size (bytes) | Source | Pin |
|---|---:|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 37,623,740,192 | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (`setup.py:66`) |
| `…-00002-of-00002.gguf` | 28,800,138,432 | same | same |
| 31 `mtp.*` tensors (range-fetched, not the 360 GB checkpoint) | ~3.9 GB | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` (`tools/mtp_fetch.py:32`) |
| llama.cpp source (code, not a model) | — | `ggml-org/llama.cpp` | `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`CMakeLists.txt:996`) |

Both repos are public and non-gated; either of the user's HF tokens works (and anonymous works too).
The two shards are byte-identical to the pinned sizes above. Shard 2 holds the PLE n-gram table and
is shared by all Qwen3.8 variants, so one download covers every variant's PLE needs; the MTP head is
absent from the GGUF release and can only be range-fetched from the BF16 checkpoint.

Not downloaded (optional variants, all out of the minimal set): qwen IQ2_XS/IQ3_XXS/IQ3_S shard 1s
(39–55 GB each), swift (67–76 GB), coder IQ1_M (58 GB), unsloth UD-Q4_K_XL (111 GB) / UD-IQ4_XS
(94 GB), mmproj (0.9 GB).

## §2 Where it lives

```
models/qwen-Q2_0/                      # gitignored (.gitignore:32  /models/)
  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf   37.6 GB
  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf   28.8 GB
mtp/                                   # gitignored (.gitignore:34  /mtp/)
  bf16/                                4.6 GB  (31 tensors, 3.9 GB, + inventory/manifest)
  bf16/mtp-q2_0.gguf                   0.889 GB (the packed MTP head)
  rt/                                  0.82 GB (dense.bin 116 MB, experts.bin 708 MB, draft_vocab.bin 425 KB)
third_party/llama.cpp/                 # gguf-py for ref/load.py + the two iq_parity ctests
pack/                                  # gitignored (.gitignore:13  pack/)
  full/                                35 GB on disk  (canonical Q2_0 pack: experts 31.6 GiB, dense 5.4 GiB, embd 0.44 GiB, index.txt, tokenizer/)
  q2_0/                                the native pack (hard-linked dense.bin, index.txt, native_experts.txt, tokenizer/)
strata-q2_0.json                       # the dev server config    (.gitignore:37  /strata-*.json)
```

`pack/q2_0/dense.bin` is a hard link to `pack/full/dense.bin` (`link count 2`), so the 5.8 GB of
dense weights is stored once.

## §3 The build chain (local generation, reproducible)

```bash
# MTP head: verify the range-fetched tensors, pack them, build the runtime
python tools/mtp_fetch.py verify --out mtp/bf16
python tools/mtp_pack.py --src mtp/bf16 --experts q2_0 --out mtp/bf16/mtp-q2_0.gguf
python tools/mtp_rt.py  --gguf mtp/bf16/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin     # see §5.4

# Canonical pack from shard 1 (derives shard 2 beside it), then its flat index
python tools/strata_pack.py build --gguf models/qwen-Q2_0/…-00001-of-00002.gguf --out pack/full
python tools/strata_tokenizer.py --gguf models/qwen-Q2_0/…-00001-of-00002.gguf --out pack/full --check   # appends tokenizer/
python tools/pack_index.py --pack pack/full        # writes pack/full/index.txt

# The native pack the engine actually loads (experts read from the GGUF via --native)
export PYTHONPATH="$PWD/third_party/llama.cpp/gguf-py"
python tools/iq_pack.py --gguf models/qwen-Q2_0/…-00001-of-00002.gguf --base pack/full --out pack/q2_0
```

`pack/q2_0/native_experts.txt` is the pack's completion marker (v3: 512 experts, type 42 = Q2_0,
absolute offsets into shard 1); a pack without it is not finished. `strata_pack.py verify` re-reads
the canonical pack against the source shards; `iq_pack.py` writes the native pack's index and shares
the base's dense tensors.

## §4 Verification

| Check | Command | Result |
|---|---|---|
| **A** pack bit-exact | `strata_pack.py verify --gguf <shard1> --out pack/full` | 1079 tensors, 4,947,698,560 elements, **0 bad**; 2304 layer/expert/role checks (48 layers, every 32nd expert), **0 bad** |
| **B** oracle, real shard | `canonical_xcheck.py --gguf <shard1> --tensors 4` | mapped types cover **1223 of 1223** tensors — PASS |
| **C1** batched == solo | `batch_test.py --batch 4 --n 4 --max-new 64 --extra "--pcie-frac 0 --adapt-every 1000000"` | **exit 0, 4/4 slots IDENTICAL** (solo 161.8–306.4 tok/s; batch aggregate 102.0 tok/s) |
| **C2** early close | `early_close_test.py http://127.0.0.1:PORT` | exit 0 (solo OK, batch OK) |
| **C3** HTTP conversation cache | `conversation_cache_http_smoke.py --url … --output … --run` | **PASS**: HTTP reuse, slot eviction, cancellation/recovery |
| bonus | `ctest -R iq_parity` | `iq_parity_fixtures` PASS; `iq_parity` FAILS on IQ2_S/IQ3_S (§7) |

The server came up in ~10 s once the model pages were in the OS page cache; the engine decoded at
140–306 tok/s solo and ~102–121 tok/s aggregate for 4 slots on the RTX 5090.

## §5 Prerequisites and gotchas

1. **The native pack is mandatory on a non-AVX-512 CPU.** `expert_layout_load` reads
   `<pack>/native_experts.txt`; absent, it uses the canonical layout and `cpu_require_expert_support()`
   hard-exits on CPUs without AVX-512-VNNI/VBMI. With a native pack the same engine runs on AVX-2
   (`generate.cpp:2054`, `expert_layout.cpp:238`). This host (Zen 3) exits without it.
2. **A native pack requires `--spec T` (T ≥ 2) and `--prefill CHUNK`** (`generate.cpp:2164-2169`);
   the engine then reads the experts from the `--native` GGUF itself (no pack `experts.bin`).
3. **`ref/load.py` imports the vendored `gguf`**, which the dev shell's Python does not put on its
   path: export `PYTHONPATH="$PWD/third_party/llama.cpp/gguf-py"` before `strata_pack verify` and
   `canonical_xcheck` (only `iq_fixture.py` self-inserts the path).
4. **`draft_vocab.py` without `--base` writes an empty `draft_vocab.bin`** (0 ids) — the MTP tools
   (`mtp_rt`) write only `experts.bin`/`dense.bin`/`dense.txt` and never a draft vocab. The committed
   base subset is `data/draft_vocab.bin` (106,299 ids); copy it over `mtp/rt/draft_vocab.bin`.
5. **Exact batch comparisons need `--pcie-frac 0 --adapt-every 1000000`** (`docs/BATCHING.md:186`).
   With the defaults, adaptation between runs perturbs the numerics and a slot can diverge (observed:
   slot 0 at token 34); with the flags, 4/4 are identical.
6. **`conversation_cache_http_smoke` needs the cache enabled and a big enough context**: the engine
   must run with `--conversation-cache-mib N --conversation-cache-slots M` (the test refuses
   otherwise), and the context must fit its ~1200-token prompt + 4096-token completion
   (`--max-context 8192` is enough; the default 4096 returns HTTP 400).
7. **The GPU must be free.** This host's own `frinfer-qwen38-27b.service` holds ~30.8 GiB of the
   32 GB card; with it running, the engine cannot even place its 260 MiB native embedding ("cannot
   pin …, nor place it in VRAM"). Stop it (`systemctl --user stop frinfer-qwen38-27b.service`) for a
   representative pass and start it again afterwards.
8. **`strata_tokenizer.py --out <dir>` appends `tokenizer/`** under the given directory, so
   `--out pack/full` is the natural call; `--out pack/full/tokenizer` nests it (`tokenizer/tokenizer/`).

The dev server config that ties these together (`strata-q2_0.json`, gitignored): `--pack pack/q2_0
--native <shard1> --ple-gguf <shard2> --expert-profile data/expert-profile.bin --spec 4
--spec-min-p 0.5 --prefill auto --conversation-cache-mib 8192 --conversation-cache-slots 4
--max-context 8192 --mtp mtp/rt`.

## §6 Disk footprint

`models/` 62 GB + `mtp/` 5.2 GB + `pack/` 35 GB = **~102 GB**, of which the two shards are 66 GB.
The hard-linked dense.bin saves a second 5.8 GB. Built once, the tree is reusable for every later
run; nothing needs re-downloading unless a pin moves.

## §7 Not proven here (out of scope for this ticket)

- `ple_parity` — the one registered ctest needing a real model — cannot pass with any download: its
  ggml captures and capture tooling are not in this checkout nor in public upstream
  (`CMakeLists.txt:87`).
- `iq_parity` reports two failures on the synthetic fixtures, in **IQ2_S** and **IQ3_S** (`mmvq rel`
  ≈ 0.7; the dequant side is exact). `Q2_0` — the type this tree downloads — is `ok`. This is an
  engine/kernel finding, independent of the fixtures, and worth its own issue.
- The IQ* / swift / coder / unsloth variants were not downloaded (§1), so their paths are untested.

## §8 Extended coverage — the remaining model-consuming tests (ticket #20)

Run after the representative pass, same fixtures, same free GPU:

| Test | Result |
|---|---|
| `batch_interleave_test` | **exit 0** — every slot and continuation solo-identical (3 slots over a 4888-token long prompt, a give-way/`YIELDED` exchange, a next turn, and a checkpoint continuation) |
| `parking_test` | **exit 0** — follow-up **IDENTICAL** live vs restored from the parking cache (restore: 266-token checkpoint in 31 ms; the second park retained ~106 MB) |
| `needle_bench` | **9 of 9 found** — 2k/4k/6k tokens × 10/50/90 % depth, through `serve/server.py` |
| `calibrate` | **exit 0** — recommends `--pcie-frac 0.20` and `--spec-min-p 0.70` (as a pair +6.4 % over the default in the interleaved confirmation; `--pcie-frac` alone +9 %, `--spec-min-p` alone +2.7 %, under the tool's own 3 % bar); 7 CPU workers best (221 tok/s); 98 s of measurement |

Two prerequisites these added (continuing §5):

9. **A batch graph needs VRAM headroom: pass `--vram-reserve-mib 2048`.** With nothing reserved, a
   3-slot batch over a ~5000-token prompt fails at `verify: batch instantiate: out of memory`
   (`cudaGraphInstantiate`, `verify.cpp:1803`). 2048 MiB reserved makes every slot identical.
   Removing the 8 GiB conversation cache does not help — the cache-free config OOMs too (the expert
   cache presumably grows into whatever VRAM is freed; that mechanism is inferred, not instrumented).
10. **`parking_test` needs both exactness flags, `--pcie-frac 0` *and* `--adapt-every 1000000`.**
    With adaptation left on, the live and parked follow-ups diverge (token 10 in one run, 16 in
    another — adaptation perturbs the numerics nondeterministically); with both flags they are
    identical — the same pair `batch_test` needs (§5.5).

One harness quirk: `parking_test.py:55` reads the hardcoded `/tmp/batch_test_engine.log`, while the
engine writes its log to `$TMPDIR` (`batch_test.py:49`). Under `nix develop` (where `$TMPDIR` is a
per-shell directory) set `BATCH_TEST_LOG=/tmp/batch_test_engine.log` or the script raises
`FileNotFoundError`.
