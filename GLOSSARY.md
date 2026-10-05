# Strata / Vsrata

The language this repository uses for running the Qwen3.8-Flash-Next moE engine locally: what a user
configures, and what the engine loads.

## Language

**Profile**:
One named model-serving instance — a single config entry, which a `programs.vsrata` module will
declare as one entry per profile once the module exists.
_Avoid_: instance, deployment, expert profile

**Model**:
The Hugging Face repository a profile serves, named `owner/name` (e.g. `unsloth/Qwen3.8-Flash-Next-GGUF`).
_Avoid_: checkpoint, weights, file

**Quant**:
The quantization selected within a model — one of the registry's names (e.g. `Q2_0`, `UD-IQ4_XS`).
A model repository hosts one or more (the Coder repository hosts only `IQ1_M`).
_Avoid_: quantization level, precision, variant, size

**Context window**:
The token context length a profile serves — one of the seven sizes setup offers (8192, 32768, 65536,
131072, 262144, 393216, 524288); an explicitly given value outside them is kept.
_Avoid_: context size, ctx, max-context

**Provisioning**:
Making a profile's model runnable: fetching its files and building what the engine needs to load
them. It is the work a profile's first run does, and it is not the same as installing the program.
_Avoid_: install, setup, download

**Pack**:
The engine's prepared directory for a model: the load table, the dense (non-expert) tensors and the
tokenizer, plus the expert form the build produced — the expert layout, the converted expert arena,
or both (a low-RAM build writes both).
_Avoid_: model file, weights, cache

**Native pack**:
A pack that carries its own expert layout, so the engine reads the experts from the GGUF and needs
no AVX-512 CPU. A pack without one is a *canonical* pack, which is not what a service loads on a
CPU without AVX-512.
_Avoid_: iq pack, converted pack

**MTP draft layer**:
The head the engine uses to propose tokens that a full pass then verifies, which makes decoding
faster. Accepted drafts are the tokens a non-speculative pass would emit, but with the IQ models a
greedy answer can differ unless `STRATA_IQ_MT_MIN=1`, and byte-identical repeats through the server
also need `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0` (`docs/DETAILS.md`, issues #152 and #410).
_Avoid_: speculator, draft model, MTP (alone)
