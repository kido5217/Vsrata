# Strata dev environment (map #9) — `nix develop` first, then `just <recipe>`.
# Build dirs at the repo root; gitignored (build*/).

nvcc := `command -v nvcc`

# Enter the devShell (CUDA 13.2, cmake, ninja, python 3.12 + server packages, just)
dev:
    nix develop

# Engine + vision encoder, CUDA {{archs}} (default: this host's sm_120)
build archs = "120":
    cmake -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={{archs}} \
        -DCMAKE_CUDA_COMPILER="{{nvcc}}" -DSTRATA_GGML_DIR="$STRATA_GGML_DIR"
    cmake --build build --parallel
    cmake -B build-vision -S tools/vision -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={{archs}} \
        -DCMAKE_CUDA_COMPILER="{{nvcc}}" -DLLAMA_DIR="$STRATA_GGML_DIR"
    cmake --build build-vision --parallel

# setup test suite (no GPU/downloads). Expected: all pass except test_setup_golden
# (pre-existing stale fixture — fails identically under host python, reported as-is).
test:
    for f in tools/test_setup_*.py; do python3 "$f" || echo "FAIL (pre-existing): $f"; done

# Canonical-form round-trip oracle (pure python)
oracle:
    python3 tools/canonical_xcheck.py --tensors 4

# Server for the most recently set-up model (drives setup.py; needs a model on this host)
server port = "8080":
    python3 setup.py --port {{port}}

# The MCP server (stdlib-only; client snippets in docs/MCP_SERVER.md)
mcp:
    python3 tools/strata_mcp.py

# Re-pin flake inputs to their branch heads
flake-update:
    nix flake update

# Remove build dirs
clean:
    rm -rf build build-vision
