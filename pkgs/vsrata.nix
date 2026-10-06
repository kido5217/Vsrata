# The package the flake exposes: the Strata engine, the vision encoder, the Python server, the tools
# and the data the engine reads — everything a profile needs to be provisioned and served.
#
# The layout is the contract from docs/design/vsrata-module.md §3: the CLI runs
# `share/vsrata/serve/server.py`, whose own ROOT is `share/vsrata`, so the engine's expert profile
# and draft vocab resolve to `share/vsrata/data/` without any configuration of ours.
#
# Neither CMakeLists has an install() rule (research #24), so the binaries are installed by hand:
# the engine lands at `build/strata`, the encoder at `build-vision/bin/strata-vision`.
{ lib
, pkgs
, llamaCpp          # the pinned llama.cpp/ggml source
, src ? ../.
, cudaArchitectures ? [ "120a" ]   # sm_120a, the verified default (ticket #26)
, version ? "0.1.39"
}:

let
  cuda = pkgs.cudaPackages_13.cudatoolkit;
  archs = lib.concatStringsSep ";" cudaArchitectures;

  # The server's own pinned dependencies (root requirements.txt). Python is also what runs the
  # tools the CLI drives (`tools/*.py`), so it must be reachable at runtime.
  python = pkgs.python312.withPackages (ps: with ps;
    [ numpy jinja2 regex pyyaml tqdm requests pillow psutil ]);
in
pkgs.stdenv.mkDerivation {
  pname = "vsrata";
  inherit version src;

  nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.makeWrapper pkgs.patchelf ];
  buildInputs = [ cuda python ];

  # CMake's CUDA compiler-ID probe needs the merged toolkit's include dir (research #10 §3).
  NVCC_PREPEND_FLAGS = "-I${cuda}/include";

  enableParallelBuilding = true;

  # We run cmake ourselves: there are two projects (the engine and the encoder), and neither
  # installs anything, so the default phases would only get in the way.
  dontUseCmakeConfigure = true;

  buildPhase = ''
    runHook preBuild

    cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_CUDA=ON \
      -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=${archs} \
      -DCMAKE_CUDA_COMPILER=${cuda}/bin/nvcc \
      -DSTRATA_GGML_DIR=${llamaCpp}
    cmake --build build --parallel

    # The image encoder. It builds llama.cpp/mtmd from the pinned source itself.
    cmake -B build-vision -S tools/vision -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_VISION_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${archs} \
      -DCMAKE_CUDA_COMPILER=${cuda}/bin/nvcc \
      -DLLAMA_DIR=${llamaCpp}
    cmake --build build-vision --parallel

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin $out/share/vsrata
    install -m755 build/strata $out/bin/strata
    install -m755 build-vision/bin/strata-vision $out/bin/strata-vision

    # server/, tools/ and data/ sit together because server.py and the tools resolve paths
    # relative to their own location — ROOT becomes $out/share/vsrata. `ref/` is the plan reader
    # strata_pack imports, and setup.py is what the CLI reads the model tables from.
    for d in serve tools data cli ref; do
      cp -r "$d" $out/share/vsrata/
    done
    # The tests are not part of the installed tool set (and keeping them out means editing one does
    # not invalidate a built package).
    find $out/share/vsrata/serve $out/share/vsrata/tools -maxdepth 1 -name 'test_*.py' -delete
    install -m644 setup.py $out/share/vsrata/setup.py

    # tools/_paths.py looks for llama.cpp's gguf-py first at <root>/third_party/llama.cpp/gguf-py,
    # and mtp_pack/mtp_rt exit without it — so the package ships it rather than needing
    # STRATA_GGUF_PY set in every unit.
    mkdir -p $out/share/vsrata/third_party/llama.cpp
    cp -r ${llamaCpp}/gguf-py $out/share/vsrata/third_party/llama.cpp/gguf-py

    find $out/share/vsrata -name __pycache__ -type d -prune -exec rm -rf {} +

    # The CLI. `python` (with the server's packages) is baked in, and PYTHONPATH lets it import
    # `serve`/`tools` from the share tree.
    makeWrapper ${python}/bin/python $out/bin/vsrata \
      --add-flags $out/share/vsrata/cli/vsrata.py \
      --prefix PYTHONPATH : "$out/share/vsrata" \
      --set VSRATA_ROOT $out/share/vsrata \
      --set-default PYTHONUNBUFFERED 1

    runHook postInstall
  '';

  # Both cmake projects link CUDA, and llama.cpp's (the encoder) leaves the CUDA *stubs*
  # directory on the binary's RUNPATH. A runtime binary must not search it: it shadows the
  # installed driver, and ggml then reports "CUDA driver is a stub library". Drop the entry
  # and put the real driver directory first so libcuda resolves to the driver (NixOS:
  # /run/opengl-driver/lib, via addDriverRunpath's driverLink).
  postFixup = ''
    for b in strata strata-vision; do
      rp=$(patchelf --print-rpath "$out/bin/$b" | tr ':' '\n' | grep -v '^$' | grep -v '/stubs$' | paste -sd: -)
      patchelf --set-rpath "${pkgs.addDriverRunpath.driverLink}/lib''${rp:+:$rp}" "$out/bin/$b"
    done
  '';

  # The engine is not useful to a build consumer, but the data dir is where the module points
  # the engine's expert profile, so expose it for the module and for tests.
  passthru.dataDir = "$out/share/vsrata/data";

  meta = {
    description = "Strata: the Qwen3.8-Flash-Next moE engine, its server and its CLI";
    homepage = "https://github.com/kido5217/Vsrata";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
