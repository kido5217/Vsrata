{
  description = "Strata dev environment (nixos-26.05 branch, CUDA 13.2, pinned llama.cpp)";

  inputs = {
    # nixos-26.05 branch (user decision, ticket #11): tracks backports; flake.lock records the rev.
    # flake = false so the outputs function gets the source path: the nixpkgs flake's
    # legacyPackages is evaluated without a config (flake.nix:224-231), and the CUDA
    # toolkit is unfree (EULA) — we import the source with our own config instead (#10 §3).
    nixpkgs = {
      url = "github:NixOS/nixpkgs/nixos-26.05";
      flake = false;
    };
    # ggml source for engine + vision (setup.py's LLAMA_CPP_COMMIT pin)
    llamaCpp = {
      url = "github:ggml-org/llama.cpp/3cf03257f219afbe7334045ff7c6a06ac68c627d";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, llamaCpp }:
    let
      pkgs   = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
      cuda   = pkgs.cudaPackages_13.cudatoolkit;      # cuda-merged-13.2: nvcc 13.2.51 + headers + libs
      python = pkgs.python312.withPackages (ps: with ps;
        [ numpy jinja2 regex pyyaml tqdm requests pillow psutil ]);
      devPackages = [
        pkgs.cmake     # 4.1.6 at the measured snapshot: satisfies every cmake_minimum_required
          pkgs.ninja
          pkgs.just pkgs.git pkgs.gh
          cuda
          python
      ];
    in
    {
      packages.x86_64-linux.default = pkgs.buildEnv { name = "strata-dev"; paths = devPackages; };

      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = devPackages;
        # CMake's CUDA compiler-ID probe needs the merged toolkit's include dir (#10 §3)
        env.NVCC_PREPEND_FLAGS = "-I${cuda}/include";
        env.STRATA_GGML_DIR = "${llamaCpp}";
      };
    };
}
