# The home-manager module: one profile = one systemd user service plus the oneshot that provisions
# it. The spec is docs/design/vsrata-module.md; this file is that spec in Nix.
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.vsrata;

  # The registry is generated from setup.py by tools/make_registry.py (see §2 of the spec); it is
  # what makes "an unknown model or quant fails nix eval" possible.
  registry = import ../nix/registry.nix;

  enabled = lib.filterAttrs (_: p: p.enable) cfg.profiles;
  names = lib.attrNames enabled;
  # The port a profile without an explicit one gets: basePort + its alphabetical position.
  indices = lib.listToAttrs (lib.imap0 (i: n: { name = n; value = i; }) names);

  stateDir = "${config.xdg.stateHome}/vsrata";       # the marker, the log, and the 0600 token file
  trainedContext = 262144;                            # what the rope rule is measured against

  # The seven sizes setup offers (setup.py:181). An explicit value outside them is kept there, but
  # the module refuses one: it cannot guess the rope treatment for an untested window.
  contexts = [ 8192 32768 65536 131072 262144 393216 524288 ];

  # "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf" → the given shard's file name.
  shardName = pattern: quant: i:
    lib.replaceStrings [ "{q}" "{i}" ] [ quant (toString i) ] pattern;

  profileOf = name: p:
    let
      entry = registry.${p.model} or (throw
        "programs.vsrata.profiles.${name}.model: '${p.model}' is not in nix/registry.nix. Known models: ${
          lib.concatStringsSep ", " (lib.attrNames registry)}");
      quant = entry.quants.${p.quant} or (throw
        "programs.vsrata.profiles.${name}.quant: '${p.quant}' is not a quant of ${p.model}. Known quants: ${
          lib.concatStringsSep ", " (lib.attrNames entry.quants)}");

      modelsDir = "${cfg.modelDir}/models/${name}";
      packDir   = "${cfg.modelDir}/packs/${name}";
      mtpDir    = "${cfg.modelDir}/mtp/${name}";
      file      = quant.file or entry.file;           # a quant may name its own shards (UD-IQ4_XS)
      firstShard = "${modelsDir}/${shardName file p.quant 1}";
      # --ple-gguf wants the shard holding the PLE table: shard 2, or shard 1 for Swift (research #25).
      pleShard  = entry.pleShard or (if entry.family == "swift" then 1 else 2);
      plePath   = "${modelsDir}/${shardName file p.quant pleShard}";
      mmprojPath = "${modelsDir}/${entry.mmproj.file}";
      expertProfile = "${cfg.package}/share/vsrata/data/${entry.profile or "expert-profile.bin"}";
      port = if p.port != null then p.port else cfg.basePort + indices.${name};
      # The extension factor past the trained window, mirroring setup.py's derived_factor.
      ropeScale = (p.contextWindow + 0.0) / trainedContext;

      args =
        [ "--pack" packDir "--native" firstShard ]
        ++ lib.optionals (quant.shards <= 2) [ "--ple-gguf" plePath ]
        ++ [ "--expert-profile" expertProfile "--expert-cache" "auto"
             "--prefill" "auto" "--spec" "4" "--spec-min-p" "0.5"
             "--mtp" "${mtpDir}/rt" "--max-context" (toString p.contextWindow) ]
        ++ lib.optionals (p.contextWindow > trainedContext)
             [ "--rope-scaling" "yarn" "--rope-scale" (toString ropeScale) ]
        ++ lib.optionals (p.contextWindow > 8192) [ "--kv" "int8" ]   # setup.py's default (setup.py:3545)
        ++ lib.optionals p.vision [ "--vision" "--vram-reserve-mib" "700" ]
        ++ p.extraArgs;

      configJson = pkgs.writeText "vsrata-${name}.json" (builtins.toJSON ({
        exe        = "${cfg.package}/bin/strata";
        inherit args;
        cwd        = stateDir;
        tokenizer  = "${packDir}/tokenizer";
        model_name = "${entry.name}-${lib.toLower p.quant}";
        log        = "${stateDir}/${name}.log";
        lib_dirs   = [ ];
        host       = cfg.host;
        inherit port;
        provision  = {
          inherit (p) model quant contextWindow;
          vision = p.vision;
          visionAccel = p.visionAccel;
          dataRoot   = cfg.modelDir;
          profileDir = modelsDir;
          packDir    = packDir;
          mtpDir     = mtpDir;
          shards     = quant.shards;
          file       = file;
          packArgs   = entry.packArgs;
          doneFile   = "${stateDir}/${name}.done";
          mmproj     = entry.mmproj;
        };
      } // lib.optionalAttrs p.vision {
        vision = {
          exe        = "${cfg.package}/bin/strata-vision";
          mmproj     = mmprojPath;
          model      = firstShard;
          gpu        = p.visionAccel == "gpu";
          max_tokens = if p.visionAccel == "gpu" then 1024 else 300;
          threads    = 0;
          cuda_device = null;
        };
      }));

      # The credential the provisioning unit can read at runtime. A literal `hfToken` has to be
      # written to the state dir at 0600 — and, because an activation script is a store file, the
      # literal is in the store either way; the option's description says so.
      tokenPath =
        if cfg.hfTokenFile != null then cfg.hfTokenFile
        else if cfg.hfToken != null then "${stateDir}/hf-token"
        else null;

      provisionUnit = "vsrata-provision-${name}";
      serveUnit = "vsrata-${name}";
    in
    {
      inherit name p configJson port tokenPath provisionUnit serveUnit;
      doneFile = "${stateDir}/${name}.done";
    };

  profiles = map (n: profileOf n enabled.${n}) names;
in
{
  options.programs.vsrata = {
    enable = lib.mkEnableOption "the Vsrata model server";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.vsrata;
      defaultText = lib.literalExpression "pkgs.vsrata";
      description = "The Vsrata package. Defaults to `pkgs.vsrata`, which needs this flake's `overlays.default` imported (or set this explicitly).";
    };

    hfToken = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "A Hugging Face token, for convenience. **It is written into the world-readable Nix store** — prefer `hfTokenFile`.";
    };

    hfTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "A file holding a Hugging Face token, read by the provisioning unit at runtime (0600, never in the store).";
    };

    modelDir = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "/home/alice/models";
      description = "The data root: `models/<profile>/`, `packs/<profile>/` and `mtp/<profile>/` are created inside it.";
    };

    basePort = lib.mkOption {
      type = lib.types.port;
      default = 8095;
      description = "The port of the first profile; each further profile gets the next one.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "The address the servers bind. A non-loopback address requires an `apiKeyFile` on every profile.";
    };

    profiles = lib.mkOption {
      default = { };
      description = "One model-serving instance per attribute name (see `programs.vsrata.profiles.<name>`).";
      type = lib.types.attrsOf (lib.types.submodule ({ name, ... }: {
        options = {
          enable = lib.mkOption {
            type = lib.types.bool;
            default = true;
            description = "Whether this profile has units.";
          };
          model = lib.mkOption {
            type = lib.types.str;
            example = "unsloth/Qwen3.8-Flash-Next-GGUF";
            description = "The Hugging Face repository, as named in nix/registry.nix.";
          };
          quant = lib.mkOption {
            type = lib.types.str;
            example = "UD-Q4_K_XL";
            description = "The quantization within that repository.";
          };
          contextWindow = lib.mkOption {
            type = lib.types.ints.positive;
            example = 262144;
            description = "The token context length: one of ${lib.concatStringsSep ", " (map toString contexts)}.";
          };
          vision = lib.mkOption {
            type = lib.types.bool;
            default = false;
            description = "Serve images as well as text (needs the repository's mmproj encoder).";
          };
          visionAccel = lib.mkOption {
            type = lib.types.enum [ "gpu" "cpu" ];
            default = "gpu";
            description = "Which encoder to run when `vision` is on.";
          };
          port = lib.mkOption {
            type = lib.types.nullOr lib.types.port;
            default = null;
            description = "This profile's port; null means `basePort` plus its alphabetical position.";
          };
          apiKeyFile = lib.mkOption {
            type = lib.types.nullOr lib.types.path;
            default = null;
            description = "A file holding this profile's API key, read by the serve unit at runtime.";
          };
          extraArgs = lib.mkOption {
            type = lib.types.listOf lib.types.str;
            default = [ ];
            description = "Appended to the engine's arguments verbatim. The only engine-tuning escape hatch — profiling several profiles against one GPU is not this module's job.";
          };
        };
      }));
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.modelDir != null;
        message = "programs.vsrata.modelDir must be set when programs.vsrata.enable is true.";
      }
      {
        assertion = cfg.hfToken == null || cfg.hfTokenFile == null;
        message = "programs.vsrata: set at most one of hfToken and hfTokenFile.";
      }
      {
        assertion = !(cfg.host != "127.0.0.1" && cfg.host != "::1" && cfg.host != "localhost")
          -> lib.all (p: p.p.apiKeyFile != null) profiles;
        message = "programs.vsrata.host is not loopback: every profile needs an apiKeyFile.";
      }
    ] ++ map (pr: {
      assertion = lib.elem pr.p.contextWindow contexts;
      message = "programs.vsrata.profiles.${pr.name}.contextWindow: ${toString pr.p.contextWindow} is not one of ${lib.concatStringsSep ", " (map toString contexts)}.";
    }) profiles ++ map (pr: {
      assertion = builtins.match "[A-Za-z0-9_.@:-]+" pr.name != null;
      message = "programs.vsrata.profiles.${pr.name}: a profile name must be usable in a unit name.";
    }) profiles;

    # A literal token has to exist as a file before the unit reads it. 0600, in the state dir.
    home.activation.vsrataHfToken = lib.mkIf (cfg.hfToken != null) (
      lib.hm.dag.entryAfter [ "writeBoundary" ] ''
        run mkdir -p ${lib.escapeShellArg stateDir}
        run install -m600 /dev/null ${lib.escapeShellArg "${stateDir}/hf-token"}
        printf '%s' ${lib.escapeShellArg cfg.hfToken} > ${lib.escapeShellArg "${stateDir}/hf-token"}
      ''
    );

    systemd.user.services = lib.listToAttrs (map (pr: {
      name = pr.provisionUnit;
      value = {
        Unit = {
          Description = "Provision the Vsrata profile ${pr.name}";
          # A finished profile is skipped by systemd itself, before the CLI runs.
          ConditionPathExists = "!${pr.doneFile}";
        };
        Service = {
          Type = "oneshot";
          RemainAfterExit = true;
          ExecStart = "${cfg.package}/bin/vsrata provision --config ${pr.configJson}";
        } // lib.optionalAttrs (pr.tokenPath != null) {
          LoadCredential = "hf-token:${toString pr.tokenPath}";
          Environment = [ "VSRATA_HF_TOKEN_FILE=%d/hf-token" ];
        };
      };
    }) profiles) // lib.listToAttrs (map (pr: {
      name = pr.serveUnit;
      value = {
        Unit = {
          Description = "The Vsrata profile ${pr.name} (${pr.p.model} ${pr.p.quant})";
          Requires = [ "${pr.provisionUnit}.service" ];
          After = [ "${pr.provisionUnit}.service" ];
          # No WantedBy: services are manual-start. Starting this one pulls the provision unit in,
          # so the model is downloaded on first run and then served.
        };
        Service = {
          Type = "simple";
          ExecStart = "${cfg.package}/bin/vsrata serve --config ${pr.configJson} --port ${toString pr.port}";
          Restart = "on-failure";
          RestartSec = 5;
          # The engine pins its expert arena and the PLE table; without this it degrades (research #25).
          # Raising a user's hard limit needs a system-level setting home-manager cannot make.
          LimitMEMLOCK = "infinity";
        } // lib.optionalAttrs (pr.p.apiKeyFile != null) {
          LoadCredential = "api-key:${toString pr.p.apiKeyFile}";
          Environment = [ "STRATA_API_KEY_FILE=%d/api-key" ];
        };
      };
    }) profiles);
  };
}
