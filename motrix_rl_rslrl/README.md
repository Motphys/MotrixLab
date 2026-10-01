# Motrix RL RSLRL

`motrix-rl-rslrl` owns MotrixLab's RSLRL PPO integration, wrapping the native `rsl_rl` runner with provider configuration,
environment wrappers, training/playback and export implementation. Its Python namespace is `motrix_rl_rslrl`; typed
configuration is `motrix_rl_rslrl.cfg.RslrlCfg`.

The method is `rslrl.ppo`, with the Torch training backend. Install it in the workspace using `sh install.sh --rslrl`;
this selects the root `rslrl` extra (`motrix-rl-rslrl`). FastSAC and SKRL Torch remain installed as required root dependencies.

The plugin directly depends on `motrix-rl` control-plane services and contracts, `rsl-rl-lib`, and Torch. RSLRL is optional
at the workspace root, not inside the plugin: installing the plugin includes its required algorithm library.
Discovery uses the `motrix_rl.frameworks` entry-point group: `rslrl = "motrix_rl_rslrl.plugin:register"`. Consumers use the
control-plane registry rather than importing plugin internals to trigger registration.
