# Motrix RL RSLRL

`motrix-rl-rslrl` owns MotrixLab's RSLRL PPO integration, using native `rsl_rl` algorithms with provider configuration,
environment wrappers, training/playback and export implementation. Its Python namespace is `motrix_rl_rslrl`; typed
configuration is `motrix_rl_rslrl.cfg.RslrlCfg`.

The method is `rslrl.ppo`, with the Torch training backend. Install it in the workspace using `sh install.sh --rslrl`;
this selects the root `rslrl` extra (`motrix-rl-rslrl`). FastSAC and SKRL Torch remain installed as required root dependencies.

The plugin directly depends on `motrix-rl` control-plane services and contracts, `rsl-rl-lib`, and Torch. RSLRL is optional
at the workspace root, not inside the plugin: installing the plugin includes its required algorithm library.
Discovery uses the `motrix_rl.frameworks` entry-point group: `rslrl = "motrix_rl_rslrl.plugin:register"`. Consumers use the
control-plane registry rather than importing plugin internals to trigger registration.

Training uses the shared [synchronous execution layer](../motrix_rl/src/motrix_rl/execution/README.md) and the same
console panel as FastSAC, while RSLRL retains rollout storage, returns and PPO updates. TensorBoard, W&B and Neptune
writers retain their native RSLRL scalar tags; all outputs publish at MotrixLab's learner-iteration logging interval. Playback and
export continue to accept native RSLRL checkpoints. Resume through the MotrixLab trainer remains unsupported.
