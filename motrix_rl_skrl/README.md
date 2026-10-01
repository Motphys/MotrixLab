# Motrix RL SKRL

`motrix-rl-skrl` owns MotrixLab's SKRL integration: provider configuration, environment wrappers, PPO trainers and export
implementation. Its Python namespace is `motrix_rl_skrl`; typed configuration is `motrix_rl_skrl.config.SkrlCfg`.

The method is `skrl.ppo`, with Torch or JAX training backends. Install the `torch` or `jax` extra for the selected backend
(`motrix-rl-skrl[torch]` or `motrix-rl-skrl[jax]`); the JAX installation is Linux-only. Workspace installation is managed by
`sh install.sh --skrl-torch` or `sh install.sh --skrl-jax`, which also selects the workspace GPU wheel profile.

The plugin depends on `motrix-rl` control-plane services and `motrix-rl-interface` contracts. SKRL and backend-specific
libraries belong to this plugin's extras, not the interface package. Discovery uses the `motrix_rl.frameworks` entry-point
group: `skrl = "motrix_rl_skrl.plugin:register"`. Registration declares provider capabilities and installs the algorithm
schema before Hydra task composition; it does not create a parallel task registry.
