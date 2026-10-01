# Motrix RL SKRL

`motrix-rl-skrl` owns MotrixLab's SKRL integration: provider configuration, environment wrappers, PPO trainers and export
implementation. Its Python namespace is `motrix_rl_skrl`; typed configuration is `motrix_rl_skrl.config.SkrlCfg`.

The method is `skrl.ppo`, with Torch or JAX training backends. Installing `motrix-rl-skrl` includes SKRL and Torch;
`motrix-rl-skrl[jax]` adds the Linux-only JAX runtime. Workspace installation is managed by
`sh install.sh --skrl-torch` or `sh install.sh --skrl-jax`, which also selects the workspace GPU wheel profile.

The plugin depends on `motrix-rl` control-plane services and contracts, SKRL, and Torch. Only the additional JAX backend
libraries are optional within the plugin. This plugin is a required workspace root dependency for the built-in PPO demos;
the core package never depends on it. Discovery uses the `motrix_rl.frameworks` entry-point group:
`skrl = "motrix_rl_skrl.plugin:register"`. Registration declares provider capabilities and installs the algorithm
schema before Hydra task composition; it does not create a parallel task registry.
