# Motrix RL RSLRL

`motrix-rl-rslrl` owns MotrixLab's RSLRL PPO integration, wrapping the native `rsl_rl` runner with provider configuration,
environment wrappers, training/playback and export implementation. Its Python namespace is `motrix_rl_rslrl`; typed
configuration is `motrix_rl_rslrl.cfg.RslrlCfg`.

The method is `rslrl.ppo`, with the Torch training backend. Install it in the workspace using `sh install.sh --rslrl`;
the root `rslrl` extra selects this optional plugin. An ordinary runtime installation does not install every workspace
package and therefore does not force RSLRL into a SKRL/FastSAC environment.

The plugin depends on `motrix-rl` control-plane services, `motrix-rl-interface` contracts and its RSLRL/Torch runtime.
Discovery uses the `motrix_rl.frameworks` entry-point group: `rslrl = "motrix_rl_rslrl.plugin:register"`. Consumers use the
control-plane registry rather than importing plugin internals to trigger registration.
