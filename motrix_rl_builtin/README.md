# Motrix RL Builtin

`motrix-rl-builtin` owns MotrixLab's built-in reinforcement-learning integrations, currently FastSAC. Its Python namespace
is `motrix_rl_builtin`; FastSAC configuration and implementation live under `motrix_rl_builtin.fastsac`.

The plugin owns FastSAC models, replay buffers, environment wrappers, synchronous/asynchronous trainers and policy-export
implementation. The method remains `motrix.fastsac`, with the Torch training backend; `algo.asynchronous` selects the
execution topology rather than a separate method.

It depends only on `motrix-rl` for control-plane services and provider/trainer contracts, plus its
Torch runtime dependencies. It registers via the `motrix_rl.frameworks` entry-point group:
`builtin = "motrix_rl_builtin.plugin:register"`. Consumers query `motrix_rl.frameworks` rather than importing plugin
internals to trigger registration.
