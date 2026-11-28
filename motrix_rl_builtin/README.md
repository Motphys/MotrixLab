# Motrix RL Builtin

`motrix-rl-builtin` integrates MotrixLab's built-in reinforcement-learning algorithms under the `motrix` framework:
`motrix.fastsac`, `motrix.supermimic`, and `motrix.appo`, using the Torch training backend. Its Python namespace is
`motrix_rl_builtin`; FastSAC configuration and implementation live under `motrix_rl_builtin.fastsac`.

The plugin owns FastSAC models, replay buffers, environment wrappers, synchronous/asynchronous trainers and policy-export
implementation. The method remains `motrix.fastsac`, with the Torch training backend; `algo.asynchronous` selects the
execution topology rather than a separate method.

SuperMimic configuration, provider and training implementation remain in the independently packaged
[`motrix-rl-supermimic`](../motrix_rl_supermimic/README.md). Builtin depends on that package and owns its registration under
`motrix`, including the `motrix.supermimic` algorithm defaults. SuperMimic does not depend on builtin.

APPO implementation remains in the independently packaged [`motrix-rl-appo`](../motrix_rl_appo/README.md). Builtin depends
on that package and owns its provider registration and public ConfigStore name, `motrix.appo`, including the corresponding
algorithm defaults. APPO does not expose its own framework entry point or public provider name.

Control-plane services and provider/trainer contracts come from `motrix-rl`. Builtin registers via the
`motrix_rl.frameworks` entry-point group:
`builtin = "motrix_rl_builtin.plugin:register"`. Consumers query `motrix_rl.frameworks` rather than importing plugin
internals to trigger registration. The builtin dependency set includes APPO by default; it is not an optional `appo` extra.
