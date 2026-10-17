# Motrix Environment Core

`motrix-env-core` provides the backend-neutral framework APIs used to build reinforcement learning
environments:

- environment base classes and configuration protocols;
- environment registry and registration decorators;
- backend-independent scene configuration and the simulator boundary (`motrix_env_core.sim`);
- NumPy environment runtime and rendering;
- framework-level math and reward utilities.

It intentionally contains no built-in tasks, robot models, meshes, textures, or motion data, and no
simulator: importing `motrix_env_core` never imports MotrixSim or MuJoCo. Install `motrix-envs` when
the built-in MotrixLab environments are required.

```python
from motrix_env_core import EnvCfg, SceneCfg, configclass, registry
```

## Environment discovery

Registry queries automatically discover installed environment providers through the
`motrix_env_core.envs` entry-point group. Each package supplies a registration callback that uses the
existing environment decorators; applications need no registration imports. Environment discovery
is independent of training and Hydra Task configuration.

The first query loads all installed providers, not just the provider for the requested Env ID.
Provider dependency and registration errors are reported to the caller.

See [Environment registration](../docs/source/en/user_guide/tutorial/building_envs/direct_env.md#environment-registration)
for the package declaration and registration example shared by direct and manager environments.

## Simulator backends

Simulator bindings live in their own distributions and register through the
`motrix_env.sim_backends` entry-point group:

- [`motrix-env-motrixsim`](../motrix_env_motrixsim) — the MotrixSim backend (live simulation,
  rendering, and the torch-tensor frontend); registered as the default simulator;
- [`motrix-env-mujoco`](../motrix_env_mujoco) — a compile-only MuJoCo backend that lowers
  `SceneCfg` into `mujoco.MjModel` (sim2sim, preview, conversion tools).

Installing a backend package registers it; the registry discovers entry points lazily on first use
and never imports a native simulator until a backend is actually constructed.

Compile-only consumers (viewers, converters, previews) import the backend's scene compiler directly,
for example `from motrix_env_mujoco.compiler import MuJoCoSceneCompiler`.

Requesting a backend whose package is not installed fails loudly with an installation hint, and the
other registered backends remain usable.
