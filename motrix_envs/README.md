# Motrix Environments

`motrix-envs` contains the built-in MotrixLab reinforcement learning environments and their robot models, meshes,
textures, and motion data. It depends on the standalone `motrix-env-core` framework package.

Importing `motrix_envs` registers all built-in environments in the core registry:

```python
from motrix_envs import registry

env = registry.make("cartpole", num_envs=1)
```

The package includes the `basic`, `locomotion`, and `manipulation` environment families. Environment names used by
training and playback commands remain unchanged.

Install the `deploy` extra to use the built-in deployment profile compilers with `mxexport deploy`:

```bash
mxexport deploy env=go2-walk-rough output=artifacts/go2-walk-rough.deploy
```

The installed package publishes these `motrix_deploy.profiles` entry points:

| Environment name | Compiler |
| --- | --- |
| `go2-walk-flat`, `go2-walk-rough` | `motrix_envs.deploy.go2_walk:build_go2_walk_profile` |
| `g1-wbt-dance` | `motrix_envs.deploy.g1_wbt:build_g1_wbt_profile` |

No registration import is needed for profile discovery. See the
[deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md#add-a-deployment-profile-compiler)
for the plugin contract and an external-package example.

## Framework API

`motrix_envs.core` mirrors the complete public API exported by `motrix_env_core`:

```python
from motrix_envs.core import EnvCfg, NpEnv, SceneCfg, configclass, registry
```

Projects that only need the environment framework should install `motrix-env-core` and import it directly:

```python
from motrix_env_core import EnvCfg, NpEnv, SceneCfg, configclass, registry
```

Framework imports such as `motrix_envs.np.env` move to `motrix_env_core` or the `motrix_envs.core` facade. Concrete
environment paths under `motrix_envs.basic`, `motrix_envs.locomotion`, and `motrix_envs.manipulation` remain stable.

## Registration

The framework never imports or discovers concrete environment packages. Each implementation package owns its
registration side effects. For the built-in environments this happens when `motrix_envs` is imported.
