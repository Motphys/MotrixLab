# Motrix Environments

`motrix-envs` contains the built-in MotrixLab reinforcement learning environments and their robot models, meshes,
textures, and motion data. It depends on the standalone `motrix-env-core` framework package.

Install the package to make built-in environments available through automatic core registry discovery;
no registration import is needed:

```python
from motrix_env_core import registry

env = registry.make("cartpole", num_envs=1)
```

The package includes the `basic`, `locomotion`, and `manipulation` environment families. Environment names used by
training and playback commands remain unchanged.

Install the `deploy` extra to use the built-in deployment profile compilers with `mxlab export-deploy`:

```bash
mxlab export-deploy env=go2-walk-rough output=artifacts/go2-walk-rough.deploy
```

The installed package publishes one provider for its built-in profile compilers:

```toml
[project.entry-points."motrix_deploy.profiles"]
builtin = "motrix_envs.deploy:register"
```

Its callback registers profiles for `go2-walk-flat`, `go2-walk-rough`, and `g1-wbt-dance`.
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

The installed package exposes one callback for all built-in environments:

```toml
[project.entry-points."motrix_env_core.envs"]
builtin = "motrix_envs:register"
```

The callback includes the environments under `basic`, `locomotion`, and `manipulation`.
Training/Hydra Task configs belong to the consuming project, not this package. See
[core discovery](../motrix_env_core/README.md#environment-discovery) for the shared discovery behavior and
[Environment registration](../docs/source/en/user_guide/tutorial/building_envs/direct_env.md#environment-registration)
for publishing an external environment package.
