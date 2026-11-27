# MuJoCo Backend

`motrix-env-mujoco` binds the declarative `SceneCfg` of
[`motrix-env-core`](../motrix_env_core) to MuJoCo: it compiles scenes into
`mujoco.MjModel` for sim2sim deployment, preview, and conversion tools. It
provides no live simulation and cannot be selected to create runtime environments
with `registry.make(..., sim="mujoco")`. Use `MuJoCoSceneCompiler` directly for scene compilation.

Installing the package registers the backend under the name `"mujoco"` through the
`motrix_env.sim_backends` entry-point group. Importing `motrix_env_mujoco` is cheap; the native
`mujoco` module loads only when a deep module is imported.

```python
from motrix_env_mujoco.compiler import MuJoCoSceneCompiler
from motrix_env_core.config import SimCfg

model = MuJoCoSceneCompiler().compile(scene_cfg, SimCfg())
```
