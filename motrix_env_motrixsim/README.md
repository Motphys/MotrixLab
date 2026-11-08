# MotrixSim Backend

`motrix-env-motrixsim` binds the backend-neutral sim boundary of
[`motrix-env-core`](../motrix_env_core) to the MotrixSim simulator:

- `MotrixSimSceneCompiler` — compiles a declarative `SceneCfg` into a MotrixSim model;
- `MotrixSimBackend` — the live `SimBackend` (reads, writes, stepping, reset, rendering).

Installing the package registers the backend under the name `"motrixsim"` through the
`motrix_env.sim_backends` entry-point group, and marks it as the default simulator. Importing
`motrix_env_motrixsim` is cheap; the native `motrixsim` module loads only when a deep module is
imported.

```python
from motrix_env_motrixsim.compiler import build_scene_model

model = build_scene_model(scene_cfg)
```
