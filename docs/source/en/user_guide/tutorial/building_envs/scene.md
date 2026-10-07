# SceneCfg: Setting Up the Physics Scene

The physics scene defines simulation parameters and scene settings for reinforcement learning training.
MotrixLab uses [MotrixSim](https://motrixsim.readthedocs.io/en/latest/user_guide/index.html) as the physics simulation backend.

## Supported File Formats

-   [**MJCF**](https://mujoco.readthedocs.io/en/stable/XMLreference.html) (MuJoCo XML format) - Provides rich physics features and simulation configuration

## Scene File Configuration

Use `SceneCfg.file` to load a complete model file as the base scene. Assets, visual settings, and scene objects declared
on the same `SceneCfg` are applied to that base world before the model is built:

```python
from motrix_env_core import registry
from motrix_env_core.base import EnvCfg, SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg


@registry.envcfg("my-task")
@configclass
class MyTaskEnvCfg(EnvCfg):
    scene: SceneCfg = SceneCfg(file="my_model.xml")

    # Simulation and control parameters
    sim: SimCfg = SimCfg(
        dt=0.002,
        solver_iterations=3,
        solver_tolerance=1e-4,
    )
    ctrl_dt: float = 0.02

    # Episode parameters
    max_episode_seconds: float = 20.0
    reset_noise_scale: float = 0.01
```

### Recommended Directory Structure

```
my_environment_package/my_task/
├── __init__.py          # Module initialization
├── cfg.py               # Environment configuration
├── my_model.xml         # Physics model file
└── my_env.py            # Environment implementation
```

For complex models with many referenced files, it's recommended to use folder management.

## Common Configuration Issues

### File Path Issues

-   When using relative paths, ensure paths are relative to the configuration file location
-   Avoid using hardcoded absolute paths
-   Check file permissions and accessibility
-   Ensure all referenced sub-files exist

### Time Step Settings

-   `ctrl_dt` should be an integer multiple of `sim.dt`
-   `sim.dt` that is too small will affect simulation performance
-   `ctrl_dt` that is too large will affect control precision
-   Recommend `sim.dt` between 0.001-0.02 seconds

`SimCfg` configures the physics engine through MSD `World.simulate_option` before the model is built. When `solver_iterations`, `solver_tolerance`, or `gravity` is `None`, the value already provided by the MJCF or MSD World is preserved; an explicit value overrides the model source.

### Simulation Stability

-   Avoid excessively large time steps
-   Set contact parameters reasonably to avoid penetration
-   Mass and inertia distribution should be reasonable
-   Joint limits should match actual conditions

Through proper physics environment configuration, you can create accurate and efficient simulation environments for reinforcement learning training.
