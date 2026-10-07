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

### Actuators on imported bodies

Declare actuators on `BodyCfg.actuators`, not on `UrdfFileCfg`. `RobotCfg` inherits the same field, and both MJCF and URDF
models use the format-neutral declarations in `motrix_env_core.config.scene.actuator`:

```python
from motrix_env_core.config.scene import BodyCfg, MjcfFileCfg
from motrix_env_core.config.scene.actuator import PositionActuatorCfg

body = BodyCfg(
    model=MjcfFileCfg(file="my_robot.xml"),
    base_link_name="base",
    prefix="robot_",
    actuators={
        "hip_position": PositionActuatorCfg(
            joint_name="hip",
            kp=100.0,
            kv=2.0,
            inherit_joint_range=True,
        ),
    },
)
```

This example assumes the imported model contains a `base` link and a limited `hip` joint. The dictionary key names the
actuator; `joint_name` uses the imported joint name **before** `prefix` / `suffix` decoration. Here, the assembled actuator
and joint names are `robot_hip_position` and `robot_hip`. Dictionary insertion order defines actuator order.

- `actuators=None` (the default) preserves the imported model's actuators.
- `actuators={}` removes all imported actuators.
- A non-empty dictionary replaces the entire imported actuator set, rather than appending to it. Declare every actuator
  needed by the body.

`PositionActuatorCfg` inherits `joint_name`, `ctrl_range`, and `force_range` from `ActuatorCfg`. `kp` is the position gain;
`kv` is the absolute damping coefficient. `inherit_joint_range=True` uses the imported joint limits as the control range
and cannot be combined with an explicit `ctrl_range`. For direct joint effort control, use `MotorActuatorCfg` from the
same module, for example `MotorActuatorCfg(joint_name="hip", ctrl_range=(-10.0, 10.0))`; it uses unit gear and gain.

Replacement or removal resets actuator control (`ctrl`) and activation (`act`, where represented) values in imported
keyframes while preserving joint positions (`qpos`) and velocities (`qvel`). MuJoCo resets both `ctrl` and `act`;
MotrixSim resets `ctrl` because MSD keyframes do not expose activation state.

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
