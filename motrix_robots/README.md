# Motrix Robot Models

`motrix-robots` provides reusable robot configurations, default poses, and model
assets for simulation.

## Simulation configuration

```python
from motrix_robots import Microduck, UnitreeGo2Robot
from motrix_env_core.config.scene.base import RobotCfg

microduck = Microduck()
robot = UnitreeGo2Robot(prefix="robot_")
robot.validate("robot")
assert isinstance(robot, RobotCfg)
assert robot.resolved_base_link_name == "robot_base"
```

Use `key_pose` for named joint poses and `init_key_pose` to select the initial pose.
Resolve prefixed model names with `robot.resolve_name(name)`. Tasks supply the scene
and choose their sensors.

## Selecting built-in models by ID

Installed robot packages are discovered automatically. Select a built-in model
by its registered ID:

```python
from motrix_env_core.registry import make_robot_config

robot = make_robot_config("go2")
```

The IDs are `anymal_c`, `g1-29dof`, `go1`, `go2`, `k1`, `dex-evt`, and `microduck`.

## Programmatic actuator overrides

`BodyCfg` (and therefore `RobotCfg`) owns format-neutral `actuators` for both MJCF
and URDF models. `None` preserves imported actuators; a dictionary replaces the
entire model instance's actuator set; `{}` removes all of them. Keys supply actuator
names, so `ActuatorCfg` does not duplicate a `name` field. Override joint names are
model-local, before prefix/suffix decoration. Other instances and source asset
files are not modified.

```python
from motrix_env_core.config.scene.actuator import MotorActuatorCfg

robot = UnitreeGo2Robot(
    actuators={
        "FL_hip_motor": MotorActuatorCfg(joint_name="FL_hip_joint", force_range=(-20.0, 20.0)),
        # Declare the other controlled joints here: this replaces, not appends.
    },
)
```

`MotorActuatorCfg` declares unit-transmission direct effort. `PositionActuatorCfg`
declares a native position servo using `kp`, absolute damping `kv`, and optional
control/force ranges. Actuator order follows dictionary insertion order.
Replacement resets keyframe actuator controls/activations while retaining joint
poses and velocities. Dex-EVT declares its drives on the robot instance.

## Model modules and assets

Import these model classes and helpers directly from `motrix_robots`:

| Defining module | Public configuration classes |
| --- | --- |
| `motrix_robots.unitree` | `UnitreeG129Dof`, `UnitreeGo1Robot`, `UnitreeGo2Robot` |
| `motrix_robots.anymal` | `AnymalC` |
| `motrix_robots.booster` | `BoosterK1` |
| `motrix_robots.dex_evt` | `DexEvt` |
| `motrix_robots.microduck` | `Microduck` |
| `motrix_robots.quadruped` | `QuadrupedLegCfg`, `QuadrupedLegsCfg`, `QuadrupedRobotCfg` |
| `motrix_robots.humanoid` | `HumanoidRobotCfg` |

Bundled model descriptions, meshes, and model textures are under `src/motrix_robots/assets/`.
Task scenes, ground textures, and terrain assets belong to `motrix_envs`. Asset licenses and attribution
remain with their asset directories; see the root `THIRD_PARTY_NOTICES.md` for the
release inventory.
