# Supported Robots

`motrix_robots` provides reusable robot models that you can preview independently or use in tasks from `motrix_envs`.

## Built-in robots

<!-- ROBOT_TABLE_START -->

<!-- This table is generated; do not edit this block manually. -->
| Screenshot | Registry name | Configuration class | Type | Model format | DoF |
| --- | --- | --- | --- | --- | --- |
| <img src="../_static/images/robots/anymal_c.png" alt="anymal_c" width="180"> | `anymal_c` | `AnymalC` | Quadruped | MJCF | 12 |
| <img src="../_static/images/robots/dex-evt.png" alt="dex-evt" width="180"> | `dex-evt` | `DexEvt` | Humanoid | URDF | 23 |
| <img src="../_static/images/robots/g1-29dof.png" alt="g1-29dof" width="180"> | `g1-29dof` | `UnitreeG129Dof` | Humanoid | MJCF | 29 |
| <img src="../_static/images/robots/go1.png" alt="go1" width="180"> | `go1` | `UnitreeGo1Robot` | Quadruped | MJCF | 12 |
| <img src="../_static/images/robots/go2.png" alt="go2" width="180"> | `go2` | `UnitreeGo2Robot` | Quadruped | MJCF | 12 |
| <img src="../_static/images/robots/k1.png" alt="k1" width="180"> | `k1` | `BoosterK1` | Humanoid | MJCF | 22 |
| <img src="../_static/images/robots/microduck.png" alt="microduck" width="180"> | `microduck` | `Microduck` | Humanoid | MJCF | 14 |

<!-- ROBOT_TABLE_END -->

## Preview a robot

Use `view.py` to inspect a registered robot in its default pose without creating an RL environment:

```bash
python scripts/view.py robot=go2
```

Robot view mode builds a static standard scene. It does not sample actions or run a physics rollout.

## Python API

Import reusable configurations directly from `motrix_robots`:

```python
from motrix_robots import Microduck, UnitreeGo2Robot

microduck = Microduck()
go2 = UnitreeGo2Robot(prefix="robot_")
```

You can also select a robot by its registry name and place it in a standard scene:

```python
from motrix_env_core import registry
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg

print(registry.list_registered_robots())
robot = registry.make_robot_config("go2")
scene = StandardSceneCfg(objs=StandardSceneObjsCfg(robot=robot))
```

## Define a new robot

A reusable robot should be defined as a `RobotCfg` that describes only the robot model and how it is instantiated.
Robot assets, actuators, simulation sites, and reusable key poses belong to `RobotCfg`; the ground, lights, task
markers, and task-created sensors belong to `SceneCfg` or the task configuration.

### 1. Prepare the robot assets

Use the following layout for a built-in robot:

```text
motrix_robots/src/motrix_robots/
├── my_robot.py
└── assets/my_robot/
    ├── my_robot.xml              # or my_robot.urdf
    └── meshes/
```

An MJCF file should contain only the reusable robot model, without a floor, lights, or task-specific objects. For a
URDF robot, `UrdfFileCfg` can add simulation properties through its `geoms`, `sites`, and `joints` fields.
Declare MJCF and URDF actuators on the robot instance through its inherited `BodyCfg.actuators` dictionary.
Import `ActuatorCfg`, `PositionActuatorCfg`, and `MotorActuatorCfg` from `motrix_env_core.config.scene.actuator`.

### 2. Declare the `RobotCfg`

Define the configuration in `motrix_robots/src/motrix_robots/my_robot.py` with `@configclass(kw_only=True)`. This is a minimal MJCF example:

```python
from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg, RobotCfg

MY_ROBOT_ASSET_DIR = Path(__file__).parent / "assets" / "my_robot"


@configclass(kw_only=True)
class MyRobot(RobotCfg):
    # Load a model file that contains only the robot.
    model: MjcfFileCfg = MjcfFileCfg(file=MY_ROBOT_ASSET_DIR / "my_robot.xml")
    # Select the robot root link used when attaching the model to a scene.
    base_link_name: str = "base"
    # Define named joint poses shared across tasks; locomotion tasks commonly require "default".
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=["left_hip", "left_knee", "right_hip", "right_knee"],
        poses={
            "default": [0.0, 0.5, 0.0, 0.5],
        },
    )
```

The public `RobotCfg` fields are:

| Field                      | Description                                                                    |
| -------------------------- | ------------------------------------------------------------------------------ |
| `model`                    | An `MjcfFileCfg` or `UrdfFileCfg` that loads the reusable robot model          |
| `base_link_name`           | Root link used to attach the model to a scene                                  |
| `translation` / `rotation` | Optional default instance pose; task configs can override it                   |
| `prefix` / `suffix`        | Optional element-name affixes for name isolation within a scene                |
| `actuators`                | `None` preserves imported actuators; `{}` removes them; a non-empty dictionary replaces them |
| `key_pose`                 | Named poses that share one explicit joint order, such as `default` or `crouch` |

`KeyPoseCfg.joint_names` must be unique and non-empty. Every pose must contain one finite value per joint. Keep only
robot joint poses here; ground names, task sensor names, and other scene semantics do not belong to `RobotCfg`.

A humanoid robot should inherit `HumanoidRobotCfg` and define its left and right foot links:

```python
from motrix_robots import HumanoidRobotCfg


@configclass(kw_only=True)
class MyHumanoid(HumanoidRobotCfg):
    model: MjcfFileCfg = MjcfFileCfg(file=MY_ROBOT_ASSET_DIR / "my_humanoid.xml")
    base_link_name: str = "pelvis"
    left_foot_link_name: str = "left_foot"
    right_foot_link_name: str = "right_foot"
```

Both foot-link names must be non-empty and distinct. `resolved_foot_link_names` automatically applies the robot
instance's `prefix` and `suffix`.

A URDF robot uses the same config hierarchy and replaces `model` with `UrdfFileCfg`. Sites remain on the model config;
actuators belong to `RobotCfg`. This minimal example assumes `base` and `left_foot` links and a limited `left_hip` joint:

```python
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.config.scene.actuator import PositionActuatorCfg
from motrix_env_core.config.scene.urdf import SiteCfg, UrdfFileCfg

robot = RobotCfg(
    model=UrdfFileCfg(
        file="my_robot.urdf",
        sites=[SiteCfg(name="left_sole", parent_link_name="left_foot")],
    ),
    base_link_name="base",
    actuators={
        "left_hip_position": PositionActuatorCfg(
            joint_name="left_hip",
            kp=100.0,
            kv=2.0,
            inherit_joint_range=True,
        ),
    },
)
```

Dictionary keys name actuators; `joint_name` uses imported names before instance `prefix` / `suffix` decoration.
Declare all controlled joints in the dictionary: it replaces rather than appends to the imported actuator set.
Omit `actuators` or set it to `None` to preserve imported actuators; use `{}` to remove them.

`PositionActuatorCfg` uses position gain `kp` and absolute damping coefficient `kv` (default `0.0`).
It inherits `joint_name`, `ctrl_range` (control input limits), and `force_range` (actuator output limits) from `ActuatorCfg`.
Set `inherit_joint_range=True` to use imported joint limits as the control range; this cannot be combined with an explicit
`ctrl_range`. For direct joint effort control with unit transmission and gain, use `MotorActuatorCfg`.

### 3. Register the robot

Register a built-in robot inside the `register()` callback in
`motrix_robots/src/motrix_robots/registration.py`. Import the new class from its defining module
and add its registration to the callback:

```python
from motrix_env_core import registry
from motrix_robots.my_robot import MyRobot


def register() -> None:
    # Keep the other built-in model registrations here.
    registry.robotcfg("my-robot")(MyRobot)
```

Installed robot packages are discovered automatically; no manual registration import is needed.
For a separate package, define the same `register()` callback using your own model import and declare it in that
package's `pyproject.toml`:

```toml
[project.entry-points."motrix_env_core.robots"]
my_robot = "my_robot_package.registration:register"
```

### 4. Validate

Construct the config through the registry and preview it before using it in a task:

```bash
python scripts/view.py robot=my-robot
python -m pytest motrix_envs/tests/test_robot_cfg.py -q
```

At minimum, verify that the model builds, `base_link_name` exists, joints and actuators correspond, the default key
pose is complete, and task-required collision geoms and sites are addressable by name. Then compose `MyRobot()` into a
task-specific `SceneCfg`; follow the velocity-tracking guide for [adding a robot task](envs/humanoid_velocity_tracking/adding_robot.md),
and keep task model semantics and rewards out of the reusable `RobotCfg`.

To include the robot in the generated table above, add its type and screenshot parameters to `_ROBOT_METADATA` in
`docs/scripts/generate_robot_docs.py`, then generate the screenshot and table:

```bash
python docs/scripts/generate_robot_docs.py --screenshots my-robot
python docs/scripts/generate_robot_docs.py --check
```
