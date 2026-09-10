# Add a Robot Task

`QuadrupedWalkTask` reads robot and scene differences from `QuadrupedWalkEnvCfg`. Adding a robot task consists of
defining its quadruped semantics in `QuadrupedRobotCfg`, providing the sensors required by the shared task, and
registering a flat-ground environment config. The robot then reuses velocity commands, action and observation
construction, diagonal-trot rewards, reset, and termination.

Use the following layout:

```text
motrix_envs/src/motrix_envs/
├── robot/
│   ├── <robot>.py                # QuadrupedRobotCfg and default key pose
│   └── assets/<robot>/           # Robot assets reusable across tasks
└── locomotion/
    └── quadruped/
        └── <robot>.py            # Flat/rough configs and environment registration

configs/task/
├── <robot>-walk-flat/
│   ├── rslrl.ppo.yaml
│   └── skrl.ppo.yaml
└── <robot>-walk-rough/
    ├── rslrl.ppo.yaml
    └── skrl.ppo.yaml
```

## 1. Prepare a `QuadrupedRobotCfg`

If the robot appears in the [built-in robots](../../robots.md#built-in-robots) table, create its config through the
registry:

```python
from motrix_env_core import registry

robot = registry.make_robot_config("go2")
```

Otherwise, follow [Define a new robot](../../robots.md#define-a-new-robot) to add its assets and public `RobotCfg`, then
subclass `QuadrupedRobotCfg`. In addition to the model, base link, and default key pose, declare one contact geom for each
foot in front-left, front-right, rear-left, rear-right order:

```python
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg
from motrix_envs.robot import QuadrupedLegCfg, QuadrupedLegsCfg, QuadrupedRobotCfg


@configclass(kw_only=True)
class MyQuadruped(QuadrupedRobotCfg):
    model: MjcfFileCfg = MjcfFileCfg(file=MY_ROBOT_ASSET_DIR / "robot.xml")
    base_link_name: str = "base"
    legs: QuadrupedLegsCfg = QuadrupedLegsCfg(
        front_left=QuadrupedLegCfg(contact_geom_name="FL_foot"),
        front_right=QuadrupedLegCfg(contact_geom_name="FR_foot"),
        rear_left=QuadrupedLegCfg(contact_geom_name="RL_foot"),
        rear_right=QuadrupedLegCfg(contact_geom_name="RR_foot"),
    )
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[...],
        poses={"default": [...]},
    )
```

The four `contact_geom_name` values must be non-empty and unique. `key_pose.poses["default"]` should be a stable standing
pose. At task construction, its joint names must exactly match every actuator's joint target; missing or extra joints are
rejected. Every actuator must target a joint and expose a valid control range.

After registering the robot config, preview it independently:

```bash
python scripts/view.py robot=<robot-config-id>
```

## 2. Provide task sensors

The shared environment requires these states:

| State                                           | Count and order                                                                 | Purpose                                                 |
| ----------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Body-frame linear velocity, gyro, and up-vector | One three-dimensional sensor each                                               | Observations, velocity tracking, and tip-over detection |
| Foot position                                   | One three-dimensional sensor for front-left, front-right, rear-left, rear-right | Swing-foot height in a body reference frame             |
| Foot-ground contact                             | One sensor for front-left, front-right, rear-left, rear-right                   | Stance and swing contact rewards                        |

`QuadrupedSceneCfg` reads `QuadrupedRobotCfg.legs` and automatically creates `front_left_contact`,
`front_right_contact`, `rear_left_contact`, and `rear_right_contact`. Do not duplicate this contact logic in the
environment implementation. Built-in configs use a ground geom named `floor`.

When the MJCF already provides the remaining sensors, map their assembled names with `Sensor`:

```python
sensor = Sensor(
    local_linvel="local_linvel",
    gyro="gyro",
    upvector="upvector",
    foot_positions=("FL_pos", "FR_pos", "RL_pos", "RR_pos"),
)
```

When an asset lacks a required state, follow `AnymalCWalkSensorsCfg`: subclass `QuadrupedTaskSensorsCfg` and add
`FrameSensorCfg` entries in the scene config. Foot `framepos` sensors must use a body or IMU site as their reference so
that z represents foot height relative to the body.

## 3. Define the flat-ground environment config

Register one complete flat-ground config in `locomotion/quadruped/<robot>.py`:

```python
from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import (
    ControlConfig,
    QuadrupedSceneCfg,
    QuadrupedWalkEnvCfg,
    RewardConfig,
    Sensor,
)


@registry.envcfg("<robot>-walk-flat")
@configclass
class RobotWalkCfg(QuadrupedWalkEnvCfg):
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(material="mat_ground"),
            robot=registry.make_robot_config("<robot-config-id>"),
        ),
    )
    control_config: ControlConfig = ControlConfig(action_scale=...)
    sensor: Sensor = Sensor(...)
    reward_config: RewardConfig = RewardConfig(...)
    initial_base_position: tuple[float, float, float] = (0.0, 0.0, ...)
    spawn_xy_range: float = 4.0
```

Set `initial_base_position[2]` and `reward_config.base_height_target` from the default standing pose before tuning action
scale, target foot height, and reward weights. See [Config Overrides and Tuning](config_tuning.md) for field semantics and
coupled adjustments.

## 4. Derive the rough-terrain config

A rough-terrain config should inherit the flat config and override only `scene`:

```python
@registry.envcfg("<robot>-walk-rough")
@configclass
class RobotWalkRoughCfg(RobotWalkCfg):
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        assets=QuadrupedWalkTerrainSceneAssetsCfg(),
        objs=StandardSceneObjsCfg(
            floor=HFieldTerrainCfg(hfield="terrain", material="mat_ground"),
            robot=registry.make_robot_config("<robot-config-id>"),
        ),
    )
```

The built-in height field uses seed `0`, a `32 m × 32 m` area, a `320 × 320` grid, and a `0.1 m` height scale. A custom
height field should retain the `floor` ground geom and verify that both spawn height and body-height rewards query terrain
correctly.

## 5. Register the shared environment implementation

Bind each environment ID to `QuadrupedWalkTask` in the same module:

```python
registry.env("<robot>-walk-flat")(QuadrupedWalkTask)
registry.env("<robot>-walk-rough")(QuadrupedWalkTask)
```

Import the module from `motrix_envs/locomotion/quadruped/__init__.py`, so `import motrix_envs` performs registration.
Each `registry.env(...)` ID must exactly match its `@registry.envcfg(...)` ID. Register only the flat ID when no rough
config is provided.

## 6. Add training configs

Create `configs/task/<env-id>/` for every trainable environment. This is the minimal task-selection portion for RSL-RL PPO:

```yaml
defaults:
    - /algo_base@algo: rslrl.ppo
    - _self_
task:
    env: <robot>-walk-flat
    rllib: rslrl
    algo: ppo
num_envs: 1024
play_num_envs: 16
```

An SKRL config selects `/algo_base@algo: skrl.ppo` and sets `task.rllib` to `skrl`. A rough-terrain training config can
inherit the same robot's flat-ground recipe and override only `task.env`.

## 7. Validate

Use this progression:

```bash
python scripts/view.py robot=<robot-config-id>
python scripts/view.py env=<robot>-walk-flat
python scripts/view.py env=<robot>-walk-rough
python scripts/train.py task=<robot>-walk-flat/rslrl.ppo
```

During preview, inspect the default pose, action direction, all four foot contacts, foot-position reference frames, and
rough-terrain spawn height.
