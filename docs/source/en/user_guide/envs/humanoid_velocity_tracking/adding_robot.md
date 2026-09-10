# Add a Robot Task

`HumanoidVelocityTrackingEnv` reads robot-specific differences from `HumanoidVelocityTrackingEnvCfg`. Adding a robot
task consists primarily of defining and registering a `HumanoidVelocityTrackingEnvCfg`. The robot then reuses the shared
velocity commands, action and observation construction, gait rewards, curriculum, reset, and termination logic.

## 1. Define `HumanoidVelocityTrackingEnvCfg`

Define the flat-ground config factory in `motrix_envs/src/motrix_envs/locomotion/humanoid/<robot>.py` and register its
environment config ID with `@registry.envcfg`:

```python
@registry.envcfg("<robot>-walk-flat")
def make_robot_walk_flat_cfg() -> HumanoidVelocityTrackingEnvCfg:
    robot = registry.make_robot_config("<robot-id>")
    return HumanoidVelocityTrackingEnvCfg(
        # Place the RobotCfg used by this task in the scene.
        scene=StandardSceneCfg(
            objs=StandardSceneObjsCfg(
                robot=robot,
            ),
        ),
        # Scale joint-position residuals around the default pose.
        control_config=ControlCfg(action_scale=...),
        # Set shared reward scales and one pose weight for every joint.
        reward_config=RewardCfg(
            scales=RewardScales(...),
            pose_weights={...},
        ),
        # Map environment semantics to model element names.
        asset=AssetCfg(...),
    )
```

This skeleton contains the fields needed to add a robot task. See [Config Overrides and Tuning](config_tuning.md) for
nested config overrides, field details, and tuning guidance.

Use the following layout:

```text
motrix_envs/src/motrix_envs/
├── robot/
│   ├── <robot>.py                # RobotCfg, including its default key pose
│   └── assets/<robot>/           # Reusable robot assets
└── locomotion/
    └── humanoid/
        └── <robot>.py            # Flat/terrain configs and environment registration

configs/task/
├── <robot>-walk-flat/
│   └── motrix.fastsac.yaml
└── <robot>-walk-terrain/
    └── motrix.fastsac.yaml
```

## 2. `scene`: configure and place the robot

`HumanoidVelocityTrackingEnvCfg.scene` requires a reusable `HumanoidRobotCfg`. Select the appropriate path below, then assign
the resulting config instance to `scene.objs.robot`.

### 2.1 The robot is built in

Find the robot registry ID in the [built-in robots](../../robots.md#built-in-robots) table and create its config through
the registry:

```python
from motrix_env_core import registry

robot = registry.make_robot_config("g1-29dof")
```

### 2.2 A new robot config is required

If the target robot is absent from the supported-robot table, follow [Define a new robot](../../robots.md#define-a-new-robot)
to add its assets, declare and register its `HumanoidRobotCfg`, and validate it in the standalone viewer. Then obtain a config
instance through `registry.make_robot_config()`.

Place the config instance in the scene:

```python
scene = StandardSceneCfg(
    objs=StandardSceneObjsCfg(
        robot=robot,
    ),
)
```

## 3. `asset`: map model semantics

`AssetCfg` maps generic environment semantics to model element names in the scene:

```python
asset = AssetCfg(
    foot_height_site_names=("<left-sole-site>", "<right-sole-site>"),
    ground_geom_name="floor",
    terminate_contact_geom_names=(
        "<pelvis-geom>",
        "<torso-geom>",
        "<head-geom>",
    ),
)
```

| Field                          | Value                                                                |
| ------------------------------ | -------------------------------------------------------------------- |
| `foot_height_site_names`       | Left and right sole-height sites, in that order                      |
| `ground_geom_name`             | Ground geom for flat or height-field terrain                         |
| `terminate_contact_geom_names` | Explicit robot geom names whose ground contact terminates an episode |

List the final model geom names for torso, head, and other disallowed contacts while avoiding foot geoms that normally
contact the ground. When `RobotCfg` defines a `prefix` or `suffix`, use the final element names after the model is inserted
into the scene.

Define the default standing pose in `RobotCfg.key_pose.poses["default"]`. Its `key_pose.joint_names` must exactly match
the joint names on the robot body. Missing, unknown, or duplicate joints and non-finite pose values fail config or
environment construction.

Preview the robot and verify joint ordering, the default pose, PD parameters, sole sites, and collision-geom names:

```bash
python scripts/view.py robot=<robot-config-id>
```

## 4. Register the environment implementation

Bind each environment ID to the shared implementation in the same
`motrix_envs/src/motrix_envs/locomotion/humanoid/<robot>.py` module:

```python
registry.env("<robot>-walk-flat")(HumanoidVelocityTrackingEnv)
registry.env("<robot>-walk-terrain")(HumanoidVelocityTrackingEnv)
```

Import the new `<robot>` module from `motrix_envs/locomotion/humanoid/__init__.py`, so `import motrix_envs` performs both
config and environment registration. Each `registry.env(...)` ID must exactly match its `@registry.envcfg(...)` ID. If
the task only has a flat-ground config, register only that ID. See
[Config Overrides and Tuning](config_tuning.md) for deriving a terrain config.

## 5. Add training configs

Create `configs/task/<env-id>/motrix.fastsac.yaml` for every trainable environment. The config minimally selects the shared
algorithm and environment:

```yaml
defaults:
    - /algo_base@algo: motrix.fastsac
    - _self_
task:
    env: <robot>-walk-flat
    rllib: motrix
    algo: fastsac
num_envs: 2048
play_num_envs: 16
algo:
    asynchronous: true
```

`algo.asynchronous: true` is the default and uses the collector/learner trainer. Set it to `false` for synchronous execution
without changing the `motrix.fastsac` algorithm identity. Override agent and async-only parameters as required by the robot
dynamics and training behavior.

## 6. Validate

Use this progression:

```bash
python scripts/view.py robot=<robot-config-id>
python scripts/view.py env=<robot>-walk-flat
python scripts/view.py env=<robot>-walk-terrain
python scripts/train.py task=<robot>-walk-flat/motrix.fastsac
python scripts/train.py task=<robot>-walk-flat/motrix.fastsac algo.asynchronous=false
python -m pytest motrix_envs/tests/test_humanoid_walk.py -q
python -m pytest motrix_rl/tests/test_task_configs.py -q
```

During preview, inspect the default pose, sole height, ground contacts, action direction, and terrain spawn position. For
a new built-in environment config, add its ID to the shared humanoid tests, assert the action, actor-observation, and
critic-observation dimensions, and verify that flat and terrain configs differ only where intended. Skip the terrain
preview when the task only defines a flat-ground config.
