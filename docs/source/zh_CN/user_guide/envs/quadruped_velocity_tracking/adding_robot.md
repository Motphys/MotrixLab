# 新增机器人任务

`QuadrupedWalkTask` 通过 `QuadrupedWalkEnvCfg` 读取机器人与场景差异。接入新机器人任务时，核心工作是定义
`QuadrupedRobotCfg` 的四足语义，提供共享任务所需的传感器，再注册一份平地环境配置。完成后，新机器人会复用速度
命令、动作与观察构造、对角小跑奖励、reset 和终止逻辑。

推荐使用以下目录布局：

```text
motrix_envs/src/motrix_envs/
├── robot/
│   ├── <robot>.py                # QuadrupedRobotCfg 与默认关键姿态
│   └── assets/<robot>/           # 可跨任务复用的机器人资产
└── locomotion/
    └── quadruped/
        └── <robot>.py            # 平地/粗糙地形配置与环境注册

configs/task/
├── <robot>-walk-flat/
│   ├── rslrl.ppo.yaml
│   └── skrl.ppo.yaml
└── <robot>-walk-rough/
    ├── rslrl.ppo.yaml
    └── skrl.ppo.yaml
```

## 1. 准备 `QuadrupedRobotCfg`

如果机器人已经列在[“内置机器人”](../../robots.md#内置机器人)中，可以直接通过 registry 创建配置：

```python
from motrix_env_core import registry

robot = registry.make_robot_config("go2")
```

否则，先按照[“定义新机器人”](../../robots.md#定义新机器人)准备资产和公共 `RobotCfg`，并让配置继承
`QuadrupedRobotCfg`。除模型、基座 link 和默认关键姿态外，还必须按前左、前右、后左、后右顺序声明四只脚用于接触
检测的 geom：

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

四个 `contact_geom_name` 必须非空且互不相同。`key_pose.poses["default"]` 应给出稳定站立姿态；任务构造时，
关键姿态的 joint 名称必须与所有 actuator 的 joint target 完全一致，缺少或多出关节都会报错。每个 actuator 还必须
以 joint 为目标并提供有效控制范围。

完成 robot registry 注册后，先独立预览模型：

```bash
python scripts/view.py robot=<robot-config-id>
```

## 2. 提供任务传感器

共享环境需要以下状态：

| 状态                              | 数量与顺序                               | 用途                     |
| --------------------------------- | ---------------------------------------- | ------------------------ |
| 机体局部线速度、陀螺仪、Up-vector | 各一个 3 维 sensor                       | 观察、速度跟踪和倾倒检测 |
| 足端位置                          | 前左、前右、后左、后右各一个 3 维 sensor | 机体参考系中的摆动脚高度 |
| 足地接触                          | 前左、前右、后左、后右各一个             | 支撑与摆动接触奖励       |

`QuadrupedSceneCfg` 会读取 `QuadrupedRobotCfg.legs`，自动创建名为 `front_left_contact`、
`front_right_contact`、`rear_left_contact` 和 `rear_right_contact` 的接触 sensor，因此不要在环境实现中重复定义
接触逻辑。内置配置使用名为 `floor` 的地面 geom。

若 MJCF 已经提供其余 sensor，只需用 `Sensor` 映射组装后的名称：

```python
sensor = Sensor(
    local_linvel="local_linvel",
    gyro="gyro",
    upvector="upvector",
    foot_positions=("FL_pos", "FR_pos", "RL_pos", "RR_pos"),
)
```

若资产缺少某个状态，应像 `AnymalCWalkSensorsCfg` 一样继承 `QuadrupedTaskSensorsCfg`，在场景配置中加入
`FrameSensorCfg`。足端 `framepos` 必须以机体或 IMU site 为参考，使其 z 分量能表示相对机体的足端高度。

## 3. 定义平地环境配置

在 `locomotion/quadruped/<robot>.py` 中注册一份完整的平地配置：

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

先根据默认站立姿态填写 `initial_base_position[2]` 和 `reward_config.base_height_target`，再选择动作缩放、足端
目标高度与奖励权重。字段含义与调节关系见[“配置覆盖与调参”](config_tuning.md)。

## 4. 派生粗糙地形配置

粗糙地形配置应继承平地配置并只覆盖 `scene`：

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

内置高度场使用固定 seed `0`、`32 m × 32 m` 尺寸、`320 × 320` 采样点和 `0.1 m` 高度尺度。若使用自定义
高度场，应继续保留 `floor` 地面 geom，并验证出生高度和机体高度奖励都能正确查询地形。

## 5. 注册共享环境实现

在同一个模块中将 Env ID 绑定到 `QuadrupedWalkTask`：

```python
registry.env("<robot>-walk-flat")(QuadrupedWalkTask)
registry.env("<robot>-walk-rough")(QuadrupedWalkTask)
```

随后从 `motrix_envs/locomotion/quadruped/__init__.py` 导入新模块，确保 `import motrix_envs` 时完成注册。
`registry.env(...)` 与 `@registry.envcfg(...)` 的 ID 必须完全一致；只提供平地配置时，只注册平地 Env ID。

## 6. 增加训练配置

为每个需要训练的 Env ID 建立 `configs/task/<env-id>/`。以下是 RSL-RL PPO 的最小任务选择部分：

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

SKRL 配置选择 `/algo_base@algo: skrl.ppo`，并将 `task.rllib` 设为 `skrl`。粗糙地形配置可以继承同一机器人的
平地训练配置，只覆盖 `task.env`。

## 7. 验证

建议按以下顺序验证：

```bash
python scripts/view.py robot=<robot-config-id>
python scripts/view.py env=<robot>-walk-flat
python scripts/view.py env=<robot>-walk-rough
python scripts/train.py task=<robot>-walk-flat/rslrl.ppo
```

预览时重点检查默认姿态、动作方向、四脚接触、足端位置参考系和粗糙地形出生高度。
