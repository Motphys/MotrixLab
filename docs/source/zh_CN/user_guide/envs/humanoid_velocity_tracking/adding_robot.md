# 新增机器人任务

`HumanoidVelocityTrackingEnv` 通过 `HumanoidVelocityTrackingEnvCfg` 读取机器人差异。接入新机器人任务时，核心工作是
定义并注册一份 `HumanoidVelocityTrackingEnvCfg`。配置完成后，新机器人会复用速度命令、动作与观察构造、步态奖励、
课程、reset 和终止逻辑。

## 1. 定义 `HumanoidVelocityTrackingEnvCfg`

在 `motrix_envs/src/motrix_envs/locomotion/humanoid/<robot>.py` 中定义平地配置函数，并用 `@registry.envcfg` 注册环境
配置 ID。先建立完整配置骨架，再填写机器人相关字段：

```python
@registry.envcfg("<robot>-walk-flat")
def make_robot_walk_flat_cfg() -> HumanoidVelocityTrackingEnvCfg:
    robot = registry.make_robot_config("<robot-id>")
    return HumanoidVelocityTrackingEnvCfg(
        # 将当前任务使用的 RobotCfg 放入场景。
        scene=StandardSceneCfg(
            objs=StandardSceneObjsCfg(
                robot=robot,
            ),
        ),
        # 设置策略动作相对默认姿态的关节位置残差缩放。
        control_config=ControlCfg(action_scale=...),
        # 设置共享奖励项的权重，以及与机器人关节一一对应的姿态权重。
        reward_config=RewardCfg(
            scales=RewardScales(...),
            pose_weights={...},
        ),
        # 将环境语义映射到模型元素名称。
        asset=AssetCfg(...),
    )
```

这里展示接入机器人任务所需的配置骨架。各子配置的覆盖方式、字段含义和调节建议见
[“配置覆盖与调参”](config_tuning.md)。

推荐将文件放在以下位置：

```text
motrix_envs/src/motrix_envs/
├── robot/
│   ├── <robot>.py                # RobotCfg，包含默认 key pose
│   └── assets/<robot>/           # 可跨任务复用的机器人资产
└── locomotion/
    └── humanoid/
        └── <robot>.py            # 平地/地形配置与环境注册

configs/task/
├── <robot>-walk-flat/
│   └── motrix.fastsac.yaml
└── <robot>-walk-terrain/
    └── motrix.fastsac.yaml
```

## 2. `scene`：配置并放置机器人

`HumanoidVelocityTrackingEnvCfg.scene` 需要一份可复用的 `HumanoidRobotCfg`。根据机器人是否已经内置选择对应方式，再将得到的
配置实例赋给 `scene.objs.robot`。

### 2.1 机器人已经内置

从[“支持的机器人”](../../robots.md#内置机器人)表格中找到 robot registry ID，然后通过 registry 创建配置实例：

```python
from motrix_env_core import registry

robot = registry.make_robot_config("g1-29dof")
```

### 2.2 需要新增机器人配置

如果“支持的机器人”中没有目标机器人，应先按照[“定义新机器人”](../../robots.md#定义新机器人)创建模型资产、声明
`HumanoidRobotCfg`、注册 robot ID，并完成独立预览和模型验证。注册完成后，通过 `registry.make_robot_config()` 获得配置实例。

将配置实例放入场景：

```python
scene = StandardSceneCfg(
    objs=StandardSceneObjsCfg(
        robot=robot,
    ),
)
```

## 3. `asset`：映射模型语义

`AssetCfg` 记录通用环境语义与 `scene` 中模型元素名称的映射：

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

| 字段                           | 配置内容                                     |
| ------------------------------ | -------------------------------------------- |
| `foot_height_site_names`       | 按左脚、右脚顺序填写的两个脚底测高 site      |
| `ground_geom_name`             | 平面或高度场对应的地面 geom                  |
| `terminate_contact_geom_names` | 与地面接触后应终止回合的机器人 geom 完整名称 |

请显式填写躯干、头部和其他不允许着地部件的最终 geom 名称，同时避免加入正常接触地面的脚部碰撞体。若
`RobotCfg` 配置了 `prefix` 或 `suffix`，这里应填写模型加入场景后的最终元素名称。

默认站立姿态必须定义在所选 `RobotCfg.key_pose.poses["default"]` 中。`key_pose.joint_names` 必须与机器人 body 上的
关节名称完全一致；缺少关节、包含未知关节、存在重复名称或姿态值非有限数时，配置或环境构造会直接报错。

建议先运行机器人预览命令检查关节顺序、默认姿态、PD 参数、脚底 site 和碰撞体名称：

```bash
python scripts/view.py robot=<robot-config-id>
```

## 4. 注册环境实现

在同一个 `motrix_envs/src/motrix_envs/locomotion/humanoid/<robot>.py` 中，将环境 ID 绑定到共享实现：

```python
registry.env("<robot>-walk-flat")(HumanoidVelocityTrackingEnv)
registry.env("<robot>-walk-terrain")(HumanoidVelocityTrackingEnv)
```

随后在 `motrix_envs/locomotion/humanoid/__init__.py` 中导入新的 `<robot>` 模块，确保 `import motrix_envs` 时配置和环境
实现都会完成注册。`registry.env(...)` 的环境 ID 必须与对应 `@registry.envcfg(...)` 的配置 ID 完全一致。若只定义平地
配置，则只注册平地环境 ID；起伏地形配置的派生方式见[“配置覆盖与调参”](config_tuning.md)。

## 5. 增加训练配置

为每个需要训练的环境 ID 创建 `configs/task/<env-id>/motrix.fastsac.yaml`。至少需要指定：

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

`algo.asynchronous: true` 是默认值，使用 Collector/Learner 异步 trainer；设为 `false` 可切换为同步执行，算法身份
仍保持 `motrix.fastsac`。其余 agent 和异步专用参数可根据机器人动力学与训练表现覆盖。

## 6. 验证

建议按以下顺序验证：

```bash
python scripts/view.py robot=<robot-config-id>
python scripts/view.py env=<robot>-walk-flat
python scripts/view.py env=<robot>-walk-terrain
python scripts/train.py task=<robot>-walk-flat/motrix.fastsac
python scripts/train.py task=<robot>-walk-flat/motrix.fastsac algo.asynchronous=false
python -m pytest motrix_envs/tests/test_humanoid_walk.py -q
python -m pytest motrix_rl/tests/test_task_configs.py -q
```

预览阶段重点检查默认姿态、脚底高度、地面碰撞、动作方向和起伏地形出生位置。新增内置环境配置时，还应在共享人形环境
测试中加入环境 ID，断言 action、actor observation 和 critic observation 的维度，并验证平地/地形配置只存在预期
差异。只定义平地配置时，可以跳过起伏地形预览。
