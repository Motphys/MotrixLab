# 支持的机器人

`motrix_robots` 提供可复用的机器人模型，可以独立预览，也可以用于 `motrix_envs` 中的任务。

## 内置机器人

<!-- ROBOT_TABLE_START -->

<!-- This table is generated; do not edit this block manually. -->
| 截图 | Registry 名称 | 配置类 | 类型 | 模型格式 | 自由度 |
| --- | --- | --- | --- | --- | --- |
| <img src="../_static/images/robots/anymal_c.png" alt="anymal_c" width="180"> | `anymal_c` | `AnymalC` | 四足机器人 | MJCF | 12 |
| <img src="../_static/images/robots/dex-evt.png" alt="dex-evt" width="180"> | `dex-evt` | `DexEvt` | 人形机器人 | URDF | 23 |
| <img src="../_static/images/robots/g1-29dof.png" alt="g1-29dof" width="180"> | `g1-29dof` | `UnitreeG129Dof` | 人形机器人 | MJCF | 29 |
| <img src="../_static/images/robots/go1.png" alt="go1" width="180"> | `go1` | `UnitreeGo1Robot` | 四足机器人 | MJCF | 12 |
| <img src="../_static/images/robots/go2.png" alt="go2" width="180"> | `go2` | `UnitreeGo2Robot` | 四足机器人 | MJCF | 12 |
| <img src="../_static/images/robots/k1.png" alt="k1" width="180"> | `k1` | `BoosterK1` | 人形机器人 | MJCF | 22 |
| <img src="../_static/images/robots/microduck.png" alt="microduck" width="180"> | `microduck` | `Microduck` | 人形机器人 | MJCF | 14 |

<!-- ROBOT_TABLE_END -->

## 独立预览

使用 `view.py` 可以在默认姿态下查看已注册 robot，无需创建 RL environment：

```bash
python scripts/view.py robot=go2
```

robot 模式会构建一个静态标准场景，不采样 action，也不执行 physics rollout。

## Python API

从 `motrix_robots` 直接导入可复用配置：

```python
from motrix_robots import Microduck, UnitreeGo2Robot

microduck = Microduck()
go2 = UnitreeGo2Robot(prefix="robot_")
```

也可以按 registry 名称选择机器人，并放入标准场景：

```python
from motrix_env_core import registry
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg

print(registry.list_registered_robots())
robot = registry.make_robot_config("go2")
scene = StandardSceneCfg(objs=StandardSceneObjsCfg(robot=robot))
```

## 定义新机器人

新的可复用机器人应定义为 `RobotCfg`，只描述机器人本体及其实例化方式。机器人资产、执行器、仿真所需的 site 和通用
默认姿态属于 `RobotCfg`；地面、灯光、任务 marker 和由任务创建的 sensor 属于 `SceneCfg` 或任务配置。

### 1. 准备机器人资产

内置机器人推荐使用以下目录结构：

```text
motrix_robots/src/motrix_robots/
├── my_robot.py
└── assets/my_robot/
    ├── my_robot.xml              # 或 my_robot.urdf
    └── meshes/
```

MJCF 文件应只包含可跨场景复用的机器人模型，不要把地面、灯光或任务专属物体放进机器人文件。若使用 URDF，可以通过
`UrdfFileCfg` 的 `geoms`、`sites` 和 `joints` 补充 URDF 本身不包含的仿真属性。
MJCF 和 URDF 的执行器均通过机器人实例继承的 `BodyCfg.actuators` 字典声明。
从 `motrix_env_core.config.scene.actuator` 导入 `ActuatorCfg`、`PositionActuatorCfg` 和 `MotorActuatorCfg`。

### 2. 声明 `RobotCfg`

在 `motrix_robots/src/motrix_robots/my_robot.py` 中使用 `@configclass(kw_only=True)` 定义配置类。下面是一份最小的 MJCF 示例：

```python
from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg, RobotCfg

MY_ROBOT_ASSET_DIR = Path(__file__).parent / "assets" / "my_robot"


@configclass(kw_only=True)
class MyRobot(RobotCfg):
    # 加载只包含机器人本体的模型文件。
    model: MjcfFileCfg = MjcfFileCfg(file=MY_ROBOT_ASSET_DIR / "my_robot.xml")
    # 指定模型连接到场景时使用的机器人根链接。
    base_link_name: str = "base"
    # 定义跨任务复用的有名关节姿态；locomotion 任务通常要求提供 "default"。
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=["left_hip", "left_knee", "right_hip", "right_knee"],
        poses={
            "default": [0.0, 0.5, 0.0, 0.5],
        },
    )
```

`RobotCfg` 的公共字段如下：

| 字段                       | 说明                                                     |
| -------------------------- | -------------------------------------------------------- |
| `model`                    | `MjcfFileCfg` 或 `UrdfFileCfg`，负责加载可复用机器人模型 |
| `base_link_name`           | 模型连接到场景时使用的根链接名称                         |
| `translation` / `rotation` | 可选的默认实例位姿；任务配置可以在实例化时覆盖           |
| `prefix` / `suffix`        | 可选的模型元素名称前后缀，用于同一场景中的名称隔离       |
| `actuators`                | `None` 保留导入执行器；`{}` 移除全部执行器；非空字典替换全部执行器 |
| `key_pose`                 | 共享一套明确关节顺序的有名姿态，例如 `default`、`crouch` |

`KeyPoseCfg.joint_names` 必须唯一且非空；每个姿态的数值数量必须与关节数量一致，所有值必须为有限数。这里只保存机器人
自身的关节姿态，不应保存地面名称、任务 sensor 名称或其他场景语义。

人形机器人应继承 `HumanoidRobotCfg`，并定义左右脚 link：

```python
from motrix_robots import HumanoidRobotCfg


@configclass(kw_only=True)
class MyHumanoid(HumanoidRobotCfg):
    model: MjcfFileCfg = MjcfFileCfg(file=MY_ROBOT_ASSET_DIR / "my_humanoid.xml")
    base_link_name: str = "pelvis"
    left_foot_link_name: str = "left_foot"
    right_foot_link_name: str = "right_foot"
```

两个脚部名称必须非空且互不相同。`resolved_foot_link_names` 会自动应用机器人实例的 `prefix` 和 `suffix`。

URDF 机器人沿用相同的配置层次，并将 `model` 换成 `UrdfFileCfg`。site 仍属于模型配置，执行器属于 `RobotCfg`。
下面的最小示例假定模型包含 `base` 和 `left_foot` link，以及具有限位的 `left_hip` joint：

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

字典键定义执行器名称；`joint_name` 使用应用实例 `prefix` / `suffix` 之前的导入名称。
字典替换而非追加导入执行器集合，因此需声明全部受控关节。省略 `actuators` 或设为 `None` 可保留导入执行器，
`{}` 则移除全部执行器。

`PositionActuatorCfg` 使用位置增益 `kp` 和绝对阻尼系数 `kv`（默认 `0.0`），并从 `ActuatorCfg` 继承
`joint_name`、`ctrl_range`（控制输入限幅）和 `force_range`（执行器输出限幅）。
设置 `inherit_joint_range=True` 可将导入关节限位用作控制范围，不能与显式 `ctrl_range` 同时设置。
直接控制关节力或力矩时，使用传动比与增益均为 1 的 `MotorActuatorCfg`。

### 3. 注册机器人

内置机器人在 `motrix_robots/src/motrix_robots/registration.py` 的 `register()` 回调中注册。
从新类的定义模块导入，并将注册语句加入回调：

```python
from motrix_env_core import registry
from motrix_robots.my_robot import MyRobot


def register() -> None:
    # 保留其他内置模型的注册语句。
    registry.robotcfg("my-robot")(MyRobot)
```

已安装的机器人 package 会自动发现，无需手动导入注册模块。
若使用独立 package，在自己的注册模块中采用同样的 `register()` 回调并导入自己的模型，再在该 package 的
`pyproject.toml` 中声明：

```toml
[project.entry-points."motrix_env_core.robots"]
my_robot = "my_robot_package.registration:register"
```

### 4. 验证

先通过 registry 构造配置并预览机器人：

```bash
python scripts/view.py robot=my-robot
python -m pytest motrix_envs/tests/test_robot_cfg.py -q
```

至少应验证模型能够构建、`base_link_name` 存在、关节与 actuator 对应、默认 key pose 完整，以及任务需要的碰撞体和 site
可以按名称找到。随后再将 `MyRobot()` 组合进具体任务的 `SceneCfg`；例如人形速度跟踪任务的模型语义和奖励配置应继续
按照[“新增机器人任务”](envs/humanoid_velocity_tracking/adding_robot.md)页面定义，并保留在任务配置中。

若新机器人需要出现在上方自动生成的内置机器人表格中，还需要在 `docs/scripts/generate_robot_docs.py` 的
`_ROBOT_METADATA` 中补充类型和截图参数，然后生成截图与表格：

```bash
python docs/scripts/generate_robot_docs.py --screenshots my-robot
python docs/scripts/generate_robot_docs.py --check
```
