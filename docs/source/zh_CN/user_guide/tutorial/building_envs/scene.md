# SceneCfg：搭建物理场景

物理场景搭建定义了强化学习训练中的仿真参数和场景设置。
MotrixLab 使用了[MotrixSim](https://motrixsim.readthedocs.io/zh-cn/latest/user_guide/index.html)作为物理仿真后端。

## 支持的文件格式

-   [**MJCF**](https://mujoco.readthedocs.io/en/stable/XMLreference.html)(MuJoCo XML 格式) - 提供丰富的物理特性和仿真配置

## 场景文件配置

使用 `SceneCfg.file` 将完整模型文件加载为基础场景。同一个 `SceneCfg` 中声明的 asset、视觉设置和场景对象会在模型 build 前继续应用到该基础 World：

```python
from motrix_env_core import registry
from motrix_env_core.base import EnvCfg, SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg


@registry.envcfg("my-task")
@configclass
class MyTaskEnvCfg(EnvCfg):
    scene: SceneCfg = SceneCfg(file="my_model.xml")

    # 仿真与控制参数
    sim: SimCfg = SimCfg(
        dt=0.002,
        solver_iterations=3,
        solver_tolerance=1e-4,
    )
    ctrl_dt: float = 0.02
```

### 导入本体的执行器

执行器通过 `BodyCfg.actuators` 声明，不属于 `UrdfFileCfg`。`RobotCfg` 继承同一字段；MJCF 和 URDF 模型均使用
`motrix_env_core.config.scene.actuator` 中与文件格式无关的声明：

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

示例假定导入模型包含 `base` link 和具有限位的 `hip` joint。字典键定义执行器名称；`joint_name` 使用应用
`prefix` / `suffix` **之前**的导入关节名称。此处组装后的执行器和关节名称分别为 `robot_hip_position` 和
`robot_hip`。字典插入顺序决定执行器顺序。

- `actuators=None`（默认值）保留导入模型的执行器。
- `actuators={}` 移除全部导入执行器。
- 非空字典替换整个导入执行器集合，而非追加；必须声明本体需要的全部执行器。

`PositionActuatorCfg` 从 `ActuatorCfg` 继承 `joint_name`、`ctrl_range` 和 `force_range`。`kp` 是位置增益，
`kv` 是绝对阻尼系数。`inherit_joint_range=True` 将导入关节限位用作控制范围，不能与显式 `ctrl_range` 同时设置。
直接控制关节力或力矩时，使用同一模块中的 `MotorActuatorCfg`，例如
`MotorActuatorCfg(joint_name="hip", ctrl_range=(-10.0, 10.0))`；其传动比与增益均为 1。

替换或移除执行器会重置导入关键帧中的执行器控制值（`ctrl`）与 activation 状态（后端表示该状态时的 `act`），
保留关节位置（`qpos`）和速度（`qvel`）。MuJoCo 重置 `ctrl` 和 `act`；MotrixSim 重置 `ctrl`，因为 MSD 关键帧
不暴露 activation 状态。

### 推荐目录结构

```
my_environment_package/my_task/
├── __init__.py          # 模块初始化
├── cfg.py               # 环境配置
├── my_model.xml         # 物理模型文件
└── my_env.py            # 环境实现
```

对于结构复杂，引用文件较多的模型，推荐使用文件夹管理。

## 常见配置问题

### 文件路径问题

-   使用相对路径时，确保路径相对于配置文件位置
-   避免使用硬编码的绝对路径
-   检查文件权限和可访问性
-   确保所有引用的子文件都存在

### 时间步设置

-   `ctrl_dt` 应该是 `sim.dt` 的整数倍
-   `sim.dt` 过小会影响仿真性能
-   `ctrl_dt` 过大会影响控制精度
-   推荐 `sim.dt` 在 0.001-0.02 秒之间

`SimCfg` 通过 MSD `World.simulate_option` 在模型 build 前配置物理引擎。`solver_iterations`、`solver_tolerance` 与 `gravity` 为 `None` 时保留 MJCF 或 MSD World 中已有的值；显式配置时覆盖模型来源中的设置。

### 仿真稳定性

-   避免过大的时间步长
-   合理设置接触参数避免穿透
-   质量和惯性分布要合理
-   关节限制要符合实际情况

通过合理的物理环境配置，您可以为强化学习训练创建准确且高效的仿真环境。
