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
