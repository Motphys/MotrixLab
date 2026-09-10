# 新增 WBT 训练任务

`ManagerEnv` 的复用单元是一份完整的 `WbtManagerEnvCfg`。为现有机器人增加一段动作时，通常只需新增 motion 文件、环境配置
factory、Env 注册和对应的 Hydra Training Task，不需要复制环境实现。本章以 G1 的 `dance1_subject1.npz` 为例，
使用 Env ID `g1-wbt-dance1-subject1`。

## 1. 定义完整环境配置

从目标机器人的 WBT 配置子类构造顶层配置。编辑
`motrix_envs/src/motrix_envs/locomotion/wbt/g1.py`：

```python
from pathlib import Path

from motrix_env_core import registry
from motrix_env_core.manager import ManagerEnv

from motrix_envs.locomotion.wbt.g1 import G1WbtManagerCfg


_MOTION_DIR = Path(__file__).parent / "assets" / "motion" / "g1"


@registry.envcfg("g1-wbt-dance1-subject1")
def make_g129dof_wbt_dance1_subject1_cfg() -> G1WbtManagerCfg:
    return G1WbtManagerCfg(motion_file=str(_MOTION_DIR / "dance1_subject1.npz"))
```

`G1WbtManagerCfg` 通过继承 `WbtManagerEnvCfg`，提供 G1 的机器人场景、tracked bodies、参考身体、控制缩放、奖励和
终止条件。新 motion 使用同一机器人和同一跟踪语义时，只需通过构造参数传入新的 `motion_file`。不要为每个 clip
复制一份 `ManagerEnv`。

## 2. 准备并 Replay Motion

将符合 [Motion 文件格式](motion_format.md)的文件放到 factory 引用的位置：

```text
motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/dance1_subject1.npz
```

内置 WBT 的 `ctrl_dt` 为 0.02 s，因此该 motion 应转换为 50 FPS。注册训练入口前先进行运动学 replay：

```bash
python scripts/motion/replay.py \
  --robot g1-29dof \
  --motion motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/dance1_subject1.npz
```

只有在根轨迹、关节方向、左右映射和完整 clip 均正确后，才进入物理训练。Replay 不使用 WBT reward 或 termination，
因此可以把数据问题与控制问题分开定位。

## 3. 确认配置责任边界

同一机器人上的新 motion 通常可以复用现有配置子类；以下变化才需要新增或调整机器人专用配置子类：

| 配置                                                       | 责任                                                                              |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `scene.objs.robot`                                         | 机器人资产、默认 key pose、base link、joint/actuator 位置范围与物理 `force_range` |
| `motion_file`                                              | MotrixLab Motion NPZ 路径                                                         |
| `tracked_body_names`                                       | 参与相对 body 位姿和速度奖励的 link 集合                                          |
| `reference_body_name`                                      | 全局参考位姿及局部对齐使用的 link                                                 |
| `control_config.action_scale`                              | 策略动作到 position target 的基础缩放                                             |
| `control_config.action_scales_by_effort_limit_over_p_gain` | 是否根据 actuator `force_range` 与 `kp` 推导各 position target scale              |
| `reward_config`                                            | Reward scale、误差核宽度、soft joint range 和允许接触 links                       |
| `termination_config`                                       | 参考高度/朝向、关键 body 高度、hard joint range 和速度异常阈值                    |
| `observation_noise`、`reset_noise`                         | Actor 观察噪声和训练初始状态噪声                                                  |
| `adaptive_timestep_sampler`                                | 基于失败 motion 帧的起始位置采样课程                                              |
| `diagnostics`                                              | 可选的机器人专用 body 与 actuator 日志                                            |

物理 actuator force limit 属于 RobotCfg/MJCF，不在 WBT config 中重复定义。开启
`action_scales_by_effort_limit_over_p_gain` 时，每个 position actuator 都必须定义 `force_range`；WBT 取其两个端点
绝对值的最大值作为 effort，并用于 position target 缩放。关闭时，WBT 直接使用标量 `action_scale`，不读取 force
range。

现有机器人配置类可作为起点：

-   G1：`G1WbtManagerCfg(motion_file=...)`
-   Dex-EVT：`DexEvtWbtManagerCfg()`
-   K1：`K1WbtManagerCfg(commands=_k1_commands(...), rewards=...)`

## 4. 绑定共享环境实现

在同一个 `g1.py` 模块中，将 Env ID 绑定到共享实现：

```python
registry.env("g1-wbt-dance1-subject1")(ManagerEnv)
```

该模块已经由 `motrix_envs.locomotion.wbt` package 导入，因此 `import motrix_envs` 会执行注册。`envcfg` 和 `env`
名称必须完全一致。

## 5. 新增 Training Task

创建 `configs/task/g1-wbt-dance1-subject1/motrix.fastsac.yaml`：

```yaml
# @package _global_
defaults:
    - /algo_base@algo: motrix.fastsac
    - _self_
task:
    env: g1-wbt-dance1-subject1
    rllib: motrix
    algo: fastsac
num_envs: 2048
play_num_envs: 16
seed: 1
checkpoint:
    interval: 1000
algo:
    asynchronous: true
    agent:
        num_updates: 4
        policy_frequency: 2
        gamma: 0.99
        tau: 0.05
        target_entropy_ratio: 0.5
        num_atoms: 501
    trainer:
        num_learning_iterations: 40000
        async_options:
            utd_mode: strict
```

默认使用 Collector/Learner 异步执行；将 `algo.asynchronous` 设为 `false` 可切换为同步执行。两种拓扑共用同一个
`motrix.fastsac` Task、算法配置和 checkpoint 格式。优先复制同一机器人的现有 WBT Training Task，只在基线稳定后
调整算法超参数。

## 6. 验证训练与回放

先用小规模 smoke test 验证 Hydra composition、registry、tensor shape 和一步训练流程：

```bash
python scripts/train.py task=g1-wbt-dance1-subject1/motrix.fastsac \
  algo.asynchronous=true num_envs=64 algo.trainer.num_learning_iterations=100
```

检查 motion/joint/body 名称没有缺失，reset 后没有系统性 NaN、joint limit 违规或立即 bad-tracking，并确认
`info["Reward"]` 与 `info["metrics"]` 能进入日志。随后使用默认规模训练：

```bash
python scripts/train.py task=g1-wbt-dance1-subject1/motrix.fastsac algo.asynchronous=true
```

生成 metadata-backed run 后回放策略：

```bash
python scripts/play.py env=g1-wbt-dance1-subject1 num_envs=16
```

Play 模式自动从第 0 帧开始，关闭 reset noise 和 adaptive sampler，并在 clip 结束后从头重播，无需单独注册 play 环境。

## 验收清单

-   Motion 通过 schema 加载和目标机器人运动学 replay，且 `fps == 1 / ctrl_dt`。
-   `envcfg`、`env` 和 Training Task 使用相同 Env ID。
-   Package 导入路径会执行新配置与环境注册。
-   Hydra 可以发现 `motrix.fastsac` Training Task，且 `algo.asynchronous` 能选择两种执行拓扑。
-   小规模训练的 observation、action、reward、terminated 和 truncated shape 正确。
-   Reset 没有系统性 NaN、hard-limit 违规或立即 bad-tracking。
-   Reward、termination 和 adaptive-sampling 指标能够写入日志。
-   `play.py` 可以发现 run，并从第 0 帧完整回放策略。
-   新 motion 包含来源与许可说明，并由 Git LFS 管理。
