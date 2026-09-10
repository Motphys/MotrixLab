# 通用人形速度跟踪环境

`HumanoidVelocityTrackingEnv` 是 MotrixLab 面向双足人形机器人的通用速度跟踪环境。它将机体坐标系下的
`[vx, vy, yaw_rate]` 命令、动作与观察构造、奖励、终止和 reset 统一为一套共享任务，不绑定具体机器人模型，
也不假设固定的关节数量。接入新机器人时，只需提供 `HumanoidRobotCfg`、任务所需的足端语义和一份
`HumanoidVelocityTrackingEnvCfg` 配置，无需复制环境实现。

## 起伏地形演示

以下视频分别展示 Unitree G1 和 Booster K1 在程序化起伏地形上的速度跟踪效果。

::::{grid} 1 1 2 2
:gutter: 2 2 2 2

:::{grid-item-card} Unitree G1

```{video} /_static/videos/g1-walk-rough.mp4
:poster: /_static/images/poster/g1-walk-rough.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} Booster K1

```{video} /_static/videos/k1-walk-rough.mp4
:poster: /_static/images/poster/k1-walk-rough.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

::::

```{toctree}
:hidden:
:maxdepth: 1

humanoid_velocity_tracking/adding_robot
humanoid_velocity_tracking/env_design
humanoid_velocity_tracking/config_tuning

```

## 内置机器人

MotrixLab 已为下列机器人提供平地和程序化起伏高度场配置。所有 Env ID 都注册到同一个
`HumanoidVelocityTrackingEnv`，但分别选择一份完整配置。点击训练曲线缩略图可查看 SVG 大图。

:::{div} task-table
| Env ID | 机器人 | 地形 | 已提供的训练配置 | 训练曲线 |
| --- | --- | --- | --- | --- |
| `g1-walk-flat` | Unitree G1 29-DoF | 平地 | `motrix.fastsac` | — |
| `g1-walk-rough` | Unitree G1 29-DoF | 程序化起伏高度场 | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="g1-walk-rough-curve" aria-label="放大 Unitree G1 起伏地形训练曲线"><img src="../../_static/images/performance/g1-walk-rough.svg" alt="Unitree G1 起伏地形 FastSAC 训练曲线" width="180"></button> |
| `dex-evt-walk-flat` | Dex-EVT | 平地 | `motrix.fastsac` | — |
| `dex-evt-walk-rough` | Dex-EVT | 程序化起伏高度场 | `motrix.fastsac` | — |
| `k1-walk-flat` | Booster K1 | 平地 | `motrix.fastsac` | — |
| `k1-walk-rough` | Booster K1 | 程序化起伏高度场 | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="k1-walk-rough-curve" aria-label="放大 Booster K1 起伏地形训练曲线"><img src="../../_static/images/performance/k1-walk-rough.svg" alt="Booster K1 起伏地形 FastSAC 训练曲线" width="180"></button> |
:::

<dialog id="g1-walk-rough-curve" class="training-curve-dialog" aria-labelledby="g1-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="关闭训练曲线">×</button>
  <img src="../../_static/images/performance/g1-walk-rough.svg" alt="Unitree G1 起伏地形 FastSAC 训练曲线">
  <p id="g1-walk-rough-curve-caption">Unitree G1（<code>g1-walk-rough</code>）训练曲线</p>
</dialog>

<dialog id="k1-walk-rough-curve" class="training-curve-dialog" aria-labelledby="k1-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="关闭训练曲线">×</button>
  <img src="../../_static/images/performance/k1-walk-rough.svg" alt="Booster K1 起伏地形 FastSAC 训练曲线">
  <p id="k1-walk-rough-curve-caption">Booster K1（<code>k1-walk-rough</code>）训练曲线</p>
</dialog>

同一机器人的平地与起伏地形配置复用观察、动作、奖励和终止逻辑，仅在场景地形与出生范围上区分。表中的曲线展示
起伏地形任务的 FastSAC 异步训练过程：左轴为平均 Episode 回报，右轴为惩罚课程的 `penalty_scale`。整体回报在
训练早期快速提升，并在约 2 ～ 3 分钟内进入稳定阶段。随后 `penalty_scale` 提升会增大惩罚项权重，因此平均回报可能
下降；这是奖励尺度发生变化，不代表策略性能退化。

## 执行命令

从“内置机器人”表格中选择 Env ID 和对应的训练配置，并替换下列命令中的 `ENV_ID` 与 `TRAINING_CONFIG`：

```bash
python scripts/view.py env=ENV_ID num_envs=1
python scripts/train.py task=ENV_ID/TRAINING_CONFIG
python scripts/play.py env=ENV_ID num_envs=16
```

例如，使用 FastSAC 异步训练 K1 起伏地形任务：

```bash
python scripts/train.py task=k1-walk-rough/motrix.fastsac algo.asynchronous=true
```

`view.py` 使用随机动作，仅用于检查场景和模型；训练后的步态需要通过 `play.py` 回放策略查看。
