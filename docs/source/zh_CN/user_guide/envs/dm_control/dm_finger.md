# Finger 机械手指

Finger 是 DeepMind Control Suite 中的经典操控任务：一个由两段连杆组成的“手指”通过施加关节力矩与旋转拨片（spinner）交互。MotrixLab 当前提供了 3 个 Finger 相关环境：

```{video} /_static/videos/dm_finger_spin.mp4
:poster: _static/images/poster/dm_finger_spin.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

```{video} /_static/videos/dm_finger_turn.mp4
:poster: _static/images/poster/dm_finger_turn.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

-   `dm-finger-spin`：让 spinner 持续向指定方向旋转
-   `dm-finger-turn-easy`：将 spinner 的顶端（tip）对准目标点（较大目标半径）
-   `dm-finger-turn-hard`：同 Turn，但目标半径更小

---

## 任务描述

Finger 是一个二维平面（x-z）内的手指与旋转拨片交互任务：

-   手指有 2 个受控关节：`proximal`、`distal`
-   spinner 通过关节 `hinge` 转动，`tip` 表示 spinner 顶端位置
-   Turn 任务会在 spinner 周围随机采样一个目标点（target）

---

## 动作空间（Action Space）

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (2,), float32)` |
| **维度** | 2                               |

动作对应如下：

| 序号 | 动作含义（施加在关节的力矩） | 最小值 | 最大值 | 对应 XML 中名称 |
| ---: | ---------------------------- | :----: | :----: | :-------------: |
|    0 | 近端关节 `proximal` 驱动扭矩 |   -1   |   1    |   `proximal`    |
|    1 | 远端关节 `distal` 驱动扭矩   |   -1   |   1    |    `distal`     |

---

## 观察空间

Finger 环境的观测以 dm_control 的 observation dict 为参考，但在 MotrixLab 中被拼成一个向量。

### Spin 观察空间

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-inf, inf, (9,), float32)` |
| **维度** | 9                               |

组成如下（按顺序）：

| 部分         | 内容说明                                      | 维度 | 备注                          |
| ------------ | --------------------------------------------- | ---- | ----------------------------- |
| **position** | `qpos(proximal, distal)` + `tip_xz`           | 4    | tip 相对 spinner 的 x、z 位置 |
| **velocity** | `qvel(proximal, distal, hinge)`               | 3    | hinge 角速度用于 Spin 奖励    |
| **touch**    | `log(1 + touch_top)`、`log(1 + touch_bottom)` | 2    | 触觉传感器的对数压缩          |

| 序号 | 观察量                             | 最小值 | 最大值 | XML/Sensor 名称         | 类型 (单位)    |
| ---- | ---------------------------------- | ------ | ------ | ----------------------- | -------------- |
| 0    | `proximal` 关节角度                | -Inf   | Inf    | `proximal`              | 角度 (rad)     |
| 1    | `distal` 关节角度                  | -Inf   | Inf    | `distal`                | 角度 (rad)     |
| 2    | tip 相对 spinner 的 x 位移         | -Inf   | Inf    | `framepos(tip/spinner)` | 位置 (m)       |
| 3    | tip 相对 spinner 的 z 位移         | -Inf   | Inf    | `framepos(tip/spinner)` | 位置 (m)       |
| 4    | `proximal` 角速度                  | -Inf   | Inf    | `proximal_velocity`     | 角速度 (rad/s) |
| 5    | `distal` 角速度                    | -Inf   | Inf    | `distal_velocity`       | 角速度 (rad/s) |
| 6    | spinner 的 `hinge` 角速度          | -Inf   | Inf    | `hinge_velocity`        | 角速度 (rad/s) |
| 7    | 触觉（上侧）`log(1 + touchtop)`    | -Inf   | Inf    | `touchtop`              | 无量纲         |
| 8    | 触觉（下侧）`log(1 + touchbottom)` | -Inf   | Inf    | `touchbottom`           | 无量纲         |

### Turn 观察空间

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (12,), float32)` |
| **维度** | 12                               |

相较 Spin 额外增加：

| 部分                | 内容说明                         | 维度 | 备注                      |
| ------------------- | -------------------------------- | ---- | ------------------------- |
| **target_position** | target 相对 spinner 的 x、z 坐标 | 2    | target 在 reset 时采样    |
| **dist_to_target**  | tip 到 target 球面距离（带符号） | 1    | 负值表示 tip 落在目标球内 |

向量最后 3 个维度为：`target_x`, `target_z`, `dist_to_target`。

---

## 奖励函数设计

### Spin

在 dm_control 中，Spin 的稀疏奖励通常由 spinner 的角速度阈值触发。MotrixLab 默认使用更易训练的 dense/shaped 奖励，并同时在 info 中记录稀疏版本：

```python
# hinge_velocity 为 spinner 关节角速度
spin_sparse = 1 if hinge_velocity <= -15 else 0

# shaped: clip(-hinge_velocity / 15, 0..1)
spin = clip(-hinge_velocity / 15, 0, 1)
```

### Turn（Easy / Hard）

Turn 的核心是 tip 触达并对准目标点：目标点位于 spinner 周围一圈，目标球半径在 easy/hard 中不同。

-   `turn_sparse = 1` 当 tip 进入目标球内部（`dist_to_target <= 0`）
-   默认 shaped 奖励以 `dist_to_target` 的指数衰减为主，并额外加入“靠近 spinner、增加接触、抑制动作抖动”等项（最终裁剪到 `[0, 1]`）

---

## 初始状态

-   `proximal`、`distal` 关节角：在各自关节限制范围内均匀采样
-   spinner 的 `hinge` 角：`[-pi, pi]` 均匀采样
-   Turn 任务：每个 episode 在 spinner 周围采样 target（角度均匀采样，位置落在 x-z 平面）

## Episode 终止条件

-   若观测中出现 `NaN` 值则终止

---

## 使用指南

### 1. 环境预览（随机动作）

```bash
python scripts/view.py env=dm-finger-spin
```

```bash
python scripts/view.py env=dm-finger-turn-easy
```

```bash
python scripts/view.py env=dm-finger-turn-hard
```

### 2. 开始训练

建议显式指定训练后端（JAX / PyTorch 二选一）：

```bash
python scripts/train.py task=dm-finger-spin/skrl.ppo task.train_backend=torch
```

```bash
python scripts/train.py task=dm-finger-turn-easy/skrl.ppo task.train_backend=torch
```

```bash
python scripts/train.py task=dm-finger-turn-hard/skrl.ppo task.train_backend=torch
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/dm-finger-spin
```

### 4. 测试训练结果

`scripts/play.py` 默认会在 `runs/{env-name}/` 下查找最新的 metadata-backed run，并加载其 `best_policy` artifact；也可以用 `policy=...` 显式指定 checkpoint：

```bash
python scripts/play.py env=dm-finger-turn-hard
```

---

## 预期训练结果

1. `dm-finger-spin`：spinner 能稳定持续向目标方向旋转（hinge_velocity 达到阈值附近）
2. `dm-finger-turn-easy`：手指能稳定接触并将 tip 对准目标点（成功率较高、抖动较小）
3. `dm-finger-turn-hard`：能对准更小目标，但更容易出现“接触不足/动作抖动”的训练难点
