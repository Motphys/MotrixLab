# 机械手抓球

Bring Ball 是 DeepMind Control Suite 中的 Manipulator 经典任务：平面机械手（带拇指与手指）需要抓取球体并将其移动到目标球位置。MotrixLab 当前提供 1 个 Bring Ball 环境：

-   `dm-manipulator-bring-ball`：抓取球体并移动到目标位置

```{video} /_static/videos/bring-ball.mp4
:poster: _static/images/poster/bring-ball.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## 任务描述

Bring Ball 是一个二维平面（x-z）内的抓取与搬运任务：

-   机械手包含 4 个手臂关节（`arm_root`、`arm_shoulder`、`arm_elbow`、`arm_wrist`）以及拇指/手指关节
-   夹持通过腱 `grasp` 同时驱动 `thumb` 与 `finger` 关节闭合
-   球体可在平面内滑动（`ball_x`、`ball_z`）并绕 y 轴旋转（`ball_y`）
-   目标球为 mocap 物体，位置在 reset 时随机采样

---

## 动作空间（Action Space）

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (5,), float32)` |
| **维度** | 5                               |

动作对应如下（控制信号施加在关节/腱上）：

| 序号 | 动作含义                      | 最小值 | 最大值 | 对应 XML 中名称 |
| ---: | ----------------------------- | :----: | :----: | :-------------: |
|    0 | 根部关节驱动                  |   -1   |   1    |     `root`      |
|    1 | 肩部关节驱动                  |   -1   |   1    |   `shoulder`    |
|    2 | 肘部关节驱动                  |   -1   |   1    |     `elbow`     |
|    3 | 腕部关节驱动                  |   -1   |   1    |     `wrist`     |
|    4 | 夹持驱动（拇指/手指耦合闭合） |   -1   |   1    |     `grasp`     |

---

## 观察空间

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (41,), float32)` |
| **维度** | 41                               |

观察向量由以下部分组成（按顺序）：

| 部分           | 内容说明                    | 维度 | 备注                                 |
| -------------- | --------------------------- | ---- | ------------------------------------ |
| **arm_pos**    | 8 个关节角度的 `sin`/`cos`  | 16   | 关节顺序见下表，`sin`/`cos` 交替排列 |
| **arm_vel**    | 8 个关节角速度              | 8    | 顺序同 arm_pos                       |
| **touch**      | 触觉传感器 `log(1 + touch)` | 5    | palm/finger/thumb/fingertip/thumbtip |
| **hand_pos**   | 抓取点 `grasp` 的世界坐标   | 3    | x, y, z                              |
| **object_pos** | 球体位置                    | 3    | x, y, z                              |
| **target_pos** | 目标球位置                  | 3    | x, y, z                              |
| **rel**        | `object_pos - target_pos`   | 3    | 相对位置                             |

| 序号  | 观察量范围                                     | 维度 | 备注                          |
| ----- | ---------------------------------------------- | ---- | ----------------------------- |
| 0-15  | 8 个关节角度的 `sin`/`cos` 交替排列            | 16   | 关节顺序：arm_root → thumbtip |
| 16-23 | 8 个关节角速度                                 | 8    | 顺序同上                      |
| 24-28 | 触觉：palm, finger, thumb, fingertip, thumbtip | 5    | `log(1 + touch)`              |
| 29-31 | grasp 位置 (x, y, z)                           | 3    | hand_pos                      |
| 32-34 | 球体位置 (x, y, z)                             | 3    | object_pos                    |
| 35-37 | 目标球位置 (x, y, z)                           | 3    | target_pos                    |
| 38-40 | 球体相对目标位置 (x, y, z)                     | 3    | rel                           |

关节顺序为：`arm_root`、`arm_shoulder`、`arm_elbow`、`arm_wrist`、`finger`、`fingertip`、`thumb`、`thumbtip`。

---

## 奖励函数设计

Bring Ball 使用 shaped 奖励，由多个子项加权组合，并加入惩罚项：

```python
# R1: Reach - 手指接近球
r_reach = tolerance(avg_tip_dist)

# R2: Orient - 手掌朝向球体
r_orient = clip(1 - orient_bound + dot(hand_dir, unit_vec_to_ball), 0..1)

# R3: Pause - 靠近球体时抑制臂部抖动
r_pause = tolerance(arm_speed_step) * is_close_to_ball

# R4: Close - 夹持动作与接触条件联合
r_close = r_close_intent * (approach_or_grasp)

# R5: Lift & Transport - 抬升高度 + 接近目标
r_lift_height = tolerance(ball_z)
r_transport = tolerance(move_dist_to_target)
r_lift = mix(r_lift_height, r_transport)

# Precision/Progress - 目标精度与移动进度
r_precision = tolerance(move_dist_to_target, gaussian)
r_progress = (prev_dist - curr_dist) * scale

# Penalty - 侧面接触与悬停惩罚
penalty_side + penalty_hover
```

默认权重（`BringBallCfg`）：

-   reach 1.0、orient 1.5、pause 0.5、close 2.0、lift 6.0、precision 1.0
-   lift 内部由 `lift_height_weight` 与 `transport_weight` 组合
-   进度奖励由 `transport_progress_scale` 控制

---

## 初始状态

-   **手臂初始化**：默认使用模型初始姿态（`randomize_arm=False`），拇指/手指对称
-   **目标位置**：`x ∈ [-0.4, 0.4]`，`z ∈ [0.1, 0.4]`，`y = 0.001`
-   **球体位置**：`x ∈ [-0.4, 0.4]`，`z ∈ [0.2, 0.7]`，并与手部保持最小距离
-   **物理稳定**：reset 后会进行若干步 settle（`settle_steps=300`）

---

## Episode 终止条件

-   若观测中出现 `NaN` 值则终止

---

## 使用指南

### 1. 环境预览（随机动作）

```bash
python scripts/view.py env=dm-manipulator-bring-ball
```

### 2. 开始训练

建议显式指定训练后端（JAX / PyTorch 二选一）：

```bash
python scripts/train.py task=dm-manipulator-bring-ball/skrl.ppo.torch
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/dm-manipulator-bring-ball
```

### 4. 测试训练结果

```bash
python scripts/play.py env=dm-manipulator-bring-ball
```

---

## 预期训练结果

1. 机械手能稳定接近并夹持球体
2. 球体被抬离地面并保持稳定高度
3. 球体最终能稳定到达目标球附近
