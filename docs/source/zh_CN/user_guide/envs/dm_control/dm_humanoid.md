# 三维人形机器人

三维人形机器人（Humanoid）是 DeepMind Control Suite 中的经典双足行走任务。其目标是训练一个模拟的三维人形机器人，通过控制其关节力矩，实现站立、行走和奔跑。

```{video} /_static/videos/dm_humanoid_run.mp4
:poster: _static/images/poster/dm_humanoid_run.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

Humanoid 是一个三维空间的双足人形机器人任务。机器人由头部、躯干、双臂和双腿组成，拥有 21 个受控关节（执行器），智能体通过向这些关节施加扭矩作为动作，让机器人实现站立平衡、向前行走或快速奔跑。该任务要求协调的双足步态、平衡控制和三维空间姿态稳定能力。

---

## 动作空间（Action Space）

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-1.0, 1.0, (21,), float32)` |
| **维度** | 21                               |

动作对应如下：

| 序号 | 动作含义（施加在关节的力矩） | 最小值 | 最大值 | 对应 XML 中名称   |
| ---: | ---------------------------- | :----: | :----: | :---------------- |
|    0 | 腹部 Y 轴旋转关节驱动扭矩    |  -1.0  |  1.0   | `abdomen_y`       |
|    1 | 腹部 Z 轴旋转关节驱动扭矩    |  -1.0  |  1.0   | `abdomen_z`       |
|    2 | 腹部 X 轴旋转关节驱动扭矩    |  -1.0  |  1.0   | `abdomen_x`       |
|    3 | 右髋关节 X 轴驱动扭矩        |  -1.0  |  1.0   | `right_hip_x`     |
|    4 | 右髋关节 Z 轴驱动扭矩        |  -1.0  |  1.0   | `right_hip_z`     |
|    5 | 右髋关节 Y 轴驱动扭矩        |  -1.0  |  1.0   | `right_hip_y`     |
|    6 | 右膝关节驱动扭矩             |  -1.0  |  1.0   | `right_knee`      |
|    7 | 右踝关节 X 轴驱动扭矩        |  -1.0  |  1.0   | `right_ankle_x`   |
|    8 | 右踝关节 Y 轴驱动扭矩        |  -1.0  |  1.0   | `right_ankle_y`   |
|    9 | 左髋关节 X 轴驱动扭矩        |  -1.0  |  1.0   | `left_hip_x`      |
|   10 | 左髋关节 Z 轴驱动扭矩        |  -1.0  |  1.0   | `left_hip_z`      |
|   11 | 左髋关节 Y 轴驱动扭矩        |  -1.0  |  1.0   | `left_hip_y`      |
|   12 | 左膝关节驱动扭矩             |  -1.0  |  1.0   | `left_knee`       |
|   13 | 左踝关节 X 轴驱动扭矩        |  -1.0  |  1.0   | `left_ankle_x`    |
|   14 | 左踝关节 Y 轴驱动扭矩        |  -1.0  |  1.0   | `left_ankle_y`    |
|   15 | 右肩关节 1 驱动扭矩          |  -1.0  |  1.0   | `right_shoulder1` |
|   16 | 右肩关节 2 驱动扭矩          |  -1.0  |  1.0   | `right_shoulder2` |
|   17 | 右肘关节驱动扭矩             |  -1.0  |  1.0   | `right_elbow`     |
|   18 | 左肩关节 1 驱动扭矩          |  -1.0  |  1.0   | `left_shoulder1`  |
|   19 | 左肩关节 2 驱动扭矩          |  -1.0  |  1.0   | `left_shoulder2`  |
|   20 | 左肘关节驱动扭矩             |  -1.0  |  1.0   | `left_elbow`      |

---

## 观察空间

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (73,), float32)` |
| **维度** | 73                               |

Humanoid 环境的观测空间由以下部分组成（按顺序）：

| 部分               | 内容说明                              | 维度 | 备注                                              |
| ------------------ | ------------------------------------- | ---- | ------------------------------------------------- |
| **joint_angles**   | 各关节角度（排除根关节的 7 个自由度） | 22   | 22 个关节的角度信息                               |
| **head_height**    | 头部高度                              | 1    | 头部相对于地面的高度                              |
| **extremities**    | 四肢末端位置（相对于躯干）            | 12   | 左手、左脚、右手、右脚的位置（各 3 维，按此顺序） |
| **torso_vertical** | 躯干垂直方向向量                      | 3    | 躯干在局部坐标系中的垂直方向                      |
| **com_vel**        | 质心线速度                            | 3    | 躯干子树的线速度                                  |
| **qvel**           | 所有关节和根部的速度信息              | 29   | 包括根关节的 6 个自由度                           |
| **target_local**   | 目标方向（局部坐标系）                | 3    | 目标方向在躯干局部坐标系中的表示                  |

---

## 奖励函数设计

Humanoid 的奖励函数根据任务类型（站立、行走、奔跑）有所不同，但都包含以下核心组件：

### 姿态奖励（Posture Reward）

```python
# 头部高度奖励：保持头部在目标高度（标准站立高度的 95%，约 1.33m）以上
stand_reward = tolerance(head_height, bounds=(stand_height * 0.95, inf), margin=0.5)

# 躯干直立奖励：保持躯干直立
upright_reward = tolerance(torso_upright, bounds=(0.9, inf), sigmoid="linear", margin=0.9)

# 盆骨高度奖励：保持盆骨在合理高度（标准站立高度的 60%，约 0.84m）以上
pelvis_height_reward = tolerance(
    pelvis_height, bounds=(stand_height * 0.6, inf), sigmoid="linear", margin=stand_height * 0.6
)

# 姿态奖励 = 头部高度奖励 × 躯干直立奖励 × 盆骨高度奖励
posture_reward = stand_reward * upright_reward * pelvis_height_reward
```

### 速度奖励（Speed Reward）

根据任务类型，速度奖励的计算方式不同：

**站立任务（move_speed <= 0）**：

```python
# 速度奖励：保持接近零速度
speed_reward = tolerance(actual_speed, bounds=(0, 0), margin=1.0, value_at_margin=0.01)
```

**行走任务（0 < move_speed <= 3.0）**：

```python
# 速度奖励：在目标方向（X 轴正方向）上达到目标速度（默认 1.0 m/s）
actual_speed = dot(com_vel[:2], target_direction[:2])  # 速度在目标方向上的投影
speed_reward = tolerance(actual_speed, bounds=(move_speed, move_speed), margin=move_speed, sigmoid="linear")
```

**奔跑任务（move_speed > 3.0）**：

```python
# 速度奖励：在目标方向上达到目标速度（默认 10.0 m/s）以上
actual_speed = dot(com_vel[:2], target_direction[:2])
speed_reward = tolerance(actual_speed, bounds=(move_speed, inf), margin=move_speed, sigmoid="linear")
```

### 能量奖励（Energy Reward）

```python
energy_reward = exp(-energy_coef * mean(ctrls ^ 2))
```

### 步态奖励（Gait Reward）

```python
# 躯干朝向奖励：躯干正对目标方向
torso_heading_reward = tolerance(dot(torso_forward, target_dir), bounds=(0.9, 1.0), margin=0.3, sigmoid="linear")

# 头部朝向奖励：头部正对目标方向
head_heading_reward = tolerance(dot(head_forward, target_dir), bounds=(0.9, 1.0), margin=0.3, sigmoid="linear")

# 盆骨朝向奖励：盆骨正对目标方向
pelvis_yaw_reward = tolerance(dot(pelvis_forward, target_dir), bounds=(0.9, 1.0), margin=0.3, sigmoid="linear")

# 盆骨水平奖励：盆骨保持水平
pelvis_level_reward = tolerance(pelvis_up, bounds=(0.9, 1.0), margin=0.3, sigmoid="linear")

# 足部高度奖励：足部保持贴近地面
feet_height_reward = tolerance(max_foot_height, bounds=(0.0, 0.3), margin=0.5, sigmoid="quadratic")

# 步态奖励 = 所有朝向和姿态奖励的乘积
gait_reward = torso_heading_reward * head_heading_reward * pelvis_yaw_reward * pelvis_level_reward * feet_height_reward
```

### 总奖励

```python
total_reward = posture_reward * speed_reward * energy_reward * gait_reward
```

---

## 初始状态

-   **机器人位置**：躯干初始高度为 1.33 米（标准站立高度的 95%）
-   **机器人姿态**：躯干保持直立，四元数设置为 (1.0, 0.0, 0.0, 0.0)
-   **关节角度**：在关节限位范围内随机初始化
    -   躯干/髋部基础关节：在较小范围内随机化（±15 度）
    -   腿部关节：对称初始化，确保左右腿独立随机化，膝盖初始弯曲
    -   手臂关节：对称初始化，使用关节限位的中间 80% 范围
-   **初始速度**：所有关节速度和线速度初始化为接近零的小随机值（-0.01 到 0.01）
-   **初始控制**：所有执行器控制量初始化为接近零的小随机值（-0.02 到 0.02）

## Episode 终止条件

-   机器人状态观测值出现异常数值（NaN 或 Inf）
-   头部高度过低：头部高度低于标准站立高度的 50%（0.7 米）
-   躯干倾斜过大：躯干垂直分量小于 0.2（躯干严重倾斜）
-   速度异常：任何关节速度的绝对值超过 200.0 rad/s 或 m/s
-   Episode 最大时长：25 秒

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=dm-humanoid-stand
python scripts/view.py env=dm-humanoid-walk
python scripts/view.py env=dm-humanoid-run
```

### 2. 开始训练

```bash
python scripts/train.py task=dm-humanoid-stand/skrl.ppo
python scripts/train.py task=dm-humanoid-walk/skrl.ppo
python scripts/train.py task=dm-humanoid-run/skrl.ppo
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/dm-humanoid-walk
```

### 4. 测试训练结果

```bash
python scripts/play.py env=dm-humanoid-stand
python scripts/play.py env=dm-humanoid-walk
python scripts/play.py env=dm-humanoid-run
```

---

## 预期训练结果

### 站立任务 (dm-humanoid-stand)

1. 头部高度保持在 1.3-1.5m 范围
2. 躯干直立角度偏差小于 15 度
3. 能够稳定站立不倒
4. 速度接近零，无明显移动

### 行走任务 (dm-humanoid-walk)

1. 实际行走速度接近 1.0 m/s
2. 步态协调，无明显摔倒
3. 能够持续稳定行走
4. 躯干和头部朝向目标方向

### 奔跑任务 (dm-humanoid-run)

1. 奔跑速度达到 4.0-10.0 m/s
2. 出现飞行相（双脚同时离地）
3. 步态协调稳定
4. 能够保持高速奔跑姿态
