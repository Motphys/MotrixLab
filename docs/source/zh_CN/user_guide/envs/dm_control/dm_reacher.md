# 双关节机械臂控制

双关节机械臂（Reacher）是 DeepMind Control Suite 中的经典操作任务。其目标是训练一个由两段连杆组成的机械臂，通过控制关节力矩，使末端执行器（fingertip）尽可能靠近随机生成的目标点。

```{video} /_static/videos/dm_reacher.mp4
:poster: _static/images/poster/dm_reacher.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

Reacher 是一个二维平面内的双关节机械臂控制任务。由两个通过铰链关节（hinge）连接的连杆组成，拥有 2 个受控关节（joint0 根部关节、joint1 中间关节），智能体通过向这些关节施加扭矩作为动作，使机械臂末端移动到目标点位置。目标点在每个 episode 初始化时随机采样。

---

## 动作空间（Action Space）

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (2,), float32)` |
| **维度** | 2                               |

动作对应如下：

| 序号 | 动作含义（施加在关节的力矩） | 最小值 | 最大值 | 对应 XML 中名称 |
| ---: | ---------------------------- | :----: | :----: | :-------------: |
|    0 | 根部关节驱动扭矩             |   -1   |   1    |    `joint0`     |
|    1 | 中间关节驱动扭矩             |   -1   |   1    |    `joint1`     |

---

## 观察空间

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-inf, inf, (6,), float32)` |
| **维度** | 6                               |

Reacher 环境的观测空间由以下部分组成（按顺序）：

| 部分                   | 内容说明             | 维度 | 备注         |
| ---------------------- | -------------------- | ---- | ------------ |
| **qpos**               | 2 个关节角度         | 2    | 关节位置信息 |
| **fingertip → target** | 末端到目标的向量差值 | 2    | x、y 两维    |
| **qvel**               | 2 个关节角速度       | 2    | 关节速度信息 |

| 序号 | 观察量                       | 最小值 | 最大值 | XML 名称   | 关节  | 类型 (单位)    |
| ---- | ---------------------------- | ------ | ------ | ---------- | ----- | -------------- |
| 0    | 第一个关节角度               | -Inf   | Inf    | joint0_pos | hinge | 角度 (rad)     |
| 1    | 第二个关节角度               | -Inf   | Inf    | joint1_pos | hinge | 角度 (rad)     |
| 2    | fingertip - target 的 x 差值 | -Inf   | Inf    | NA         | slide | 位置 (m)       |
| 3    | fingertip - target 的 y 差值 | -Inf   | Inf    | NA         | slide | 位置 (m)       |
| 4    | 第一个关节角速度             | -Inf   | Inf    | joint0_vel | hinge | 角速度 (rad/s) |
| 5    | 第二个关节角速度             | -Inf   | Inf    | joint1_vel | hinge | 角速度 (rad/s) |

---

## 奖励函数设计

reacher 的奖励函数基于指尖与目标的距离：

```python
# 距离奖励：指尖越接近目标，奖励越高
reward = tolerance(|| fingertip - target ||)

# 其中 tolerance 是一个单调递减函数
# 距离为 0 时奖励最大，距离越大奖励越小
```

---

## 初始状态

初始状态通过随机分布采样：

-   **手臂角度**：均匀分布
-   **手臂角速度**：小范围随机值
-   **目标点位置**：随机一个圆形区域

## Episode 终止条件

-   若观测中出现 `NaN` 值则终止

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=dm-reacher
```

### 2. 开始训练

```bash
python scripts/train.py task=dm-reacher/skrl.ppo
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/dm-reacher
```

### 4. 测试训练结果

```bash
python scripts/play.py env=dm-reacher
```

---

## 预期训练结果

1. 机械臂迅速精准触达目标
2. 末端与目标的平均距离小于 0.01 米
3. 动作平滑，无震荡
