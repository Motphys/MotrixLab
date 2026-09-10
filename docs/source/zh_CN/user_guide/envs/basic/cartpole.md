# 倒立摆

倒立摆（CartPole）是强化学习中的经典控制任务。其目标是训练一个通过控制小车左右移动来保持杆子平衡的智能体。

```{video} /_static/videos/cartpole.mp4
:poster: _static/images/poster/cartpole.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

CartPole 是一个经典的平衡控制任务。环境由一个小车和一根通过铰链连接在小车上的杆组成。智能体通过向左或向右施加力来控制小车移动，使杆子保持直立不倒。该任务要求精确的时序控制和平衡能力。

---

## 动作空间（Action Space）

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-3.0, 3.0, (1,), float32)` |
| **维度** | 1                               |

动作对应如下：

| 序号 | 动作含义（施加在小车上的力） | 最小值 | 最大值 | 对应 XML 中名称 |
| ---: | ---------------------------- | :----: | :----: | :-------------: |
|    0 | 水平方向的力                 |  -3.0  |  3.0   |    `slider`     |

---

## 观察空间

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-inf, inf, (4,), float32)` |
| **维度** | 4                               |

CartPole 环境的观测空间由以下部分组成（按顺序）：

| 部分     | 内容说明             | 维度 | 备注           |
| -------- | -------------------- | ---- | -------------- |
| **qpos** | 小车位置和杆子角度   | 2    | 位置和角度信息 |
| **qvel** | 小车速度和杆子角速度 | 2    | 速度为位置导数 |

| 序号 | 观察量     | 最小值 | 最大值 | XML 名称 | 关节  | 类型 (单位)    |
| ---- | ---------- | ------ | ------ | -------- | ----- | -------------- |
| 0    | 小车位置   | -Inf   | Inf    | slider   | slide | 位置 (m)       |
| 1    | 杆子角度   | -Inf   | Inf    | hinge    | hinge | 角度 (rad)     |
| 2    | 小车速度   | -Inf   | Inf    | slider   | slide | 速度 (m/s)     |
| 3    | 杆子角速度 | -Inf   | Inf    | hinge    | hinge | 角速度 (rad/s) |

---

## 奖励函数设计

cartpole 的奖励函数设计如下：

```python
# 每步保持杆子不倒下获得 +1 奖励
reward = 1.0  # 每步固定奖励
```

---

## 初始状态

-   **小车初始位置**：0.0 米（中心位置）
-   **杆子初始角度**：0.0 弧度（直立）
-   **初始速度**：均为 0
-   在初始状态上添加小幅随机噪声（`reset_noise_scale = 0.01`）以增加训练多样性

## Episode 终止条件

-   杆子角度超过 ±0.2 弧度（约 ±11.5 度）
-   小车位置超出 [-0.8, 0.8] 米范围
-   杆子角度出现 NaN 值
-   Episode 最大时长：10 秒

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=cartpole
```

### 2. 开始训练

```bash
python scripts/train.py task=cartpole/skrl.ppo
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/cartpole
```

### 4. 测试训练结果

```bash
python scripts/play.py env=cartpole
```

---

## 预期训练结果

1. 杆子角度大部分时间保持在 ±5 度以内
2. Episode 时长接近或超过 10 秒
3. 小车在合理范围内移动以保持平衡

## Performance

![Episode 回报](/_static/images/performance/cartpole.svg)

当前曲线来自单个随机种子 `42`，不显示置信区间。下方横轴为累计环境交互步数。上方横轴为训练耗时。

| Training Task       | TensorBoard 指标               | Seeds |
| ------------------- | ------------------------------ | ----- |
| `cartpole/skrl.ppo` | `Reward / Total reward (mean)` | `42`  |
