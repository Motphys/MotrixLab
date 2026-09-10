# 质点环境

质点（Point Mass）环境是一个简单但基础的 2D 导航任务，智能体通过控制一个质点来到达目标位置。这个环境是强化学习概念和连续动作空间的优秀入门案例。

```{video} /_static/videos/point_mass.mp4
:poster: _static/images/poster/point_mass.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

质点环境是一个 2D 导航任务。智能体需要通过施加力来控制一个质点，使其移动到随机生成的目标位置。该任务要求智能体学习高效的导航策略，以最小的控制成本到达目标。

---

## 动作空间（Action Space）

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (2,), float32)` |
| **维度** | 2                               |

动作对应如下：

| 序号 | 动作含义（施加的力） | 最小值 | 最大值 | 对应 XML 中名称 |
| ---: | -------------------- | :----: | :----: | :-------------: |
|    0 | x 方向力             |   -1   |   1    |    `x_force`    |
|    1 | y 方向力             |   -1   |   1    |    `y_force`    |

---

## 观察空间

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-inf, inf, (9,), float32)` |
| **维度** | 9                               |

质点环境的观测空间由以下部分组成（按顺序）：

| 部分     | 内容说明                   | 维度 | 备注 |
| -------- | -------------------------- | ---- | ---- |
| **位置** | 质点的 x、y 坐标           | 2    |      |
| **速度** | 质点的 x、y 方向速度       | 2    |      |
| **目标** | 目标的 x、y 坐标           | 2    |      |
| **距离** | 到目标的 x、y 方向距离向量 | 2    |      |
| **距离** | 到目标的欧几里得距离       | 1    |      |

---

## 奖励函数设计

质点环境的奖励函数由以下几个部分组成：

### 距离奖励

```python
# 指数距离奖励 - 离目标越近奖励越强
distance_reward = np.exp(-10 * dist_to_target)
```

### 目标到达和停留奖励

```python
# 到达目标的大额奖励
target_bonus = 100.0 * in_target

# 在目标内持续停留的奖励
continuous_reward = 30.0 * in_target
```

### 控制和路径优化

```python
# 在目标内时，距离目标中心越远的惩罚
center_penalty = np.where(in_target, 10.0 * dist_to_target, 0.0)

# 控制惩罚，鼓励平滑移动
control_penalty = 0.1 * vel_magnitude

# 路径优化奖励，鼓励直线移动
path_reward = 0.5 * direction_alignment
```

### 总奖励计算

```python
# 组合所有奖励组件
rwd = distance_reward + target_bonus + continuous_reward + path_reward - center_penalty - control_penalty
```

---

## 初始状态

-   质点位置在[-1.0, 1.0]范围内随机初始化
-   目标位置在[-1.5, 1.5]范围内随机初始化
-   质点速度初始化为 0

## Episode 终止条件

-   质点到达目标并在目标内停留 0.5 秒
-   模拟时间达到 10 秒
-   观测值出现异常数值（NaN）

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=point_mass
```

### 2. 开始训练

```bash
python scripts/train.py task=point_mass/skrl.ppo
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/point_mass
```

### 4. 测试训练结果

```bash
python scripts/play.py env=point_mass
```

---

## 预期训练结果

### 导航性能

1. 智能体学会直接向目标移动
2. 移动平滑，控制 effort 最小
3. 在 episode 持续时间内一致地到达目标

### 学习进度

1. 初始学习阶段迅速，智能体发现基本导航策略
2. 控制策略逐渐精细化
3. 在不同目标位置上表现稳定

### 行为特征

1. 朝向目标的高效路径规划
2. 平滑接近目标中心
3. 最小化过冲或振荡行为
