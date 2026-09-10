# 乒乓球颠球

使用单臂机器人控制挡板颠球，实现连续弹跳并保持球在目标位置。

```{video} /_static/videos/bounce_ball.mp4
:poster: _static/images/poster/bounce_ball.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

Bounce Ball 是一个单臂机器人操作任务，使用 6 自由度的配天 AIR4-560 工业机械臂控制末端挡板（球拍）的位置。智能体通过控制机械臂 6 个关节的位置变化作为动作，使乒乓球在挡板上持续弹跳，并尽可能将球保持在目标高度和目标水平位置。

---

## 动作空间（Action Space）

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (6,), float32)` |
| **维度** | 6                               |

关节对应如下：

| 序号 | 动作含义（关节位置变化）   | 最小值 | 最大值 | 对应 XML 中名称 |
| ---: | -------------------------- | :----: | :----: | :-------------: |
|    0 | Joint1（基座旋转）位置变化 |   -1   |   1    |    `Joint1`     |
|    1 | Joint2（大臂）位置变化     |   -1   |   1    |    `Joint2`     |
|    2 | Joint3（小臂）位置变化     |   -1   |   1    |    `Joint3`     |
|    3 | Joint4（手腕旋转）位置变化 |   -1   |   1    |    `Joint4`     |
|    4 | Joint5（手腕俯仰）位置变化 |   -1   |   1    |    `Joint5`     |
|    5 | Joint6（手腕旋转）位置变化 |   -1   |   1    |    `Joint6`     |

---

## 观察空间

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (29,), float32)` |
| **维度** | 29                               |

观察空间由以下部分组成（按顺序）：

| 部分              | 内容说明             | 维度 | 备注                                                            |
| ----------------- | -------------------- | ---- | --------------------------------------------------------------- |
| **dof_pos**       | 各关节自由度位置信息 | 13   | 前 6 个为机械臂关节，后 7 个为球的自由关节（3 位置 + 4 四元数） |
| **dof_vel**       | 各关节自由度速度信息 | 12   | 速度为位置导数                                                  |
| **paddle_pos**    | 挡板位置信息         | 3    | 挡板中心的 x, y, z 坐标                                         |
| **target_height** | 目标高度             | 1    | 当前环境的目标高度                                              |

| 序号  | 观察量                     | 最小值 | 最大值 | XML 名称  | 类型 (单位)       |
| ----- | -------------------------- | ------ | ------ | --------- | ----------------- |
| 0-5   | 机械臂关节角度             | -Inf   | Inf    | Joint1-6  | 角度 (rad)        |
| 6-8   | 球位置 [x, y, z]           | -Inf   | Inf    | ball_link | 位置 (m)          |
| 9-12  | 球姿态四元数 [w,x,y,z]     | -Inf   | Inf    | ball_link | 四元数            |
| 13-18 | 机械臂关节角速度           | -Inf   | Inf    | Joint1-6  | 角速度 (rad/s)    |
| 19-24 | 球速度 [vx,vy,vz,wx,wy,wz] | -Inf   | Inf    | ball_link | 速度 (m/s, rad/s) |
| 25-27 | 挡板位置 [x, y, z]         | -Inf   | Inf    | blocker   | 位置 (m)          |
| 28    | 目标高度                   | -Inf   | Inf    | -         | 位置 (m)          |

---

## 奖励函数

奖励函数采用复合设计，包含多个奖励和惩罚项，引导机器人学习稳定的颠球策略。所有奖励参数可通过配置文件调整。

### 主要奖励项

#### 1. 水平位置奖励

**设计意义**：这是最核心的奖励项，确保球始终保持在挡板正上方。通过垂直距离加权机制，当球接近挡板时（即将击打时刻）对水平位置的要求更严格，引导策略在关键时刻精确对齐。

**计算公式**：

$$
\begin{aligned}
\text{err}_{xy} &= \sqrt{(x_{ball} - x_{target})^2 + (y_{ball} - y_{target})^2} &&\text{（水平位置误差）} \\
d_{vert} &= |z_{ball} - z_{paddle}| &&\text{（垂直距离）} \\
w_{vert} &= e^{-d_{vert} / \sigma_{vert}} &&\text{（垂直距离权重）} \\
\sigma_{pos} &= \sigma_{base} \times (1.0 + k_{weight} \times w_{vert}) &&\text{（自适应尺度）} \\
r_{pos} &= e^{-\frac{\text{err}_{xy}^2}{2\sigma_{pos}^2}} &&\text{（高斯奖励）} \\
\\
\text{其中：} \quad &\sigma_{vert} = 0.15 \text{ m} &&\text{（垂直距离尺度）} \\
&\sigma_{base} = 0.1 \text{ m} &&\text{（基础水平尺度）} \\
&k_{weight} = 3.0 &&\text{（权重因子）} \\
&x_{target} = 0.58856 \text{ m}, \, y_{target} = 0.0 \text{ m} &&\text{（目标位置）}
\end{aligned}
$$

**权重**：2.0

#### 2. 位置偏离惩罚

**设计意义**：对严重偏离目标位置的情况施加强惩罚，防止球飞出控制范围。使用 sigmoid 函数实现平滑过渡，避免奖励函数不连续。

**计算公式**：

$$
r_{out} = -\frac{2.0}{1 + e^{-(\text{err}_{xy} - 0.05) / 0.03}} \qquad \text{（sigmoid 惩罚）}
$$

**权重**：1.0

#### 3. 速度匹配奖励

**设计意义**：基于抛体运动物理规律，激励球的运动轨迹能够在目标高度时具有期望的速度（0.5 m/s）。这确保球不会过快或过慢地通过目标高度，有利于稳定控制。

**计算公式**：

$$
\begin{aligned}
\Delta h &= h_{target} - z_{ball} &&\text{（高度差）} \\
v_{desired} &= 0.5 \text{ m/s} &&\text{（期望速度）} \\
\\
\text{情况1：}&\text{球向上运动且低于目标高度} \\
v_{z,up}^2 &= v_z^2 - 2g\Delta h &&\text{（能量守恒）} \\
v_{at\_target,up} &= \sqrt{\max(0, v_{z,up}^2)} &&\text{（向上到达速度）} \\
\\
\text{情况2：}&\text{球向下运动且高于目标高度} \\
v_{z,down}^2 &= v_z^2 + 2g|\Delta h| &&\text{（能量守恒）} \\
v_{at\_target,down} &= -\sqrt{\max(0, v_{z,down}^2)} &&\text{（向下到达速度）} \\
\\
\text{情况3：}&\text{球接近目标高度（} |\Delta h| < 0.05 \text{ m）} \\
v_{at\_target,near} &= v_z &&\text{（当前速度）} \\
\\
\text{平滑组合：}& \\
\sigma_{up} &= \frac{1}{1 + e^{-v_z / 0.2}} &&\text{（向上运动权重）} \\
\sigma_{below} &= \frac{1}{1 + e^{-\Delta h / 0.02}} &&\text{（低于目标权重）} \\
\sigma_{down} &= 1 - \sigma_{up} &&\text{（向下运动权重）} \\
\sigma_{above} &= 1 - \sigma_{below} &&\text{（高于目标权重）} \\
w_{near} &= e^{-\frac{\Delta h^2}{2 \times 0.01^2}} &&\text{（接近目标权重）} \\
\\
v_{at\_target} &= v_{at\_target,up} \cdot \sigma_{up} \cdot \sigma_{below} \\
&\quad + v_{at\_target,down} \cdot \sigma_{down} \cdot \sigma_{above} \\
&\quad + v_{at\_target,near} \cdot w_{near} &&\text{（加权组合）} \\
\\
\text{err}_{vel} &= |v_{at\_target} - v_{desired}| &&\text{（速度误差）} \\
r_{vel} &= e^{-\frac{\text{err}_{vel}^2}{2 \times 0.8^2}} &&\text{（高斯奖励）}
\end{aligned}
$$

**权重**：2.0

#### 4. 高度奖励

**设计意义**：直接激励球接近目标高度，这是任务的核心目标之一。较高的权重（4.5）确保策略优先考虑高度控制。目标高度在每个环境中随机采样（0.3-0.6 m），提升策略的泛化能力。

**计算公式**：

$$
\begin{aligned}
\text{err}_h &= |z_{ball} - h_{target}| &&\text{（高度误差）} \\
r_h &= e^{-\frac{\text{err}_h^2}{2 \times 0.15^2}} &&\text{（高斯奖励）}
\end{aligned}
$$

**权重**：4.5

#### 5. 高度进步奖励

**设计意义**：激励球达到更高位置，帮助策略在训练初期快速学会向上击打球，避免陷入"不击打"的局部最优。

**计算公式**：

$$
r_{progress} = \max(0, z_{ball} - 0.2) \times 2.0 \qquad \text{（线性奖励）}
$$

**权重**：1.0

#### 6. 受控向上速度奖励

**设计意义**：只有当球水平位置良好时才奖励向上速度，避免"乱打"行为。理想速度根据物理公式计算，确保球能恰好达到目标高度。这个奖励引导策略学习精确的击打力度。

**计算公式**：

$$
\begin{aligned}
q_{pos} &= e^{-\frac{\text{err}_{xy}^2}{2 \times 0.02^2}} &&\text{（位置质量）} \\
v_{ideal} &= \sqrt{2g \times \max(0, \Delta h)} &&\text{（理想发射速度）} \\
v_{ideal} &\in [0.5, 3.0] \text{ m/s} &&\text{（限制范围）} \\
q_{vel} &= e^{-\frac{(v_z - v_{ideal})^2}{2 \times 0.5^2}} &&\text{（速度质量）} \\
\sigma_{up} &= \frac{1}{1 + e^{-v_z / 0.1}} &&\text{（向上掩码）} \\
r_{controlled} &= q_{pos} \times q_{vel} \times \sigma_{up} \times \text{clip}(v_z, 0, 1.5) &&\text{（组合奖励）}
\end{aligned}
$$

**权重**：1.5

#### 7. 连续弹跳奖励

**设计意义**：激励多次连续成功弹跳，引导策略学习稳定的长期控制。使用对数函数避免奖励无限增长，同时要求球位置良好才给予奖励。

**计算公式**：

$$
\begin{aligned}
q_{bounce} &= e^{-\frac{\text{err}_{xy}^2}{2 \times 0.05^2}} &&\text{（弹跳位置质量）} \\
r_{bounce\_log} &= 0.5 \times \log(n_{bounces} + 1) &&\text{（对数奖励）} \\
r_{bounce} &= r_{bounce\_log} \times q_{bounce} \times \mathbb{1}_{n_{bounces} > 0} &&\text{（条件奖励）}
\end{aligned}
$$

**权重**：0.8

#### 8. 高弹跳次数奖励

**设计意义**：对高弹跳次数（≥3 次）给予额外奖励，进一步激励长期稳定控制。使用 sigmoid 激活函数使奖励平滑增长。

**计算公式**：

$$
\begin{aligned}
\sigma_{high} &= \frac{1}{1 + e^{-(n_{bounces} - 2.0) / 0.5}} &&\text{（高弹跳激活）} \\
r_{high} &= n_{bounces} \times 0.15 \times q_{bounce} \times \sigma_{high} &&\text{（额外奖励）}
\end{aligned}
$$

**权重**：0.3

#### 9. 挡板-球水平对齐奖励

**设计意义**：激励挡板主动移动到球的正下方，而不是等待球落下。垂直距离越近权重越大，在即将击打时刻要求更精确的对齐。弹跳时刻给予额外奖励（boost），强化正确的击打行为。

**计算公式**：

$$
\begin{aligned}
\text{err}_{align} &= \|(\mathbf{x}_{ball}, \mathbf{y}_{ball}) - (\mathbf{x}_{paddle}, \mathbf{y}_{paddle})\|_2 &&\text{（对齐误差）} \\
w_{prox} &= e^{-d_{vert} / 0.1} &&\text{（垂直接近权重）} \\
q_{align} &= e^{-\frac{\text{err}_{align}^2}{2 \times 0.03^2}} \times (1.0 + 2.0 \times w_{prox}) &&\text{（对齐质量）} \\
k_{boost} &= \begin{cases} 3.0 & \text{if bounce detected} \\ 1.0 & \text{otherwise} \end{cases} &&\text{（弹跳增益）} \\
r_{align} &= q_{align} \times k_{boost} \times 0.3 &&\text{（对齐奖励）}
\end{aligned}
$$

**权重**：0.6

#### 10. 挡板原位奖励

**设计意义**：激励挡板在球远离时回到原位（home position），避免挡板长时间停留在高位。使用距离动态因子：当球远离挡板时（$d_{vert} > 0.15$ m）增大奖励，鼓励挡板快速回位；当球接近时减小奖励，允许挡板上移准备击打。这种设计使挡板运动更加节能和自然。

**计算公式**：

$$
\begin{aligned}
\text{err}_{home} &= |z_{paddle} - z_{home}| &&\text{（原位偏离）} \\
k_{dist} &= 1.0 + \frac{0.5}{1 + e^{-(d_{vert} - 0.15) / 0.03}} &&\text{（距离因子）} \\
q_{home} &= e^{-\frac{\text{err}_{home}^2}{2 \times 0.05^2}} &&\text{（原位质量）} \\
r_{home} &= q_{home} \times k_{dist} &&\text{（原位奖励）} \\
\\
\text{其中：} \quad &z_{home} = 0.05 \text{ m} &&\text{（挡板原位高度）}
\end{aligned}
$$

**权重**：1.5

### 惩罚项

#### 1. 过高向上速度惩罚

**设计意义**：防止球速度过快（>3.5 m/s）失去控制，确保球的运动在可控范围内。

**计算公式**：

$$
r_{excess} = -\frac{1.0}{1 + e^{-(v_z - 3.5) / 0.3}} \qquad \text{（sigmoid 惩罚）}
$$

**权重**：1.0

#### 2. 向下速度惩罚

**设计意义**：惩罚球向下运动（$v_z < -0.2$ m/s），激励策略及时击打球，避免球自由下落。

**计算公式**：

$$
\begin{aligned}
\text{mag}_{down} &= -v_z \times \text{clip}(-v_z \times 0.3, 0, 0.5) &&\text{（向下幅度）} \\
\sigma_{down} &= \frac{1}{1 + e^{(v_z + 0.2) / 0.2}} &&\text{（向下激活）} \\
r_{down} &= \text{mag}_{down} \times \sigma_{down} &&\text{（向下惩罚）}
\end{aligned}
$$

**权重**：1.0

#### 3. 挡板高度违规惩罚

**设计意义**：对挡板偏离原位过远（>0.1 m）施加强惩罚，确保挡板不会长时间停留在高位。

**计算公式**：

$$
\begin{aligned}
\text{viol}_{height} &= \max(0, \text{err}_{home} - 0.1) &&\text{（违规量）} \\
r_{violation} &= -20.0 \times \text{viol}_{height} &&\text{（强惩罚）}
\end{aligned}
$$

**权重**：1.0

#### 4. 动作变化率惩罚

**设计意义**：惩罚动作的剧烈变化，鼓励平滑的控制策略。

**计算公式**：

$$
\begin{aligned}
r_{action} &= -\|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 &&\text{（L2 惩罚）} \\
\\
\text{其中：} \quad &\mathbf{a}_t \in \mathbb{R}^6 &&\text{（当前动作向量）}
\end{aligned}
$$

**权重**：$10^{-4}$

#### 5. 关节速度惩罚

**设计意义**：惩罚过高的关节速度，鼓励节能和平滑的运动。

**计算公式**：

$$
\begin{aligned}
r_{joint\_vel} &= -\sum_{i=1}^{6} \dot{q}_i^2 &&\text{（速度平方和惩罚）} \\
\\
\text{其中：} \quad &\dot{\mathbf{q}} = [\dot{q}_1, \dot{q}_2, \dot{q}_3, \dot{q}_4, \dot{q}_5, \dot{q}_6]^T &&\text{（关节角速度向量）}
\end{aligned}
$$

**权重**：$10^{-4}$

### 总奖励计算

$$
\begin{aligned}
R_{total} = &\ 2.0 \cdot r_{pos} \\
&+ 1.0 \cdot r_{out} \\
&+ 2.0 \cdot r_{vel} \\
&+ 4.5 \cdot r_h \\
&+ 1.0 \cdot r_{progress} \\
&+ 1.5 \cdot r_{controlled} \\
&+ 0.8 \cdot r_{bounce} \\
&+ 0.3 \cdot r_{high} \\
&+ 0.6 \cdot r_{align} \\
&+ 1.5 \cdot r_{home} \\
&+ 1.0 \cdot r_{excess} \\
&+ 1.0 \cdot r_{down} \\
&+ 1.0 \cdot r_{violation} \\
&+ 10^{-4} \cdot r_{action} \\
&+ 10^{-4} \cdot r_{joint\_vel}
\end{aligned}
$$

---

## Episode 终止条件

-   **球掉落**：球的 z 坐标 < 0.05m（接近地面）
-   **球过高**：球的 z 坐标 > 目标高度 + 1.0m（失去控制）
-   **水平偏离过远**：球的 x 或 y 坐标绝对值 > 1.5m
-   **关节速度过高**：任一关节角速度 > 2π rad/s（360°/s）
-   **超时**：Episode 时长超过最大允许时间

---

## 初始状态

### 机器人初始化

**关节角度**：

-   默认角度：[0°, 40°, 110°, 0°, -60°, 0°]
-   随机噪声：每个关节在 [-0.1, 0.1] 弧度范围内添加均匀随机噪声

**关节速度**：

-   初始化为零，带有小幅随机噪声

### 球初始化

**位置**：

-   基准位置：配置文件中的 `ball_init_pos`（默认 [0.58856, 0, 0.45] m）
-   随机噪声：在 [-0.01, 0.01] m 范围内添加均匀随机噪声

**速度**：

-   初始化为零

**姿态**：

-   四元数：[0, 0, 0, 1]（单位四元数，无旋转）

### 目标高度

每个环境的目标高度在 [0.4, 0.6] m 范围内随机采样，提升策略的泛化能力。

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=bounce_ball
```

### 2. 开始训练

```bash
python scripts/train.py task=bounce_ball/skrl.ppo
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/bounce_ball
```

### 4. 测试训练结果

```bash
python scripts/play.py env=bounce_ball
```

---

## 预期训练结果

1. **连续弹跳**：能够实现 3 次以上的连续弹跳
2. **位置控制**：球的水平位置稳定在目标位置 ± 0.05m 范围内
3. **高度控制**：球的高度稳定在目标高度 ± 0.1m 范围内
4. **速度控制**：球的向上速度保持在合理范围（0.1-1.5 m/s）内
5. **稳定控制**：能够持续 20 秒的稳定颠球而不触发终止条件

---

## 已知问题

-   **JAX 后端训练效果不佳**：当前 JAX 版本的训练效果不理想。建议使用 PyTorch 后端进行该环境的训练以获得更好的效果。
