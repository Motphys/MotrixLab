# Ping Pong Ball Bouncing

Train a single-arm robotic manipulator to control a paddle for continuous ball bouncing, maintaining the ball at a target height and position.

```{video} /_static/videos/bounce_ball.mp4
:poster: _static/images/poster/bounce_ball.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

Bounce Ball is a single-arm robotic manipulation task using a 6-DOF Peitian AIR4-560 industrial robotic arm to control the position of an end-effector paddle. The agent controls the position changes of the arm's 6 joints as actions, making the ping pong ball bounce continuously on the paddle and keeping it as close as possible to the target height and target horizontal position.

---

## Action Space

| Item          | Details                         |
| ------------- | ------------------------------- |
| **Type**      | `Box(-1.0, 1.0, (6,), float32)` |
| **Dimension** | 6                               |

The joints correspond as follows:

| Index | Action Meaning (Joint Position Change)  | Min Value | Max Value | Corresponding XML Name |
| ----: | --------------------------------------- | :-------: | :-------: | :--------------------: |
|     0 | Joint1 (Base Rotation) Position Change  |    -1     |     1     |        `Joint1`        |
|     1 | Joint2 (Upper Arm) Position Change      |    -1     |     1     |        `Joint2`        |
|     2 | Joint3 (Forearm) Position Change        |    -1     |     1     |        `Joint3`        |
|     3 | Joint4 (Wrist Rotation) Position Change |    -1     |     1     |        `Joint4`        |
|     4 | Joint5 (Wrist Pitch) Position Change    |    -1     |     1     |        `Joint5`        |
|     5 | Joint6 (Wrist Rotation) Position Change |    -1     |     1     |        `Joint6`        |

---

## Observation Space

| Item          | Details                          |
| ------------- | -------------------------------- |
| **Type**      | `Box(-inf, inf, (29,), float32)` |
| **Dimension** | 29                               |

The observation space consists of the following parts (in order):

| Part              | Content Description                             | Dimension | Remarks                                                                          |
| ----------------- | ----------------------------------------------- | --------- | -------------------------------------------------------------------------------- |
| **dof_pos**       | Position information for each degree of freedom | 13        | First 6 are arm joints, last 7 are ball's free joint (3 position + 4 quaternion) |
| **dof_vel**       | Velocity information for each degree of freedom | 12        | Velocity is derivative of position                                               |
| **paddle_pos**    | Paddle position information                     | 3         | x, y, z coordinates of paddle center                                             |
| **target_height** | Target height                                   | 1         | Target height for current environment                                            |

| Index | Observation                           | Min Value | Max Value | XML Name  | Type (Unit)              |
| ----- | ------------------------------------- | --------- | --------- | --------- | ------------------------ |
| 0-5   | Arm Joint Angles                      | -Inf      | Inf       | Joint1-6  | Angle (rad)              |
| 6-8   | Ball Position [x, y, z]               | -Inf      | Inf       | ball_link | Position (m)             |
| 9-12  | Ball Orientation Quaternion [w,x,y,z] | -Inf      | Inf       | ball_link | Quaternion               |
| 13-18 | Arm Joint Angular Velocities          | -Inf      | Inf       | Joint1-6  | Angular Velocity (rad/s) |
| 19-24 | Ball Velocity [vx,vy,vz,wx,wy,wz]     | -Inf      | Inf       | ball_link | Velocity (m/s, rad/s)    |
| 25-27 | Paddle Position [x, y, z]             | -Inf      | Inf       | blocker   | Position (m)             |
| 28    | Target Height                         | -Inf      | Inf       | -         | Position (m)             |

---

## Reward Function

The reward function uses a composite design with multiple reward and penalty terms to guide the robot to learn a stable ball bouncing strategy. All reward parameters can be adjusted through the configuration file.

### Main Reward Terms

#### 1. Horizontal Position Reward

**Design Rationale**: This is the core reward term, ensuring the ball stays directly above the paddle. Through a vertical distance weighting mechanism, when the ball is close to the paddle (about to hit), the horizontal position requirement is stricter, guiding the strategy to align precisely at critical moments.

**Formula**:

$$
\begin{aligned}
\text{err}_{xy} &= \sqrt{(x_{ball} - x_{target})^2 + (y_{ball} - y_{target})^2} &&\text{(Horizontal position error)} \\
d_{vert} &= |z_{ball} - z_{paddle}| &&\text{(Vertical distance)} \\
w_{vert} &= e^{-d_{vert} / \sigma_{vert}} &&\text{(Vertical distance weight)} \\
\sigma_{pos} &= \sigma_{base} \times (1.0 + k_{weight} \times w_{vert}) &&\text{(Adaptive scale)} \\
r_{pos} &= e^{-\frac{\text{err}_{xy}^2}{2\sigma_{pos}^2}} &&\text{(Gaussian reward)} \\
\\
\text{Where:} \quad &\sigma_{vert} = 0.15 \text{ m} &&\text{(Vertical distance scale)} \\
&\sigma_{base} = 0.1 \text{ m} &&\text{(Base horizontal scale)} \\
&k_{weight} = 3.0 &&\text{(Weight factor)} \\
&x_{target} = 0.58856 \text{ m}, \, y_{target} = 0.0 \text{ m} &&\text{(Target position)}
\end{aligned}
$$

**Weight**: 2.0

#### 2. Out of Position Penalty

**Design Rationale**: Applies strong penalty for severe deviation from target position to prevent the ball from flying out of control range. Uses sigmoid function for smooth transition, avoiding discontinuous reward function.

**Formula**:

$$
r_{out} = -\frac{2.0}{1 + e^{-(\text{err}_{xy} - 0.05) / 0.03}} \qquad \text{(Sigmoid penalty)}
$$

**Weight**: 1.0

#### 3. Velocity Matching Reward

**Design Rationale**: Based on projectile motion physics, encourages the ball's trajectory to have the desired velocity (0.5 m/s) at target height. This ensures the ball doesn't pass through the target height too fast or too slow, facilitating stable control.

**Formula**:

$$
\begin{aligned}
\Delta h &= h_{target} - z_{ball} &&\text{(Height difference)} \\
v_{desired} &= 0.5 \text{ m/s} &&\text{(Desired velocity)} \\
\\
\text{Case 1:}&\text{ Ball moving upward and below target height} \\
v_{z,up}^2 &= v_z^2 - 2g\Delta h &&\text{(Energy conservation)} \\
v_{at\_target,up} &= \sqrt{\max(0, v_{z,up}^2)} &&\text{(Upward arrival velocity)} \\
\\
\text{Case 2:}&\text{ Ball moving downward and above target height} \\
v_{z,down}^2 &= v_z^2 + 2g|\Delta h| &&\text{(Energy conservation)} \\
v_{at\_target,down} &= -\sqrt{\max(0, v_{z,down}^2)} &&\text{(Downward arrival velocity)} \\
\\
\text{Case 3:}&\text{ Ball near target height (} |\Delta h| < 0.05 \text{ m)} \\
v_{at\_target,near} &= v_z &&\text{(Current velocity)} \\
\\
\text{Smooth combination:}& \\
\sigma_{up} &= \frac{1}{1 + e^{-v_z / 0.2}} &&\text{(Upward motion weight)} \\
\sigma_{below} &= \frac{1}{1 + e^{-\Delta h / 0.02}} &&\text{(Below target weight)} \\
\sigma_{down} &= 1 - \sigma_{up} &&\text{(Downward motion weight)} \\
\sigma_{above} &= 1 - \sigma_{below} &&\text{(Above target weight)} \\
w_{near} &= e^{-\frac{\Delta h^2}{2 \times 0.01^2}} &&\text{(Near target weight)} \\
\\
v_{at\_target} &= v_{at\_target,up} \cdot \sigma_{up} \cdot \sigma_{below} \\
&\quad + v_{at\_target,down} \cdot \sigma_{down} \cdot \sigma_{above} \\
&\quad + v_{at\_target,near} \cdot w_{near} &&\text{(Weighted combination)} \\
\\
\text{err}_{vel} &= |v_{at\_target} - v_{desired}| &&\text{(Velocity error)} \\
r_{vel} &= e^{-\frac{\text{err}_{vel}^2}{2 \times 0.8^2}} &&\text{(Gaussian reward)}
\end{aligned}
$$

**Weight**: 2.0

#### 4. Height Reward

**Design Rationale**: Directly encourages the ball to approach target height, one of the core task objectives. Higher weight (4.5) ensures the strategy prioritizes height control. Target height is randomly sampled (0.3-0.6 m) in each environment to improve policy generalization.

**Formula**:

$$
\begin{aligned}
\text{err}_h &= |z_{ball} - h_{target}| &&\text{(Height error)} \\
r_h &= e^{-\frac{\text{err}_h^2}{2 \times 0.15^2}} &&\text{(Gaussian reward)}
\end{aligned}
$$

**Weight**: 4.5

#### 5. Height Progress Reward

**Design Rationale**: Encourages the ball to reach higher positions, helping the strategy quickly learn to hit the ball upward in early training, avoiding the "no-hit" local optimum.

**Formula**:

$$
r_{progress} = \max(0, z_{ball} - 0.2) \times 2.0 \qquad \text{(Linear reward)}
$$

**Weight**: 1.0

#### 6. Controlled Upward Velocity Reward

**Design Rationale**: Only rewards upward velocity when the ball's horizontal position is good, avoiding "random hitting" behavior. Ideal velocity is calculated from physics formula, ensuring the ball can exactly reach target height. This reward guides the strategy to learn precise hitting force.

**Formula**:

$$
\begin{aligned}
q_{pos} &= e^{-\frac{\text{err}_{xy}^2}{2 \times 0.02^2}} &&\text{(Position quality)} \\
v_{ideal} &= \sqrt{2g \times \max(0, \Delta h)} &&\text{(Ideal launch velocity)} \\
v_{ideal} &\in [0.5, 3.0] \text{ m/s} &&\text{(Limit range)} \\
q_{vel} &= e^{-\frac{(v_z - v_{ideal})^2}{2 \times 0.5^2}} &&\text{(Velocity quality)} \\
\sigma_{up} &= \frac{1}{1 + e^{-v_z / 0.1}} &&\text{(Upward mask)} \\
r_{controlled} &= q_{pos} \times q_{vel} \times \sigma_{up} \times \text{clip}(v_z, 0, 1.5) &&\text{(Combined reward)}
\end{aligned}
$$

**Weight**: 1.5

#### 7. Consecutive Bounces Reward

**Design Rationale**: Encourages multiple consecutive successful bounces, guiding the strategy to learn stable long-term control. Uses logarithmic function to avoid infinite reward growth, while requiring good ball position to give reward.

**Formula**:

$$
\begin{aligned}
q_{bounce} &= e^{-\frac{\text{err}_{xy}^2}{2 \times 0.05^2}} &&\text{(Bounce position quality)} \\
r_{bounce\_log} &= 0.5 \times \log(n_{bounces} + 1) &&\text{(Logarithmic reward)} \\
r_{bounce} &= r_{bounce\_log} \times q_{bounce} \times \mathbb{1}_{n_{bounces} > 0} &&\text{(Conditional reward)}
\end{aligned}
$$

**Weight**: 0.8

#### 8. High Bounce Count Reward

**Design Rationale**: Gives extra reward for high bounce counts (≥3), further encouraging long-term stable control. Uses sigmoid activation function for smooth reward growth.

**Formula**:

$$
\begin{aligned}
\sigma_{high} &= \frac{1}{1 + e^{-(n_{bounces} - 2.0) / 0.5}} &&\text{(High bounce activation)} \\
r_{high} &= n_{bounces} \times 0.15 \times q_{bounce} \times \sigma_{high} &&\text{(Extra reward)}
\end{aligned}
$$

**Weight**: 0.3

#### 9. Paddle-Ball Horizontal Alignment Reward

**Design Rationale**: Encourages the paddle to actively move directly below the ball rather than waiting for the ball to fall. The closer the vertical distance, the greater the weight, requiring more precise alignment at the moment of hitting. Extra reward (boost) is given at bounce moment to reinforce correct hitting behavior.

**Formula**:

$$
\begin{aligned}
\text{err}_{align} &= \|(\mathbf{x}_{ball}, \mathbf{y}_{ball}) - (\mathbf{x}_{paddle}, \mathbf{y}_{paddle})\|_2 &&\text{(Alignment error)} \\
w_{prox} &= e^{-d_{vert} / 0.1} &&\text{(Vertical proximity weight)} \\
q_{align} &= e^{-\frac{\text{err}_{align}^2}{2 \times 0.03^2}} \times (1.0 + 2.0 \times w_{prox}) &&\text{(Alignment quality)} \\
k_{boost} &= \begin{cases} 3.0 & \text{if bounce detected} \\ 1.0 & \text{otherwise} \end{cases} &&\text{(Bounce boost)} \\
r_{align} &= q_{align} \times k_{boost} \times 0.3 &&\text{(Alignment reward)}
\end{aligned}
$$

**Weight**: 0.6

#### 10. Paddle Home Position Reward

**Design Rationale**: Encourages the paddle to return to home position when the ball is far away, avoiding the paddle staying at high position for long time. Uses distance dynamic factor: when ball is far from paddle ($d_{vert} > 0.15$ m), increase reward to encourage quick return; when ball is close, decrease reward to allow paddle to move up for hitting. This design makes paddle motion more energy-efficient and natural.

**Formula**:

$$
\begin{aligned}
\text{err}_{home} &= |z_{paddle} - z_{home}| &&\text{(Home deviation)} \\
k_{dist} &= 1.0 + \frac{0.5}{1 + e^{-(d_{vert} - 0.15) / 0.03}} &&\text{(Distance factor)} \\
q_{home} &= e^{-\frac{\text{err}_{home}^2}{2 \times 0.05^2}} &&\text{(Home quality)} \\
r_{home} &= q_{home} \times k_{dist} &&\text{(Home reward)} \\
\\
\text{Where:} \quad &z_{home} = 0.05 \text{ m} &&\text{(Paddle home height)}
\end{aligned}
$$

**Weight**: 1.5

### Penalty Terms

#### 1. Excessive Upward Velocity Penalty

**Design Rationale**: Prevents ball velocity from being too fast (>3.5 m/s) and losing control, ensuring ball motion stays within controllable range.

**Formula**:

$$
r_{excess} = -\frac{1.0}{1 + e^{-(v_z - 3.5) / 0.3}} \qquad \text{(Sigmoid penalty)}
$$

**Weight**: 1.0

#### 2. Downward Velocity Penalty

**Design Rationale**: Penalizes ball moving downward ($v_z < -0.2$ m/s), encouraging the strategy to hit the ball in time, avoiding free fall.

**Formula**:

$$
\begin{aligned}
\text{mag}_{down} &= -v_z \times \text{clip}(-v_z \times 0.3, 0, 0.5) &&\text{(Downward magnitude)} \\
\sigma_{down} &= \frac{1}{1 + e^{(v_z + 0.2) / 0.2}} &&\text{(Downward activation)} \\
r_{down} &= \text{mag}_{down} \times \sigma_{down} &&\text{(Downward penalty)}
\end{aligned}
$$

**Weight**: 1.0

#### 3. Paddle Height Violation Penalty

**Design Rationale**: Applies strong penalty when paddle deviates too far from home position (>0.1 m), ensuring paddle doesn't stay at high position for long time.

**Formula**:

$$
\begin{aligned}
\text{viol}_{height} &= \max(0, \text{err}_{home} - 0.1) &&\text{(Violation amount)} \\
r_{violation} &= -20.0 \times \text{viol}_{height} &&\text{(Strong penalty)}
\end{aligned}
$$

**Weight**: 1.0

#### 4. Action Change Rate Penalty

**Design Rationale**: Penalizes drastic action changes, encouraging smooth control strategy.

**Formula**:

$$
\begin{aligned}
r_{action} &= -\|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 &&\text{(L2 penalty)} \\
\\
\text{Where:} \quad &\mathbf{a}_t \in \mathbb{R}^6 &&\text{(Current action vector)}
\end{aligned}
$$

**Weight**: $10^{-4}$

#### 5. Joint Velocity Penalty

**Design Rationale**: Penalizes excessive joint velocities, encouraging energy-efficient and smooth motion.

**Formula**:

$$
\begin{aligned}
r_{joint\_vel} &= -\sum_{i=1}^{6} \dot{q}_i^2 &&\text{(Velocity squared sum penalty)} \\
\\
\text{Where:} \quad &\dot{\mathbf{q}} = [\dot{q}_1, \dot{q}_2, \dot{q}_3, \dot{q}_4, \dot{q}_5, \dot{q}_6]^T &&\text{(Joint angular velocity vector)}
\end{aligned}
$$

**Weight**: $10^{-4}$

### Total Reward Calculation

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

## Initial State

### Robot Initialization

**Joint Angles**:

-   Default angles: [0°, 40°, 110°, 0°, -60°, 0°]
-   Random noise: Uniform random noise in [-0.1, 0.1] radians added to each joint

**Joint Velocities**:

-   Initialized to zero with small random noise

### Ball Initialization

**Position**:

-   Base position: `ball_init_pos` from config file (default [0.58856, 0, 0.45] m)
-   Random noise: Uniform random noise in [-0.01, 0.01] m

**Velocity**:

-   Initialized to zero

**Orientation**:

-   Quaternion: [0, 0, 0, 1] (identity quaternion, no rotation)

### Target Height

Target height for each environment is randomly sampled in [0.4, 0.6] m range to improve policy generalization.

---

## Episode Termination Conditions

-   **Ball Falls**: Ball z-coordinate < 0.05m (near ground)
-   **Ball Too High**: Ball z-coordinate > target height + 1.0m (lost control)
-   **Horizontal Deviation Too Far**: Ball x or y coordinate absolute value > 1.5m
-   **Joint Velocity Too High**: Any joint angular velocity > 2π rad/s (360°/s)
-   **Timeout**: Episode duration exceeds maximum allowed time

---

## Usage Guide

### 1. Environment Preview

```bash
python scripts/view.py env=bounce_ball
```

### 2. Start Training

```bash
python scripts/train.py task=bounce_ball/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/bounce_ball
```

### 4. Test Training Results

```bash
python scripts/play.py env=bounce_ball
```

---

## Expected Training Results

1. Consecutive Bouncing: Capable of achieving 3 or more consecutive bounces
2. Position Control: Ball's horizontal position (x-coordinate) stable within target position ± 0.05m range
3. Height Control: Ball's height stable within target height 0.8 ± 0.1m range
4. Velocity Control: Ball's upward velocity maintained within reasonable range (0.1-1.5 m/s)
5. Stable Control: Capable of maintaining stable bouncing for 20 seconds without dropping

---

## Known Issues

-   **JAX Backend Training Performance**: The JAX version currently shows suboptimal training performance. For better results, it is recommended to use the PyTorch backend for this environment.
