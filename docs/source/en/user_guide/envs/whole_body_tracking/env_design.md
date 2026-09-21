# Task Environment Design

`ManagerEnv` advances one reference-motion frame per control step and compares the robot state with that frame's targets.
`WbtManagerEnvCfg` selects the motion, `tracked_body_names`, and `reference_body_name`; `scene.objs.robot` supplies the robot model,
default key pose, base link, and actuators. Reference joint states are policy commands in the observation, but the action is
still a position residual around the robot's default pose, and rewards primarily compare body poses and velocities.

Let $A$ be the number of position actuators and $B$ the number of entries in `tracked_body_names`.

## Action space

The action space is an $A$-dimensional continuous `Box`, with one dimension per position actuator. Let
$q_{default,i}$ be the default joint angle, $[q_{min,i},q_{max,i}]$ the actuator `ctrl_range` declared directly or inherited
from its target joint, and $\alpha=control\_config.action\_scale$. The action bound is

$$
b_i=\frac{\max\left(\left|q_{min,i}-q_{default,i}\right|,
\left|q_{max,i}-q_{default,i}\right|\right)}{\alpha},
\qquad a_i\in[-b_i,b_i].
$$

The environment writes the following target to the position actuator:

$$
q_{target,i}=q_{default,i}+s_i a_i.
$$

Here $s_i=\alpha$. When `action_scales_by_effort_limit_over_p_gain=True`,
$s_i=\alpha\,e_i/k_{p,i}$. For actuator `force_range` $(f_{min,i},f_{max,i})$,
$e_i=\max(|f_{min,i}|,|f_{max,i}|)$, and $k_{p,i}$ comes from the same position actuator. Every position actuator must define
`force_range` when this mode is enabled. RobotCfg/MJCF remains the single source of physical force limits: the same runtime
range supplies both this target scaling input and the actual actuator force clamp.

## Observation space

Reference joint arrays are reordered to model actuator order. Position and orientation differences are expressed in the
current `reference_body_name` frame, and orientation uses the first two columns of a rotation matrix as a 6D representation.
Base velocities are expressed in the robot base-link frame.

| Observation                                | Actor | Critic | Meaning                                                                           |
| ------------------------------------------ | ----: | -----: | --------------------------------------------------------------------------------- |
| Reference joint position and velocity      |  $2A$ |   $2A$ | `joint_pos` and `joint_vel` at the current motion frame                           |
| Relative reference-body position           |     — |      3 | Motion reference-body position relative to the current robot reference body       |
| Relative reference-body orientation        |     6 |      6 | 6D motion reference-body orientation relative to the current robot reference body |
| Current tracked-body relative positions    |     — |   $3B$ | Each tracked body relative to the current robot reference body                    |
| Current tracked-body relative orientations |     — |   $6B$ | 6D orientation of each tracked body relative to the current robot reference body  |
| Base-frame linear velocity                 |     — |      3 | Base-link linear velocity in the base-link frame                                  |
| Base-frame angular velocity                |     3 |      3 | Base-link angular velocity in the base-link frame                                 |
| Joint-position residual                    |   $A$ |    $A$ | Current joint angle minus the default pose                                        |
| Joint velocity                             |   $A$ |    $A$ | Current joint velocity                                                            |
| Current action                             |   $A$ |    $A$ | Most recent policy action                                                         |

The actor dimension is $5A+9$, and the critic dimension is $5A+9B+15$. The built-in tasks have these dimensions:

| Robot             | $A$ | $B$ | Actor | Critic |
| ----------------- | --: | --: | ----: | -----: |
| Unitree G1 29-DoF |  29 |  14 |   154 |    286 |
| Dex-EVT           |  23 |  14 |   124 |    256 |
| Booster K1        |  22 |  13 |   119 |    242 |

The actor receives uniform noise on reference-body orientation, base angular velocity, joint position, and joint velocity,
with amplitudes from `observation_noise`. Critic observations are noise-free. The current implementation applies no separate
observation-scale normalization.

## Reward design

Global reference terms compare the motion and robot `reference_body_name` directly in world coordinates. Relative-body terms
first align the motion with the current robot reference body's horizontal position and yaw, then compare all tracked bodies.
This makes the relative body-configuration reward independent of the current horizontal translation and yaw.

| Reward term                                  | Computation                                                                                                 | Design purpose                                       |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `motion_global_ref_position_error_exp`       | Apply an exponential kernel to squared reference-body world-position error                                  | Track the motion's overall translation and height    |
| `motion_global_ref_orientation_error_exp`    | Apply an exponential kernel to squared reference-body rotation distance                                     | Track overall body orientation                       |
| `motion_relative_body_position_error_exp`    | Apply an exponential kernel to the mean squared relative-position error over tracked bodies                 | Reproduce the spatial arrangement of limbs and torso |
| `motion_relative_body_orientation_error_exp` | Apply an exponential kernel to the mean squared rotation distance over tracked bodies                       | Reproduce the orientation of each body part          |
| `motion_global_body_lin_vel`                 | Apply an exponential kernel to mean squared world linear-velocity error over tracked bodies                 | Match motion timing and translational velocity       |
| `motion_global_body_ang_vel`                 | Apply an exponential kernel to mean squared world angular-velocity error over tracked bodies                | Match body rotational velocity                       |
| `action_rate_l2`                             | Sum squared differences between current and previous actions                                                | Reduce abrupt control-target changes                 |
| `limits_dof_pos`                             | Sum distance outside the soft joint range and cap it with `limits_dof_pos_cap`                              | Keep the policy away from joint limits               |
| `undesired_contacts`                         | Count robot links whose net contact force exceeds the threshold and are absent from `allowed_contact_links` | Suppress body contacts not required by the motion    |

Each raw term is multiplied by its `WbtRewardScales` weight and then by `ctrl_dt`. Negative weights turn `action_rate_l2`,
`limits_dof_pos`, and `undesired_contacts` into penalties. Final weighted terms are written to `state.reward_terms`.

## Termination conditions

| Type                        | Status       | Condition                                                                                                              | Meaning                                                                         |
| --------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Reference-height error      | `terminated` | Reference-body world-Z error exceeds `bad_ref_pos_threshold`                                                           | Overall robot height has clearly diverged from the motion                       |
| Reference-orientation error | `terminated` | The projected-gravity Z-component difference between motion and robot reference bodies exceeds `bad_ref_ori_threshold` | The robot has fallen or its overall orientation has diverged                    |
| Key-body height error       | `terminated` | Maximum Z error over `bad_motion_body_pos_body_names` exceeds `bad_motion_body_pos_threshold`                          | A key part such as a foot or hand is badly misplaced                            |
| Invalid joint position      | `terminated` | A joint position is non-finite, or maximum distance beyond a hard limit exceeds the threshold                          | Stop invalid or diverging joint states                                          |
| Invalid joint velocity      | `terminated` | A joint velocity is non-finite, or its maximum absolute value exceeds `bad_dof_vel_abs`                                | Stop numerical velocity spikes before they grow                                 |
| Time limit                  | `truncated`  | Training reaches `max_episode_seconds`, 10 s in built-in configs                                                       | Normal training time limit, not bad tracking                                    |
| Motion final frame          | Neither      | `motion_steps` reaches the clip end                                                                                    | Resample and reset motion state during training; restart at frame 0 during play |

`undesired_contacts` contributes only a reward penalty and does not terminate the episode. Termination rates and error means
are written to `state.metrics`.

## Reset logic

On reset, the environment copies floating-root pose and velocity plus joint position and velocity from a selected motion
frame, resets the model, and recomputes kinematics. Current and previous actions are both cleared.

| Randomized quantity              | Sampling                                                                                                        | Timing                              | Purpose                                                     |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------- |
| Motion start frame               | Sample from the adaptive distribution by default; `start_at_timestep_zero_prob` can force a fraction to frame 0 | Episode reset and training clip end | Cover the complete motion and revisit failure-heavy regions |
| Joint position                   | Add uniform noise to reference joint angles, then clip to hard joint ranges                                     | Training reset                      | Improve recovery from initial pose error                    |
| Root position and orientation    | Add uniform noise using `reset_noise.root_pos` and `root_rot`                                                   | Training reset                      | Broaden the initial global-pose distribution                |
| Root linear and angular velocity | Add uniform noise using `reset_noise.root_lin_vel` and `root_ang_vel`                                           | Training reset                      | Improve robustness to initial velocity disturbances         |
| Actor observation noise          | Add uniform noise to reference orientation, base angular velocity, joint position, and joint velocity           | Every observation build             | Improve robustness to observation error                     |

The adaptive sampler records only `terminated` motion frames as failures and updates its probabilities with an exponential
moving average plus a uniform exploration floor. These are initial-state, observation, and task-curriculum randomization.
The current WBT environment does not randomize physical parameters such as mass, inertia, friction, PD gains, or actuator delay.
