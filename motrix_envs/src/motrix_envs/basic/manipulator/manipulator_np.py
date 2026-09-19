# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct import reward as reward_utils
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math import quaternion
from motrix_env_core.sim import (
    DofPositionLimitsQuery,
    GeomPairCollidingQuery,
    JointPositionQuery,
    JointPositionWrite,
    JointVelocityQuery,
    SensorValuesQuery,
    SitePositionQuery,
    SiteQuaternionQuery,
)
from motrix_env_core.sim.write import (
    CtrlTargetsWrite,
    JointVelocityWrite,
    KinematicBodyPositionWrite,
    KinematicBodyRotationWrite,
)
from motrix_envs.basic.manipulator.cfg import BringBallCfg

_ARM_JOINTS = (
    "arm_root",
    "arm_shoulder",
    "arm_elbow",
    "arm_wrist",
    "finger",
    "fingertip",
    "thumb",
    "thumbtip",
)

_TOUCH_SENSORS = ("palm_touch", "finger_touch", "thumb_touch", "fingertip_touch", "thumbtip_touch")

_HAND_GEOMS = (
    "hand",
    "palm1",
    "palm2",
    "thumb1",
    "thumb2",
    "thumbtip1",
    "thumbtip2",
    "finger1",
    "finger2",
    "fingertip1",
    "fingertip2",
)

_HAND_OBJECT_PAIRS = tuple((name, "ball") for name in _HAND_GEOMS)

_SIM_DATA_QUERIES = {
    "arm_pos": JointPositionQuery(joints=_ARM_JOINTS),
    "arm_vel": JointVelocityQuery(joints=_ARM_JOINTS),
    "grasp_pos": SitePositionQuery(site="grasp"),
    "grasp_quat": SiteQuaternionQuery(site="grasp"),
    "ball_pos": SitePositionQuery(site="ball"),
    "target_pos": SitePositionQuery(site="target_ball"),
    "fingertip_pos": SitePositionQuery(site="fingertip_touch"),
    "thumbtip_pos": SitePositionQuery(site="thumbtip_touch"),
    # Columns follow the declared sensors order (_TOUCH_SENSORS).
    "touch": SensorValuesQuery(sensors=_TOUCH_SENSORS),
    "hand_object_colliding": GeomPairCollidingQuery(pairs=_HAND_OBJECT_PAIRS),
}
_SIM_MODEL_QUERIES = {"dof_position_limits": DofPositionLimitsQuery()}


def _sanitize_joint_limits(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = low.copy()
    high = high.copy()
    low = np.where(np.isfinite(low), low, -np.pi)
    high = np.where(np.isfinite(high), high, np.pi)
    return low.astype(np.float32), high.astype(np.float32)


def _quat_from_y_angle(angle: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(angle)
    return quaternion.from_euler(zeros, angle, zeros)


def _quat_to_z_axis(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    return quaternion.rotate_vector(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)


def _tolerance(
    x: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> np.ndarray:
    """Vectorized tolerance reward (ported from dm_control-style reward_utils)."""
    return reward_utils.tolerance(x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=value_at_margin)


class ManipulatorBase(DirectEnv):
    _cfg: BringBallCfg
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: BringBallCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._target_writes = self.sim.write_compiler.compile(
            {
                "target_pos": KinematicBodyPositionWrite(("target_ball",)),
                "target_rot": KinematicBodyRotationWrite(("target_ball",)),
            }
        )
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        object_joints = ("ball_x", "ball_z", "ball_y")
        self._reset_program = self.sim.write_compiler.compile(
            {
                "arm_position": JointPositionWrite(_ARM_JOINTS),
                "arm_velocity": JointVelocityWrite(_ARM_JOINTS),
                "object_position": JointPositionWrite(object_joints),
                "object_velocity": JointVelocityWrite(object_joints),
            },
            reset=True,
        )
        self._arm_reset_position = self._reset_program.buffer("arm_position")
        self._arm_reset_velocity = self._reset_program.buffer("arm_velocity")
        self._object_reset_position = self._reset_program.buffer("object_position")
        self._object_reset_velocity = self._reset_program.buffer("object_velocity")
        self._cfg = cfg

        dof_lower, dof_upper = self.model.others["dof_position_limits"]
        self._joint_limit_low, self._joint_limit_high = _sanitize_joint_limits(dof_lower, dof_upper)

        # Canonical actuator order comes straight from the declared specs.
        actuator_indices = {spec.name: index for index, spec in enumerate(self.model.actuators)}
        self._grasp_act_i = int(actuator_indices["grasp"])

        self._init_action_space()
        self._init_obs_space()

        # Episode-scoped action history: full-batch buffers, reset writes the
        # done rows in place.
        self._actions = np.zeros((self._num_envs, self.num_actuators), dtype=np.float32)
        self._last_actions = np.zeros((self._num_envs, self.num_actuators), dtype=np.float32)
        self._prev_move_dist = np.zeros(self._num_envs, dtype=np.float32)

    def _init_action_space(self):
        ctrl_ranges = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32)
        self._action_space = gym.spaces.Box(
            ctrl_ranges[:, 0], ctrl_ranges[:, 1], (self.num_actuators,), dtype=np.float32
        )

    def _init_obs_space(self):
        # arm_pos(sin,cos)=16 + arm_vel=8 + touch=5 + hand=3 + object=3 + target=3 + rel=3
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (41,), dtype=np.float32)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
        actions = np.asarray(actions, dtype=np.float32)
        # Enforce actuator control limits to avoid out-of-range impulses.
        actions = np.clip(actions, self._action_space.low, self._action_space.high).astype(np.float32)
        self._last_actions[:] = self._actions
        self._actions[:] = actions
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = actions
        self._ctrl_writes.execute()
        return state

    def _touch_raw(self, rows) -> np.ndarray:
        # Columns follow the declared sensors order in the cfg.
        return self.sim_data["touch"][rows].astype(np.float32)

    def _touch_log(self, rows) -> np.ndarray:
        return np.log1p(self._touch_raw(rows))

    def _hand_pos(self, rows) -> np.ndarray:
        return self.sim_data["grasp_pos"][rows].astype(np.float32)

    def _object_pos(self, rows) -> np.ndarray:
        return self.sim_data["ball_pos"][rows].astype(np.float32)

    def _target_pos(self, rows) -> np.ndarray:
        return self.sim_data["target_pos"][rows].astype(np.float32)

    def _contact_with_object(self, rows) -> np.ndarray:
        colliding = self.sim_data["hand_object_colliding"][rows]
        return colliding.any(axis=-1)

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        qpos = inputs["arm_pos"]
        arm_pos = np.stack([np.sin(qpos), np.cos(qpos)], axis=-1).reshape(-1, 16)
        arm_vel = inputs["arm_vel"]
        touch = self._touch_log(slice(None))

        hand_pos = self._hand_pos(slice(None))
        object_pos = self._object_pos(slice(None))
        target_pos = self._target_pos(slice(None))
        rel = object_pos - target_pos

        obs = np.concatenate([arm_pos, arm_vel, touch, hand_pos, object_pos, target_pos, rel], axis=-1)
        assert obs.shape[1] == self._observation_space.shape[0]
        return state.replace(obs=obs.astype(np.float32))

    def _sample_arm_joint_angles(self, num: int) -> np.ndarray:
        low = self._joint_limit_low[: len(_ARM_JOINTS)]
        high = self._joint_limit_high[: len(_ARM_JOINTS)]
        return np.random.uniform(low=low, high=high, size=(num, len(_ARM_JOINTS))).astype(np.float32)

    def _sample_target_pose(self, num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self._cfg
        target_x = np.random.uniform(cfg.target_x_range[0], cfg.target_x_range[1], size=(num,)).astype(np.float32)
        target_z = np.random.uniform(cfg.target_z_range[0], cfg.target_z_range[1], size=(num,)).astype(np.float32)
        target_angle = np.random.uniform(cfg.target_angle_range[0], cfg.target_angle_range[1], size=(num,)).astype(
            np.float32
        )
        return target_x, target_z, target_angle

    def _set_target_mocap(
        self, env_ids: np.ndarray, target_x: np.ndarray, target_z: np.ndarray, target_angle: np.ndarray
    ):
        num = len(env_ids)
        pose = np.zeros((num, 7), dtype=np.float32)
        pose[:, 0] = target_x
        pose[:, 1] = float(self._cfg.target_y)
        pose[:, 2] = target_z
        pose[:, 3:7] = _quat_from_y_angle(target_angle)
        env_ids = np.asarray(env_ids, dtype=np.int64)
        self._target_writes.buffer("target_pos")[env_ids, 0] = pose[:, :3]
        self._target_writes.buffer("target_rot")[env_ids, 0] = pose[:, 3:7]
        self._target_writes.execute(env_ids)

    def _set_object_state(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        target_x: np.ndarray,
        target_z: np.ndarray,
        target_angle: np.ndarray,
        grasp_pos: np.ndarray,
    ):
        cfg = self._cfg
        num = dof_pos.shape[0]

        # Default: uniform in workspace.
        object_x = np.random.uniform(cfg.object_x_range[0], cfg.object_x_range[1], size=(num,)).astype(np.float32)
        object_z = np.random.uniform(cfg.object_z_range[0], cfg.object_z_range[1], size=(num,)).astype(np.float32)
        object_angle = np.random.uniform(cfg.object_angle_range[0], cfg.object_angle_range[1], size=(num,)).astype(
            np.float32
        )

        # dm_control-style object init distribution.
        r = np.random.uniform(0.0, 1.0, size=(num,)).astype(np.float32)
        in_hand = r < float(cfg.p_in_hand)
        in_target = (r >= float(cfg.p_in_hand)) & (r < float(cfg.p_in_hand + cfg.p_in_target))
        uniform = ~(in_hand | in_target)

        # Avoid initializing the object too close to the hand to prevent interpenetration / impulse explosions.
        min_dist = float(getattr(cfg, "min_object_hand_dist", 0.0))
        if min_dist > 0.0 and uniform.any():
            min_dist_sq = np.float32(min_dist * min_dist)
            max_attempts = 50
            pending = uniform.copy()
            for _ in range(max_attempts):
                if not pending.any():
                    break
                dx = object_x - grasp_pos[:, 0]
                dz = object_z - grasp_pos[:, 2]
                too_close = (dx * dx + dz * dz) < min_dist_sq
                pending = pending & too_close
                if not pending.any():
                    break
                n = int(pending.sum())
                object_x[pending] = np.random.uniform(cfg.object_x_range[0], cfg.object_x_range[1], size=(n,)).astype(
                    np.float32
                )
                object_z[pending] = np.random.uniform(cfg.object_z_range[0], cfg.object_z_range[1], size=(n,)).astype(
                    np.float32
                )

        object_x[in_target] = target_x[in_target]
        object_z[in_target] = target_z[in_target]
        object_angle[in_target] = target_angle[in_target]

        object_x[in_hand] = grasp_pos[in_hand, 0]
        object_z[in_hand] = grasp_pos[in_hand, 2]
        object_angle[in_hand] = 0.0

        dof_pos[:] = np.stack([object_x, object_z, object_angle], axis=-1)

        dof_vel[:] = 0.0
        if uniform.any():
            dof_vel[uniform, 0] = np.random.uniform(
                cfg.object_x_vel_range[0], cfg.object_x_vel_range[1], size=(int(uniform.sum()),)
            ).astype(np.float32)

    def initialize_episode(self, env_ids: np.ndarray) -> None:
        """Episode initialization: direct state construction, no physics stepping."""
        num = len(env_ids)
        row_ids = np.asarray(env_ids, dtype=np.int64)
        arm_pos = np.zeros((num, len(_ARM_JOINTS)), dtype=np.float32)
        arm_vel = np.zeros_like(arm_pos)
        object_pos = np.zeros((num, 3), dtype=np.float32)
        object_vel = np.zeros((num, 3), dtype=np.float32)

        # Optionally randomize arm joint angles and symmetrize the hand.
        if getattr(self._cfg, "randomize_arm", True):
            arm_pos[:] = self._sample_arm_joint_angles(num)
        arm_pos[:, _ARM_JOINTS.index("finger")] = arm_pos[:, _ARM_JOINTS.index("thumb")]
        arm_pos[:, _ARM_JOINTS.index("fingertip")] = arm_pos[:, _ARM_JOINTS.index("thumbtip")]

        self._arm_reset_position[env_ids] = arm_pos
        self._arm_reset_velocity[env_ids] = arm_vel
        self._object_reset_position[env_ids] = object_pos
        self._object_reset_velocity[env_ids] = object_vel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(row_ids)

        target_x, target_z, target_angle = self._sample_target_pose(num)
        grasp_pos = self.sim_data["grasp_pos"][env_ids].copy()
        self._set_object_state(object_pos, object_vel, target_x, target_z, target_angle, grasp_pos)
        self._object_reset_position[env_ids] = object_pos
        self._object_reset_velocity[env_ids] = object_vel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(row_ids)

        # The mocap target is written after the last row reset (a reset restores
        # the default mocap pose).
        self._set_target_mocap(env_ids, target_x, target_z, target_angle)
        self.sim_data.execute(row_ids)

    def reset(self, env_ids: np.ndarray) -> None:
        self.initialize_episode(env_ids)

        # Write episode-scoped state for the reset rows
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._prev_move_dist[env_ids] = 0.0


@registry.env("dm-manipulator-bring-ball")
class BringBall(ManipulatorBase):
    _cfg: BringBallCfg

    def _compute_hand_direction(self, rows) -> np.ndarray:
        """Calculates the Z-axis vector of the hand (grasp site)."""
        grasp_quat = self.sim_data["grasp_quat"][rows]
        return _quat_to_z_axis(grasp_quat)

    def _get_tip_positions(self, rows) -> tuple[np.ndarray, np.ndarray]:
        fingertip_pos = self.sim_data["fingertip_pos"][rows].astype(np.float32)
        thumbtip_pos = self.sim_data["thumbtip_pos"][rows].astype(np.float32)
        return fingertip_pos, thumbtip_pos

    def _compute_aim_direction(self, object_pos: np.ndarray, grasp_pos: np.ndarray) -> np.ndarray:
        vec_to_aim = object_pos - grasp_pos
        dist_to_aim = np.linalg.norm(vec_to_aim, axis=-1, keepdims=True)
        return vec_to_aim / (dist_to_aim + 1e-6)

    def _strict_grasp_condition(self, rows, object_pos: np.ndarray) -> np.ndarray:
        cfg = self._cfg
        height_ok = object_pos[:, 2] > float(cfg.lift_height_threshold)
        all_touch = self._touch_raw(rows)

        touch_threshold = float(cfg.touch_threshold)

        touch_ok = (
            (all_touch[..., 0] > touch_threshold)
            | (all_touch[..., 3] > touch_threshold)
            | (all_touch[..., 4] > touch_threshold)
        )

        object_contact_ok = self._contact_with_object(rows)
        return height_ok & touch_ok & object_contact_ok

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        cfg = self._cfg

        # 1. Positions
        object_pos = self._object_pos(slice(None))
        target_pos = self._target_pos(slice(None))
        grasp_pos = self._hand_pos(slice(None))

        # 2. Terminate on non-finite simulator quantities
        inputs = self.sim_data
        terminated = (
            np.isnan(inputs["arm_pos"]).any(axis=-1)
            | np.isnan(inputs["arm_vel"]).any(axis=-1)
            | np.isnan(inputs["touch"]).any(axis=-1)
            | np.isnan(grasp_pos).any(axis=-1)
            | np.isnan(object_pos).any(axis=-1)
            | np.isnan(target_pos).any(axis=-1)
        )

        # 3. Kinematics
        fingertip_pos, thumbtip_pos = self._get_tip_positions(slice(None))
        dist_finger = np.linalg.norm(fingertip_pos - object_pos, axis=-1)
        dist_thumb = np.linalg.norm(thumbtip_pos - object_pos, axis=-1)
        avg_tip_dist = ((dist_finger + dist_thumb) / 2.0).astype(np.float32)
        move_dist = np.linalg.norm(object_pos - target_pos, axis=-1).astype(np.float32)

        # 4. Dynamics
        arm_vel = self.sim_data["arm_vel"][:, :4].astype(np.float32)
        arm_speed = np.linalg.norm(arm_vel, axis=-1).astype(np.float32)
        arm_speed_step = (arm_speed * float(cfg.ctrl_dt)).astype(np.float32)

        # 5. Logic Checks
        is_grasped = self._strict_grasp_condition(slice(None), object_pos)
        contact_with_obj = self._contact_with_object(slice(None))
        hover_threshold = float(cfg.hover_close_threshold)
        is_close_to_ball = (avg_tip_dist < hover_threshold).astype(np.float32)
        grasp_mask = is_grasped.astype(np.float32)
        post_grasp_scale = 1.0 - grasp_mask * float(cfg.post_grasp_discount)

        # --- Rewards ---

        # R1: Reach
        r_reach = _tolerance(avg_tip_dist, bounds=(0.0, 0.02), margin=0.25, sigmoid="linear").astype(np.float32)
        r_reach = (r_reach * post_grasp_scale).astype(np.float32)

        # R2: Orient
        hand_dir = self._compute_hand_direction(slice(None))
        unit_vec_to_aim = self._compute_aim_direction(object_pos, grasp_pos)
        pointing_dot = np.sum(hand_dir * unit_vec_to_aim, axis=-1)

        # Dynamic tolerance
        dist_from_base = np.linalg.norm(object_pos[:, :2], axis=-1)
        orient_bound_lower = 0.95 * np.clip(dist_from_base / 0.5, 0.0, 1.0)

        r_orient_raw = 1.0 - orient_bound_lower + pointing_dot
        r_orient = np.clip(r_orient_raw, 0.0, 1.0).astype(np.float32)
        r_orient = (r_orient * post_grasp_scale).astype(np.float32)

        # R3: Pause
        r_pause = (
            _tolerance(arm_speed_step, bounds=(0.0, 0.05), margin=0.3, sigmoid="linear").astype(np.float32)
            * is_close_to_ball
        )
        r_pause = (r_pause * post_grasp_scale).astype(np.float32)

        # R4: Close
        grasp_action = self._actions[:, self._grasp_act_i].astype(np.float32)
        r_close_intent = _tolerance(
            grasp_action, bounds=(0.8, 1.0), margin=1.0, sigmoid="linear", value_at_margin=0.01
        ).astype(np.float32)

        r_approach_grasp = r_close_intent * is_close_to_ball * r_orient * r_pause * contact_with_obj.astype(np.float32)
        r_sustain_grasp = r_close_intent * grasp_mask
        r_close = (r_approach_grasp * (1.0 - grasp_mask) + r_sustain_grasp).astype(np.float32)

        # R5: Lift & Transport
        lift_h = float(cfg.lift_height_threshold)
        ball_z = object_pos[:, 2].astype(np.float32)
        r_lift_height = (
            _tolerance(ball_z, bounds=(lift_h, lift_h + 0.15), margin=0.02, sigmoid="linear", value_at_margin=0.01)
            * grasp_mask
        ).astype(np.float32)
        r_transport = (_tolerance(move_dist, bounds=(0.0, 0.01), margin=0.3, sigmoid="linear") * grasp_mask).astype(
            np.float32
        )
        r_precision = (
            _tolerance(
                move_dist,
                bounds=(0.0, 0.0),
                margin=float(cfg.precision_margin),
                sigmoid="gaussian",
                value_at_margin=float(cfg.precision_value_at_margin),
            )
            * grasp_mask
        ).astype(np.float32)
        lift_height_weight = float(cfg.lift_height_weight)
        transport_weight = float(cfg.transport_weight)
        lift_norm = max(lift_height_weight + transport_weight, 1e-6)
        r_lift = ((lift_height_weight * r_lift_height + transport_weight * r_transport) / lift_norm).astype(np.float32)

        first_step = state.episode_steps == 0
        prev_move_dist = np.where(first_step, move_dist, self._prev_move_dist)
        progress_clip = float(cfg.transport_progress_clip)
        progress = (prev_move_dist - move_dist) / max(progress_clip, 1e-6)
        progress = np.clip(progress, -1.0, 1.0).astype(np.float32)
        r_progress = (progress * float(cfg.transport_progress_scale) * grasp_mask).astype(np.float32)
        self._prev_move_dist[:] = move_dist.astype(np.float32)

        # --- Penalties ---
        all_touch = self._touch_raw(slice(None))
        side_touch_sum = (all_touch[..., 1] + all_touch[..., 2]).astype(np.float32)
        penalty_side = (
            -float(cfg.side_penalty_scale) * np.tanh(side_touch_sum * float(cfg.side_penalty_tanh_scale))
        ).astype(np.float32)

        hover_phase = (is_close_to_ball > 0.5) & (~contact_with_obj)
        penalty_hover = (-float(cfg.hover_penalty_scale) * hover_phase.astype(np.float32)).astype(np.float32)

        # --- Total ---
        reach_w = float(cfg.reach_weight)
        orient_w = float(cfg.orient_weight)
        pause_w = float(cfg.pause_weight)
        close_w = float(cfg.close_weight)
        lift_w = float(cfg.lift_reward_weight)
        precision_w = float(cfg.precision_weight)
        weight_sum = max(reach_w + orient_w + pause_w + close_w + lift_w + precision_w, 1e-6)
        reward = (
            (
                reach_w * r_reach
                + orient_w * r_orient
                + pause_w * r_pause
                + close_w * r_close
                + lift_w * r_lift
                + precision_w * r_precision
            )
            / weight_sum
            + penalty_side
            + penalty_hover
            + r_progress
        ).astype(np.float32)

        reward = np.where(terminated, 0.0, reward)

        state.reward_terms = {
            "reach": r_reach,
            "orient": r_orient,
            "close": r_close,
            "lift": r_lift,
            "transport": r_transport,
            "precision": r_precision,
            "progress": r_progress,
            "total": reward,
        }

        state.metrics = {
            "pointing_dot": pointing_dot,
            "is_grasped": is_grasped.astype(np.float32),
            "avg_tip_dist": avg_tip_dist,
            "move_dist": move_dist,
            "transport_reward": r_transport,
            "precision_reward": r_precision,
            "progress_reward": r_progress,
        }

        return state.replace(reward=reward, terminated=terminated)
