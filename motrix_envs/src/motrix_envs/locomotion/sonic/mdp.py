# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import cast

import gymnasium as gym
import numpy as np
from omegaconf import MISSING

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    ActionCfg,
    ActionTerm,
    CommandCfg,
    CommandTerm,
    ManagerContext,
    ManagerEnv,
    ObservationTermCfg,
    ObsTerm,
    ResetTerm,
    ResetTermCfg,
    RewardTerm,
    RewardTermCfg,
    SharedArray,
    TerminationTerm,
    TerminationTermCfg,
    dispatch,
    kernel_data,
    metric,
    njit,
)
from motrix_env_core.manager.math.quaternion import inverse as quat_inverse
from motrix_env_core.manager.math.quaternion import mul as quat_mul
from motrix_env_core.manager.math.quaternion import (
    rotate_inverse,
    rotate_vector,
    rotation_distance,
)
from motrix_env_core.mdp.rewards import (
    ActionRateRewardCfg as ActionRateRewardCfg,
)
from motrix_env_core.numba.kernel_data import Map
from motrix_env_core.numba.manager.commands import ResetContext
from motrix_env_core.sim.write import ActuatorDampingWrite, ActuatorKpWrite
from motrix_envs.locomotion.wbt.mdp.command import _adaptive_sampling_probabilities, _sample_motion_step
from motrix_envs.locomotion.wbt.mdp.rewards import (
    DofLimitRewardCfg as DofLimitRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalBodyAngularVelocityRewardCfg as GlobalBodyAngularVelocityRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalBodyLinearVelocityRewardCfg as GlobalBodyLinearVelocityRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalRefOrientationRewardCfg as GlobalRefOrientationRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalRefPositionRewardCfg as GlobalRefPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    RelativeBodyOrientationRewardCfg as RelativeBodyOrientationRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    RelativeBodyPositionRewardCfg as RelativeBodyPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    UndesiredContactsRewardCfg as UndesiredContactsRewardCfg,
)
from motrix_envs.motion import MotrixMotion
from motrix_envs.motion.sonic import SonicMotionClip

G1_SONIC_JOINTS = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
G1_SONIC_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
G1_SONIC_EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
SONIC_TERMINATION_REASON_NAMES = ("anchor_pos_z", "anchor_ori", "ee_body_pos_z", "feet_pos", "time_out", "clip_end")
G1_SONIC_ACTION_SCALE = 2.0
SONIC_VECTOR_DIM = 3
SONIC_ROTATION_REPRESENTATION_DIM = 2 * SONIC_VECTOR_DIM
SONIC_ACTOR_JOINT_FEATURES = 3
SONIC_ENCODER_COUNT = 2
SONIC_WRIST_POLICY_INDICES = (23, 24, 25, 26, 27, 28)
# motrixlab SONIC actuator contract: (kp, kd, effort_limit, armature).
_SONIC_ACTUATOR_PARAMETERS = {
    **{
        n: (99.098427777, 6.308801854, 139.0, 0.025101925)
        for n in (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
        )
    },
    **{
        n: (40.179238471, 2.557889765, 88.0, 0.010177520)
        for n in ("left_hip_yaw_joint", "right_hip_yaw_joint", "waist_yaw_joint")
    },
    **{
        n: (28.501246196, 1.814445687, 50.0, 0.00721945)
        for n in (
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        )
    },
    **{
        n: (14.250623098, 0.907222843, 25.0, 0.003609725)
        for n in (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        )
    },
    **{
        n: (16.778327481, 1.068141502, 5.0, 0.00425)
        for n in ("left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
    },
}
G1_POLICY_JOINT_NAMES = G1_SONIC_JOINTS
SONIC_ROOT_BODY_INDEX = 0  # ``pelvis`` is the first tracked body in SONIC order.


@njit(inline="always")
def _sonic_actor_frame_dim(joint_count: int) -> int:
    return SONIC_ACTOR_JOINT_FEATURES * joint_count + 2 * SONIC_VECTOR_DIM


@njit(inline="always")
def _g1_reference_frame_dim(joint_count: int) -> int:
    return 2 * joint_count + SONIC_ROTATION_REPRESENTATION_DIM


@njit(inline="always")
def _smpl_reference_frame_dim(joint_count: int) -> int:
    return joint_count * SONIC_VECTOR_DIM + SONIC_ROTATION_REPRESENTATION_DIM + len(SONIC_WRIST_POLICY_INDICES)


def _sonic_action_scale(base_scale: float) -> np.ndarray:
    """Expand a scalar scale as ``scale * effort_limit / kp`` in policy order."""
    if not np.isfinite(base_scale) or base_scale <= 0.0:
        raise ValueError("SONIC base action scale must be positive and finite")
    return np.asarray(
        [base_scale * _SONIC_ACTUATOR_PARAMETERS[n][2] / _SONIC_ACTUATOR_PARAMETERS[n][0] for n in G1_SONIC_JOINTS],
        dtype=np.float32,
    )


def _sonic_policy_action_scale() -> np.ndarray:
    return _sonic_action_scale(0.25)


@kernel_data
class SonicMotionCommand(CommandTerm):
    clip: SonicMotionClip
    reference_index: np.int64
    target_body_position_relative: np.ndarray
    target_body_orientation_relative: np.ndarray
    adaptive_bin_failed_count: SharedArray
    adaptive_current_bin_failed_count: SharedArray
    sampling_cdf: SharedArray
    start_at_timestep_zero_prob: np.float32
    hold_at_clip_end: bool
    uniform_ratio: np.float32
    alpha: np.float32
    kernel_size: np.int64
    kernel_lambda: np.float32
    encoder_index: np.ndarray
    reset_counter: np.ndarray
    num_future_frames: np.int64
    steps: np.ndarray = metric(name="motion_step", dtype=np.float32)
    clip_ended: np.ndarray = metric()

    @dispatch
    def update(self, ctx: ManagerContext) -> None:
        step = self.steps[0]
        motion_reference_body_pos_w = self.clip.reference_body_pos_w[step]
        motion_reference_body_quat_w = self.clip.reference_body_quat_w[step]
        motion_tracked_bodies_pos_w = self.clip.tracked_bodies_pos_w[step]
        motion_tracked_bodies_quat_w = self.clip.tracked_bodies_quat_w[step]
        robot_reference_body_pos_w = ctx.sim["tracked_body_pos"][self.reference_index]
        robot_reference_body_quat_w = ctx.sim["tracked_body_quat"][self.reference_index]
        out_body_pos = self.target_body_position_relative
        out_body_quat = self.target_body_orientation_relative

        delta_quat = out_body_quat[0]
        quat_inverse(motion_reference_body_quat_w, delta_quat)
        quat_mul(robot_reference_body_quat_w, delta_quat, delta_quat)
        dx, dy, dz, dw = delta_quat
        half_yaw = np.float32(0.5 * math.atan2(2.0 * (dw * dz + dx * dy), 1.0 - 2.0 * (dy * dy + dz * dz)))
        yaw_z = math.sin(half_yaw)
        yaw_w = math.cos(half_yaw)
        height_delta = motion_reference_body_pos_w[2] - robot_reference_body_pos_w[2]
        for body_id in range(motion_tracked_bodies_pos_w.shape[0]):
            out_quat = out_body_quat[body_id]
            out_quat[0] = 0.0
            out_quat[1] = 0.0
            out_quat[2] = yaw_z
            out_quat[3] = yaw_w
            out_pos = out_body_pos[body_id]
            rotate_vector(
                out_quat,
                (
                    motion_tracked_bodies_pos_w[body_id, 0] - motion_reference_body_pos_w[0],
                    motion_tracked_bodies_pos_w[body_id, 1] - motion_reference_body_pos_w[1],
                    motion_tracked_bodies_pos_w[body_id, 2] - motion_reference_body_pos_w[2],
                ),
                out_pos,
            )
            out_pos[0] += robot_reference_body_pos_w[0]
            out_pos[1] += robot_reference_body_pos_w[1]
            out_pos[2] += robot_reference_body_pos_w[2] + height_delta
            quat_mul(out_quat, motion_tracked_bodies_quat_w[body_id], out_quat)

    def reset(self, ctx: ResetContext) -> None:
        env_ids = ctx.env_ids
        if self.sampling_cdf.size:
            episode_failed = ctx.terminated[env_ids]
            if np.any(episode_failed):
                failed_steps = self.steps[env_ids, 0][episode_failed]
                num_bins = self.adaptive_bin_failed_count.size
                failed_bins = (failed_steps.astype(np.int64) * num_bins) // max(self.clip.joint_pos.shape[0], 1)
                failed_bins = np.clip(failed_bins, 0, num_bins - 1)
                self.adaptive_current_bin_failed_count[:] += np.bincount(failed_bins, minlength=num_bins).astype(
                    np.float32
                )
            probabilities = self._sampling_probabilities()
            entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
            ctx.metrics["adaptive_sampling_entropy"] = entropy / float(np.log(max(probabilities.size, 2)))
            top_bin = int(np.argmax(probabilities))
            ctx.metrics["adaptive_sampling_top1_prob"] = float(probabilities[top_bin])
            ctx.metrics["adaptive_sampling_top1_bin"] = float(top_bin) / float(probabilities.size)
            ctx.metrics["adaptive_failure_mass"] = float(np.sum(self.adaptive_bin_failed_count))
            self.sampling_cdf[:] = np.cumsum(probabilities, dtype=np.float32)
            self.sampling_cdf[-1] = 1.0
        self.encoder_index[ctx.env_ids] = (1.0, 0.0)
        self.reset_counter[ctx.env_ids] += 1

    def on_transition(self) -> None:
        if self.adaptive_bin_failed_count.size:
            self.adaptive_bin_failed_count[:] = (
                self.alpha * self.adaptive_current_bin_failed_count
                + (np.float32(1.0) - self.alpha) * self.adaptive_bin_failed_count
            )
            self.adaptive_current_bin_failed_count.fill(0.0)

    @dispatch
    def reset_env(self, ctx: ManagerContext) -> None:
        self.steps[0] = _sample_motion_step(
            ctx.rand,
            self.sampling_cdf,
            np.int64(self.clip.joint_pos.shape[0]),
            self.start_at_timestep_zero_prob,
        )

    @dispatch
    def advance(self, ctx: ManagerContext) -> None:
        step = self.steps[0]
        clip_end = self.clip.frame_clip_end[step]
        self.steps[0] = min(step + 1, clip_end)
        self.clip_ended[0] = self.steps[0] >= clip_end

    def _sampling_probabilities(self) -> np.ndarray:
        return _adaptive_sampling_probabilities(
            self.adaptive_bin_failed_count,
            self.uniform_ratio,
            self.kernel_size,
            self.kernel_lambda,
        )


@configclass(kw_only=True)
class SonicMotionCommandCfg(CommandCfg):
    motion_file: str = MISSING
    packed_store: str | None = None
    packed_clip_limit: int | None = None
    joint_names: tuple[str, ...] = G1_SONIC_JOINTS
    tracked_body_names: tuple[str, ...] = G1_SONIC_BODY_NAMES
    reference_body_name: str = "pelvis"
    num_future_frames: int = 10
    adaptive_sampling_enabled: bool = True
    start_at_timestep_zero_prob: float = 0.0
    hold_at_clip_end: bool = False
    uniform_ratio: float = 0.2
    alpha: float = 0.001
    kernel_size: int = 3
    kernel_lambda: float = 0.8
    reward_point_body_names: tuple[str, ...] = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
    reward_point_body_offsets: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    def __post_init__(self) -> None:
        if isinstance(self.num_future_frames, bool) or self.num_future_frames <= 0:
            raise ValueError("SONIC num_future_frames must be a positive integer")
        if self.packed_clip_limit is not None and (
            isinstance(self.packed_clip_limit, bool) or self.packed_clip_limit <= 0
        ):
            raise ValueError("SONIC packed_clip_limit must be a positive integer or None")
        if len(self.reward_point_body_names) != len(self.reward_point_body_offsets):
            raise ValueError("SONIC reward-point body names and offsets must have equal length")
        if len(set(self.reward_point_body_names)) != len(self.reward_point_body_names):
            raise ValueError("SONIC reward-point body names must be unique")
        if any(name not in self.tracked_body_names for name in self.reward_point_body_names):
            raise ValueError("SONIC reward-point body names must be tracked bodies")
        offsets = np.asarray(self.reward_point_body_offsets, dtype=np.float32)
        if offsets.shape != (len(self.reward_point_body_names), 3) or not np.isfinite(offsets).all():
            raise ValueError("SONIC reward-point offsets must be finite XYZ vectors")

    def __call__(self, env: ManagerEnv) -> CommandTerm:
        robot = cast(RobotCfg, env.cfg.scene.objs.robot)
        if self.packed_store:
            clip = SonicMotionClip.from_packed(
                self.packed_store,
                joint_names=self.joint_names,
                body_names=self.tracked_body_names,
                reference_body_name=self.reference_body_name,
                root_body_name=robot.base_link_name,
                clip_limit=self.packed_clip_limit,
            )
        else:
            clip = SonicMotionClip.from_motion(
                MotrixMotion(self.motion_file),
                self.joint_names,
                self.tracked_body_names,
                self.reference_body_name,
                robot.base_link_name,
            )
        reference_index = self.tracked_body_names.index(self.reference_body_name)
        nbin = (
            clip.joint_pos.shape[0] // max(int(round(1.0 / env.cfg.ctrl_dt)), 1) + 1
            if self.adaptive_sampling_enabled
            else 0
        )
        # Term builds run before the read program exists, so derive the
        # tracked-body layout statically instead of reading env.sim_data.
        shape = (len(self.tracked_body_names), 3)
        return SonicMotionCommand(
            clip=clip,
            reference_index=np.int64(reference_index),
            steps=np.zeros((env.num_envs, 1), np.int64),
            clip_ended=np.zeros((env.num_envs, 1), bool),
            reset_counter=np.zeros((env.num_envs, 1), np.int64),
            encoder_index=np.tile(np.asarray((1.0, 0.0), np.float32), (env.num_envs, 1)),
            command=np.empty((env.num_envs, 2 * len(self.joint_names)), np.float32),
            target_body_position_relative=np.empty((env.num_envs, *shape), np.float32),
            target_body_orientation_relative=np.empty((env.num_envs, shape[0], 4), np.float32),
            adaptive_bin_failed_count=np.zeros(nbin, np.float32),
            adaptive_current_bin_failed_count=np.zeros(nbin, np.float32),
            sampling_cdf=np.ones(nbin, np.float32) if nbin else np.empty((0,), np.float32),
            start_at_timestep_zero_prob=np.float32(self.start_at_timestep_zero_prob),
            hold_at_clip_end=self.hold_at_clip_end,
            uniform_ratio=np.float32(self.uniform_ratio),
            alpha=np.float32(self.alpha),
            kernel_size=np.int64(self.kernel_size),
            kernel_lambda=np.float32(self.kernel_lambda),
            num_future_frames=np.int64(self.num_future_frames),
        )


@kernel_data
class SonicJointPositionAction(ActionTerm):
    current: np.ndarray
    previous: np.ndarray
    default_angles: SharedArray
    joint_lower: SharedArray
    joint_upper: SharedArray
    action_scales: SharedArray
    encoder_bias: np.ndarray
    processed: np.ndarray
    joint_velocity_before_action: np.ndarray
    foot_joint_policy_indices: SharedArray
    simulate_action_latency: bool

    def action_space(self, env: ManagerEnv, actuators) -> gym.spaces.Box:
        return gym.spaces.Box(-20.0, 20.0, shape=(len(actuators),), dtype=np.float32)

    def process(self, actions: np.ndarray) -> np.ndarray:
        np.copyto(self.previous, self.current)
        np.copyto(self.current, actions, casting="unsafe")
        executed = self.previous if self.simulate_action_latency else self.current
        np.multiply(executed, self.action_scales, out=self.processed)
        # ``@kernel_data`` instances are frozen dataclasses; augmented
        # assignment attempts to rebind the field even when mutating an
        # ndarray in-place.  Use the ufunc ``out`` form instead.
        np.add(self.processed, self.default_angles - self.encoder_bias, out=self.processed)
        np.clip(self.processed, self.joint_lower, self.joint_upper, out=self.processed)
        return self.processed

    def prepare(self, sim_data) -> None:
        velocities = sim_data["robot_dof_vel"]
        self.joint_velocity_before_action[:] = velocities[:, self.foot_joint_policy_indices]

    def reset(self, env_ids: np.ndarray) -> None:
        self.current[env_ids] = 0.0
        self.previous[env_ids] = 0.0
        self.encoder_bias[env_ids] = 0.0
        self.joint_velocity_before_action[env_ids] = 0.0


@dispatch
def _reset_sonic_actuator_dynamics(
    ctx: ManagerContext,
    sim_writes: Map[np.ndarray],
    kp: tuple[np.float32, ...],
    damping: tuple[np.float32, ...],
) -> None:
    kp_out = sim_writes["kp"]
    damping_out = sim_writes["damping"]
    for index in range(len(kp)):
        kp_out[index] = kp[index]
        damping_out[index] = damping[index]


@configclass(kw_only=True)
class SonicActuatorDynamicsResetCfg(ResetTermCfg):
    """Apply the official SONIC PPO actuator gains on every simulator reset."""

    def __call__(self, env: ManagerEnv) -> ResetTerm:
        model_names = tuple(actuator.name for actuator in env.model.actuators)
        missing = tuple(name for name in G1_SONIC_JOINTS if name not in model_names)
        if missing:
            raise ValueError(f"SONIC actuator dynamics targets are missing: {missing}")
        return ResetTerm(
            _reset_sonic_actuator_dynamics,
            tuple(np.float32(_SONIC_ACTUATOR_PARAMETERS[name][0]) for name in G1_SONIC_JOINTS),
            tuple(np.float32(_SONIC_ACTUATOR_PARAMETERS[name][1]) for name in G1_SONIC_JOINTS),
            writes={
                "kp": ActuatorKpWrite(G1_SONIC_JOINTS),
                "damping": ActuatorDampingWrite(G1_SONIC_JOINTS),
            },
        )


@configclass(kw_only=True)
class SonicJointPositionActionCfg(ActionCfg):
    scale: float = 2.0
    simulate_action_latency: bool = False
    use_default_offset: bool = True

    def __call__(self, env: ManagerEnv, actuators) -> ActionTerm:
        assert actuators is not None
        robot = cast(RobotCfg, env.cfg.scene.objs.robot)
        limits = env.model.others["robot_joint_position_limits"]
        lower_all, upper_all = limits
        names = tuple(a.name for a in actuators)
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("SONIC action scale must be positive and finite")
        model_names = tuple(a.name for a in env.model.actuators)
        if any(n not in _SONIC_ACTUATOR_PARAMETERS for n in names):
            raise ValueError("SONIC actuator set is missing a parameter contract entry")
        if names != G1_SONIC_JOINTS:
            raise ValueError("SONIC action actuators must preserve G1 policy joint order")
        # The official checkpoint was trained with gear_sonic's actuator
        # contract.  Do not rederive this from the generic G1 XML: its two
        # hip-pitch actuators intentionally use different gains/limits.
        scales = _sonic_action_scale(self.scale)
        index = np.asarray([model_names.index(n) for n in names], dtype=np.int64)
        lower = np.asarray(lower_all, dtype=np.float32)[index]
        upper = np.asarray(upper_all, dtype=np.float32)[index]
        defaults = dict(
            zip(
                (robot.resolve_name(n) for n in robot.key_pose.joint_names),
                robot.key_pose.poses["default"],
                strict=True,
            )
        )
        default = (
            np.asarray([defaults[n] for n in names], np.float32)
            if self.use_default_offset
            else np.zeros(len(names), np.float32)
        )
        return SonicJointPositionAction(
            current=np.zeros((env.num_envs, len(names)), np.float32),
            previous=np.zeros((env.num_envs, len(names)), np.float32),
            default_angles=default,
            joint_lower=lower,
            joint_upper=upper,
            action_scales=scales,
            encoder_bias=np.zeros((env.num_envs, len(names)), np.float32),
            processed=np.zeros((env.num_envs, len(names)), np.float32),
            joint_velocity_before_action=np.zeros((env.num_envs, 4), np.float32),
            foot_joint_policy_indices=np.asarray((13, 14, 17, 18), dtype=np.int64),
            simulate_action_latency=self.simulate_action_latency,
        )


@njit(inline="always")
def _sonic_release_rotation6(quat: np.ndarray, out: np.ndarray) -> None:
    """Write the release checkpoint's first-two-columns rotation encoding."""
    x, y, z, w = quat
    out[0] = 1.0 - 2.0 * (y * y + z * z)
    out[1] = 2.0 * (x * y - w * z)
    out[2] = 2.0 * (x * y + w * z)
    out[3] = 1.0 - 2.0 * (x * x + z * z)
    out[4] = 2.0 * (x * z - w * y)
    out[5] = 2.0 * (y * z + w * x)


@kernel_data
class SonicActorObservation:
    history: np.ndarray
    seen_reset: np.ndarray

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray, state: SonicActorObservation) -> None:
        motion = ctx.commands["motion"]
        action = ctx.actions["joint_position"]
        # Observation receivers live under ``ctx.observations`` and are
        # therefore shared by all lanes.  Select this lane's history row
        # explicitly; otherwise ``self.history[0]`` is the whole history matrix
        # array and quaternion helpers receive a 2-D output during Numba
        # compilation.
        env_id = ctx.env_id
        history = state.history[env_id]
        # Proprioception is expressed in the robot root (pelvis) frame.  The
        # SONIC's actor observation uses the robot root (pelvis) state.
        root_q = ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX]
        root_av = ctx.sim["tracked_body_angular_velocity"][SONIC_ROOT_BODY_INDEX]
        reset_history = state.seen_reset[env_id] != motion.reset_counter[0]
        h = history.shape[0]
        if reset_history:
            frame = history[0]
        else:
            # The released local_dir_hist contract is oldest-to-newest.
            for i in range(h - 1):
                history[i] = history[i + 1]
            frame = history[h - 1]
        joint_count = action.current.shape[0]
        root_dim = SONIC_VECTOR_DIM
        pos_end = root_dim + joint_count
        action_start = pos_end + joint_count
        gravity_start = action_start + joint_count
        rotate_inverse(root_q, root_av, frame[:root_dim])
        # Explicit loop instead of an array subtraction, which would allocate
        # a temporary inside the fused kernel lane.
        dof_pos = ctx.sim["robot_dof_pos"]
        for j in range(joint_count):
            frame[root_dim + j] = dof_pos[j] - action.default_angles[j]
        frame[pos_end:action_start] = ctx.sim["robot_dof_vel"]
        frame[action_start:gravity_start] = action.current
        rotate_inverse(root_q, (0.0, 0.0, -1.0), frame[gravity_start : gravity_start + SONIC_VECTOR_DIM])
        if reset_history:
            # ``np.tile`` is not supported by Numba's nopython mode.  Keep
            # this reset path allocation-free with explicit row copies.
            for i in range(1, h):
                history[i] = frame
            state.seen_reset[env_id] = motion.reset_counter[0]
        o0, o1, o2, o3 = (
            root_dim * h,
            (root_dim + joint_count) * h,
            (root_dim + 2 * joint_count) * h,
            (root_dim + 3 * joint_count) * h,
        )
        # Slices such as ``history[:, :3]`` are strided views and Numba
        # cannot reshape them without first materialising a contiguous array.
        # Emit the feature-major layout directly.
        for i in range(h):
            out[root_dim * i : root_dim * i + root_dim] = history[i, :root_dim]
            out[o0 + joint_count * i : o0 + joint_count * i + joint_count] = history[i, root_dim:pos_end]
            out[o1 + joint_count * i : o1 + joint_count * i + joint_count] = history[i, pos_end:action_start]
            out[o2 + joint_count * i : o2 + joint_count * i + joint_count] = history[i, action_start:gravity_start]
            out[o3 + SONIC_VECTOR_DIM * i : o3 + SONIC_VECTOR_DIM * (i + 1)] = history[i, gravity_start:]


@configclass(kw_only=True)
class SonicActorObservationCfg(ObservationTermCfg):
    history_length: int = 10

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        joint_count = len(env.model.bodies["robot"].joint_names)
        frame_dim = _sonic_actor_frame_dim(joint_count)
        state = SonicActorObservation(
            np.zeros((env.num_envs, self.history_length, frame_dim), np.float32),
            np.full(env.num_envs, -1, np.int64),
        )
        return ObsTerm(self.history_length * frame_dim, SonicActorObservation.compute, state)


@kernel_data
class SonicG1ReferenceObservation:
    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray) -> None:
        m = ctx.commands["motion"]
        clip = m.clip
        q = ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX]
        n = int(m.num_future_frames)
        stride = 5
        joint_count = clip.joint_pos.shape[1]
        position_width = n * joint_count
        velocity_start = position_width
        rotation_start = 2 * position_width
        inverse_root = np.empty(4, np.float32)
        quat_inverse(q, inverse_root)
        # ``quat_mul`` overwrites every component, so one scratch quaternion
        # serves all frames without per-frame Numba allocations.
        rel = np.empty(4, np.float32)
        for i in range(n):
            step = min(int(m.steps[0]) + i * stride, clip.frame_clip_end[int(m.steps[0])])
            position_offset = i * joint_count
            velocity_offset = velocity_start + position_offset
            rotation_offset = rotation_start + i * SONIC_ROTATION_REPRESENTATION_DIM
            out[position_offset : position_offset + joint_count] = clip.joint_pos[step]
            out[velocity_offset : velocity_offset + joint_count] = clip.joint_vel[step]
            quat_mul(inverse_root, clip.tracked_bodies_quat_w[step, SONIC_ROOT_BODY_INDEX], rel)
            _sonic_release_rotation6(
                rel,
                out[rotation_offset : rotation_offset + SONIC_ROTATION_REPRESENTATION_DIM],
            )


@configclass(kw_only=True)
class SonicG1ReferenceObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        motion = env.command_terms["motion"]
        size = int(motion.num_future_frames) * _g1_reference_frame_dim(motion.clip.joint_pos.shape[1])
        return ObsTerm(size, SonicG1ReferenceObservation.compute)


@kernel_data
class SonicSmplReferenceObservation:
    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray) -> None:
        m = ctx.commands["motion"]
        clip = m.clip
        n = int(m.num_future_frames)
        joint_count = clip.smpl_joints.shape[1]
        joints_dim = joint_count * SONIC_VECTOR_DIM
        human_frame_dim = joints_dim + SONIC_ROTATION_REPRESENTATION_DIM
        wrist_start = n * human_frame_dim
        inverse_robot_root = np.empty(4, np.float32)
        quat_inverse(ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX], inverse_robot_root)
        # Both scratch quaternions are fully overwritten per frame, so they are
        # allocated once instead of once per future frame.
        inverse_human_root = np.empty(4, np.float32)
        relative_human_root = np.empty(4, np.float32)
        for i in range(n):
            step = min(int(m.steps[0]) + i, clip.frame_clip_end[int(m.steps[0])])
            start = i * human_frame_dim
            rotation_start = start + joints_dim
            quat_inverse(clip.smpl_root_quat[step], inverse_human_root)
            for joint_index in range(joint_count):
                joint_start = start + joint_index * SONIC_VECTOR_DIM
                rotate_vector(
                    inverse_human_root,
                    clip.smpl_joints[step, joint_index],
                    out[joint_start : joint_start + SONIC_VECTOR_DIM],
                )
            quat_mul(inverse_robot_root, clip.smpl_root_quat[step], relative_human_root)
            _sonic_release_rotation6(
                relative_human_root,
                out[rotation_start : rotation_start + SONIC_ROTATION_REPRESENTATION_DIM],
            )
            wrist_offset = wrist_start + i * len(SONIC_WRIST_POLICY_INDICES)
            for wrist_index, policy_index in enumerate(SONIC_WRIST_POLICY_INDICES):
                out[wrist_offset + wrist_index] = clip.joint_pos[step, policy_index]


@configclass(kw_only=True)
class SonicSmplReferenceObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        motion = env.command_terms["motion"]
        size = int(motion.num_future_frames) * _smpl_reference_frame_dim(motion.clip.smpl_joints.shape[1])
        return ObsTerm(size, SonicSmplReferenceObservation.compute)


@kernel_data
class SonicEncoderIndexObservation:
    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray) -> None:
        out[:] = ctx.commands["motion"].encoder_index


@configclass(kw_only=True)
class SonicEncoderIndexObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(SONIC_ENCODER_COUNT, SonicEncoderIndexObservation.compute)


@kernel_data
class SonicCriticObservation:
    history: np.ndarray
    seen_reset: np.ndarray

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray, state: SonicCriticObservation) -> None:
        motion = ctx.commands["motion"]
        clip = motion.clip
        action = ctx.actions["joint_position"]
        joint_count = clip.joint_pos.shape[1]
        step = int(motion.steps[0])
        n = int(motion.num_future_frames)
        width = n * joint_count
        for i in range(n):
            future = min(step + 5 * i, clip.frame_clip_end[step])
            start = i * joint_count
            out[start : start + joint_count] = clip.joint_pos[future]
            out[width + start : width + start + joint_count] = clip.joint_vel[future]
        offset = 2 * width
        root_pos = ctx.sim["tracked_body_pos"][motion.reference_index]
        root_quat = ctx.sim["tracked_body_quat"][motion.reference_index]
        inverse = np.empty(4, np.float32)
        relative = np.empty(4, np.float32)
        quat_inverse(root_quat, inverse)
        # Component tuples avoid one Numba array allocation per subtraction
        # inside the fused kernel lane.
        ref_root_pos = clip.reference_body_pos_w[step]
        rotate_vector(
            inverse,
            (ref_root_pos[0] - root_pos[0], ref_root_pos[1] - root_pos[1], ref_root_pos[2] - root_pos[2]),
            out[offset : offset + 3],
        )
        quat_mul(inverse, clip.reference_body_quat_w[step], relative)
        _sonic_release_rotation6(relative, out[offset + 3 : offset + 9])
        offset += 9
        body_count = clip.tracked_bodies_pos_w.shape[1]
        tracked_body_pos = ctx.sim["tracked_body_pos"]
        for i in range(body_count):
            body_pos = tracked_body_pos[i]
            start = offset + 3 * i
            rotate_vector(
                inverse,
                (body_pos[0] - root_pos[0], body_pos[1] - root_pos[1], body_pos[2] - root_pos[2]),
                out[start : start + 3],
            )
            start = offset + 3 * body_count + 6 * i
            quat_mul(inverse, ctx.sim["tracked_body_quat"][i], relative)
            _sonic_release_rotation6(relative, out[start : start + 6])
        offset += 9 * body_count
        history = state.history[ctx.env_id]
        reset = state.seen_reset[ctx.env_id] != motion.reset_counter[0]
        for i in range(n - 1):
            history[i] = history[i + 1]
        frame = history[n - 1]
        pelvis_quat = ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX]
        rotate_inverse(pelvis_quat, ctx.sim["tracked_body_linear_velocity"][SONIC_ROOT_BODY_INDEX], frame[:3])
        rotate_inverse(pelvis_quat, ctx.sim["tracked_body_angular_velocity"][SONIC_ROOT_BODY_INDEX], frame[3:6])
        dof_pos = ctx.sim["robot_dof_pos"]
        for j in range(joint_count):
            frame[6 + j] = dof_pos[j] - action.default_angles[j]
        frame[6 + joint_count : 6 + 2 * joint_count] = ctx.sim["robot_dof_vel"]
        frame[6 + 2 * joint_count :] = action.current
        if reset:
            for i in range(n - 1):
                history[i] = frame
            state.seen_reset[ctx.env_id] = motion.reset_counter[0]
        feature_start = 0
        for feature_width in (3, 3, joint_count, joint_count, joint_count):
            for i in range(n):
                start = offset + i * feature_width
                out[start : start + feature_width] = history[i, feature_start : feature_start + feature_width]
            offset += n * feature_width
            feature_start += feature_width


@configclass(kw_only=True)
class SonicCriticObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        motion = env.command_terms["motion"]
        joint_count = motion.clip.joint_pos.shape[1]
        body_count = motion.clip.tracked_bodies_pos_w.shape[1]
        actor_frame_dim = _sonic_actor_frame_dim(joint_count)
        current_frame_dim = 9 + body_count * 9
        future_frame_dim = 2 * joint_count + actor_frame_dim
        size = current_frame_dim + int(motion.num_future_frames) * future_frame_dim
        state = SonicCriticObservation(
            np.zeros(
                (env.num_envs, int(motion.num_future_frames), _sonic_actor_frame_dim(joint_count)),
                np.float32,
            ),
            np.full(env.num_envs, -1, np.int64),
        )
        return ObsTerm(size, SonicCriticObservation.compute, state)


@kernel_data
class SonicAntiShakeReward:
    body_indices: SharedArray

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicAntiShakeReward) -> float:
        v = ctx.sim["tracked_body_angular_velocity"]
        total = 0.0
        for i in state.body_indices:
            norm = math.sqrt(v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1] + v[i, 2] * v[i, 2])
            excess = max(norm - 1.5, 0.0)
            total += excess * excess
        return total / state.body_indices.shape[0]


@configclass(kw_only=True)
class SonicAntiShakeRewardCfg(RewardTermCfg):
    def __call__(self, env: ManagerEnv) -> RewardTerm:
        names = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
        tracked = env.cfg.commands.motion.tracked_body_names
        state = SonicAntiShakeReward(np.asarray([tracked.index(name) for name in names], dtype=np.int64))
        return RewardTerm(SonicAntiShakeReward.compute, state)


@kernel_data
class SonicFeetAccelerationReward:
    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, step_dt: np.float32) -> float:
        action = ctx.actions["joint_position"]
        total = 0.0
        for i in range(action.joint_velocity_before_action.shape[0]):
            acceleration = (
                ctx.sim["robot_dof_vel"][action.foot_joint_policy_indices[i]] - action.joint_velocity_before_action[i]
            ) / step_dt
            total += acceleration * acceleration
        return total


@configclass(kw_only=True)
class SonicFeetAccelerationRewardCfg(RewardTermCfg):
    def __call__(self, env: ManagerEnv) -> RewardTerm:
        return RewardTerm(SonicFeetAccelerationReward.compute, np.float32(env.cfg.ctrl_dt))


@kernel_data
class SonicAnchorPosTermination:
    threshold: np.float32

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicAnchorPosTermination) -> bool:
        m = ctx.commands["motion"]
        p = ctx.sim["tracked_body_pos"][m.reference_index]
        return abs(m.clip.root_body_pos_w[m.steps[0], 2] - p[2]) > state.threshold


@configclass(kw_only=True)
class SonicAnchorPosTerminationCfg(TerminationTermCfg):
    threshold: float = 0.15

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        state = SonicAnchorPosTermination(np.float32(self.threshold))
        return TerminationTerm(SonicAnchorPosTermination.compute, state)


@kernel_data
class SonicAnchorOriTermination:
    threshold: np.float32

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicAnchorOriTermination) -> bool:
        m = ctx.commands["motion"]
        q = ctx.sim["tracked_body_quat"][m.reference_index]
        error = rotation_distance(m.clip.root_body_quat_w[m.steps[0]], q)
        return error * error > state.threshold


@configclass(kw_only=True)
class SonicAnchorOriTerminationCfg(TerminationTermCfg):
    threshold: float = 0.2

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        state = SonicAnchorOriTermination(np.float32(self.threshold))
        return TerminationTerm(SonicAnchorOriTermination.compute, state)


@kernel_data
class SonicFeetPosTermination:
    threshold: np.float32

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicFeetPosTermination) -> bool:
        m = ctx.commands["motion"]
        p = ctx.sim["tracked_body_pos"]
        step = m.steps[0]
        for i in (3, 6):
            dx = p[i, 0] - m.clip.tracked_bodies_pos_w[step, i, 0]
            dy = p[i, 1] - m.clip.tracked_bodies_pos_w[step, i, 1]
            dz = p[i, 2] - m.clip.tracked_bodies_pos_w[step, i, 2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) > state.threshold:
                return True
        return False


@configclass(kw_only=True)
class SonicFeetPosTerminationCfg(TerminationTermCfg):
    threshold: float = 0.2

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        state = SonicFeetPosTermination(np.float32(self.threshold))
        return TerminationTerm(SonicFeetPosTermination.compute, state)


@kernel_data
class SonicEeBodyPosTermination:
    threshold: np.float32

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicEeBodyPosTermination) -> bool:
        m = ctx.commands["motion"]
        p = ctx.sim["tracked_body_pos"]
        step = m.steps[0]
        for i in (3, 6, 10, 13):
            if abs(p[i, 2] - m.clip.tracked_bodies_pos_w[step, i, 2]) > state.threshold:
                return True
        return False


@configclass(kw_only=True)
class SonicEeBodyPosTerminationCfg(TerminationTermCfg):
    threshold: float = 0.15

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        state = SonicEeBodyPosTermination(np.float32(self.threshold))
        return TerminationTerm(SonicEeBodyPosTermination.compute, state)


@dispatch
def _sonic_clip_end_termination(ctx: ManagerContext) -> bool:
    motion = ctx.commands["motion"]
    step = motion.steps[0]
    return not motion.hold_at_clip_end and step >= motion.clip.frame_clip_end[step]


@configclass(kw_only=True)
class SonicClipEndTerminationCfg(TerminationTermCfg):
    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        return TerminationTerm(_sonic_clip_end_termination)


@dispatch
def _sonic_time_out_termination(ctx: ManagerContext) -> bool:
    return False


@configclass(kw_only=True)
class SonicTimeOutTerminationCfg(TerminationTermCfg):
    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        return TerminationTerm(_sonic_time_out_termination)


@kernel_data
class SonicTrackingReward:
    point_indices: SharedArray
    point_offsets: SharedArray
    std: np.float32
    scratch_ref_point: np.ndarray
    scratch_robot_point: np.ndarray
    scratch_ref_local: np.ndarray
    scratch_robot_local: np.ndarray

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicTrackingReward) -> float:
        m = ctx.commands["motion"]
        p = ctx.sim["tracked_body_pos"]
        q = ctx.sim["tracked_body_quat"]
        step = m.steps[0]
        anchor = m.reference_index
        ref_anchor_p = m.clip.tracked_bodies_pos_w[step, anchor]
        robot_anchor_p = p[anchor]
        ref_anchor_q = m.clip.tracked_bodies_quat_w[step, anchor]
        robot_anchor_q = q[anchor]
        total = 0.0
        for point_id in range(state.point_indices.shape[0]):
            body = state.point_indices[point_id]
            ref_point = state.scratch_ref_point[ctx.env_id]
            robot_point = state.scratch_robot_point[ctx.env_id]
            rotate_vector(m.clip.tracked_bodies_quat_w[step, body], state.point_offsets[point_id], ref_point)
            rotate_vector(q[body], state.point_offsets[point_id], robot_point)
            ref_point += m.clip.tracked_bodies_pos_w[step, body] - ref_anchor_p
            robot_point += p[body] - robot_anchor_p
            ref_local = state.scratch_ref_local[ctx.env_id]
            robot_local = state.scratch_robot_local[ctx.env_id]
            rotate_inverse(ref_anchor_q, ref_point, ref_local)
            rotate_inverse(robot_anchor_q, robot_point, robot_local)
            dx = ref_local[0] - robot_local[0]
            dy = ref_local[1] - robot_local[1]
            dz = ref_local[2] - robot_local[2]
            total += dx * dx + dy * dy + dz * dz
        mean_error = total / max(state.point_indices.shape[0], 1)
        return math.exp(-mean_error / (state.std * state.std))


@configclass(kw_only=True)
class SonicTrackingRewardCfg(RewardTermCfg):
    std: float = 0.1

    def __post_init__(self) -> None:
        if not np.isfinite(self.std) or self.std <= 0.0:
            raise ValueError("SONIC tracking reward std must be positive and finite")

    def __call__(self, env: ManagerEnv) -> RewardTerm:
        motion = env.cfg.commands.motion
        indices = np.asarray(
            [motion.tracked_body_names.index(name) for name in motion.reward_point_body_names], dtype=np.int64
        )
        offsets = np.asarray(motion.reward_point_body_offsets, dtype=np.float32)
        scratch = np.zeros((env.num_envs, 3), dtype=np.float32)
        state = SonicTrackingReward(
            indices, offsets, np.float32(self.std), scratch.copy(), scratch.copy(), scratch.copy(), scratch.copy()
        )
        return RewardTerm(SonicTrackingReward.compute, state)
