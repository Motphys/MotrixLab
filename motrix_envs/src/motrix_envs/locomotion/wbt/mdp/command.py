# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Command terms for manager-based whole-body tracking."""

import math

import numpy as np
from numba import njit
from omegaconf import MISSING

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    CommandCfg,
    CommandTerm,
    ManagerContext,
    ManagerEnv,
    SharedArray,
    kernel_data,
    metric,
)
from motrix_env_core.manager.math.quaternion import inverse as quat_inverse
from motrix_env_core.manager.math.quaternion import mul as quat_mul
from motrix_env_core.manager.math.quaternion import rotate_vector
from motrix_env_core.numba.manager.commands import ResetContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_envs.motion import MotrixMotion, WbtMotionClip


@njit(inline="always")
def _sample_motion_step(
    rand,
    sampling_cdf,
    num_frames: np.int64,
    start_at_timestep_zero_prob: np.float32,
    phase_start_min: np.int64,
    phase_start_max: np.int64,
):
    """Draw one start frame from the adaptive-bin CDF (or uniformly when disabled).

    ``phase_start_min`` / ``phase_start_max`` are static start-frame gates: the
    sampled frame clamps into ``[min, max]``. ``max`` bounds resets to episodes
    that must still pass through the aerial phase of the clip, so completion
    statistics cannot be inflated by tail-frame starts.
    """
    unit = (rand.next_uniform() + np.float32(1.0)) * np.float32(0.5)
    if sampling_cdf.size == 0:
        step_value = unit * np.float32(num_frames - 1)
    else:
        bin_id = np.int64(0)
        while bin_id + 1 < sampling_cdf.size and unit > sampling_cdf[bin_id]:
            bin_id += 1
        bin_unit = (rand.next_uniform() + np.float32(1.0)) * np.float32(0.5)
        phase = (np.float32(bin_id) + bin_unit) / np.float32(sampling_cdf.size)
        step_value = phase * np.float32(num_frames)
    max_value = np.float32(num_frames - 2)
    step_value = min(max(step_value, np.float32(0.0)), max_value)
    step = np.int64(step_value)
    if (rand.next_uniform() + np.float32(1.0)) * np.float32(0.5) < start_at_timestep_zero_prob:
        step = np.int64(0)
    if step < phase_start_min:
        step = phase_start_min
    if phase_start_max >= 0 and step > phase_start_max:
        step = phase_start_max
    return step


@kernel_data
class WbtMotionCommand(CommandTerm):
    """Drive each environment through a shared whole-body reference-motion clip.

    Clip arrays are shared and indexed by the per-environment ``steps`` vector.
    :meth:`evaluate` aligns the selected frame to the simulated robot's current
    reference body, the fused-kernel hooks :meth:`advance` / :meth:`reset_env`
    advance and resample the per-lane frame index, and host-side lifecycle
    methods maintain adaptive-sampling statistics and the sampling distribution.

    Attributes:
        clip: Shared numeric reference-motion clip in model and tracked-body order.
        reference_index: Index of the alignment body in the tracked-body order.
        command: Per-environment joint-position and joint-velocity command buffer
            (inherited ``CommandTerm.command``).
        target_body_position_relative: Tracked-body targets aligned to the current robot pose.
        target_body_orientation_relative: Aligned tracked-body target quaternions.
        adaptive_bin_failed_count: Exponential moving failure count for each sampling bin.
        adaptive_current_bin_failed_count: Failure counts accumulated in the current update.
        start_at_timestep_zero_prob: Probability that a reset starts at frame zero.
        hold_at_clip_end: Whether environments remain on the final frame instead of resetting.
        uniform_ratio: Uniform prior mass added to adaptive sampling probabilities.
        alpha: Update rate for adaptive failure-count statistics.
        kernel_size: Number of bins used to smooth adaptive probabilities.
        kernel_lambda: Exponential decay factor of the smoothing kernel.
        steps: Current reference-frame index for every environment.
        clip_ended: Whether each environment advanced beyond the final frame.
    """

    # Immutable shared reference data loaded from the motion clip.
    clip: WbtMotionClip

    # Runtime buffers and tracked-body alignment metadata.
    reference_index: np.int64
    target_body_position_relative: np.ndarray
    target_body_orientation_relative: np.ndarray

    # Host-side adaptive frame-sampling state and parameters.
    adaptive_bin_failed_count: SharedArray
    adaptive_current_bin_failed_count: SharedArray
    sampling_cdf: SharedArray
    start_at_timestep_zero_prob: np.float32
    hold_at_clip_end: bool
    uniform_ratio: np.float32
    alpha: np.float32
    kernel_size: np.int64
    kernel_lambda: np.float32

    # Host-side action-rate curriculum state: EMA of episode length and the
    # current penalty multiplier, host-written and reward-kernel-read.
    action_rate_curriculum: bool
    curriculum_high: np.float32
    curriculum_low: np.float32
    curriculum_step: np.float32
    curriculum_max_scale: np.float32
    curriculum_ema_alpha: np.float32
    action_rate_scale: SharedArray
    curriculum_ep_len_ema: SharedArray

    # Reverse start-frame curriculum state: grounded candidate start frames
    # (sorted), the current gate index into them, and the kernel-read earliest
    # allowed start frame. Host-written on episode resets, kernel-read in
    # reset_env / advance resampling.
    phase_curriculum: bool
    phase_curriculum_high: np.float32
    phase_curriculum_low: np.float32
    phase_curriculum_step: np.int64
    # Candidate grounded start frames (sorted, read-only in kernels — but
    # lowered as SHARED because the frame axis, not the env axis, is leading).
    grounded_start_steps: SharedArray
    phase_gate_index: SharedArray
    phase_start_min: SharedArray

    # Aerial-phase ("flight window") honest-skill metrics. The window is
    # derived once from the clip's root-z profile; the kernel accumulates
    # per-lane max pelvis height and unwrapped pitch rotation inside it and
    # stamps an upright-landing check right after it. Static window bounds are
    # baked in at compile time (never mutated at runtime).
    flight_metrics: bool
    flight_start: np.int64
    flight_end: np.int64
    land_check_step: np.int64
    # Static upper bound on sampled start frames (-1 = unlimited).
    phase_start_max: np.int64
    flight_max_z: np.ndarray = metric(name="flight_max_pelvis_z", dtype=np.float32)
    flight_pitch_acc: np.ndarray = metric(name="flight_pitch_rotation", dtype=np.float32)
    landed_upright: np.ndarray = metric()

    # Per-environment frame state exposed through the manager metrics system.
    # Kept as ``(num_envs, 1)`` per-env arrays: the kernel lowering hands each
    # lane a writable row view, so advance/reset_env can update the lane's
    # frame in place without a shared backing.
    steps: np.ndarray = metric(name="motion_step", dtype=np.float32)
    clip_ended: np.ndarray = metric()
    episode_length: np.ndarray = metric(name="episode_length", dtype=np.float32)

    @dispatch
    def update(self, ctx: ManagerContext) -> None:
        """Align the selected motion frame to one simulated robot lane."""
        # Select the current motion frame for this environment lane.
        step = self.steps[0]
        motion_reference_body_pos_w = self.clip.reference_body_pos_w[step]
        motion_reference_body_quat_w = self.clip.reference_body_quat_w[step]
        motion_tracked_bodies_pos_w = self.clip.tracked_bodies_pos_w[step]
        motion_tracked_bodies_quat_w = self.clip.tracked_bodies_quat_w[step]
        # Anchor the motion to the current simulated pose of the configured
        # reference body, rather than to the clip's absolute world transform.
        reference_index = self.reference_index
        robot_reference_body_pos_w = ctx.sim["tracked_body_pos"][reference_index]
        robot_reference_body_quat_w = ctx.sim["tracked_body_quat"][reference_index]
        out_body_pos = self.target_body_position_relative
        out_body_quat = self.target_body_orientation_relative

        # Keep only the relative yaw between motion and robot. Roll and pitch
        # remain those of each body in the source motion after composition.
        delta_quat = out_body_quat[0]
        quat_inverse(motion_reference_body_quat_w, delta_quat)
        quat_mul(robot_reference_body_quat_w, delta_quat, delta_quat)
        dx, dy, dz, dw = delta_quat
        half_yaw = np.float32(0.5 * math.atan2(2.0 * (dw * dz + dx * dy), 1.0 - 2.0 * (dy * dy + dz * dz)))
        yaw_z = math.sin(half_yaw)
        yaw_w = math.cos(half_yaw)
        height_delta = motion_reference_body_pos_w[2] - robot_reference_body_pos_w[2]

        # Apply the same planar rigid transform to every tracked body, then
        # compose the yaw alignment with its source-motion orientation.
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

        if self.flight_metrics:
            # Honest flip metrics, evaluated inside the aerial window only.
            # Rotation progress integrates the pelvis pitch rate (angular
            # velocity y-component): an asin-based euler pitch folds beyond
            # ±90° (asin(sinθ) is non-monotonic through inversion), so the
            # old euler-delta accumulator returned ~0 for a full flip and
            # mixed signs for partial flips past vertical. Integrating the
            # rate accumulates exactly ±2π over a complete rotation.
            pelvis_pos = ctx.sim["tracked_body_pos"][0]
            pelvis_ang_vel = ctx.sim["tracked_body_angular_velocity"][0]
            step = self.steps[0]
            if self.flight_start <= step <= self.flight_end:
                self.flight_pitch_acc[0] += pelvis_ang_vel[1] * ctx.dt
                if pelvis_pos[2] > self.flight_max_z[0]:
                    self.flight_max_z[0] = pelvis_pos[2]
            if step == self.land_check_step:
                pelvis_quat = ctx.sim["tracked_body_quat"][0]
                sin_pitch = 2.0 * (pelvis_quat[3] * pelvis_quat[1] - pelvis_quat[2] * pelvis_quat[0])
                sin_pitch = min(max(sin_pitch, -1.0), 1.0)
                pitch = math.asin(sin_pitch)
                self.landed_upright[0] = abs(pitch) < 0.3

    def reset(self, ctx: ResetContext) -> None:
        """Update curriculum/adaptive statistics and prepare sampling for selected environments."""
        env_ids = ctx.env_ids
        self._update_curricula(ctx)
        if self.flight_metrics:
            self.flight_max_z[env_ids, 0] = 0.0
            self.flight_pitch_acc[env_ids, 0] = 0.0
            self.landed_upright[env_ids, 0] = False
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
            entropy_norm = entropy / float(np.log(max(probabilities.size, 2)))
            top_bin = int(np.argmax(probabilities))
            ctx.metrics["adaptive_sampling_entropy"] = entropy_norm
            ctx.metrics["adaptive_sampling_top1_prob"] = float(probabilities[top_bin])
            ctx.metrics["adaptive_sampling_top1_bin"] = float(top_bin) / float(probabilities.size)
            ctx.metrics["adaptive_failure_mass"] = float(np.sum(self.adaptive_bin_failed_count))
            num_frames = self.clip.joint_pos.shape[0]
            max_step = num_frames - 1
            max_bin = int(np.clip(np.ceil(max_step * probabilities.size / max(num_frames, 1)), 1, probabilities.size))
            probabilities = probabilities.copy()
            probabilities[max_bin:] = 0.0
            total = np.sum(probabilities)
            if total <= 0.0:
                probabilities[:max_bin] = 1.0 / max_bin
            else:
                probabilities /= total
            self.sampling_cdf[:] = np.cumsum(probabilities, dtype=np.float32)
            self.sampling_cdf[-1] = 1.0

    def _update_curricula(self, ctx: ResetContext) -> None:
        """Shared episode-length EMA feeding the closed-loop controllers.

        zh_CN: 共享的 episode 长度 EMA，驱动动作率惩罚与起始帧课程两个控制器。

        The EMA of episode length is computed once per reset batch; the
        action-rate hysteresis and the reverse start-frame curriculum both
        consume it. ``episode_length`` is zeroed here (not in the kernel reset
        hook) because that hook also fires for clip-wrap rematerialization,
        which is not an episode boundary.
        """
        if not (self.action_rate_curriculum or self.phase_curriculum):
            return
        lengths = self.episode_length[ctx.env_ids]
        batch_mean = float(np.mean(lengths)) if lengths.size else 0.0
        self.episode_length[ctx.env_ids] = 0.0
        alpha = float(self.curriculum_ema_alpha)
        ema = alpha * batch_mean + (1.0 - alpha) * float(self.curriculum_ep_len_ema[0])
        self.curriculum_ep_len_ema[0] = np.float32(ema)
        ctx.metrics["curriculum_ep_len_ema"] = ema

        if self.action_rate_curriculum:
            # Once the EMA clears ``curriculum_high`` the action-rate penalty
            # ramps up (anti-jitter phase); falling back below
            # ``curriculum_low`` relaxes it again, protecting a fragile skill.
            scale = float(self.action_rate_scale[0])
            if ema >= float(self.curriculum_high):
                scale = min(float(self.curriculum_max_scale), scale + float(self.curriculum_step))
            elif ema <= float(self.curriculum_low):
                scale = max(1.0, scale - float(self.curriculum_step))
            self.action_rate_scale[0] = np.float32(scale)
            ctx.metrics["action_rate_scale"] = scale

        if self.phase_curriculum:
            # Reverse start-frame curriculum: while the policy survives only
            # short episodes the gate stays at the clip tail (landing hold);
            # sustained long episodes unlock earlier grounded frames, moving
            # the start toward frame zero (the full flip). Short episodes
            # retreat the gate back toward the tail.
            num_frames = max(self.clip.joint_pos.shape[0], 1)
            gate = int(self.phase_gate_index[0])
            if ema >= float(self.phase_curriculum_high):
                gate = max(0, gate - int(self.phase_curriculum_step))
            elif ema <= float(self.phase_curriculum_low):
                gate = min(self.grounded_start_steps.size - 1, gate + int(self.phase_curriculum_step))
            self.phase_gate_index[0] = np.int64(gate)
            self.phase_start_min[0] = np.int64(self.grounded_start_steps[gate])
            ctx.metrics["phase_start_fraction"] = float(self.grounded_start_steps[gate]) / num_frames

    def on_transition(self) -> None:
        """Fold accumulated failures into the adaptive sampler once per step."""
        if self.adaptive_bin_failed_count.size:
            self.adaptive_bin_failed_count[:] = (
                self.alpha * self.adaptive_current_bin_failed_count
                + (np.float32(1.0) - self.alpha) * self.adaptive_bin_failed_count
            )
            self.adaptive_current_bin_failed_count.fill(0.0)

    @dispatch
    def reset_env(self, ctx: ManagerContext) -> None:
        """Sample the starting frame for one reset environment lane."""
        num_frames = self.clip.joint_pos.shape[0]
        self.steps[0] = _sample_motion_step(
            ctx.rand,
            self.sampling_cdf,
            np.int64(num_frames),
            self.start_at_timestep_zero_prob,
            self.phase_start_min[0],
            self.phase_start_max,
        )
        # ``clip_ended`` is intentionally left untouched: advance recomputes it
        # every transition, so a lane that just wrapped keeps its flag for the
        # metrics/observation pass that follows rematerialization.

    @dispatch
    def advance(self, ctx: ManagerContext) -> None:
        """Advance one frame for the current environment lane.

        The lowering already binds this lane's writable row view to
        ``self.steps`` / ``self.clip_ended``; only the wrap branch needs ``ctx``
        to draw the replacement frame.
        """
        num_frames = self.clip.joint_pos.shape[0]
        self.steps[0] += 1
        self.episode_length[0] += 1.0
        self.clip_ended[0] = self.steps[0] >= num_frames
        if self.hold_at_clip_end:
            self.steps[0] = min(self.steps[0], num_frames - 1)
        elif self.clip_ended[0]:
            # Request sim-only rematerialization: the reset pipeline resamples
            # this lane's frame via reset_env from the freshly rebuilt CDF and
            # the configured sim reset terms teleport the robot there. Episode
            # bookkeeping and action-term state are untouched. The inline
            # resample keeps steps valid and consistently distributed between
            # this kernel and the reset pipeline.
            self.steps[0] = _sample_motion_step(
                ctx.rand,
                self.sampling_cdf,
                np.int64(num_frames),
                self.start_at_timestep_zero_prob,
                self.phase_start_min[0],
                self.phase_start_max,
            )
            ctx.sim_reset_requested[0] = True

    def _sampling_probabilities(self) -> np.ndarray:
        """Build normalized frame-bin probabilities from failure history."""
        num_bins = self.adaptive_bin_failed_count.size
        probability = self.adaptive_bin_failed_count + self.uniform_ratio / num_bins
        kernel_size = max(self.kernel_size, 1)
        if kernel_size > 1:
            kernel = np.asarray([self.kernel_lambda**i for i in range(kernel_size)], dtype=np.float32)
            kernel /= np.sum(kernel)
            padded = np.pad(probability, (0, kernel_size - 1), mode="edge")
            probability = np.asarray(
                [np.sum(padded[index : index + kernel_size] * kernel) for index in range(num_bins)],
                dtype=np.float32,
            )
        total = np.sum(probability)
        if total <= 0.0:
            return np.full((num_bins,), 1.0 / num_bins, dtype=np.float32)
        return (probability / total).astype(np.float32)


@configclass(kw_only=True)
class WbtMotionCommandCfg(CommandCfg):
    motion_file: str = MISSING
    joint_names: tuple[str, ...] = MISSING
    tracked_body_names: tuple[str, ...] = MISSING
    reference_body_name: str = MISSING
    adaptive_sampling_enabled: bool = True
    start_at_timestep_zero_prob: float = 0.0
    hold_at_clip_end: bool = False
    uniform_ratio: float = 0.1
    alpha: float = 0.001
    kernel_size: int = 1
    kernel_lambda: float = 0.8
    # Action-rate curriculum (hysteresis on mean episode length).
    # The step is per reset batch (~33 batches/s): 0.002 ramps 1.0 -> 4.0 over
    # ~1500 batches so the policy can adapt incrementally; 0.05 hit the cap in
    # seconds and collapsed the skill (observed 350 -> 250).
    action_rate_curriculum: bool = False
    curriculum_high: float = 400.0
    curriculum_low: float = 300.0
    curriculum_step: float = 0.002
    curriculum_max_scale: float = 4.0
    curriculum_ema_alpha: float = 0.05
    # Reverse start-frame curriculum: resets start from grounded frames only
    # (root z below ``phase_grounded_z``, so never mid-flight), initially at
    # the clip tail (landing hold). Sustained episode-length EMA above
    # ``phase_curriculum_high`` unlocks earlier grounded frames toward frame
    # zero; EMA below ``phase_curriculum_low`` retreats back toward the tail.
    phase_curriculum: bool = False
    phase_grounded_z: float = 0.85
    phase_curriculum_high: float = 180.0
    phase_curriculum_low: float = 120.0
    phase_curriculum_step: int = 1
    # Honest aerial-skill metrics: the flight window is derived from the clip
    # root-z profile (frames above min + flight_z_fraction * (max - min)).
    # Inside it the command accumulates per-env max pelvis height, unwrapped
    # pelvis pitch rotation, and an upright-landing check half a second after
    # touchdown. ``start_before_flight_only`` additionally clamps sampled start
    # frames to <= flight_start so every episode must pass through the aerial
    # phase — clip completion can then no longer be inflated by tail starts.
    flight_metrics_enabled: bool = False
    flight_z_fraction: float = 0.5
    start_before_flight_only: bool = False

    def __call__(self, env: ManagerEnv) -> CommandTerm:
        robot = env.cfg.scene.objs.robot
        if not isinstance(robot, RobotCfg):
            raise TypeError(f"WBT scene robot must be RobotCfg, got {type(robot).__name__}")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("WBT motion joint_names must be non-empty and unique.")
        body_joint_names = env.model.bodies["robot"].joint_names
        if body_joint_names != tuple(self.joint_names):
            raise ValueError("WBT robot body joint order must match commands.motion.joint_names order.")
        if robot.resolved_base_link_name not in self.tracked_body_names:
            raise ValueError(f"tracked_body_names must include the robot base link {robot.resolved_base_link_name!r}")
        try:
            reference_index = self.tracked_body_names.index(self.reference_body_name)
        except ValueError:
            raise ValueError(
                f"tracked_body_names must include the reference body {self.reference_body_name!r}"
            ) from None
        source = WbtMotionClip.create(
            MotrixMotion(self.motion_file),
            list(self.joint_names),
            self.tracked_body_names,
            self.reference_body_name,
            robot.base_link_name,
        )
        if self.adaptive_sampling_enabled:
            env_fps = max(int(round(1.0 / env.cfg.ctrl_dt)), 1)
            num_bins = source.joint_pos.shape[0] // env_fps + 1
        else:
            num_bins = 0
        if self.phase_curriculum:
            root_z = np.asarray(source.root_body_pos_w)[:, 2]
            grounded_steps = np.flatnonzero(root_z < self.phase_grounded_z).astype(np.int64)
            if grounded_steps.size < 2:
                raise ValueError(
                    f"WBT phase curriculum found <2 grounded frames below z={self.phase_grounded_z} "
                    f"in {self.motion_file!r}; adjust phase_grounded_z."
                )
            num_frames = source.joint_pos.shape[0]
            if grounded_steps[-1] > num_frames - 2:
                grounded_steps = grounded_steps[grounded_steps <= num_frames - 2]
            initial_gate = grounded_steps.size - 1
        else:
            grounded_steps = np.zeros((1,), dtype=np.int64)
            initial_gate = 0
        if self.flight_metrics_enabled:
            root_z = np.asarray(source.root_body_pos_w)[:, 2]
            threshold = root_z.min() + self.flight_z_fraction * (root_z.max() - root_z.min())
            airborne = np.flatnonzero(root_z > threshold)
            if airborne.size < 2:
                raise ValueError(
                    f"WBT flight metrics found <2 airborne frames above z={threshold:.3f} "
                    f"in {self.motion_file!r}; adjust flight_z_fraction."
                )
            flight_start = int(airborne[0])
            flight_end = int(airborne[-1])
            env_fps = max(int(round(1.0 / env.cfg.ctrl_dt)), 1)
            land_check_step = flight_end + env_fps // 2
        else:
            flight_start = 0
            flight_end = -1
            land_check_step = -1
        phase_start_max = flight_start if self.start_before_flight_only else -1
        tracked_shape = (len(self.tracked_body_names), 3)
        return WbtMotionCommand(
            clip=source,
            reference_index=np.int64(reference_index),
            steps=np.zeros((env.num_envs, 1), dtype=np.int64),
            clip_ended=np.zeros((env.num_envs, 1), dtype=bool),
            episode_length=np.zeros((env.num_envs, 1), dtype=np.float32),
            command=np.empty((env.num_envs, 2 * env.num_actuators), dtype=np.float32),
            target_body_position_relative=np.empty((env.num_envs, *tracked_shape), dtype=np.float32),
            target_body_orientation_relative=np.empty((env.num_envs, tracked_shape[0], 4), dtype=np.float32),
            adaptive_bin_failed_count=np.zeros((num_bins,), dtype=np.float32),
            adaptive_current_bin_failed_count=np.zeros((num_bins,), dtype=np.float32),
            sampling_cdf=np.ones((num_bins,), dtype=np.float32) if num_bins else np.empty((0,), dtype=np.float32),
            start_at_timestep_zero_prob=np.float32(self.start_at_timestep_zero_prob),
            hold_at_clip_end=self.hold_at_clip_end,
            uniform_ratio=np.float32(self.uniform_ratio),
            alpha=np.float32(self.alpha),
            kernel_size=np.int64(self.kernel_size),
            kernel_lambda=np.float32(self.kernel_lambda),
            action_rate_curriculum=self.action_rate_curriculum,
            curriculum_high=np.float32(self.curriculum_high),
            curriculum_low=np.float32(self.curriculum_low),
            curriculum_step=np.float32(self.curriculum_step),
            curriculum_max_scale=np.float32(self.curriculum_max_scale),
            curriculum_ema_alpha=np.float32(self.curriculum_ema_alpha),
            action_rate_scale=np.ones((1,), dtype=np.float32),
            curriculum_ep_len_ema=np.zeros((1,), dtype=np.float32),
            phase_curriculum=self.phase_curriculum,
            phase_curriculum_high=np.float32(self.phase_curriculum_high),
            phase_curriculum_low=np.float32(self.phase_curriculum_low),
            phase_curriculum_step=np.int64(self.phase_curriculum_step),
            grounded_start_steps=grounded_steps,
            phase_gate_index=np.full((1,), initial_gate, dtype=np.int64),
            phase_start_min=np.full((1,), grounded_steps[initial_gate], dtype=np.int64),
            flight_metrics=self.flight_metrics_enabled,
            flight_start=np.int64(flight_start),
            flight_end=np.int64(flight_end),
            land_check_step=np.int64(land_check_step),
            phase_start_max=np.int64(phase_start_max),
            flight_max_z=np.zeros((env.num_envs, 1), dtype=np.float32),
            flight_pitch_acc=np.zeros((env.num_envs, 1), dtype=np.float32),
            landed_upright=np.zeros((env.num_envs, 1), dtype=bool),
        )


__all__ = [
    "WbtMotionCommand",
    "WbtMotionCommandCfg",
]
