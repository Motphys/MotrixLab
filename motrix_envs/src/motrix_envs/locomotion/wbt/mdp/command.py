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
def _sample_motion_step(rand, sampling_cdf, num_frames: np.int64, start_at_timestep_zero_prob: np.float32):
    """Draw one start frame from the adaptive-bin CDF (or uniformly when disabled)."""
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
    return step


def _adaptive_sampling_probabilities(
    failed_count: np.ndarray,
    uniform_ratio: np.float32,
    kernel_size: np.int64,
    kernel_lambda: np.float32,
) -> np.ndarray:
    """Build normalized frame-bin probabilities with edge-extended smoothing.

    Command host resets run once per training step whenever any lane is done,
    so the smoothing must stay a single vectorized pass: one Python-level
    numpy call per bin is measurably expensive for packed stores with
    thousands of one-second bins.
    """
    num_bins = failed_count.size
    probability = failed_count + uniform_ratio / num_bins
    kernel_size = max(kernel_size, 1)
    if kernel_size > 1:
        kernel = np.asarray([kernel_lambda**i for i in range(kernel_size)], dtype=np.float32)
        kernel /= np.sum(kernel)
        padded = np.pad(probability, (0, kernel_size - 1), mode="edge")
        # ``sliding_window_view`` rows equal ``padded[i : i + kernel_size]``,
        # so the matvec reproduces the per-bin windowed dot exactly.
        probability = np.lib.stride_tricks.sliding_window_view(padded, kernel_size) @ kernel
    total = np.sum(probability)
    if total <= 0.0:
        return np.full((num_bins,), 1.0 / num_bins, dtype=np.float32)
    return (probability / total).astype(np.float32)


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

    # Per-environment frame state exposed through the manager metrics system.
    # Kept as ``(num_envs, 1)`` per-env arrays: the kernel lowering hands each
    # lane a writable row view, so advance/reset_env can update the lane's
    # frame in place without a shared backing.
    steps: np.ndarray = metric(name="motion_step", dtype=np.float32)
    clip_ended: np.ndarray = metric()

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

    def reset(self, ctx: ResetContext) -> None:
        """Update adaptive statistics and prepare sampling for selected environments."""
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
            ctx.rand, self.sampling_cdf, np.int64(num_frames), self.start_at_timestep_zero_prob
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
                ctx.rand, self.sampling_cdf, np.int64(num_frames), self.start_at_timestep_zero_prob
            )
            ctx.sim_reset_requested[0] = True

    def _sampling_probabilities(self) -> np.ndarray:
        """Build normalized frame-bin probabilities from failure history."""
        return _adaptive_sampling_probabilities(
            self.adaptive_bin_failed_count,
            self.uniform_ratio,
            self.kernel_size,
            self.kernel_lambda,
        )


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
        tracked_shape = (len(self.tracked_body_names), 3)
        return WbtMotionCommand(
            clip=source,
            reference_index=np.int64(reference_index),
            steps=np.zeros((env.num_envs, 1), dtype=np.int64),
            clip_ended=np.zeros((env.num_envs, 1), dtype=bool),
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
        )


__all__ = [
    "WbtMotionCommand",
    "WbtMotionCommandCfg",
]
