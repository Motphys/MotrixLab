# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Command terms for manager-based whole-body tracking."""

from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import MISSING

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    CommandCfg,
    CommandTerm,
    ManagerContext,
    ManagerEnv,
    metric,
)
from motrix_env_core.manager.commands import ResetContext
from motrix_env_core.manager.dispatch import dispatch
from motrix_env_core.math.quaternion_array import align_targets
from motrix_env_core.manager.rand import draw_uniform
from motrix_envs.motion import MotionLibrary, WbtMotionClip


def _sample_motion_step(
    draws: torch.Tensor,
    sampling_cdf: torch.Tensor,
    num_frames: int,
    start_zero_prob: torch.Tensor,
) -> torch.Tensor:
    """Batched start-frame sampling from pre-drawn uniform samples.

    ``draws`` is ``(N, 2)`` without adaptive sampling or ``(N, 3)`` with it —
    one unit draw, optionally one within-bin draw, and one start-at-zero draw
    per lane, matching the fixed draw order.
    """
    unit = (draws[:, :1] + 1.0) * 0.5
    adaptive = sampling_cdf.numel() != 0
    if adaptive:
        # ``sum(cdf < unit)`` is search-left: equality selects the equal bin,
        # matching the per-lane linear scan that stops at ``unit <= cdf[bin]``.
        bin_id = (sampling_cdf[None, :] < unit).sum(dim=1, dtype=torch.int64)
        bin_unit = (draws[:, 1:2] + 1.0) * 0.5
        phase = (bin_id.to(torch.float32) + bin_unit[:, 0]) / sampling_cdf.numel()
        step_value = phase * num_frames
    else:
        step_value = unit[:, 0] * (num_frames - 1)
    step_value = torch.clamp(step_value, min=0.0, max=float(num_frames - 2))
    steps = step_value.to(torch.int64)
    zero_unit = (draws[:, -1:] + 1.0) * 0.5
    return torch.where(zero_unit[:, 0] < start_zero_prob, torch.zeros_like(steps), steps)


@dataclass(frozen=True, slots=True)
class WbtMotionCommand(CommandTerm):
    """Drive each environment through a shared whole-body reference-motion clip.

    Clip arrays are shared and indexed by the per-environment ``steps`` vector.
    :meth:`update` aligns the selected frame to the simulated robot's current
    reference body, the kernel hooks :meth:`advance` / :meth:`reset_env`
    advance and resample the per-lane frame index, and host-side lifecycle
    methods maintain adaptive-sampling statistics and the sampling distribution.

    Attributes:
        clip: Shared numeric reference-motion corpus in model and tracked-body
            order — one clip or several concatenated onto a global frame axis
            with per-frame ``frame_clip_end`` boundaries.
        reference_index: Index of the alignment body in the tracked-body order.
        command: Per-environment joint-position and joint-velocity command buffer
            (inherited ``CommandTerm.command``).
        target_body_position_relative: Tracked-body targets aligned to the current robot pose.
        target_body_orientation_relative: Aligned tracked-body target quaternions.
        adaptive_bin_failed_count: Exponential moving failure count for each sampling bin.
        adaptive_current_bin_failed_count: Failure counts accumulated in the current update.
        start_at_timestep_zero_prob: Probability that a reset starts at frame zero.
        hold_at_clip_end: Whether lanes remain on a final frame instead of
            resetting — the lane's own clip end without ``sequential_clips``,
            the corpus end with it.
        sequential_clips: Whether lanes treat the corpus as one ordered sequence —
            cross each clip boundary into the next clip's head frame, keep the
            current frame on every reset instead of resampling, and at the
            corpus end either loop back to the head or, with
            ``hold_at_clip_end``, hold on the final frame.
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
    reference_index: int
    target_body_position_relative: torch.Tensor
    target_body_orientation_relative: torch.Tensor

    # Host-side adaptive frame-sampling state and parameters. Host hooks read
    # and write them through zero-copy ``.numpy()`` views; kernels write the
    # per-env members through the tensors themselves.
    adaptive_bin_failed_count: torch.Tensor
    adaptive_current_bin_failed_count: torch.Tensor
    sampling_cdf: torch.Tensor
    start_at_timestep_zero_prob: float
    hold_at_clip_end: bool
    sequential_clips: bool
    uniform_ratio: float
    alpha: float
    kernel_size: int
    kernel_lambda: float

    # Per-environment frame state exposed through the manager metrics system.
    # Kept as ``(num_envs, 1)`` per-env arrays: the kernel lowering hands each
    # lane a writable row view, so advance/reset_env can update the lane's
    # frame in place without a shared backing.
    steps: torch.Tensor = metric(name="motion_step", dtype=torch.float32)
    clip_ended: torch.Tensor = metric()

    def reset(self, ctx: ResetContext) -> None:
        """Update adaptive statistics and prepare sampling for selected environments."""
        env_ids = ctx.env_ids
        cdf = self.sampling_cdf.numpy()
        if cdf.size:
            failed = self.adaptive_bin_failed_count.numpy()
            episode_failed = ctx.terminated[env_ids]
            if bool(episode_failed.any()):
                failed_steps = self.steps[env_ids, 0][episode_failed]
                num_bins = failed.size
                failed_bins = (failed_steps.to(torch.int64) * num_bins) // max(self.clip.joint_pos.shape[0], 1)
                failed_bins = torch.clamp(failed_bins, 0, num_bins - 1)
                self.adaptive_current_bin_failed_count.add_(
                    torch.bincount(failed_bins, minlength=num_bins).to(torch.float32)
                )
            probabilities = self._sampling_probabilities()
            entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
            entropy_norm = entropy / float(np.log(max(probabilities.size, 2)))
            top_bin = int(np.argmax(probabilities))
            ctx.metrics["adaptive_sampling_entropy"] = entropy_norm
            ctx.metrics["adaptive_sampling_top1_prob"] = float(probabilities[top_bin])
            ctx.metrics["adaptive_sampling_top1_bin"] = float(top_bin) / float(probabilities.size)
            ctx.metrics["adaptive_failure_mass"] = float(np.sum(failed))
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
            cdf[:] = np.cumsum(probabilities, dtype=np.float32)
            cdf[-1] = 1.0

    def on_transition(self) -> None:
        """Fold accumulated failures into the adaptive sampler once per step."""
        failed = self.adaptive_bin_failed_count.numpy()
        if failed.size:
            failed[:] = self.alpha * self.adaptive_current_bin_failed_count.numpy() + (np.float32(1.0) - self.alpha) * (
                failed
            )
            self.adaptive_current_bin_failed_count.numpy().fill(0.0)

    def _sampling_probabilities(self) -> np.ndarray:
        """Build normalized frame-bin probabilities from failure history."""
        failed = self.adaptive_bin_failed_count.numpy()
        num_bins = failed.size
        probability = failed + self.uniform_ratio / num_bins
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

    # -- Kernel hooks ----------------------------------------------------------
    # These static hooks receive a batched context (all tensors whole-batch)
    # and a receiver namespace over this command's kernel-data fields. Per-lane
    # state is written in place through the context tensors, so host-visible
    # buffers stay authoritative.

    @staticmethod
    @dispatch
    def update(receiver, ctx: ManagerContext) -> None:
        """Yaw-align the clip to each lane."""
        steps = receiver.steps[:, 0]
        reference_index = receiver.reference_index
        robot_position = ctx.sim["tracked_body_pos"][:, reference_index]
        robot_quat = ctx.sim["tracked_body_quat"][:, reference_index]
        target_position, target_quat = align_targets(
            robot_position,
            robot_quat,
            receiver.clip.reference_body_pos_w[steps],
            receiver.clip.reference_body_quat_w[steps],
            receiver.clip.tracked_bodies_pos_w[steps],
            receiver.clip.tracked_bodies_quat_w[steps],
        )
        receiver.target_body_position_relative.copy_(target_position)
        receiver.target_body_orientation_relative.copy_(target_quat)

    @staticmethod
    @dispatch
    def advance(receiver, ctx: ManagerContext) -> None:
        """Advance every lane one frame."""
        steps = receiver.steps
        clip_end = receiver.clip.frame_clip_end[steps[:, 0]]
        num_frames = receiver.clip.joint_pos.shape[0]
        advanced = steps + 1
        ended = advanced[:, 0] > clip_end
        receiver.clip_ended[:, 0] = ended
        if bool(receiver.sequential_clips):
            if bool(receiver.hold_at_clip_end):
                next_steps = torch.clamp(advanced, max=num_frames - 1)
                crossing = ended & (advanced[:, 0] < num_frames)
            else:
                next_steps = torch.where(advanced >= num_frames, torch.zeros_like(advanced), advanced)
                crossing = ended
            steps.copy_(next_steps)
            ctx.sim_reset_requested |= crossing[:, None]
            return
        if bool(receiver.hold_at_clip_end):
            # Freeze each lane on its own clip's final frame; no reset request.
            steps.copy_(torch.minimum(advanced, clip_end[:, None]))
            return
        # Non-sequential wrap: resample each ended lane inline. Only ended
        # lanes consume sampling draws, so lanes that did not wrap keep their
        # RNG stream: mask the batched state advance back for the rest.
        sampling_cdf = receiver.sampling_cdf
        count = 3 if sampling_cdf.numel() != 0 else 2
        draws, next_state = draw_uniform(ctx.rand.state, count)
        sampled = _sample_motion_step(
            draws,
            sampling_cdf,
            num_frames,
            receiver.start_at_timestep_zero_prob,
        )
        steps.copy_(torch.where(ended[:, None], sampled[:, None], advanced))
        ctx.rand.state.copy_(torch.where(ended[:, None], next_state, ctx.rand.state))
        ctx.sim_reset_requested |= ended[:, None]

    @staticmethod
    @dispatch
    def reset_env(receiver, ctx: ManagerContext) -> None:
        """Sample start frames for reset lanes."""
        if bool(receiver.sequential_clips):
            # Sequential playback rematerializes onto the current frame.
            return
        sampling_cdf = receiver.sampling_cdf
        count = 3 if sampling_cdf.numel() != 0 else 2
        draws = ctx.rand.next_uniform(count)
        steps = _sample_motion_step(
            draws,
            sampling_cdf,
            receiver.clip.joint_pos.shape[0],
            receiver.start_at_timestep_zero_prob,
        )
        # ``clip_ended`` is intentionally left untouched; advance recomputes it
        # on the following transition.
        receiver.steps[:, 0] = steps


@configclass(kw_only=True)
class WbtMotionCommandCfg(CommandCfg):
    motion_files: tuple[str, ...] = MISSING
    extension_channels: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = MISSING
    tracked_body_names: tuple[str, ...] = MISSING
    reference_body_name: str = MISSING
    adaptive_sampling_enabled: bool = True
    start_at_timestep_zero_prob: float = 0.0
    hold_at_clip_end: bool = False
    sequential_clips: bool = False
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
        env_fps = max(int(round(1.0 / env.cfg.ctrl_dt)), 1)
        source = MotionLibrary(
            self.motion_files,
            joint_names=list(self.joint_names),
            tracked_body_names=self.tracked_body_names,
            reference_body_name=self.reference_body_name,
            root_body_name=robot.base_link_name,
            fps=env_fps,
            extension_channels=self.extension_channels,
        ).assemble()
        if self.adaptive_sampling_enabled:
            num_bins = source.joint_pos.shape[0] // env_fps + 1
        else:
            num_bins = 0
        tracked_shape = (len(self.tracked_body_names), 3)
        return WbtMotionCommand(
            clip=source,
            reference_index=int(reference_index),
            steps=torch.zeros((env.num_envs, 1), dtype=torch.int64),
            clip_ended=torch.zeros((env.num_envs, 1), dtype=torch.bool),
            command=torch.empty((env.num_envs, 2 * env.num_actuators), dtype=torch.float32),
            target_body_position_relative=torch.empty((env.num_envs, *tracked_shape), dtype=torch.float32),
            target_body_orientation_relative=torch.empty((env.num_envs, tracked_shape[0], 4), dtype=torch.float32),
            adaptive_bin_failed_count=torch.zeros((num_bins,), dtype=torch.float32),
            adaptive_current_bin_failed_count=torch.zeros((num_bins,), dtype=torch.float32),
            sampling_cdf=torch.ones((num_bins,), dtype=torch.float32)
            if num_bins
            else torch.empty((0,), dtype=torch.float32),
            start_at_timestep_zero_prob=float(self.start_at_timestep_zero_prob),
            hold_at_clip_end=self.hold_at_clip_end,
            sequential_clips=self.sequential_clips,
            uniform_ratio=float(self.uniform_ratio),
            alpha=float(self.alpha),
            kernel_size=int(self.kernel_size),
            kernel_lambda=float(self.kernel_lambda),
        )


__all__ = [
    "WbtMotionCommand",
    "WbtMotionCommandCfg",
]
