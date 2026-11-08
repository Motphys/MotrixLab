# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Command terms for manager-based whole-body tracking."""

import math
from dataclasses import dataclass

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
from motrix_env_core.manager.rand import draw_uniform
from motrix_env_core.math.quaternion import align_targets
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
    reference body, the kernel hooks :meth:`advance` / :meth:`reset`
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

    # Host-side adaptive frame-sampling state and parameters. Host hooks
    # operate on the tensors directly; kernels write the per-env members
    # through the tensors themselves.
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
    # Kept as ``(num_envs, 1)`` whole-batch tensors: the Torch executor hooks
    # run on the live term instance, so advance/reset update the lanes' frame
    # indices in place through these tensors.
    steps: torch.Tensor = metric(name="motion_step", dtype=torch.float32)
    clip_ended: torch.Tensor = metric()

    # Batch-level adaptive-sampling gauges written by the reset kernel.
    adaptive_entropy: torch.Tensor = metric(name="adaptive_sampling_entropy", scalar=True)
    adaptive_top1_prob: torch.Tensor = metric(name="adaptive_sampling_top1_prob", scalar=True)
    adaptive_top1_bin: torch.Tensor = metric(name="adaptive_sampling_top1_bin", scalar=True)
    adaptive_failure_mass: torch.Tensor = metric(name="adaptive_failure_mass", scalar=True)

    # -- Kernel hooks ----------------------------------------------------------
    # These hooks receive a batched context (all tensors whole-batch) and run
    # on the live host instance (the compiled graph guards on its identity).
    # Per-lane state is written in place through the context tensors, so
    # host-visible buffers stay authoritative.

    def update(self: "WbtMotionCommand", ctx: ManagerContext) -> None:
        """Yaw-align the clip to each lane."""
        steps = self.steps[:, 0]
        reference_index = self.reference_index
        robot_position = ctx.sim["tracked_body_pos"][:, reference_index]
        robot_quat = ctx.sim["tracked_body_quat"][:, reference_index]
        target_position, target_quat = align_targets(
            robot_position,
            robot_quat,
            self.clip.reference_body_pos_w[steps],
            self.clip.reference_body_quat_w[steps],
            self.clip.tracked_bodies_pos_w[steps],
            self.clip.tracked_bodies_quat_w[steps],
        )
        self.target_body_position_relative.copy_(target_position)
        self.target_body_orientation_relative.copy_(target_quat)

    def advance(self: "WbtMotionCommand", ctx: ManagerContext) -> None:
        """Advance every lane one frame."""
        # Fold the transition's failure mass into the adaptive sampler here
        # (first, so every branch below runs it): whole-batch shared state,
        # previously a separate host on_transition phase.
        failed = self.adaptive_bin_failed_count
        if failed.numel():
            failed.mul_(1.0 - self.alpha).add_(self.adaptive_current_bin_failed_count, alpha=self.alpha)
            self.adaptive_current_bin_failed_count.zero_()
        steps = self.steps
        clip_end = self.clip.frame_clip_end[steps[:, 0]]
        num_frames = self.clip.joint_pos.shape[0]
        advanced = steps + 1
        ended = advanced[:, 0] > clip_end
        self.clip_ended[:, 0] = ended
        if bool(self.sequential_clips):
            if bool(self.hold_at_clip_end):
                next_steps = torch.clamp(advanced, max=num_frames - 1)
                crossing = ended & (advanced[:, 0] < num_frames)
            else:
                next_steps = torch.where(advanced >= num_frames, torch.zeros_like(advanced), advanced)
                crossing = ended
            steps.copy_(next_steps)
            ctx.sim_reset_requested |= crossing[:, None]
            return
        if bool(self.hold_at_clip_end):
            # Freeze each lane on its own clip's final frame; no reset request.
            steps.copy_(torch.minimum(advanced, clip_end[:, None]))
            return
        # Non-sequential wrap: resample each ended lane inline. Only ended
        # lanes consume sampling draws, so lanes that did not wrap keep their
        # RNG stream: mask the batched state advance back for the rest.
        sampling_cdf = self.sampling_cdf
        count = 3 if sampling_cdf.numel() != 0 else 2
        draws, next_state = draw_uniform(ctx.rand.state, count)
        sampled = _sample_motion_step(
            draws,
            sampling_cdf,
            num_frames,
            self.start_at_timestep_zero_prob,
        )
        steps.copy_(torch.where(ended[:, None], sampled[:, None], advanced))
        ctx.rand.state.copy_(torch.where(ended[:, None], next_state, ctx.rand.state))
        ctx.sim_reset_requested |= ended[:, None]

    def reset(self: "WbtMotionCommand", ctx: ManagerContext) -> None:
        """Rebuild the adaptive sampler, then resample start frames for reset lanes.

        Compact semantics: ``ctx.reset_ids`` are this pass's reset rows (K).
        Episode-boundary statistics gather just those rows — the terminated
        lanes among them (``reset_episode``) feed a ``scatter_add`` failure
        histogram, gauges publish through 0-d scalar metrics, and the sampling
        CDF is rebuilt before the reset rows draw from it and scatter their
        new frame indices back.
        """
        idx = ctx.reset_ids
        cdf = self.sampling_cdf
        if cdf.numel():
            failed = self.adaptive_bin_failed_count
            num_bins = failed.numel()
            num_frames = self.clip.joint_pos.shape[0]
            episode_failed = ctx.reset_episode & ctx.terminated[idx]
            failed_bins = (self.steps[idx, 0].to(torch.int64) * num_bins) // max(num_frames, 1)
            failed_bins = torch.clamp(failed_bins, 0, num_bins - 1)
            histogram = torch.zeros(num_bins, dtype=torch.float32)
            histogram.scatter_add_(0, failed_bins, episode_failed.to(torch.float32))
            self.adaptive_current_bin_failed_count.add_(histogram)
            probabilities = failed + self.uniform_ratio / num_bins
            kernel_size = max(self.kernel_size, 1)
            if kernel_size > 1:
                kernel = torch.tensor([self.kernel_lambda**i for i in range(kernel_size)], dtype=torch.float32)
                kernel /= kernel.sum()
                padded = torch.nn.functional.pad(probabilities[None, None, :], (0, kernel_size - 1), mode="rep")
                probability = (padded.unfold(2, kernel_size, 1) @ kernel)[0, 0]
            else:
                probability = probabilities
            total = probability.sum()
            uniform = torch.full((num_bins,), 1.0 / num_bins, dtype=torch.float32)
            positive = total > 0.0
            probabilities = torch.where(
                positive, probability / torch.where(positive, total, torch.ones_like(total)), uniform
            )
            entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
            top_bin = torch.argmax(probabilities)
            self.adaptive_entropy.copy_(entropy / math.log(max(num_bins, 2)))
            # Gather (not ``probabilities[top_bin]``): tensor indexing would
            # sync the index to a Python int and break the graph.
            self.adaptive_top1_prob.copy_(torch.gather(probabilities, 0, top_bin.reshape(1)).reshape(()))
            self.adaptive_top1_bin.copy_(top_bin.to(torch.float32) / float(num_bins))
            self.adaptive_failure_mass.copy_(failed.sum())
            max_step = num_frames - 1
            max_bin = min(max(math.ceil(max_step * num_bins / max(num_frames, 1)), 1), num_bins)
            probabilities = probabilities.clone()
            probabilities[max_bin:] = 0.0
            total = probabilities.sum()
            fallback = torch.zeros(num_bins, dtype=torch.float32)
            fallback[:max_bin] = 1.0 / max_bin
            positive = total > 0.0
            probabilities = torch.where(
                positive, probabilities / torch.where(positive, total, torch.ones_like(total)), fallback
            )
            cdf.copy_(torch.cumsum(probabilities, dim=0).to(torch.float32))
            cdf[-1] = 1.0
        if bool(self.sequential_clips):
            # Sequential playback rematerializes onto the current frame.
            return
        count = 3 if cdf.numel() != 0 else 2
        # Only reset lanes consume sampling draws, so lanes outside this pass
        # keep their RNG stream: mask the batched state advance back for them
        # (same discipline as the inline wrap resample in advance()).
        previous_state = ctx.rand.state.clone()
        draws = ctx.rand.next_uniform(count)[idx]
        steps = _sample_motion_step(
            draws,
            cdf,
            self.clip.joint_pos.shape[0],
            self.start_at_timestep_zero_prob,
        )
        in_pass = torch.zeros(self.steps.shape[0], dtype=torch.bool)
        in_pass[idx] = True
        ctx.rand.state.copy_(torch.where(in_pass[:, None], ctx.rand.state, previous_state))
        # ``clip_ended`` is intentionally left untouched; advance recomputes it
        # on the following transition.
        self.steps[idx, 0] = steps


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
