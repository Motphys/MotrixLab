# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Command terms for manager-based whole-body tracking."""

from dataclasses import dataclass

import torch
from omegaconf import MISSING

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    ManagerContext,
    metric,
)
from motrix_env_core.manager.commands import CommandCfg, CommandTerm
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.math.quaternion import align_targets
from motrix_envs.motion import MotionLibrary, WbtMotionClip
from motrix_envs.motion.sampler import (
    AdaptiveTimestepsSampler,
    MotionStartSampler,
    ReturnTimestepsSampler,
    UniformTimestepsSampler,
    max_discounted_return,
)

_SAMPLING_MODES = ("uniform", "adaptive", "return")


@dataclass(frozen=True, slots=True)
class WbtMotionCommand(CommandTerm):
    """Drive each environment through a shared whole-body reference-motion clip.

    Clip arrays are shared and indexed by the per-environment ``steps`` vector.
    :meth:`update` aligns the selected frame to the simulated robot's current
    reference body, the kernel hooks :meth:`advance` / :meth:`reset`
    advance and resample the per-lane frame index, and the nested
    :attr:`sampler` owns the start-frame curriculum (statistics, sampling
    distribution, and its gauges) for whichever ``sampling_mode`` the config
    selected.

    Attributes:
        clip: Shared numeric reference-motion corpus in model and tracked-body
            order — one clip or several concatenated onto a global frame axis
            with per-frame ``frame_clip_end`` boundaries.
        reference_index: Index of the alignment body in the tracked-body order.
        command: Per-environment joint-position and joint-velocity command buffer
            (inherited ``CommandTerm.command``).
        target_body_position_relative: Tracked-body targets aligned to the current robot pose.
        target_body_orientation_relative: Aligned tracked-body target quaternions.
        sampler: Start-frame curriculum owning all sampling statistics and
            its ``metric()`` gauges (see
            :class:`~motrix_envs.motion.sampler.MotionStartSampler`).
        start_at_timestep_zero_prob: Probability that a reset starts at frame zero.
        hold_at_clip_end: Whether lanes remain on a final frame instead of
            resetting — the lane's own clip end without ``sequential_clips``,
            the corpus end with it.
        sequential_clips: Whether lanes treat the corpus as one ordered sequence —
            cross each clip boundary into the next clip's head frame, keep the
            current frame on every reset instead of resampling, and at the
            corpus end either loop back to the head or, with
            ``hold_at_clip_end``, hold on the final frame.
        steps: Current reference-frame index for every environment.
        clip_ended: Whether each environment advanced beyond the final frame.
    """

    # Immutable shared reference data loaded from the motion clip.
    clip: WbtMotionClip

    # Runtime buffers and tracked-body alignment metadata.
    reference_index: int
    target_body_position_relative: torch.Tensor
    target_body_orientation_relative: torch.Tensor

    # Start-frame curriculum: owns the sampling statistics, distribution, and
    # gauges for the configured sampling_mode.
    sampler: MotionStartSampler

    start_at_timestep_zero_prob: float
    hold_at_clip_end: bool
    sequential_clips: bool

    # Per-environment frame state exposed through the manager metrics system.
    # Kept as ``(num_envs, 1)`` whole-batch tensors: the Torch executor hooks
    # run on the live term instance, so advance/reset update the lanes' frame
    # indices in place through these tensors.
    steps: torch.Tensor = metric(name="motion_step", dtype=torch.float32)
    clip_ended: torch.Tensor = metric()

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
        # Per-transition curriculum fold first, so every branch below runs it.
        self.sampler.fold_step(ctx)
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
        # Draw from the production default stream; only ended lanes change frames.
        draws = 2 * ctx.rand.uniform((steps.shape[0], self.sampler.draw_count), device=steps.device) - 1
        sampled = self.sampler.sample(draws, self.start_at_timestep_zero_prob)
        steps.copy_(torch.where(ended[:, None], sampled[:, None], advanced))
        ctx.sim_reset_requested |= ended[:, None]

    def reset(self, ctx: ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        """Fold curriculum statistics, then resample start frames for reset lanes.

        Compact semantics: ``env_ids`` are this pass's reset rows (K).
        The sampler first folds the episode rows' statistics and rebuilds its
        distribution, then the reset rows draw from it and scatter their new
        frame indices back; finally the sampler opens new episodes on the
        episode rows.
        """
        idx = env_ids
        self.sampler.fold_episode_stats(idx, episode_reset, ctx.terminated, self.steps[idx, 0])
        self.sampler.rebuild()
        if bool(self.sequential_clips):
            # Sequential playback rematerializes onto the current frame.
            self.sampler.mark_episode_starts(idx, episode_reset, self.steps[idx, 0])
            return
        draws = 2 * ctx.rand.uniform((idx.numel(), self.sampler.draw_count), device=self.steps.device) - 1
        steps = self.sampler.sample(draws, self.start_at_timestep_zero_prob)
        # ``clip_ended`` is intentionally left untouched; advance recomputes it
        # on the following transition.
        self.steps[idx, 0] = steps
        self.sampler.mark_episode_starts(idx, episode_reset, self.steps[idx, 0])


@configclass(kw_only=True)
class WbtMotionCommandCfg(CommandCfg):
    motion_files: tuple[str, ...] = MISSING
    extension_channels: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = MISSING
    tracked_body_names: tuple[str, ...] = MISSING
    reference_body_name: str = MISSING
    sampling_mode: str = "adaptive"
    start_at_timestep_zero_prob: float = 0.0
    hold_at_clip_end: bool = False
    sequential_clips: bool = False
    uniform_ratio: float = 0.1
    alpha: float = 0.001
    kernel_size: int = 1
    kernel_lambda: float = 0.8
    # Return-mode mixing and fold parameters (ignored by other modes).
    uniform_mix: float = 0.5
    return_mix: float = 0.4
    history_mix: float = 0.1
    sampling_gamma: float = 0.99
    return_ema: float = 0.2

    def _build_sampler(self, ctx: BuildContext, source: WbtMotionClip, env_fps: int) -> MotionStartSampler:
        if self.sampling_mode == "uniform":
            return UniformTimestepsSampler(num_frames=source.joint_pos.shape[0])
        if self.sampling_mode == "adaptive":
            return AdaptiveTimestepsSampler.from_clip(
                source,
                env_fps,
                uniform_ratio=self.uniform_ratio,
                alpha=self.alpha,
                kernel_size=self.kernel_size,
                kernel_lambda=self.kernel_lambda,
            )
        if self.sampling_mode == "return":
            # Softmax normalizer derived from the configured reward weights
            # (no copied value): the task's largest achievable discounted
            # return over the sampling horizon.
            g_max = max_discounted_return(
                ctx.cfg.ctrl_dt,
                sum(cfg.weight for cfg in ctx.cfg.reward_cfgs().values() if cfg.weight > 0.0),
                float(self.sampling_gamma),
            )
            return ReturnTimestepsSampler.from_clip(
                source,
                env_fps,
                ctx.num_envs,
                uniform_mix=self.uniform_mix,
                return_mix=self.return_mix,
                history_mix=self.history_mix,
                sampling_gamma=self.sampling_gamma,
                return_ema=self.return_ema,
                g_max=g_max,
            )
        raise ValueError(f"commands.motion sampling_mode must be one of {_SAMPLING_MODES}, got {self.sampling_mode!r}.")

    def __call__(self, ctx: BuildContext) -> CommandTerm:
        robot = ctx.cfg.scene.objs.robot
        if not isinstance(robot, RobotCfg):
            raise TypeError(f"WBT scene robot must be RobotCfg, got {type(robot).__name__}")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("WBT motion joint_names must be non-empty and unique.")
        body_joint_names = ctx.model.bodies["robot"].joint_names
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
        env_fps = max(int(round(1.0 / ctx.cfg.ctrl_dt)), 1)
        source = MotionLibrary(
            self.motion_files,
            joint_names=list(self.joint_names),
            tracked_body_names=self.tracked_body_names,
            reference_body_name=self.reference_body_name,
            root_body_name=robot.base_link_name,
            fps=env_fps,
            extension_channels=self.extension_channels,
        ).assemble()
        num_envs = ctx.num_envs
        tracked_shape = (len(self.tracked_body_names), 3)
        return WbtMotionCommand(
            clip=source,
            reference_index=int(reference_index),
            steps=torch.zeros((num_envs, 1), dtype=torch.int64),
            clip_ended=torch.zeros((num_envs, 1), dtype=torch.bool),
            command=torch.empty((num_envs, 2 * len(ctx.model.actuators)), dtype=torch.float32),
            target_body_position_relative=torch.empty((num_envs, *tracked_shape), dtype=torch.float32),
            target_body_orientation_relative=torch.empty((num_envs, tracked_shape[0], 4), dtype=torch.float32),
            sampler=self._build_sampler(ctx, source, env_fps),
            start_at_timestep_zero_prob=float(self.start_at_timestep_zero_prob),
            hold_at_clip_end=self.hold_at_clip_end,
            sequential_clips=self.sequential_clips,
        )


__all__ = [
    "WbtMotionCommand",
    "WbtMotionCommandCfg",
]
