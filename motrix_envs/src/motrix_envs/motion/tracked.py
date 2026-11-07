# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Kernel-compatible, eagerly indexed whole-body motion data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from motrix_env_core.manager.kernel_data import Map
from motrix_envs.motion.loader import MotionFormatError, MotrixMotion


def load_extension_channel(motion: MotrixMotion, name: str) -> np.ndarray:
    """Read one declared ``ext_`` channel as a contiguous per-frame float32 array.

    Args:
        motion: Loaded source motion.
        name: Channel name without the ``ext_`` prefix (e.g. ``"smpl_joints"``).

    Returns:
        Contiguous array whose first axis is the motion frame axis.

    Raises:
        MotionFormatError: If the file lacks the channel, it is not a per-frame
            numeric array matching ``num_frames``, or it holds non-finite values.
    """
    if name not in motion.extensions:
        raise MotionFormatError(f"Motion file {motion.path} does not provide declared extension channel 'ext_{name}'.")
    channel = np.ascontiguousarray(motion.extensions[name], dtype=np.float32)
    if channel.ndim < 1 or channel.shape[0] != motion.num_frames:
        raise MotionFormatError(
            f"Extension channel 'ext_{name}' in {motion.path} must be a per-frame array "
            f"with {motion.num_frames} frames, got shape {channel.shape}."
        )
    if not np.all(np.isfinite(channel)):
        raise MotionFormatError(f"Extension channel 'ext_{name}' in {motion.path} contains non-finite values.")
    return channel


@dataclass(frozen=True, slots=True)
class MotionChannel:
    """One declared ``ext_`` channel, aligned with the clip's frame axis.

    Map leaves are flattened by value type, so each
    channel is wrapped in this record whose field is shared across lanes.
    """

    data: torch.Tensor


@dataclass(frozen=True, slots=True)
class WbtMotionClip:
    """Numeric whole-body-tracking views over one clip or a concatenated corpus.

    The factory resolves all string-based joint and body selections once and
    stores only contiguous numeric arrays suitable for sharing with compiled
    manager kernels. ``T`` is the number of frames, ``N`` the selected joint
    count, and ``K`` the tracked-body count. A clip built from a single file is
    the ``frame_clip_end ≡ T-1`` special case of a multi-clip corpus built by
    :class:`~motrix_envs.motion.library.MotionLibrary`.

    Attributes:
        joint_pos: Selected joint positions shaped ``(T, N)``.
        joint_vel: Selected joint velocities shaped ``(T, N)``.
        tracked_bodies_pos_w: Tracked-body world positions shaped ``(T, K, 3)``.
        tracked_bodies_quat_w: Tracked-body world quaternions shaped ``(T, K, 4)``.
        tracked_bodies_lin_vel_w: Tracked-body world linear velocities shaped ``(T, K, 3)``.
        tracked_bodies_ang_vel_w: Tracked-body world angular velocities shaped ``(T, K, 3)``.
        root_body_pos_w: Root-body world positions shaped ``(T, 3)``.
        root_body_quat_w: Root-body world quaternions shaped ``(T, 4)``.
        root_body_lin_vel_w: Root-body world linear velocities shaped ``(T, 3)``.
        root_body_ang_vel_w: Root-body world angular velocities shaped ``(T, 3)``.
        reference_body_pos_w: Reference-body world positions shaped ``(T, 3)``.
        reference_body_quat_w: Reference-body world quaternions shaped ``(T, 4)``.
        frame_clip_end: Global frame index of the owning clip's final frame, per
            frame, shaped ``(T,)``; a lane leaves its clip when its step index
            exceeds this boundary.
        extensions: Declared ``ext_`` channels, each concatenated along the frame
            axis; empty when the task declares none.
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    tracked_bodies_pos_w: torch.Tensor
    tracked_bodies_quat_w: torch.Tensor
    tracked_bodies_lin_vel_w: torch.Tensor
    tracked_bodies_ang_vel_w: torch.Tensor
    root_body_pos_w: torch.Tensor
    root_body_quat_w: torch.Tensor
    root_body_lin_vel_w: torch.Tensor
    root_body_ang_vel_w: torch.Tensor
    reference_body_pos_w: torch.Tensor
    reference_body_quat_w: torch.Tensor
    frame_clip_end: torch.Tensor
    extensions: Map

    @staticmethod
    def create(
        motion: MotrixMotion,
        joint_names: list[str],
        tracked_body_names: tuple[str, ...],
        reference_body_name: str,
        root_body_name: str,
        extension_channels: tuple[str, ...] = (),
    ) -> WbtMotionClip:
        """Resolve names and build contiguous numeric WBT motion views.

        Args:
            motion: Loaded source motion in file-defined joint and body order.
            joint_names: Desired output joint order.
            tracked_body_names: Desired tracked-body order.
            reference_body_name: Body used to align the motion to the robot.
            root_body_name: Floating-base root body.
            extension_channels: Names (without the ``ext_`` prefix) of extension
                channels to load; undeclared channels are not loaded.

        Returns:
            Kernel-compatible motion data containing no string metadata. The
            single clip spans the whole frame axis, so ``frame_clip_end`` is the
            constant final-frame index.
        """
        if motion.num_frames < 2:
            raise ValueError(f"Tracked motion must have at least two frames: {motion.path}")

        joint_motion_idx = motion.joint_indices(joint_names)
        tracked_idx = motion.body_indices(list(tracked_body_names))
        root_index = motion.body_index(root_body_name)
        reference_index = motion.body_index(reference_body_name)

        return WbtMotionClip(
            joint_pos=torch.from_numpy(np.ascontiguousarray(motion.joint_pos[:, joint_motion_idx])),
            joint_vel=torch.from_numpy(np.ascontiguousarray(motion.joint_vel[:, joint_motion_idx])),
            tracked_bodies_pos_w=torch.from_numpy(np.ascontiguousarray(motion.body_pos_w[:, tracked_idx])),
            tracked_bodies_quat_w=torch.from_numpy(np.ascontiguousarray(motion.body_quat_w[:, tracked_idx])),
            tracked_bodies_lin_vel_w=torch.from_numpy(np.ascontiguousarray(motion.body_lin_vel_w[:, tracked_idx])),
            tracked_bodies_ang_vel_w=torch.from_numpy(np.ascontiguousarray(motion.body_ang_vel_w[:, tracked_idx])),
            root_body_pos_w=torch.from_numpy(np.ascontiguousarray(motion.body_pos_w[:, root_index])),
            root_body_quat_w=torch.from_numpy(np.ascontiguousarray(motion.body_quat_w[:, root_index])),
            root_body_lin_vel_w=torch.from_numpy(np.ascontiguousarray(motion.body_lin_vel_w[:, root_index])),
            root_body_ang_vel_w=torch.from_numpy(np.ascontiguousarray(motion.body_ang_vel_w[:, root_index])),
            reference_body_pos_w=torch.from_numpy(np.ascontiguousarray(motion.body_pos_w[:, reference_index])),
            reference_body_quat_w=torch.from_numpy(np.ascontiguousarray(motion.body_quat_w[:, reference_index])),
            frame_clip_end=torch.from_numpy(np.full(motion.num_frames, motion.num_frames - 1, dtype=np.int64)),
            extensions=Map(
                {
                    name: MotionChannel(torch.from_numpy(load_extension_channel(motion, name)))
                    for name in extension_channels
                }
            ),
        )
