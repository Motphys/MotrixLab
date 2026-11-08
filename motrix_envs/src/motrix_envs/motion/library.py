# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotionLibrary — assemble multiple MotrixLab motion NPZ v1 files into one clip.

Each file holds one clip (schema v1). The library reorders every clip to the
task joint/body order, loads only the task-declared ``ext_`` channels, validates
the corpus-level contracts, and concatenates the clips onto one global frame
axis with per-frame clip boundaries. A single file is the N=1 special case.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from motrix_env_core.manager.kernel_data import Map
from motrix_envs.motion.loader import MotrixMotion
from motrix_envs.motion.tracked import MotionChannel, WbtMotionClip

# Per-frame array fields of :class:`WbtMotionClip` that concatenate along the
# frame axis when clips are joined.
_FRAME_AXIS_FIELDS = (
    "joint_pos",
    "joint_vel",
    "tracked_bodies_pos_w",
    "tracked_bodies_quat_w",
    "tracked_bodies_lin_vel_w",
    "tracked_bodies_ang_vel_w",
    "root_body_pos_w",
    "root_body_quat_w",
    "root_body_lin_vel_w",
    "root_body_ang_vel_w",
    "reference_body_pos_w",
    "reference_body_quat_w",
)

_REQUIRED_FRAME_ARRAYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class MotionLibrary:
    """Load an ordered corpus of motion NPZ v1 files and assemble one WBT clip.

    Entries may be files or directories; a directory expands to its ``.npz``
    files in sorted order, so the corpus order is fixed and reproducible.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        joint_names: list[str],
        tracked_body_names: tuple[str, ...],
        reference_body_name: str,
        root_body_name: str,
        fps: int,
        extension_channels: tuple[str, ...] = (),
    ) -> None:
        """Configure the corpus.

        Args:
            paths: Ordered motion files and/or directories.
            joint_names: Desired output joint order (task order).
            tracked_body_names: Desired tracked-body order.
            reference_body_name: Body used to align the motion to the robot.
            root_body_name: Floating-base root body.
            fps: Control rate every clip must be recorded at.
            extension_channels: Names (without the ``ext_`` prefix) of channels
                to load from every file; undeclared channels are not loaded.
        """
        self.entries = expand_motion_paths(paths)
        self.joint_names = joint_names
        self.tracked_body_names = tracked_body_names
        self.reference_body_name = reference_body_name
        self.root_body_name = root_body_name
        self.fps = fps
        self.extension_channels = tuple(extension_channels)

    def assemble(self) -> WbtMotionClip:
        """Load, validate, and concatenate the corpus into one kernel-data clip."""
        motions = [MotrixMotion(path) for path in self.entries]
        self._validate_corpus(motions)
        segments = [
            WbtMotionClip.create(
                motion,
                self.joint_names,
                self.tracked_body_names,
                self.reference_body_name,
                self.root_body_name,
                self.extension_channels,
            )
            for motion in motions
        ]
        clip_lengths = np.asarray([segment.joint_pos.shape[0] for segment in segments], dtype=np.int64)
        clip_offsets = np.zeros_like(clip_lengths)
        np.cumsum(clip_lengths[:-1], out=clip_offsets[1:])
        return WbtMotionClip(
            **{field: torch.cat([getattr(segment, field) for segment in segments]) for field in _FRAME_AXIS_FIELDS},
            frame_clip_end=torch.cat(
                [segment.frame_clip_end + offset for segment, offset in zip(segments, clip_offsets)]
            ),
            extensions=Map(
                {
                    name: MotionChannel(torch.cat([segment.extensions[name].data for segment in segments]))
                    for name in self.extension_channels
                }
            ),
        )

    def _validate_corpus(self, motions: list[MotrixMotion]) -> None:
        first = motions[0]
        for motion in motions:
            if motion.fps != self.fps:
                raise ValueError(
                    f"Motion fps must match the control rate {self.fps}: {motion.path} is recorded at {motion.fps}."
                )
            if set(motion.joint_names) != set(first.joint_names):
                raise ValueError(f"Joint name set of {motion.path} differs from {first.path}.")
            if set(motion.body_names) != set(first.body_names):
                raise ValueError(f"Body name set of {motion.path} differs from {first.path}.")
            for field in _REQUIRED_FRAME_ARRAYS:
                if not np.all(np.isfinite(getattr(motion, field))):
                    raise ValueError(f"{field} in {motion.path} contains non-finite values.")
        for name in self.extension_channels:
            trailing_shapes = []
            for motion in motions:
                if name not in motion.extensions:
                    raise ValueError(
                        f"Motion file {motion.path} does not provide declared extension channel 'ext_{name}'."
                    )
                trailing_shapes.append(motion.extensions[name].shape[1:])
            if any(shape != trailing_shapes[0] for shape in trailing_shapes):
                raise ValueError(
                    f"Extension channel 'ext_{name}' must keep one shape across the corpus; "
                    f"got per-file trailing shapes {trailing_shapes}."
                )


def expand_motion_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    """Expand motion sources to an ordered tuple of ``.npz`` file paths.

    Directory entries expand to their ``.npz`` children in sorted order, so
    the corpus order of a directory source is fixed and reproducible.
    """
    entries: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            entries.extend(sorted(child for child in path.iterdir() if child.suffix == ".npz"))
        else:
            entries.append(path)
    if not entries:
        raise ValueError("No motion files were found in the given sources.")
    return tuple(entries)


__all__ = ["MotionLibrary"]
