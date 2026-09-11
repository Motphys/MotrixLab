# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SONIC motion data loading and packing helpers.

The runtime loader is intentionally small and NumPy-only.  A packed store is a
manifest plus memory-mapped arrays so manager kernels can consume reference
data without Python work in the control loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motrix_env_core.numba.kernel_data import SharedArray, kernel_data
from motrix_envs.motion.loader import MotrixMotion
from motrix_envs.motion.tracked import WbtMotionClip

SONIC_PACKED_FORMAT = "motrixlab_sonic_packed_v1"
SONIC_QUATERNION_ORDER = "xyzw"
SONIC_ARRAYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "smpl_joints",
    "smpl_root_quat",
)


@kernel_data
class SonicMotionClip(WbtMotionClip):
    """Kernel-compatible WBT clip with SONIC's SMPL and clip boundaries."""

    frame_clip_end: SharedArray
    smpl_joints: SharedArray
    smpl_root_quat: SharedArray

    @classmethod
    def from_motion(
        cls,
        motion: MotrixMotion,
        joint_names: tuple[str, ...],
        body_names: tuple[str, ...],
        reference_body_name: str,
        root_body_name: str,
    ) -> SonicMotionClip:
        base = WbtMotionClip.create(
            motion,
            list(joint_names),
            body_names,
            reference_body_name,
            root_body_name,
        )
        smpl_joints = np.ascontiguousarray(
            motion.extensions.get("smpl_joints", np.zeros((motion.num_frames, 24, 3), np.float32))
        )
        smpl_root_quat = np.ascontiguousarray(motion.extensions.get("smpl_root_quat", base.root_body_quat_w))
        if smpl_joints.shape != (motion.num_frames, 24, 3):
            raise ValueError("ext_smpl_joints must have shape (T, 24, 3)")
        if smpl_root_quat.shape != (motion.num_frames, 4):
            raise ValueError("ext_smpl_root_quat must have shape (T, 4)")
        return cls(
            joint_pos=base.joint_pos,
            joint_vel=base.joint_vel,
            tracked_bodies_pos_w=base.tracked_bodies_pos_w,
            tracked_bodies_quat_w=base.tracked_bodies_quat_w,
            tracked_bodies_lin_vel_w=base.tracked_bodies_lin_vel_w,
            tracked_bodies_ang_vel_w=base.tracked_bodies_ang_vel_w,
            root_body_pos_w=base.root_body_pos_w,
            root_body_quat_w=base.root_body_quat_w,
            root_body_lin_vel_w=base.root_body_lin_vel_w,
            root_body_ang_vel_w=base.root_body_ang_vel_w,
            reference_body_pos_w=base.reference_body_pos_w,
            reference_body_quat_w=base.reference_body_quat_w,
            frame_clip_end=np.full(motion.num_frames, motion.num_frames - 1, dtype=np.int64),
            smpl_joints=smpl_joints,
            smpl_root_quat=smpl_root_quat,
        )

    @classmethod
    def from_packed(
        cls,
        store: str | Path,
        *,
        joint_names: tuple[str, ...],
        body_names: tuple[str, ...],
        reference_body_name: str,
        root_body_name: str,
        clip_limit: int | None = None,
    ) -> SonicMotionClip:
        source = SonicPackedMotion(
            store,
            joint_names=joint_names,
            body_names=body_names,
            clip_limit=clip_limit,
        )
        reference_index = body_names.index(reference_body_name)
        root_index = body_names.index(root_body_name)
        clip_ends = source.clip_offsets + source.clip_lengths - 1
        return cls(
            joint_pos=source.joint_pos,
            joint_vel=source.joint_vel,
            tracked_bodies_pos_w=source.body_pos_w,
            tracked_bodies_quat_w=source.body_quat_w,
            tracked_bodies_lin_vel_w=source.body_lin_vel_w,
            tracked_bodies_ang_vel_w=source.body_ang_vel_w,
            root_body_pos_w=source.body_pos_w[:, root_index],
            root_body_quat_w=source.body_quat_w[:, root_index],
            root_body_lin_vel_w=source.body_lin_vel_w[:, root_index],
            root_body_ang_vel_w=source.body_ang_vel_w[:, root_index],
            reference_body_pos_w=source.body_pos_w[:, reference_index],
            reference_body_quat_w=source.body_quat_w[:, reference_index],
            frame_clip_end=np.repeat(clip_ends, source.clip_lengths),
            smpl_joints=source.smpl_joints,
            smpl_root_quat=source.smpl_root_quat,
        )


class SonicPackedMotion:
    """Read a versioned SONIC packed store using read-only ``np.memmap``."""

    def __init__(self, store: str | Path, *, joint_names, body_names, clip_limit: int | None = None):
        root = Path(store).expanduser().resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != SONIC_PACKED_FORMAT:
            raise ValueError(f"Unsupported SONIC packed format: {manifest.get('format')!r}")
        if manifest.get("quaternion_order") != SONIC_QUATERNION_ORDER:
            raise ValueError("SONIC packed quaternion_order must be 'xyzw'; repack wxyz sources before loading")
        stored_joint_names = tuple(manifest.get("joint_names", ()))
        if stored_joint_names != tuple(joint_names):
            raise ValueError("SONIC packed joint order does not match the task; repack it in task order")
        if tuple(manifest.get("body_names", ())) != tuple(body_names):
            raise ValueError("SONIC packed body order does not match the task")
        self.fps = int(manifest.get("fps", 0))
        stored_num_frames = int(manifest.get("num_frames", 0))
        stored_num_clips = int(manifest.get("num_clips", 0))
        if self.fps != 50 or stored_num_frames <= 0 or stored_num_clips <= 0:
            raise ValueError("Invalid SONIC packed metadata")
        stored_clip_lengths = np.asarray(
            np.load(root / manifest["clip_lengths_file"], allow_pickle=False), dtype=np.int32
        )
        if stored_clip_lengths.shape != (stored_num_clips,) or np.any(stored_clip_lengths <= 0):
            raise ValueError("Invalid SONIC packed clip lengths")
        if int(stored_clip_lengths.sum(dtype=np.int64)) != stored_num_frames:
            raise ValueError("SONIC packed clip lengths do not match num_frames")
        if clip_limit is not None and (isinstance(clip_limit, bool) or clip_limit <= 0):
            raise ValueError("SONIC packed clip_limit must be a positive integer or None")
        self.num_clips = stored_num_clips if clip_limit is None else min(clip_limit, stored_num_clips)
        self.clip_lengths = stored_clip_lengths[: self.num_clips]
        self.num_frames = int(self.clip_lengths.sum(dtype=np.int64))
        self.clip_offsets = np.zeros(self.num_clips, dtype=np.int32)
        if self.num_clips > 1:
            self.clip_offsets[1:] = np.cumsum(self.clip_lengths[:-1], dtype=np.int32)
        self.clip_end_frames = self.clip_offsets + self.clip_lengths - 1
        expected = {
            "joint_pos": (stored_num_frames, len(joint_names)),
            "joint_vel": (stored_num_frames, len(joint_names)),
            "body_pos_w": (stored_num_frames, len(body_names), 3),
            "body_quat_w": (stored_num_frames, len(body_names), 4),
            "body_lin_vel_w": (stored_num_frames, len(body_names), 3),
            "body_ang_vel_w": (stored_num_frames, len(body_names), 3),
            "smpl_joints": (stored_num_frames, 24, 3),
            "smpl_root_quat": (stored_num_frames, 4),
        }
        for name, shape in expected.items():
            spec = manifest["arrays"][name]
            arr = np.load(root / spec["file"], mmap_mode="r", allow_pickle=False)
            if arr.dtype != np.float32 or arr.shape != shape:
                raise ValueError(f"Invalid packed SONIC array {name}: {arr.shape} {arr.dtype}")
            arr = arr[: self.num_frames]
            setattr(self, name, arr)
        self.joint_names = tuple(joint_names)
        self.body_names = tuple(body_names)
        self.num_joints = len(joint_names)
        self.num_bodies = len(body_names)


def pack_sonic_store(output: str | Path, *, clips: list[dict], joint_names, body_names, fps: int = 50) -> Path:
    """Write packed arrays from already-converted clip dictionaries."""
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not clips:
        raise ValueError("At least one SONIC clip is required")
    arrays = {
        name: np.concatenate([np.asarray(c[name], dtype=np.float32) for c in clips], axis=0) for name in SONIC_ARRAYS
    }
    lengths = np.asarray([len(c["joint_pos"]) for c in clips], dtype=np.int32)
    np.save(root / "clip_lengths.npy", lengths)
    specs = {}
    for name, array in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, np.ascontiguousarray(array, dtype=np.float32))
        specs[name] = {"file": path.name, "dtype": "float32", "shape": list(array.shape)}
    manifest = {
        "format": SONIC_PACKED_FORMAT,
        "quaternion_order": SONIC_QUATERNION_ORDER,
        "fps": int(fps),
        "num_frames": int(lengths.sum()),
        "num_clips": len(clips),
        "joint_names": list(joint_names),
        "body_names": list(body_names),
        "clip_lengths_file": "clip_lengths.npy",
        "arrays": specs,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


__all__ = [
    "SONIC_PACKED_FORMAT",
    "SONIC_QUATERNION_ORDER",
    "SONIC_ARRAYS",
    "SonicMotionClip",
    "SonicPackedMotion",
    "pack_sonic_store",
]
