# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_envs.motion.sonic import SonicPackedMotion, pack_sonic_store


def test_sonic_packed_roundtrip(tmp_path):
    joints = ("j0", "j1")
    bodies = ("b0", "b1")
    clip = {
        "joint_pos": np.zeros((10, 2), np.float32),
        "joint_vel": np.ones((10, 2), np.float32),
        "body_pos_w": np.zeros((10, 2, 3), np.float32),
        "body_quat_w": np.tile(np.asarray([0, 0, 0, 1], np.float32), (10, 2, 1)),
        "body_lin_vel_w": np.zeros((10, 2, 3), np.float32),
        "body_ang_vel_w": np.zeros((10, 2, 3), np.float32),
        "smpl_joints": np.zeros((10, 24, 3), np.float32),
        "smpl_root_quat": np.tile(np.asarray([0, 0, 0, 1], np.float32), (10, 1)),
    }
    root = pack_sonic_store(tmp_path / "store", clips=[clip], joint_names=joints, body_names=bodies)
    loaded = SonicPackedMotion(root, joint_names=joints, body_names=bodies)
    assert loaded.num_frames == 10
    assert loaded.clip_lengths.tolist() == [10]
    assert isinstance(loaded.joint_pos, np.memmap)
    np.testing.assert_array_equal(loaded.joint_vel, clip["joint_vel"])

    with pytest.raises(ValueError, match="joint order"):
        SonicPackedMotion(root, joint_names=tuple(reversed(joints)), body_names=bodies, clip_limit=1)


def test_sonic_packed_clip_limit_materializes_only_the_selected_prefix(tmp_path) -> None:
    joints = ("j0", "j1")
    bodies = ("b0",)

    def clip(value: float) -> dict[str, np.ndarray]:
        return {
            "joint_pos": np.full((10, 2), value, np.float32),
            "joint_vel": np.full((10, 2), value, np.float32),
            "body_pos_w": np.zeros((10, 1, 3), np.float32),
            "body_quat_w": np.tile(np.asarray([0, 0, 0, 1], np.float32), (10, 1, 1)),
            "body_lin_vel_w": np.zeros((10, 1, 3), np.float32),
            "body_ang_vel_w": np.zeros((10, 1, 3), np.float32),
            "smpl_joints": np.zeros((10, 24, 3), np.float32),
            "smpl_root_quat": np.tile(np.asarray([0, 0, 0, 1], np.float32), (10, 1)),
        }

    root = pack_sonic_store(
        tmp_path / "store",
        clips=[clip(1.0), clip(2.0)],
        joint_names=joints,
        body_names=bodies,
    )

    loaded = SonicPackedMotion(root, joint_names=joints, body_names=bodies, clip_limit=1)

    assert loaded.num_clips == 1
    assert loaded.num_frames == 10
    np.testing.assert_array_equal(loaded.joint_pos, np.ones((10, 2), dtype=np.float32))
