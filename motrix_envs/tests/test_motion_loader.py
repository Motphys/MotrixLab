# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_envs.motion import MotrixMotion, WbtMotionClip
from motrix_envs.motion.loader import MotionFormatError


def _make_minimal_npz(path, num_frames=10, num_joints=3, num_bodies=4):
    """Write a MotrixLab v1 npz with arbitrary data for schema-level tests."""
    rng = np.random.default_rng(0)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[..., 3] = 1.0  # identity xyzw = (0,0,0,1)
    data = {
        "schema_version": np.int32(1),
        "fps": np.int32(50),
        "num_frames": np.int32(num_frames),
        "joint_names": np.asarray([f"joint_{i}" for i in range(num_joints)]),
        "body_names": np.asarray([f"body_{i}" for i in range(num_bodies)]),
        "joint_pos": rng.standard_normal((num_frames, num_joints)).astype(np.float32),
        "joint_vel": rng.standard_normal((num_frames, num_joints)).astype(np.float32),
        "body_pos_w": rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        "body_ang_vel_w": rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        "ext_object_pos_w": rng.standard_normal((num_frames, 3)).astype(np.float32),
    }
    np.savez(path, **data)


def test_loader_reads_minimal_file(tmp_path):
    path = tmp_path / "m.npz"
    _make_minimal_npz(path)
    m = MotrixMotion(path)
    assert m.schema_version == 1
    assert m.fps == 50
    assert m.num_frames == 10
    assert m.joint_names == [f"joint_{i}" for i in range(3)]
    assert m.body_names == [f"body_{i}" for i in range(4)]
    assert m.joint_pos.shape == (10, 3)
    assert m.body_pos_w.shape == (10, 4, 3)
    assert m.body_quat_w.shape == (10, 4, 4)
    assert m.extensions["object_pos_w"].shape == (10, 3)


def test_loader_missing_required_field(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, fps=np.int32(50))  # most required fields missing
    with pytest.raises(MotionFormatError, match="missing required fields"):
        MotrixMotion(path)


def test_loader_rejects_bad_schema_version(tmp_path):
    path = tmp_path / "bad.npz"
    _make_minimal_npz(path)
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    data["schema_version"] = np.int32(99)
    np.savez(path, **data)
    with pytest.raises(MotionFormatError, match="Unsupported schema_version"):
        MotrixMotion(path)


def test_loader_rejects_non_unit_quaternion(tmp_path):
    path = tmp_path / "bad.npz"
    _make_minimal_npz(path)
    # Corrupt one quaternion to be non-unit.
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    data["body_quat_w"][0, 0] = np.asarray([5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.savez(path, **data)
    with pytest.raises(MotionFormatError, match="non-unit quaternions"):
        MotrixMotion(path)


def test_loader_rejects_inconsistent_shapes(tmp_path):
    path = tmp_path / "bad.npz"
    _make_minimal_npz(path)
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    # Truncate joint_pos without updating num_frames.
    data["joint_pos"] = data["joint_pos"][:5]
    np.savez(path, **data)
    with pytest.raises(MotionFormatError, match=r"joint_pos.* shape"):
        MotrixMotion(path)


def test_loader_rejects_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        MotrixMotion(tmp_path / "nope.npz")


def test_body_index_helpers(tmp_path):
    path = tmp_path / "m.npz"
    _make_minimal_npz(path)
    m = MotrixMotion(path)
    assert m.body_index("body_2") == 2
    assert m.joint_index("joint_0") == 0
    with pytest.raises(KeyError):
        m.body_index("missing_body")
    idx = m.body_indices(["body_3", "body_0"])
    assert idx.tolist() == [3, 0]


def test_select_bodies_returns_subset(tmp_path):
    path = tmp_path / "m.npz"
    _make_minimal_npz(path)
    m = MotrixMotion(path)
    slice_ = m.select_bodies(["body_1", "body_3"])
    assert slice_.body_names == ("body_1", "body_3")
    assert slice_.body_pos_w.shape == (10, 2, 3)
    np.testing.assert_allclose(slice_.body_pos_w, m.body_pos_w[:, [1, 3]])


def test_optional_fields_default_to_none(tmp_path):
    path = tmp_path / "m.npz"
    _make_minimal_npz(path)
    m = MotrixMotion(path)
    assert m.tracked_body_names is None
    assert m.reference_body_name is None
    assert m.root_body_name is None
    assert m.clip_name is None


def test_wbt_motion_clip_builds_name_ordered_numeric_views(tmp_path):
    path = tmp_path / "m.npz"
    _make_minimal_npz(path)
    motion = MotrixMotion(path)

    wbt = WbtMotionClip.create(
        motion,
        joint_names=["joint_2", "joint_0"],
        tracked_body_names=("body_3", "body_1"),
        reference_body_name="body_2",
        root_body_name="body_0",
    )

    np.testing.assert_allclose(wbt.joint_pos, motion.joint_pos[:, [2, 0]])
    np.testing.assert_allclose(wbt.joint_vel, motion.joint_vel[:, [2, 0]])
    np.testing.assert_allclose(wbt.tracked_bodies_pos_w, motion.body_pos_w[:, [3, 1]])
    np.testing.assert_allclose(wbt.root_body_pos_w, motion.body_pos_w[:, 0])
    np.testing.assert_allclose(wbt.reference_body_quat_w, motion.body_quat_w[:, 2])
    assert wbt.joint_pos.shape[0] == motion.num_frames

    joint_pos = wbt.joint_pos
    motion.joint_pos = np.zeros_like(motion.joint_pos)
    assert wbt.joint_pos is joint_pos
    assert np.any(wbt.joint_pos)

    array_fields = (
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
    assert all(getattr(wbt, name).flags.c_contiguous for name in array_fields)
