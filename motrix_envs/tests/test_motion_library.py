# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_envs.motion import MotionLibrary, MotrixMotion, WbtMotionClip
from motrix_envs.motion.loader import MotionFormatError

# Task-order selection shared by the library tests: joints reordered away from
# the file order, a two-body tracked subset, and distinct reference/root bodies.
_TASK_ORDER = {
    "joint_names": ["joint_2", "joint_0"],
    "tracked_body_names": ("body_1", "body_3"),
    "reference_body_name": "body_2",
    "root_body_name": "body_0",
    "fps": 50,
}


def _write_motion_npz(
    path,
    *,
    num_frames=6,
    fps=50,
    joint_names=("joint_0", "joint_1", "joint_2"),
    body_names=("body_0", "body_1", "body_2", "body_3"),
    extensions=None,
):
    """Write a MotrixLab v1 npz with deterministic arbitrary data."""
    rng = np.random.default_rng(7)
    num_joints = len(joint_names)
    num_bodies = len(body_names)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[..., 3] = 1.0  # identity xyzw = (0,0,0,1)
    data = {
        "schema_version": np.int32(1),
        "fps": np.int32(fps),
        "num_frames": np.int32(num_frames),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "joint_pos": rng.standard_normal((num_frames, num_joints)).astype(np.float32),
        "joint_vel": rng.standard_normal((num_frames, num_joints)).astype(np.float32),
        "body_pos_w": rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        "body_ang_vel_w": rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
    }
    data.update(extensions or {})
    np.savez(path, **data)


def _joint_task_order(motion):
    return [motion.joint_names.index(name) for name in _TASK_ORDER["joint_names"]]


def test_library_concatenates_corpus_in_task_order(tmp_path):
    _write_motion_npz(tmp_path / "a.npz", num_frames=6)
    # Same joint/body sets as a.npz, but stored in a rotated joint order.
    _write_motion_npz(tmp_path / "b.npz", num_frames=4, joint_names=("joint_1", "joint_2", "joint_0"))
    clip = MotionLibrary([tmp_path / "a.npz", tmp_path / "b.npz"], **_TASK_ORDER).assemble()

    a = MotrixMotion(tmp_path / "a.npz")
    b = MotrixMotion(tmp_path / "b.npz")
    np.testing.assert_array_equal(
        clip.joint_pos,
        np.concatenate([a.joint_pos[:, _joint_task_order(a)], b.joint_pos[:, _joint_task_order(b)]]),
    )
    np.testing.assert_array_equal(
        clip.tracked_bodies_pos_w,
        np.concatenate([a.body_pos_w[:, [1, 3]], b.body_pos_w[:, [1, 3]]]),
    )
    np.testing.assert_array_equal(clip.root_body_quat_w, np.concatenate([a.body_quat_w[:, 0], b.body_quat_w[:, 0]]))
    np.testing.assert_array_equal(clip.reference_body_pos_w, np.concatenate([a.body_pos_w[:, 2], b.body_pos_w[:, 2]]))
    assert clip.joint_pos.shape == (10, 2)
    assert clip.tracked_bodies_ang_vel_w.shape == (10, 2, 3)
    individual_clips = [
        WbtMotionClip.create(
            motion,
            _TASK_ORDER["joint_names"],
            _TASK_ORDER["tracked_body_names"],
            _TASK_ORDER["reference_body_name"],
            _TASK_ORDER["root_body_name"],
        )
        for motion in (a, b)
    ]
    for field in (
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
    ):
        np.testing.assert_array_equal(
            getattr(clip, field), np.concatenate([getattr(part, field) for part in individual_clips])
        )


def test_library_emits_per_frame_clip_boundaries(tmp_path):
    _write_motion_npz(tmp_path / "a.npz", num_frames=6)
    _write_motion_npz(tmp_path / "b.npz", num_frames=4)
    _write_motion_npz(tmp_path / "c.npz", num_frames=5)
    clip = MotionLibrary([tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"], **_TASK_ORDER).assemble()

    np.testing.assert_array_equal(clip.frame_clip_end, [5] * 6 + [9] * 4 + [14] * 5)
    assert clip.frame_clip_end.dtype == np.int64


@pytest.mark.parametrize("directory_source", [False, True])
def test_library_single_file_matches_clip_create(tmp_path, directory_source):
    _write_motion_npz(tmp_path / "only.npz", num_frames=6)
    source = tmp_path if directory_source else tmp_path / "only.npz"
    library_clip = MotionLibrary([source], **_TASK_ORDER).assemble()
    single_clip = WbtMotionClip.create(
        MotrixMotion(tmp_path / "only.npz"),
        _TASK_ORDER["joint_names"],
        _TASK_ORDER["tracked_body_names"],
        _TASK_ORDER["reference_body_name"],
        _TASK_ORDER["root_body_name"],
    )
    for field in (
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
        "frame_clip_end",
    ):
        np.testing.assert_array_equal(getattr(library_clip, field), getattr(single_clip, field))


def test_library_directory_expands_npz_in_sorted_order(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_motion_npz(corpus / "late.npz", num_frames=3)
    _write_motion_npz(corpus / "early.npz", num_frames=5)
    ignored = corpus / "notes.txt"
    ignored.write_text("not a motion")
    library = MotionLibrary([corpus], **_TASK_ORDER)
    assert [entry.name for entry in library.entries] == ["early.npz", "late.npz"]
    clip = library.assemble()
    np.testing.assert_array_equal(clip.frame_clip_end, [4] * 5 + [7] * 3)


def test_library_loads_only_declared_extension_channels(tmp_path):
    _write_motion_npz(
        tmp_path / "a.npz",
        num_frames=6,
        extensions={"ext_smpl_joints": np.arange(18, dtype=np.float32).reshape(6, 3)},
    )
    _write_motion_npz(
        tmp_path / "b.npz",
        num_frames=4,
        extensions={"ext_smpl_joints": np.full((4, 3), 9.0, dtype=np.float32)},
    )

    undeclared = MotionLibrary([tmp_path / "a.npz", tmp_path / "b.npz"], **_TASK_ORDER).assemble()
    assert undeclared.extensions.keys_tuple == ()

    declared = MotionLibrary(
        [tmp_path / "a.npz", tmp_path / "b.npz"], extension_channels=("smpl_joints",), **_TASK_ORDER
    ).assemble()
    np.testing.assert_array_equal(
        declared.extensions["smpl_joints"].data,
        np.concatenate([np.arange(18, dtype=np.float32).reshape(6, 3), np.full((4, 3), 9.0, dtype=np.float32)]),
    )


@pytest.mark.parametrize(
    ("corpus_kwargs", "match"),
    [
        # Second file recorded at the wrong rate.
        ({"fps": 60}, "fps must match the control rate 50"),
        # Second file carries a different joint name set.
        ({"joint_names": ("joint_0", "joint_1", "joint_9")}, "Joint name set"),
        # Second file carries a different body name set.
        ({"body_names": ("body_0", "body_1", "body_2", "torso")}, "Body name set"),
    ],
)
def test_library_rejects_cross_file_contract_violations(tmp_path, corpus_kwargs, match):
    _write_motion_npz(tmp_path / "a.npz")
    _write_motion_npz(tmp_path / "b.npz", **corpus_kwargs)
    with pytest.raises(ValueError, match=match):
        MotionLibrary([tmp_path / "a.npz", tmp_path / "b.npz"], **_TASK_ORDER).assemble()


def test_library_rejects_missing_declared_channel(tmp_path):
    _write_motion_npz(tmp_path / "a.npz", extensions={"ext_smpl_joints": np.zeros((6, 3), np.float32)})
    _write_motion_npz(tmp_path / "b.npz", num_frames=4)
    with pytest.raises(ValueError, match="does not provide declared extension channel 'ext_smpl_joints'"):
        MotionLibrary(
            [tmp_path / "a.npz", tmp_path / "b.npz"], extension_channels=("smpl_joints",), **_TASK_ORDER
        ).assemble()


def test_library_rejects_channel_shape_mismatch(tmp_path):
    _write_motion_npz(tmp_path / "a.npz", extensions={"ext_smpl_joints": np.zeros((6, 3), np.float32)})
    _write_motion_npz(tmp_path / "b.npz", num_frames=4, extensions={"ext_smpl_joints": np.zeros((4, 7), np.float32)})
    with pytest.raises(ValueError, match="must keep one shape across the corpus"):
        MotionLibrary(
            [tmp_path / "a.npz", tmp_path / "b.npz"], extension_channels=("smpl_joints",), **_TASK_ORDER
        ).assemble()


def test_library_rejects_non_finite_values(tmp_path):
    _write_motion_npz(tmp_path / "a.npz")
    _write_motion_npz(tmp_path / "b.npz")
    with np.load(tmp_path / "b.npz", allow_pickle=False) as data:
        fields = {key: data[key] for key in data.files}
    fields["body_pos_w"][2, 1] = np.nan
    np.savez(tmp_path / "b.npz", **fields)
    with pytest.raises(ValueError, match="body_pos_w.*non-finite"):
        MotionLibrary([tmp_path / "a.npz", tmp_path / "b.npz"], **_TASK_ORDER).assemble()


def test_library_rejects_non_finite_extension_channel(tmp_path):
    _write_motion_npz(tmp_path / "a.npz", extensions={"ext_smpl_joints": np.zeros((6, 3), np.float32)})
    with np.load(tmp_path / "a.npz", allow_pickle=False) as data:
        fields = {key: data[key] for key in data.files}
    fields["ext_smpl_joints"][3, 1] = np.inf
    np.savez(tmp_path / "a.npz", **fields)
    with pytest.raises(MotionFormatError, match="ext_smpl_joints.*non-finite"):
        MotionLibrary([tmp_path / "a.npz"], extension_channels=("smpl_joints",), **_TASK_ORDER).assemble()


def test_library_rejects_empty_source(tmp_path):
    with pytest.raises(ValueError, match="No motion files"):
        MotionLibrary([], **_TASK_ORDER)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No motion files"):
        MotionLibrary([empty_dir], **_TASK_ORDER)


def test_wbt_motion_clip_single_clip_metadata(tmp_path):
    _write_motion_npz(tmp_path / "m.npz", num_frames=6)
    clip = WbtMotionClip.create(
        MotrixMotion(tmp_path / "m.npz"),
        _TASK_ORDER["joint_names"],
        _TASK_ORDER["tracked_body_names"],
        _TASK_ORDER["reference_body_name"],
        _TASK_ORDER["root_body_name"],
    )
    # The single clip spans the whole frame axis, so every frame ends at T-1.
    np.testing.assert_array_equal(clip.frame_clip_end, [5] * 6)
    assert clip.extensions.keys_tuple == ()


def test_wbt_motion_clip_loads_declared_extension_channel(tmp_path):
    _write_motion_npz(
        tmp_path / "m.npz",
        num_frames=6,
        extensions={"ext_smpl_joints": np.arange(18, dtype=np.float32).reshape(6, 3)},
    )
    motion = MotrixMotion(tmp_path / "m.npz")
    clip = WbtMotionClip.create(
        motion,
        _TASK_ORDER["joint_names"],
        _TASK_ORDER["tracked_body_names"],
        _TASK_ORDER["reference_body_name"],
        _TASK_ORDER["root_body_name"],
        extension_channels=("smpl_joints",),
    )
    np.testing.assert_array_equal(clip.extensions["smpl_joints"].data, motion.extensions["smpl_joints"])
    assert clip.extensions["smpl_joints"].data.flags.c_contiguous

    with pytest.raises(MotionFormatError, match="does not provide declared extension channel"):
        WbtMotionClip.create(
            motion,
            _TASK_ORDER["joint_names"],
            _TASK_ORDER["tracked_body_names"],
            _TASK_ORDER["reference_body_name"],
            _TASK_ORDER["root_body_name"],
            extension_channels=("missing_channel",),
        )

    _write_motion_npz(tmp_path / "scalar.npz", extensions={"ext_bad": np.float32(1.0)})
    scalar_motion = MotrixMotion(tmp_path / "scalar.npz")
    with pytest.raises(MotionFormatError, match="must be a per-frame array"):
        WbtMotionClip.create(
            scalar_motion,
            _TASK_ORDER["joint_names"],
            _TASK_ORDER["tracked_body_names"],
            _TASK_ORDER["reference_body_name"],
            _TASK_ORDER["root_body_name"],
            extension_channels=("bad",),
        )
