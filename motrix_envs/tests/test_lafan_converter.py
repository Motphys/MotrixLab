# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import motrixsim as mtx
import numpy as np
from motrix_robots.unitree import UNITREE_G1_ASSET_DIR

from motrix_envs.motion import MotrixMotion
from motrix_envs.motion.converters.lafan_converter import G1_CSV_JOINT_ORDER, convert_lafan

# G1 pelvis rest height (~0.79 m); keeps the synthetic root above ground.
_ROOT_Z = 0.793
_G1_MODEL_FILE = UNITREE_G1_ASSET_DIR / "g1_29dof.xml"


def _write_synthetic_csv(path, num_frames=12):
    """Fabricate a tiny LAFAN-format G1 csv: 3 pos + 4 quat(xyzw) + 29 dof."""
    rng = np.random.default_rng(0)
    n_joints = len(G1_CSV_JOINT_ORDER)
    rows = np.zeros((num_frames, 7 + n_joints), dtype=np.float64)
    t = np.linspace(0.0, 1.0, num_frames)
    rows[:, 0] = 0.10 * t  # root x drifts forward
    rows[:, 1] = 0.0
    rows[:, 2] = _ROOT_Z
    rows[:, 3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quat, xyzw
    # small smooth joint motion so velocities are well-defined
    rows[:, 7:] = 0.05 * np.sin(2.0 * np.pi * t)[:, None] * rng.uniform(-1, 1, n_joints)[None, :]
    np.savetxt(path, rows, delimiter=",")
    return rows


def test_convert_lafan_round_trip(tmp_path):
    csv_path = tmp_path / "synthetic_g1.csv"
    rows = _write_synthetic_csv(csv_path)
    out = tmp_path / "converted.npz"

    # input_fps == output_fps => no resampling, so per-frame values map 1:1.
    stats = convert_lafan(csv_path, out, input_fps=30.0, output_fps=30.0)
    assert stats["num_frames"] == rows.shape[0]
    assert stats["num_joints"] == 29
    assert stats["has_object"] is False

    motion = MotrixMotion(out)
    model = mtx.load_model(str(_G1_MODEL_FILE))
    expected_body_names = [str(name) for name in model.link_names]

    assert stats["num_bodies"] == len(expected_body_names)
    assert motion.body_names == expected_body_names
    assert motion.fps == 30
    assert motion.num_frames == rows.shape[0]
    assert motion.joint_names == list(G1_CSV_JOINT_ORDER)
    assert motion.root_body_name == "pelvis"
    assert motion.reference_body_name == "torso_link"

    # joint_pos is DOF-only and preserves the csv joint columns (identity remap).
    assert motion.joint_pos.shape == (rows.shape[0], 29)
    np.testing.assert_allclose(motion.joint_pos, rows[:, 7:], atol=1e-5)

    # Root link (pelvis) world pose reproduces the csv root pose (xyzw passthrough).
    root = motion.body_index("pelvis")
    np.testing.assert_allclose(motion.body_pos_w[:, root], rows[:, 0:3], atol=1e-5)
    np.testing.assert_allclose(motion.body_quat_w[:, root], rows[:, 3:7], atol=1e-4)

    # Bodies the WBT env tracks must all be present, and arrays finite.
    for name in ("pelvis", "torso_link", "left_wrist_yaw_link", "right_ankle_roll_link"):
        assert name in motion.body_names
    for arr in (motion.body_lin_vel_w, motion.body_ang_vel_w, motion.joint_vel):
        assert np.all(np.isfinite(arr))


def test_convert_lafan_resamples_frame_count(tmp_path):
    csv_path = tmp_path / "synthetic_g1.csv"
    _write_synthetic_csv(csv_path, num_frames=31)  # 1.0s at 30 fps
    out = tmp_path / "converted_50.npz"

    convert_lafan(csv_path, out, input_fps=30.0, output_fps=60.0)
    motion = MotrixMotion(out)
    assert motion.fps == 60
    # ~2x the source frames (duration preserved, exclusive end).
    assert 58 <= motion.num_frames <= 61


def test_convert_lafan_trims_segment(tmp_path):
    csv_path = tmp_path / "synthetic_g1.csv"
    rows = _write_synthetic_csv(csv_path, num_frames=61)  # 2.0s at 30 fps
    out = tmp_path / "trimmed.npz"

    # keep [0.5s, 1.5s) at 30 fps -> frames [15, 45) = 30 frames, no resample.
    convert_lafan(csv_path, out, input_fps=30.0, output_fps=30.0, start_sec=0.5, end_sec=1.5)
    motion = MotrixMotion(out)
    assert motion.num_frames == 30
    np.testing.assert_allclose(motion.joint_pos, rows[15:45, 7:], atol=1e-5)


def test_convert_lafan_rejects_empty_trim(tmp_path):
    csv_path = tmp_path / "synthetic_g1.csv"
    _write_synthetic_csv(csv_path, num_frames=61)
    out = tmp_path / "empty.npz"
    try:
        convert_lafan(csv_path, out, input_fps=30.0, start_sec=1.0, end_sec=1.0)
    except ValueError as exc:
        assert "trim window" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty trim window")


def test_convert_lafan_rejects_wrong_column_count(tmp_path):
    bad = tmp_path / "bad.csv"
    np.savetxt(bad, np.zeros((4, 20)), delimiter=",")
    out = tmp_path / "out.npz"
    try:
        convert_lafan(bad, out)
    except ValueError as exc:
        assert "columns" in str(exc)
    else:
        raise AssertionError("expected ValueError for wrong column count")
