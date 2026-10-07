# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert LAFAN1_Retargeting_Dataset G1 CSV motions to MotrixLab motion NPZ v1.

Source: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset

Unlike the Holosoma converter (which reformats an npz that already carries body
poses), the LAFAN CSVs contain only the reduced robot state (root pose + joint
angles). We therefore load the G1 MotrixSim model, run forward kinematics per
frame, and read the world link poses/velocities straight from the engine — the
same route ``scripts/motion/replay.py`` uses to drive a motion. This guarantees
``body_pos_w`` / ``body_names`` line up with the env's ``get_link(...)`` bindings.

CSV layout (30 fps, no header, 36 columns per row):
    [0:3]   root position   (x, y, z)                    meters
    [3:7]   root quaternion  (qx, qy, qz, qw)  -- xyzw    (already MotrixSim order)
    [7:36]  29 joint angles  (radians)         -- Unitree G1 29-dof order

The joint column order matches the model's ``joint_names`` exactly, but we remap
by name so a reordered source still converts correctly. Quaternions are xyzw end
to end (CSV root, MotrixSim dof layout, and schema output all agree) so no flip
is needed anywhere.

Output matches ``motrix_envs.motion.schema`` v1: joint_pos/joint_vel are DOF-only
(root state lives in body_pos_w/body_quat_w at the root link index), quats xyzw.
"""

from __future__ import annotations

from pathlib import Path

import motrixsim as mtx
import numpy as np
from motrix_robots.unitree import UNITREE_G1_ASSET_DIR

from motrix_env_core.math import quaternion
from motrix_envs.motion.schema import SCHEMA_VERSION

# G1 joint order as stored in the CSV columns (dataset meta_data/info.json).
G1_CSV_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Optional metadata baked into the output so replay / WBT binding is self-describing.
_G1_ROOT_BODY = "pelvis"
_G1_REFERENCE_BODY = "torso_link"


def _default_model_file(robot: str) -> Path:
    """Locate the robot-only MJCF used for forward kinematics."""
    if robot != "g1":
        raise ValueError(f"LAFAN converter only supports robot 'g1', got {robot!r}")
    return UNITREE_G1_ASSET_DIR / "g1_29dof.xml"


def _normalize(q: np.ndarray) -> np.ndarray:
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Axis-angle vector of xyzw quaternions -> (..., 3)."""
    q = _normalize(q)
    q = np.where(q[..., 3:4] < 0.0, -q, q)  # shortest path
    w = np.clip(q[..., 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    safe_s = np.where(s > 1e-8, s, 1.0)  # avoid 0/0 at identity; masked out below
    axis = np.where(s[..., None] > 1e-8, q[..., :3] / safe_s[..., None], np.zeros_like(q[..., :3]))
    return axis * angle[..., None]


def _angular_velocity_w(quat: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a sequence of xyzw world orientations -> (T, 3)."""
    if quat.shape[0] < 3:
        return np.zeros((quat.shape[0], 3), dtype=np.float64)
    q_rel = quaternion.mul(quat[2:], quaternion.conjugate(quat[:-2]))  # rotation over 2*dt
    omega = _quat_to_rotvec(q_rel) / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)  # pad ends to keep length T


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical interpolation of xyzw quats. q0,q1: (N,4); t: (N,) -> (N,4)."""
    q0, q1 = _normalize(q0), _normalize(q1)
    tt = t[:, None]
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)  # antipodal / shortest path
    dot = np.clip((q0 * q1).sum(-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    s0 = np.sin((1.0 - tt) * theta) / (sin_theta + 1e-8)
    s1 = np.sin(tt * theta) / (sin_theta + 1e-8)
    out = np.where(np.abs(sin_theta) < 1e-8, (1.0 - tt) * q0 + tt * q1, s0 * q0 + s1 * q1)
    return _normalize(out)


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a LAFAN G1 CSV -> (root_pos (T,3), root_quat_xyzw (T,4), dof_csv_order (T,29))."""
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    expected = 7 + len(G1_CSV_JOINT_ORDER)
    if arr.shape[1] != expected:
        raise ValueError(f"{path}: expected {expected} columns (3 pos + 4 quat + 29 dof), got {arr.shape[1]}")
    root_pos = arr[:, 0:3]
    root_quat_xyzw = _normalize(arr[:, 3:7])
    dof = arr[:, 7:expected]
    return root_pos, root_quat_xyzw, dof


def _trim(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    dof: np.ndarray,
    input_fps: float,
    start_sec: float,
    end_sec: float | None,
    src: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice the clip to [start_sec, end_sec) (seconds at input_fps) before resampling."""
    if start_sec <= 0.0 and end_sec is None:
        return root_pos, root_quat, dof
    n = root_pos.shape[0]
    i0 = min(int(round(max(0.0, start_sec) * input_fps)), n)
    i1 = n if end_sec is None else min(int(round(end_sec * input_fps)), n)
    if i1 - i0 < 2:
        raise ValueError(
            f"{src}: trim window [start_sec={start_sec}, end_sec={end_sec}] selects "
            f"{max(i1 - i0, 0)} frame(s) at {input_fps} fps; need >= 2. "
            f"Clip has {n} frames ({n / input_fps:.2f}s)."
        )
    return root_pos[i0:i1], root_quat[i0:i1], dof[i0:i1]


def _resample(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    dof: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample (lerp pos/dof, slerp quat) from input_fps to output_fps."""
    if output_fps == input_fps or root_pos.shape[0] < 2:
        return root_pos, root_quat, dof
    n_in = root_pos.shape[0]
    duration = (n_in - 1) / input_fps
    times = np.arange(0.0, duration, 1.0 / output_fps)
    phase = times / duration
    i0 = np.floor(phase * (n_in - 1)).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    blend = (phase * (n_in - 1) - i0)[:, None]
    pos = root_pos[i0] * (1 - blend) + root_pos[i1] * blend
    d = dof[i0] * (1 - blend) + dof[i1] * blend
    quat = _slerp(root_quat[i0], root_quat[i1], blend[:, 0])
    return pos, quat, d


def convert_lafan(
    input_path: str | Path,
    output_path: str | Path,
    *,
    robot: str = "g1",
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    model_file: str | Path | None = None,
) -> dict[str, object]:
    """Convert a LAFAN G1 retargeting CSV to a MotrixLab motion NPZ v1.

    Optionally trim to ``[start_sec, end_sec)`` (seconds, at input_fps) to carve a
    short segment out of a long LAFAN clip.

    Returns a stats dict (num_frames/num_joints/num_bodies/has_object/output_path)
    matching the other converters, for logging and testing.
    """
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input motion file does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_file).expanduser() if model_file is not None else _default_model_file(robot)
    model = mtx.load_model(str(model_path))

    joint_names = [str(n) for n in model.joint_names]
    body_names = [str(n) for n in model.link_names]
    if sorted(joint_names) != sorted(G1_CSV_JOINT_ORDER):
        raise ValueError(
            "Model joints do not match the expected LAFAN G1 joint set.\n"
            f"  model: {joint_names}\n  csv:   {list(G1_CSV_JOINT_ORDER)}"
        )
    # Reorder CSV dof columns into the model's joint order (identity for stock G1).
    csv_to_model = [G1_CSV_JOINT_ORDER.index(name) for name in joint_names]

    root_pos, root_quat, dof_csv = _load_csv(input_path)
    root_pos, root_quat, dof_csv = _trim(root_pos, root_quat, dof_csv, input_fps, start_sec, end_sec, input_path)
    root_pos, root_quat, dof_csv = _resample(root_pos, root_quat, dof_csv, input_fps, output_fps)
    dof = dof_csv[:, csv_to_model]
    num_frames = root_pos.shape[0]

    # Velocities in the world frame, consistent with the free-joint qvel layout
    # ([lin_w(3), ang_w(3), joint_vel(N)]) that forward_kinematic expects.
    dt = 1.0 / output_fps
    root_lin_vel = np.gradient(root_pos, dt, axis=0) if num_frames > 1 else np.zeros_like(root_pos)
    root_ang_vel = _angular_velocity_w(root_quat, dt)
    dof_vel = np.gradient(dof, dt, axis=0) if num_frames > 1 else np.zeros_like(dof)

    qpos = np.concatenate([root_pos, root_quat, dof], axis=1).astype(np.float32)  # (T, 36)
    qvel = np.concatenate([root_lin_vel, root_ang_vel, dof_vel], axis=1).astype(np.float32)  # (T, 35)
    if qpos.shape[1] != model.num_dof_pos or qvel.shape[1] != model.num_dof_vel:
        raise ValueError(
            f"qpos/qvel width ({qpos.shape[1]}/{qvel.shape[1]}) != model dof ({model.num_dof_pos}/{model.num_dof_vel})"
        )

    # Batched forward kinematics: one SceneData holding all T frames.
    data = mtx.SceneData(model, batch=[num_frames])
    data.set_dof_pos(qpos, model)
    data.set_dof_vel(qvel)
    model.forward_kinematic(data)

    poses = np.asarray(model.get_link_poses(data), dtype=np.float32)  # (T, B, 7) [x,y,z,i,j,k,w]
    body_pos_w = poses[:, :, 0:3].copy()
    body_quat_w = poses[:, :, 3:7].copy()  # xyzw
    body_lin_vel_w = np.asarray(model.get_link_linear_velocities(data), dtype=np.float32)  # (T, B, 3)
    body_ang_vel_w = np.asarray(model.get_link_angular_velocities(data), dtype=np.float32)  # (T, B, 3)

    output = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "fps": np.int32(round(output_fps)),
        "num_frames": np.int32(num_frames),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "joint_pos": qpos[:, 7:].copy(),  # DOF-only, no root prefix
        "joint_vel": qvel[:, 6:].copy(),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "root_body_name": np.asarray(_G1_ROOT_BODY),
        "reference_body_name": np.asarray(_G1_REFERENCE_BODY),
        "clip_name": np.asarray(input_path.stem),
    }
    np.savez(output_path, **output)

    return {
        "num_frames": num_frames,
        "num_joints": len(joint_names),
        "num_bodies": len(body_names),
        "has_object": False,
        "output_path": str(output_path),
    }
