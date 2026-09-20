# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert mjbatch G1 backflip motion NPZ to MotrixLab NPZ v1.

Source: https://github.com/kevinzakka/mjbatch ``examples/assets/flip.npz``
(Apache-2.0; a mocap flip retargeted to the Unitree G1). The file ships
``joint_pos`` / ``body_*`` arrays plus ``fps`` but no ``joint_names`` /
``body_names`` and quaternions in ``wxyz`` order — the same situation as the
K1 MuJoCo clips, so this converter follows
:mod:`scripts.private.k1_mujoco_converter`: assign the menagerie G1 29-DOF
column order, remap joints by name into the training model's order, then
re-bake all body poses/velocities with the same :class:`UnitreeG129Dof` model
used by the WBT environment.

The source is 50 fps, matching the WBT ``ctrl_dt`` of 0.02 s, so no resampling
is needed by default. mjbatch's ``g1_flip.py`` tracks only frames [0, 200)
("the ankles roll after 200"); trim with ``--end-sec 4.0`` for a clean clip.

Column-order assumption is verified at conversion time: the FK-baked
``body_pos_w`` is compared against the source ``body_pos_w`` by body name, and
the conversion aborts if the error is large (wrong joint order assumption).
"""

from __future__ import annotations

from pathlib import Path

import motrixsim as mtx
import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg, SceneCfg, SceneObjsCfg
from motrix_env_core.math import quaternion
from motrix_env_motrixsim.compiler import build_scene_model
from motrix_envs.motion.converters.lafan_converter import (
    _angular_velocity_w,
    _normalize,
    _resample,
    _trim,
)
from motrix_envs.motion.schema import SCHEMA_VERSION, XYZW_FROM_WXYZ
from motrix_envs.robot import UnitreeG129Dof

# Assumed joint column order of the mjbatch flip.npz: the MuJoCo Menagerie
# unitree_g1 29-DOF qpos order (verified indirectly by the FK cross-check).
_G1_JOINT_ORDER = (
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

# Source body column order: menagerie G1 link order, pelvis first.
_G1_ROOT_BODY = "pelvis"
_G1_REFERENCE_BODY = "torso_link"

_MAX_FK_MISMATCH = 5e-3  # m; baked vs source body position, per shared body
_MAX_ROOT_ANG_VEL_MISMATCH = 0.15  # rad/s; baked vs quat-finite-difference root angular velocity


@configclass
class _RobotOnlySceneObjsCfg(SceneObjsCfg):
    robot: RobotCfg


def _build_default_model() -> mtx.SceneModel:
    """Build the same G1 robot model used by the locomotion environments."""
    return build_scene_model(SceneCfg(objs=_RobotOnlySceneObjsCfg(robot=UnitreeG129Dof())))


def convert_g1_flip(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_fps: float | None = None,
    output_fps: float = 50.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    model_file: str | Path | None = None,
) -> dict[str, object]:
    """Convert a mjbatch G1 flip NPZ to a MotrixLab motion NPZ v1."""
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input motion file does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"}
    with np.load(input_path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"mjbatch flip missing keys {sorted(missing)}: {input_path}")
        stored_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        source_joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        source_body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        source_body_quat_wxyz = np.asarray(data["body_quat_w"], dtype=np.float64)

    if source_joint_pos.shape[1] != len(_G1_JOINT_ORDER):
        raise ValueError(f"joint_pos has {source_joint_pos.shape[1]} columns, expected {len(_G1_JOINT_ORDER)}")

    model = mtx.load_model(str(Path(model_file).expanduser())) if model_file is not None else _build_default_model()
    joint_names = [str(n) for n in model.joint_names]
    if sorted(joint_names) != sorted(_G1_JOINT_ORDER):
        raise ValueError(
            "Model joints do not match the assumed G1 joint set.\n"
            f"  model:   {joint_names}\n  assumed: {list(_G1_JOINT_ORDER)}"
        )

    # Root pose from the source pelvis row (menagerie body 0).
    root_pos = source_body_pos[:, 0, :]
    root_quat = _normalize(source_body_quat_wxyz[:, 0, :][:, list(XYZW_FROM_WXYZ)])
    dof_src = source_joint_pos  # already in _G1_JOINT_ORDER

    src_fps = stored_fps if input_fps is None else float(input_fps)
    root_pos, root_quat, dof_src = _trim(root_pos, root_quat, dof_src, src_fps, start_sec, end_sec, input_path)
    root_pos, root_quat, dof_src = _resample(root_pos, root_quat, dof_src, src_fps, output_fps)
    dof = dof_src[:, [_G1_JOINT_ORDER.index(name) for name in joint_names]]
    num_frames = root_pos.shape[0]

    dt = 1.0 / output_fps
    root_lin_vel = np.gradient(root_pos, dt, axis=0) if num_frames > 1 else np.zeros_like(root_pos)
    root_ang_vel_w = _angular_velocity_w(root_quat, dt)
    dof_vel = np.gradient(dof, dt, axis=0) if num_frames > 1 else np.zeros_like(dof)
    # MotrixSim free-joint qvel follows the MuJoCo convention: linear velocity
    # in the world frame but ANGULAR velocity in the body frame. Feeding the
    # world-frame angular velocity here makes FK re-bake every link's angular
    # velocity as R @ omega_world (rotated by the instantaneous pose), which
    # silently corrupts body_ang_vel_w while positions stay correct.
    root_ang_vel_body = quaternion.rotate_inverse(root_quat, root_ang_vel_w)

    qpos = np.concatenate([root_pos, root_quat, dof], axis=1).astype(np.float32)
    qvel = np.concatenate([root_lin_vel, root_ang_vel_body, dof_vel], axis=1).astype(np.float32)
    if qpos.shape[1] != model.num_dof_pos or qvel.shape[1] != model.num_dof_vel:
        raise ValueError(
            f"qpos/qvel width ({qpos.shape[1]}/{qvel.shape[1]}) != model dof ({model.num_dof_pos}/{model.num_dof_vel})"
        )

    data = mtx.SceneData(model, batch=[num_frames])
    data.set_dof_pos(qpos, model)
    data.set_dof_vel(qvel)
    model.forward_kinematic(data)

    poses = np.asarray(model.get_link_poses(data), dtype=np.float32)
    body_pos_w = poses[:, :, 0:3].copy()
    body_names = [str(n) for n in model.link_names]

    # Cross-check the assumed joint order: FK must reproduce the source body
    # positions (modulo the trim/resample boundary rows).
    menagerie_bodies = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ]
    src_body_index = {name: i for i, name in enumerate(menagerie_bodies)}
    n_check = min(num_frames, source_body_pos.shape[0])
    errors = [
        np.linalg.norm(body_pos_w[t, body_names.index(name)] - source_body_pos[t, src_body_index[name]])
        for t in range(n_check)
        for name in menagerie_bodies
        if name in body_names
    ]
    max_err = float(np.max(errors))
    if max_err > _MAX_FK_MISMATCH:
        raise ValueError(
            f"FK cross-check failed: max body position mismatch {max_err:.4f} m > "
            f"{_MAX_FK_MISMATCH} m; the assumed joint column order is likely wrong."
        )

    # Velocity self-check: the baked root angular velocity must match the
    # world-frame quaternion finite difference. A qvel frame-convention bug
    # (world vs body angular velocity) corrupts velocities while leaving all
    # position cross-checks green, so it needs its own guard.
    body_ang_vel_w = np.asarray(model.get_link_angular_velocities(data), dtype=np.float32)
    root_ang_err = float(
        np.abs(body_ang_vel_w[:, body_names.index("pelvis")] - root_ang_vel_w.astype(np.float32)).mean()
    )
    if root_ang_err > _MAX_ROOT_ANG_VEL_MISMATCH:
        raise ValueError(
            f"FK velocity self-check failed: mean root angular velocity mismatch {root_ang_err:.4f} rad/s > "
            f"{_MAX_ROOT_ANG_VEL_MISMATCH} rad/s; the free-joint qvel angular frame convention is likely wrong."
        )

    output = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "fps": np.int32(round(output_fps)),
        "num_frames": np.int32(num_frames),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "joint_pos": qpos[:, 7:].copy(),
        "joint_vel": qvel[:, 6:].copy(),
        "body_pos_w": body_pos_w,
        "body_quat_w": poses[:, :, 3:7].copy(),
        "body_lin_vel_w": np.asarray(model.get_link_linear_velocities(data), dtype=np.float32),
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
        "fk_max_mismatch_m": max_err,
    }
