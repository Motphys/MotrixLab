# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert xMimic (Isaac Lab BeyondMimic) Dex-EVT motion NPZ to MotrixLab NPZ v1.

xMimic ships whole-body-tracking motions as Isaac Lab npz files with per-frame
``joint_pos`` plus ``body_{pos,quat,lin_vel,ang_vel}_w`` for all 39 URDF links,
but **no** ``joint_names`` / ``body_names`` and quaternions in **wxyz** order.

Rather than trust the 39-body ordering (Isaac-internal) or the Isaac link frames,
we take only the reduced state — root pose (body index 0 = ``pelvis``) plus the
23 joint angles — remap the joints into the MotrixSim model's order by name, then
run forward kinematics through the same URDF-backed :class:`DexEvt` model used by
training and read world link poses/velocities straight from the engine. This is
the same FK-bake route as :mod:`lafan_converter`, and guarantees
``body_pos_w`` / ``body_names`` line up with the env's ``get_link(...)`` bindings
and include fixed wrist/TCP links retained by the training model.

xMimic input (per frame):
    joint_pos      (T, 23)     radians, XMIMIC_JOINT_ORDER
    body_pos_w     (T, 39, 3)  meters, body[0] = pelvis (root)
    body_quat_w    (T, 39, 4)  wxyz  <-- flipped to xyzw on the way in
    fps            (1,)        typically 100

Output matches ``motrix_envs.motion.schema`` v1 (quats xyzw, joint_pos DOF-only).
"""

from __future__ import annotations

from pathlib import Path

import motrixsim as mtx
import numpy as np
from motrix_robots.dex_evt import DexEvt

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg, SceneCfg, SceneObjsCfg
from motrix_env_motrixsim.compiler import build_scene_model
from motrix_envs.motion.converters.lafan_converter import (
    _angular_velocity_w,
    _normalize,
    _resample,
    _trim,
)
from motrix_envs.motion.schema import SCHEMA_VERSION, XYZW_FROM_WXYZ

# 23-DOF joint order of the xMimic joint_pos columns.
#
# IMPORTANT: this is NOT the ROBOT_CONFIG_DEX_EVT["joint_names"] declaration order.
# gmr_to_npz_inter.py writes the motion into Isaac's joint slots but then LOGS
# ``robot.data.joint_pos`` (Isaac Lab's *internal* order) to the npz. Isaac orders
# the articulation breadth-first from the root, interleaving left/right/waist per
# tree level — verified to reproduce xMimic's body_pos_w to 0.0 cm; the declaration
# order is off by tens of cm on distal links. Same ordering applies to body_quat_w.
XMIMIC_JOINT_ORDER = (
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "waist_yaw_joint",
    "hip_roll_l_joint",
    "hip_roll_r_joint",
    "waist_roll_joint",
    "hip_yaw_l_joint",
    "hip_yaw_r_joint",
    "waist_pitch_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint",
    "ankle_pitch_r_joint",
    "shoulder_roll_l_joint",
    "shoulder_roll_r_joint",
    "ankle_roll_l_joint",
    "ankle_roll_r_joint",
    "shoulder_yaw_l_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_l_joint",
    "elbow_pitch_r_joint",
)

# body index 0 in the xMimic arrays is the floating base / root link.
_XMIMIC_ROOT_INDEX = 0
_DEX_EVT_ROOT_BODY = "pelvis"
_DEX_EVT_REFERENCE_BODY = "waist_pitch_link"  # torso analog (mirrors G1's torso_link)


@configclass
class _RobotOnlySceneObjsCfg(SceneObjsCfg):
    robot: RobotCfg


def _build_default_model() -> mtx.SceneModel:
    """Build the same URDF-backed Dex-EVT robot model used by training."""
    return build_scene_model(SceneCfg(objs=_RobotOnlySceneObjsCfg(robot=DexEvt())))


def _load_xmimic(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load an xMimic npz -> (root_pos (T,3), root_quat_xyzw (T,4), dof (T,23), fps)."""
    d = np.load(path, allow_pickle=True)
    for key in ("joint_pos", "body_pos_w", "body_quat_w", "fps"):
        if key not in d.files:
            raise ValueError(f"{path}: missing required xMimic field '{key}' (have {sorted(d.files)})")
    dof = np.asarray(d["joint_pos"], dtype=np.float64)
    if dof.shape[1] != len(XMIMIC_JOINT_ORDER):
        raise ValueError(
            f"{path}: joint_pos has {dof.shape[1]} columns, expected {len(XMIMIC_JOINT_ORDER)} (XMIMIC_JOINT_ORDER)"
        )
    body_pos = np.asarray(d["body_pos_w"], dtype=np.float64)
    body_quat_wxyz = np.asarray(d["body_quat_w"], dtype=np.float64)
    root_pos = body_pos[:, _XMIMIC_ROOT_INDEX, :]
    root_quat_xyzw = _normalize(body_quat_wxyz[:, _XMIMIC_ROOT_INDEX, :][:, list(XYZW_FROM_WXYZ)])
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    return root_pos, root_quat_xyzw, dof, fps


def convert_xmimic(
    input_path: str | Path,
    output_path: str | Path,
    *,
    robot: str = "dex_evt",
    input_fps: float | None = None,
    output_fps: float = 50.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    model_file: str | Path | None = None,
) -> dict[str, object]:
    """Convert an xMimic Dex-EVT motion NPZ to a MotrixLab motion NPZ v1.

    ``input_fps`` defaults to the value stored in the source npz (xMimic clips are
    100 fps); the clip is resampled to ``output_fps`` (default 50, matching the WBT
    env ``ctrl_dt`` of 0.02 s so one motion frame == one control step). Optionally
    trims to ``[start_sec, end_sec)`` (seconds at input_fps) first.
    """
    if robot != "dex_evt":
        raise ValueError(f"xMimic converter only supports robot 'dex_evt', got {robot!r}")
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input motion file does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_file is None:
        model = _build_default_model()
    else:
        model = mtx.load_model(str(Path(model_file).expanduser()))

    joint_names = [str(n) for n in model.joint_names]
    body_names = [str(n) for n in model.link_names]
    if sorted(joint_names) != sorted(XMIMIC_JOINT_ORDER):
        raise ValueError(
            "Model joints do not match the expected xMimic Dex-EVT joint set.\n"
            f"  model:  {joint_names}\n  xmimic: {list(XMIMIC_JOINT_ORDER)}"
        )
    # Reorder xMimic dof columns into the model's joint order (by name).
    xmimic_to_model = [XMIMIC_JOINT_ORDER.index(name) for name in joint_names]

    root_pos, root_quat, dof_src, src_fps = _load_xmimic(input_path)
    in_fps = float(input_fps) if input_fps is not None else src_fps
    root_pos, root_quat, dof_src = _trim(root_pos, root_quat, dof_src, in_fps, start_sec, end_sec, input_path)
    root_pos, root_quat, dof_src = _resample(root_pos, root_quat, dof_src, in_fps, output_fps)
    dof = dof_src[:, xmimic_to_model]
    num_frames = root_pos.shape[0]

    # World-frame velocities consistent with the free-joint qvel layout
    # ([lin_w(3), ang_w(3), joint_vel(N)]) that forward_kinematic expects.
    dt = 1.0 / output_fps
    root_lin_vel = np.gradient(root_pos, dt, axis=0) if num_frames > 1 else np.zeros_like(root_pos)
    root_ang_vel = _angular_velocity_w(root_quat, dt)
    dof_vel = np.gradient(dof, dt, axis=0) if num_frames > 1 else np.zeros_like(dof)

    qpos = np.concatenate([root_pos, root_quat, dof], axis=1).astype(np.float32)  # (T, 7+N)
    qvel = np.concatenate([root_lin_vel, root_ang_vel, dof_vel], axis=1).astype(np.float32)  # (T, 6+N)
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
        "root_body_name": np.asarray(_DEX_EVT_ROOT_BODY),
        "reference_body_name": np.asarray(_DEX_EVT_REFERENCE_BODY),
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
