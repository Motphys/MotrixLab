# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert delivered Booster K1 MuJoCo motion NPZ files to MotrixLab NPZ v1.

The delivered files already contain K1 joint and body names, joint state, and
world-space body state. Their quaternions use ``wxyz`` order, however, and they
do not contain the MotrixLab schema metadata. The source clips are also 30 fps,
while WBT advances one frame per 50 Hz control step.

This converter reads the root pose and joint positions, remaps joints by name,
resamples the reduced state, and bakes all body poses and velocities again with
the same :class:`BoosterK1` model used by training. The result uses ``xyzw``
quaternions and matches ``motrix_envs.motion.schema`` v1.
"""

from __future__ import annotations

from pathlib import Path

import motrixsim as mtx
import numpy as np
from motrix_robots.booster import BoosterK1

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

_K1_ROOT_BODY = "Trunk"
_K1_REFERENCE_BODY = "Trunk"


@configclass
class _RobotOnlySceneObjsCfg(SceneObjsCfg):
    robot: RobotCfg


def _build_default_model() -> mtx.SceneModel:
    """Build the same K1 robot model used by the locomotion environments."""
    return build_scene_model(SceneCfg(objs=_RobotOnlySceneObjsCfg(robot=BoosterK1())))


def convert_k1_mujoco(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_fps: float | None = None,
    output_fps: float = 50.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    model_file: str | Path | None = None,
) -> dict[str, object]:
    """Convert a delivered K1 MuJoCo NPZ clip to MotrixLab motion NPZ v1."""
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input motion file does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required = {"fps", "joint_pos", "body_pos_w", "body_quat_w", "joint_names", "body_names"}
    with np.load(input_path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"K1 MuJoCo motion missing keys {sorted(missing)}: {input_path}")
        stored_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        source_joint_names = [str(name) for name in data["joint_names"].tolist()]
        source_body_names = [str(name) for name in data["body_names"].tolist()]
        source_joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        source_body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        source_body_quat_wxyz = np.asarray(data["body_quat_w"], dtype=np.float64)

    model = mtx.load_model(str(Path(model_file).expanduser())) if model_file is not None else _build_default_model()
    joint_names = [str(name) for name in model.joint_names]
    body_names = [str(name) for name in model.link_names]
    if sorted(source_joint_names) != sorted(joint_names):
        raise ValueError(
            "Source joints do not match the K1 model joint set.\n"
            f"  model:  {joint_names}\n  source: {source_joint_names}"
        )
    if sorted(source_body_names) != sorted(body_names):
        raise ValueError(
            f"Source bodies do not match the K1 model body set.\n  model:  {body_names}\n  source: {source_body_names}"
        )
    if source_joint_pos.ndim != 2 or source_joint_pos.shape[1] != len(source_joint_names):
        raise ValueError(f"joint_pos shape {source_joint_pos.shape}, expected (T, {len(source_joint_names)})")
    expected_body_shape = (source_joint_pos.shape[0], len(source_body_names))
    if source_body_pos.shape != (*expected_body_shape, 3):
        raise ValueError(f"body_pos_w shape {source_body_pos.shape}, expected {(*expected_body_shape, 3)}")
    if source_body_quat_wxyz.shape != (*expected_body_shape, 4):
        raise ValueError(f"body_quat_w shape {source_body_quat_wxyz.shape}, expected {(*expected_body_shape, 4)}")

    root_index = source_body_names.index(_K1_ROOT_BODY)
    root_pos = source_body_pos[:, root_index]
    root_quat = _normalize(source_body_quat_wxyz[:, root_index][:, XYZW_FROM_WXYZ])
    source_to_model = [source_joint_names.index(name) for name in joint_names]
    dof = source_joint_pos[:, source_to_model]

    src_fps = stored_fps if input_fps is None else float(input_fps)
    root_pos, root_quat, dof = _trim(root_pos, root_quat, dof, src_fps, start_sec, end_sec, input_path)
    root_pos, root_quat, dof = _resample(root_pos, root_quat, dof, src_fps, output_fps)
    num_frames = root_pos.shape[0]

    dt = 1.0 / output_fps
    root_lin_vel = np.gradient(root_pos, dt, axis=0) if num_frames > 1 else np.zeros_like(root_pos)
    root_ang_vel = _angular_velocity_w(root_quat, dt)
    dof_vel = np.gradient(dof, dt, axis=0) if num_frames > 1 else np.zeros_like(dof)

    qpos = np.concatenate([root_pos, root_quat, dof], axis=1).astype(np.float32)
    qvel = np.concatenate([root_lin_vel, root_ang_vel, dof_vel], axis=1).astype(np.float32)
    if qpos.shape[1] != model.num_dof_pos or qvel.shape[1] != model.num_dof_vel:
        raise ValueError(
            f"qpos/qvel width ({qpos.shape[1]}/{qvel.shape[1]}) != model dof ({model.num_dof_pos}/{model.num_dof_vel})"
        )

    data = mtx.SceneData(model, batch=[num_frames])
    data.set_dof_pos(qpos, model)
    data.set_dof_vel(qvel)
    model.forward_kinematic(data)

    poses = np.asarray(model.get_link_poses(data), dtype=np.float32)
    output = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "fps": np.int32(round(output_fps)),
        "num_frames": np.int32(num_frames),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "joint_pos": qpos[:, 7:].copy(),
        "joint_vel": qvel[:, 6:].copy(),
        "body_pos_w": poses[:, :, 0:3].copy(),
        "body_quat_w": poses[:, :, 3:7].copy(),
        "body_lin_vel_w": np.asarray(model.get_link_linear_velocities(data), dtype=np.float32),
        "body_ang_vel_w": np.asarray(model.get_link_angular_velocities(data), dtype=np.float32),
        "root_body_name": np.asarray(_K1_ROOT_BODY),
        "reference_body_name": np.asarray(_K1_REFERENCE_BODY),
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
