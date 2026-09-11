# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pack paired motrixlab SONIC NPZ clips into the native MotrixLab mmap contract."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from motrix_envs.locomotion.sonic.mdp import G1_SONIC_BODY_NAMES, G1_SONIC_JOINTS
from motrix_envs.motion.sonic import SONIC_ARRAYS, SONIC_PACKED_FORMAT, SONIC_QUATERNION_ORDER

_FPS = 50
_ROBOT_FORMAT = "motrixlab_sonic_robot_v1"
_SMPL_FORMAT = "motrixlab_sonic_smpl_v1"


def _paired_npz(robot_root: Path, smpl_root: Path) -> list[tuple[str, Path, Path]]:
    robots = {path.stem: path for path in robot_root.glob("*.npz")}
    humans = {path.stem: path for path in smpl_root.glob("*.npz")}
    if not robots or set(robots) != set(humans):
        missing_robot = sorted(set(humans) - set(robots))
        missing_smpl = sorted(set(robots) - set(humans))
        raise ValueError(
            f"motrixlab SONIC NPZ pairs do not match (missing_robot={missing_robot}, missing_smpl={missing_smpl})"
        )
    return [(name, robots[name], humans[name]) for name in sorted(robots)]


def _scalar(value: np.ndarray) -> object:
    return np.asarray(value).reshape(-1)[0].item()


def _metadata(robot_path: Path, smpl_path: Path) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    with np.load(robot_path, allow_pickle=False) as robot:
        if str(_scalar(robot["source_format"])) != _ROBOT_FORMAT:
            raise ValueError(f"Unsupported motrixlab robot NPZ format: {robot_path}")
        count = int(_scalar(robot["num_frames"]))
        fps = int(_scalar(robot["fps"]))
        joint_names = tuple(str(value) for value in robot["joint_names"])
        body_names = tuple(str(value) for value in robot["body_names"])
    with np.load(smpl_path, allow_pickle=False) as smpl:
        if str(_scalar(smpl["source_format"])) != _SMPL_FORMAT:
            raise ValueError(f"Unsupported motrixlab SMPL NPZ format: {smpl_path}")
        smpl_count = int(_scalar(smpl["num_frames"]))
        smpl_fps = int(_scalar(smpl["fps"]))
    if count <= 0 or fps != _FPS or smpl_count != count or smpl_fps != fps:
        raise ValueError(f"Invalid frame/fps metadata for motrixlab SONIC pair {robot_path.stem!r}")
    return count, joint_names, body_names


def _wxyz_to_xyzw(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value[..., (1, 2, 3, 0)], dtype=np.float32)


def _validate_quaternions(value: np.ndarray, label: str) -> None:
    deviation = np.abs(np.linalg.norm(value, axis=-1) - 1.0)
    if not np.isfinite(value).all() or float(deviation.max(initial=0.0)) > 1.0e-3:
        raise ValueError(f"{label} contains invalid unit quaternions")


def pack_motrixlab_npz_store(robot_root: Path, smpl_root: Path, output: Path) -> Path:
    """Convert motrixlab wxyz/model-order pairs to MotrixLab xyzw/task-order mmap arrays."""

    robot_root = robot_root.expanduser().resolve()
    smpl_root = smpl_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    pairs = _paired_npz(robot_root, smpl_root)

    metadata = [_metadata(robot, smpl) for _, robot, smpl in pairs]
    lengths = np.asarray([item[0] for item in metadata], dtype=np.int32)
    total_frames = int(lengths.sum(dtype=np.int64))
    source_joint_names = metadata[0][1]
    source_body_names = metadata[0][2]
    if any(item[1:] != metadata[0][1:] for item in metadata):
        raise ValueError("motrixlab SONIC clips do not share one joint/body contract")
    if set(source_joint_names) != set(G1_SONIC_JOINTS):
        raise ValueError("motrixlab SONIC joint names do not match the MotrixLab SONIC task")
    if set(source_body_names) != set(G1_SONIC_BODY_NAMES):
        raise ValueError("motrixlab SONIC body names do not match the MotrixLab SONIC task")
    joint_index = np.asarray([source_joint_names.index(name) for name in G1_SONIC_JOINTS], dtype=np.intp)
    body_index = np.asarray([source_body_names.index(name) for name in G1_SONIC_BODY_NAMES], dtype=np.intp)

    shapes = {
        "joint_pos": (total_frames, len(G1_SONIC_JOINTS)),
        "joint_vel": (total_frames, len(G1_SONIC_JOINTS)),
        "body_pos_w": (total_frames, len(G1_SONIC_BODY_NAMES), 3),
        "body_quat_w": (total_frames, len(G1_SONIC_BODY_NAMES), 4),
        "body_lin_vel_w": (total_frames, len(G1_SONIC_BODY_NAMES), 3),
        "body_ang_vel_w": (total_frames, len(G1_SONIC_BODY_NAMES), 3),
        "smpl_joints": (total_frames, 24, 3),
        "smpl_root_quat": (total_frames, 4),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    arrays: dict[str, np.memmap] = {}
    try:
        for name, shape in shapes.items():
            arrays[name] = np.lib.format.open_memmap(
                temporary / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape
            )
        offset = 0
        for clip_index, (name, robot_path, smpl_path) in enumerate(pairs, start=1):
            count = int(lengths[clip_index - 1])
            end = offset + count
            with np.load(robot_path, allow_pickle=False) as robot:
                robot_values = {field: np.asarray(robot[field], dtype=np.float32) for field in SONIC_ARRAYS[:6]}
            with np.load(smpl_path, allow_pickle=False) as smpl:
                smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
                smpl_root_quat = np.asarray(smpl["smpl_root_quat"], dtype=np.float32)
            actual_shapes = {
                **{field: value.shape for field, value in robot_values.items()},
                "smpl_joints": smpl_joints.shape,
                "smpl_root_quat": smpl_root_quat.shape,
            }
            expected_source_shapes = {
                "joint_pos": (count, len(source_joint_names)),
                "joint_vel": (count, len(source_joint_names)),
                "body_pos_w": (count, len(source_body_names), 3),
                "body_quat_w": (count, len(source_body_names), 4),
                "body_lin_vel_w": (count, len(source_body_names), 3),
                "body_ang_vel_w": (count, len(source_body_names), 3),
                "smpl_joints": (count, 24, 3),
                "smpl_root_quat": (count, 4),
            }
            if actual_shapes != expected_source_shapes:
                raise ValueError(f"motrixlab SONIC clip {name!r} has incompatible shapes: {actual_shapes}")
            if not all(np.isfinite(value).all() for value in (*robot_values.values(), smpl_joints)):
                raise ValueError(f"motrixlab SONIC clip {name!r} contains non-finite values")
            _validate_quaternions(robot_values["body_quat_w"], f"{name}.body_quat_w")
            _validate_quaternions(smpl_root_quat, f"{name}.smpl_root_quat")

            arrays["joint_pos"][offset:end] = robot_values["joint_pos"][:, joint_index]
            arrays["joint_vel"][offset:end] = robot_values["joint_vel"][:, joint_index]
            arrays["body_pos_w"][offset:end] = robot_values["body_pos_w"][:, body_index]
            arrays["body_quat_w"][offset:end] = _wxyz_to_xyzw(robot_values["body_quat_w"][:, body_index])
            arrays["body_lin_vel_w"][offset:end] = robot_values["body_lin_vel_w"][:, body_index]
            arrays["body_ang_vel_w"][offset:end] = robot_values["body_ang_vel_w"][:, body_index]
            arrays["smpl_joints"][offset:end] = smpl_joints
            arrays["smpl_root_quat"][offset:end] = _wxyz_to_xyzw(smpl_root_quat)
            offset = end
            print(f"Packed {clip_index}/{len(pairs)}: {name}", flush=True)

        for array in arrays.values():
            array.flush()
        arrays.clear()
        np.save(temporary / "clip_lengths.npy", lengths)
        (temporary / "clip_names.txt").write_text("".join(f"{name}\n" for name, _, _ in pairs), encoding="utf-8")
        manifest = {
            "format": SONIC_PACKED_FORMAT,
            "quaternion_order": SONIC_QUATERNION_ORDER,
            "fps": _FPS,
            "num_frames": total_frames,
            "num_clips": len(pairs),
            "joint_names": list(G1_SONIC_JOINTS),
            "body_names": list(G1_SONIC_BODY_NAMES),
            "clip_lengths_file": "clip_lengths.npy",
            "clip_names_file": "clip_names.txt",
            "source": {
                "format": "paired_motrixlab_sonic_npz",
                "quaternion_order": "wxyz",
            },
            "arrays": {
                name: {"file": f"{name}.npy", "dtype": "float32", "shape": list(shape)}
                for name, shape in shapes.items()
            },
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        arrays.clear()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
