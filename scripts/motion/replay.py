# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Kinematically replay a MotrixLab motion NPZ with only a RobotCfg.

For each motion frame the root pose/velocity and name-remapped joint state are
written into a standard robot scene, forward kinematics is run, and the frame is
rendered. Replay does not require a registered environment, WBT config, tracked
body list, reward, or termination settings.

Reads MotrixLab motion NPZ v1 files (see ``motrix_envs.motion``). Other source
formats must be converted first via ``scripts/motion/convert.py`` or
``scripts/private/convert.py``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import motrixsim as mtx
import numpy as np
from absl import app, flags
from motrixsim.render import RenderClosedError

from motrix_env_core.config.scene import (
    RobotCfg,
    SystemCameraCfg,
)
from motrix_env_core.renderer import RenderConfig
from motrix_env_motrixsim.compiler import build_scene_model
from motrix_env_motrixsim.renderer import MotrixSimRenderer
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.motion import MotrixMotion
from motrix_envs.robot import BoosterK1, DexEvt, UnitreeG129Dof

logger = logging.getLogger(__name__)

_ROBOT_CFG_FACTORIES: dict[str, Callable[[], RobotCfg]] = {
    "dex-evt": DexEvt,
    "g1-29dof": UnitreeG129Dof,
    "k1": BoosterK1,
}

_ROBOT = flags.DEFINE_enum(
    "robot",
    None,
    sorted(_ROBOT_CFG_FACTORIES),
    "Built-in RobotCfg used to build the replay scene.",
    required=True,
)
_MOTION = flags.DEFINE_string(
    "motion",
    None,
    "MotrixLab v1 motion NPZ file.",
    required=True,
)
_FPS = flags.DEFINE_integer("fps", None, "Override playback frame rate; defaults to the motion's fps.")
_START_STEP = flags.DEFINE_integer("start-step", 0, "Inclusive start frame index.")
_END_STEP = flags.DEFINE_integer("end-step", None, "Exclusive end frame index; defaults to the last frame.")
_LOOP = flags.DEFINE_bool("loop", False, "Loop the clip until the window is closed.")
_SPEED = flags.DEFINE_float("speed", 1.0, "Playback speed multiplier (>0).", lower_bound=0.0)


def _build_replay_scene(robot_cfg: RobotCfg) -> StandardSceneCfg:
    """Build a standard ground scene containing only the requested robot."""
    return StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
        objs=StandardSceneObjsCfg(robot=robot_cfg),
    )


def _model_joint_indices(model: mtx.SceneModel, motion: MotrixMotion) -> np.ndarray:
    """Resolve motion joint columns into model order and validate free-root layout."""
    joint_names = [str(name) for name in model.joint_names]
    expected_qpos = 7 + len(joint_names)
    expected_qvel = 6 + len(joint_names)
    if model.num_dof_pos != expected_qpos or model.num_dof_vel != expected_qvel:
        raise ValueError(
            "Replay requires a floating-base robot with one scalar DOF per named joint; "
            f"model has qpos/qvel widths {model.num_dof_pos}/{model.num_dof_vel}, "
            f"expected {expected_qpos}/{expected_qvel}."
        )
    return motion.joint_indices(joint_names)


def _root_index(motion: MotrixMotion) -> int:
    """Use explicit root metadata when available, else the schema's first body."""
    root_body_name = motion.root_body_name or motion.body_names[0]
    return motion.body_index(root_body_name)


def write_frame(
    model: mtx.SceneModel,
    data: mtx.SceneData,
    motion: MotrixMotion,
    joint_indices: np.ndarray,
    root_index: int,
    step: int,
) -> None:
    """Write one motion frame into SceneData and run forward kinematics."""
    qpos = np.zeros((1, model.num_dof_pos), dtype=np.float32)
    qvel = np.zeros((1, model.num_dof_vel), dtype=np.float32)

    qpos[0, :3] = motion.body_pos_w[step, root_index]
    qpos[0, 3:7] = motion.body_quat_w[step, root_index]
    qpos[0, 7:] = motion.joint_pos[step, joint_indices]

    qvel[0, :3] = motion.body_lin_vel_w[step, root_index]
    qvel[0, 3:6] = motion.body_ang_vel_w[step, root_index]
    qvel[0, 6:] = motion.joint_vel[step, joint_indices]

    data.set_dof_pos(qpos, model)
    data.set_dof_vel(qvel)
    model.forward_kinematic(data)


def _frame_range(motion: MotrixMotion, start: int, end: int | None) -> tuple[int, int]:
    total = motion.num_frames
    start = max(0, min(start, total - 1))
    end = total if end is None else max(start + 1, min(end, total))
    return start, end


def _launch_renderer(model: mtx.SceneModel, data: mtx.SceneData, camera: SystemCameraCfg) -> MotrixSimRenderer:
    return MotrixSimRenderer(
        model,
        lambda: data,
        RenderConfig(),
        num_envs=1,
        render_spacing=1.0,
        system_camera=camera,
    )


def replay(
    robot_cfg: RobotCfg,
    motion_path: str | Path,
    *,
    fps: int | None = None,
    start_step: int = 0,
    end_step: int | None = None,
    loop: bool = False,
    speed: float = 1.0,
) -> None:
    """Replay a motion using only its file and the robot configuration."""
    motion = MotrixMotion(motion_path)
    scene = _build_replay_scene(robot_cfg)
    model = build_scene_model(scene)
    camera = scene.system_camera
    joint_indices = _model_joint_indices(model, motion)
    root_index = _root_index(motion)
    start, end = _frame_range(motion, start_step, end_step)

    playback_fps = motion.fps if fps is None else fps
    if playback_fps <= 0:
        raise ValueError(f"fps must be positive, got {playback_fps}")
    if speed <= 0.0:
        raise ValueError(f"speed must be positive, got {speed}")
    frame_dt = 1.0 / (playback_fps * speed)

    logger.info(
        "Replaying %s with %s: frames [%d, %d) of %d @ %d fps (speed=%.2fx)",
        motion.path,
        type(robot_cfg).__name__,
        start,
        end,
        motion.num_frames,
        playback_fps,
        speed,
    )

    data = mtx.SceneData(model, batch=[1])
    data.reset(model)
    renderer = _launch_renderer(model, data, camera)
    steps = range(start, end)
    try:
        while True:
            for step in steps:
                t0 = time.monotonic()
                write_frame(model, data, motion, joint_indices, root_index, step)
                renderer.render()
                sleep_dt = frame_dt - (time.monotonic() - t0)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
            if not loop:
                break
    except RenderClosedError:
        logger.info("Render window closed.")
    finally:
        renderer.close()


def main(argv):
    del argv  # unused
    replay(
        _ROBOT_CFG_FACTORIES[_ROBOT.value](),
        _MOTION.value,
        fps=_FPS.value,
        start_step=_START_STEP.value,
        end_step=_END_STEP.value,
        loop=_LOOP.value,
        speed=_SPEED.value,
    )


if __name__ == "__main__":
    app.run(main)
