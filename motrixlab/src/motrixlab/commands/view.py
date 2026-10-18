# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import gymnasium as gym
import hydra
import motrixsim as mtx
import numpy as np
from hydra.core.config_store import ConfigStore
from motrixsim.render import RenderApp, RenderClosedError, RenderSettings
from omegaconf import DictConfig, OmegaConf

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.base import ABEnv
from motrix_env_core.config import SimCfg, configclass
from motrix_env_core.config.scene.base import RobotCfg, SceneCfg, SceneObjsCfg, SceneVisualCfg, SystemCameraCfg
from motrix_env_core.config.scene.geometry import FlatTerrainCfg
from motrix_env_core.config.scene.light import LightCfg
from motrix_env_core.renderer import RenderConfig, create_renderer
from motrix_env_motrixsim.compiler import build_scene_model

if TYPE_CHECKING:
    from motrix_env_motrixsim.torch_env import TorchEnv

DEFAULT_ENV_NAME = "cartpole"
ROBOT_VIEW_FPS = 60


@dataclass
class ViewConfig:
    """Top-level schema for environment and robot viewing."""

    env: str | None = None
    robot: str | None = None
    sim: str | None = None
    num_envs: int = 1


def _sample_random_action_np(env: ABEnv) -> np.ndarray:
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        size = (env.num_envs, *action_space.shape)
        low = action_space.low
        high = action_space.high
        low = np.where(np.isneginf(low), -1e6, low)
        high = np.where(np.isposinf(high), 1e6, high)
        return np.random.uniform(low=low, high=high, size=size).astype(action_space.dtype)
    else:
        raise NotImplementedError("Only Box action space is supported")


def _run_np(env: ArrayEnv):
    renderer = create_renderer(env, RenderConfig())
    env_dt = env.cfg.ctrl_dt
    while True:
        t0 = time.monotonic()
        actions = _sample_random_action_np(env)
        env.step(actions)
        renderer.render()
        real_dt = time.monotonic() - t0
        sleep_dt = env_dt - real_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)


@configclass
class _RobotViewObjsCfg(SceneObjsCfg):
    floor: FlatTerrainCfg = FlatTerrainCfg()
    sun: LightCfg = LightCfg(color=(0.7, 0.7, 0.7), illuminance=10_000.0)


def _make_robot_scene(robot_cfg: RobotCfg) -> SceneCfg:
    camera = SystemCameraCfg(lookat=(0.0, 0.0, 0.75), distance=3.0, elevation=-20.0, azimuth=180.0)
    return SceneCfg(
        system_camera=camera,
        visual=SceneVisualCfg(
            ambient_light_color=(0.3, 0.3, 0.3),
            ambient_light_brightness=1_000.0,
            head_light_color=(0.6, 0.6, 0.6),
            head_light_luminous_power=1_000.0,
            haze=(0.1, 0.1, 0.1, 1.0),
            tone_mapping="none",
        ),
        objs=_RobotViewObjsCfg(robot=robot_cfg),
    )


def _run_robot_motrixsim(scene: SceneCfg) -> None:
    camera = scene.system_camera
    model = build_scene_model(scene)
    data = mtx.SceneData(model, batch=[1])
    data.reset(model)
    model.forward_kinematic(data)

    settings = RenderSettings.performance()
    settings.enable_shadow = True
    renderer = RenderApp()
    try:
        renderer.launch(
            model,
            batch=1,
            render_offset=[[0.0, 0.0, 0.0]],
            render_settings=settings,
        )
        renderer.system_camera.set_view(
            camera.lookat,
            camera.distance,
            camera.elevation,
            camera.azimuth,
        )
        renderer.system_camera.active = True
        while not renderer.is_closed:
            renderer.sync(data=data)
            time.sleep(1.0 / ROBOT_VIEW_FPS)
    except RenderClosedError:
        pass
    finally:
        renderer.__exit__(None, None, None)


def _run_robot_mujoco(scene: SceneCfg) -> None:
    try:
        import mujoco as mj
        import mujoco.viewer as mj_viewer

        from motrix_env_mujoco.compiler import MuJoCoSceneCompiler
    except ModuleNotFoundError as exc:
        if exc.name == "mujoco":
            raise ModuleNotFoundError(
                "MuJoCo robot viewing requires the optional dependency; install motrix-env-mujoco"
            ) from exc
        raise

    model = MuJoCoSceneCompiler().compile(scene, SimCfg())
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    camera = scene.system_camera
    with mj_viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            if camera.lookat is not None:
                viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.elevation = camera.elevation
            viewer.cam.azimuth = camera.azimuth
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / ROBOT_VIEW_FPS)


def _run_robot(robot_cfg: RobotCfg, sim: str | None = None) -> None:
    scene = _make_robot_scene(robot_cfg)
    backend = "motrixsim" if sim is None else sim
    if backend == "motrixsim":
        _run_robot_motrixsim(scene)
    elif backend == "mujoco":
        _run_robot_mujoco(scene)
    else:
        raise ValueError(f"Unsupported robot view sim {backend!r}; expected 'motrixsim' or 'mujoco'")


def _run_torch(env: "TorchEnv"):
    import torch

    renderer = create_renderer(env, RenderConfig())
    env.init_state()
    env_dt = env.cfg.ctrl_dt
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise NotImplementedError("Only Box action space is supported")
    low = torch.as_tensor(np.where(np.isneginf(action_space.low), -1e6, action_space.low))
    high = torch.as_tensor(np.where(np.isposinf(action_space.high), 1e6, action_space.high))
    while True:
        t0 = time.monotonic()
        actions = low + torch.rand((env.num_envs, *action_space.shape)) * (high - low)
        env.step(actions)
        renderer.render()
        real_dt = time.monotonic() - t0
        sleep_dt = env_dt - real_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)


def _torch_frontend_missing(env_name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        f"Environment '{env_name}' uses the torch frontend, but PyTorch is not installed; install torch "
        f"(for example `uv sync --all-packages`) or view a NumPy-frontend environment such as '{DEFAULT_ENV_NAME}'"
    )


def run(cfg: ViewConfig) -> None:
    if cfg.env is not None and cfg.robot is not None:
        raise ValueError("env and robot are mutually exclusive; set only one")

    if cfg.robot is not None:
        if cfg.num_envs != 1:
            raise ValueError("num_envs must be 1 when viewing a robot")
        _run_robot(registry.make_robot_config(cfg.robot), cfg.sim)
        return

    env_name = cfg.env or DEFAULT_ENV_NAME
    try:
        env = registry.make(env_name, num_envs=cfg.num_envs)
    except ModuleNotFoundError as exc:
        # Torch-frontend environments import torch when their factory resolves the env class.
        if exc.name == "torch":
            raise _torch_frontend_missing(env_name) from exc
        raise

    if isinstance(env, ArrayEnv):
        _run_np(env)
        return

    try:
        from motrix_env_motrixsim.torch_env import TorchEnv
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise _torch_frontend_missing(env_name) from exc
        raise

    if isinstance(env, TorchEnv):
        _run_torch(env)
    else:
        raise TypeError(f"Unsupported environment type '{type(env).__name__}'.")


def _hydra_main(cfg: DictConfig) -> None:
    run(cast(ViewConfig, OmegaConf.to_object(cfg)))


def main() -> None:
    ConfigStore.instance().store(name="view_schema", node=ViewConfig)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config-dir")
    args, _ = parser.parse_known_args()
    config_dir = Path(args.config_dir) if args.config_dir is not None else Path.cwd() / "configs"
    config_path = str(config_dir.resolve()) if (config_dir / "view.yaml").is_file() else "../configs"
    hydra.main(version_base=None, config_path=config_path, config_name="view")(_hydra_main)()
