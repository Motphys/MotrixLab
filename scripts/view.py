# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import time
from typing import TYPE_CHECKING

import gymnasium as gym
import hydra
import motrixsim as mtx
import numpy as np
from motrixsim.render import RenderApp, RenderClosedError, RenderSettings
from omegaconf import DictConfig

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.base import ABEnv
from motrix_env_core.config import SimCfg
from motrix_env_core.config.scene import RobotCfg, SystemCameraCfg
from motrix_env_core.renderer import RenderConfig, create_renderer
from motrix_env_motrixsim.compiler import build_scene_model
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_rl.cli import to_typed_config
from motrix_rl.config import ViewConfig

if TYPE_CHECKING:
    from motrix_env_motrixsim.torch_env import TorchEnv

DEFAULT_ENV_NAME = "cartpole"
ROBOT_VIEW_FPS = 60


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


def _make_robot_scene(robot_cfg: RobotCfg) -> StandardSceneCfg:
    camera = SystemCameraCfg(lookat=(0.0, 0.0, 0.75), distance=3.0, elevation=-20.0, azimuth=180.0)
    return StandardSceneCfg(
        system_camera=camera,
        objs=StandardSceneObjsCfg(robot=robot_cfg),
    )


def _run_robot_motrixsim(scene: StandardSceneCfg) -> None:
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


def _run_robot_mujoco(scene: StandardSceneCfg) -> None:
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


@hydra.main(version_base=None, config_path="../configs", config_name="view")
def main(cfg: DictConfig) -> None:
    run(to_typed_config(cfg, ViewConfig))


if __name__ == "__main__":
    main()
