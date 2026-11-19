# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim renderer contract tests (issue #222): RenderApp wiring and modes."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

import motrix_env_motrixsim.compiler as motrixsim_compiler
import motrix_env_motrixsim.renderer as motrixsim_renderer
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.sim.backend import RecordConfig, RenderConfig, SimRenderer
from motrix_env_motrixsim.runtime import MotrixSimBackend


def _system_camera() -> SystemCameraCfg:
    return SystemCameraCfg(
        lookat=(1.0, 2.0, 3.0),
        distance=6.0,
        elevation=-10.0,
        azimuth=180.0,
    )


def test_interactive_renderer_launches_with_system_camera(monkeypatch):
    render_app = MagicMock()
    monkeypatch.setattr(motrixsim_renderer, "RenderApp", lambda headless=False: render_app)
    model = object()
    data = object()

    renderer = motrixsim_renderer.MotrixSimRenderer(
        model,
        lambda: data,
        RenderConfig(),
        num_envs=4,
        render_spacing=2.0,
        system_camera=_system_camera(),
    )

    assert isinstance(renderer, SimRenderer)
    launch_args = render_app.launch.call_args
    assert launch_args.args[0] is model
    assert launch_args.kwargs["batch"] == 4
    # 4 envs arrange into a 2x2 grid with the configured spacing.
    assert launch_args.kwargs["render_offset"] == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]]
    view_args = render_app.system_camera.set_view.call_args.args
    assert view_args[0] == [1.0, 2.0, 3.0]
    assert view_args[1:] == (6.0, -10.0, 180.0)

    renderer.render()
    render_app.sync.assert_called_once_with(data=data)
    renderer.close()
    renderer.close()
    assert render_app.__exit__.call_count == 1


def test_interactive_renderer_space_key_toggles_data_sync(monkeypatch):
    render_app = MagicMock()
    render_app.input.is_key_just_pressed.return_value = True
    monkeypatch.setattr(motrixsim_renderer, "RenderApp", lambda headless=False: render_app)
    data = object()

    renderer = motrixsim_renderer.MotrixSimRenderer(
        object(),
        lambda: data,
        RenderConfig(),
        num_envs=1,
        render_spacing=1.0,
        system_camera=_system_camera(),
    )

    renderer.render()
    assert render_app.sync.call_args.kwargs["data"] is data
    renderer.render()
    assert render_app.sync.call_args.kwargs["data"] is None


def test_interactive_renderer_rejects_capture(monkeypatch):
    render_app = MagicMock()
    monkeypatch.setattr(motrixsim_renderer, "RenderApp", lambda headless=False: render_app)

    renderer = motrixsim_renderer.MotrixSimRenderer(
        object(),
        lambda: object(),
        RenderConfig(),
        num_envs=1,
        render_spacing=1.0,
        system_camera=_system_camera(),
    )

    with pytest.raises(NotImplementedError):
        renderer.capture()


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("record", [None, RecordConfig(width=128, height=64)])
def test_compiler_configures_camera_target_and_optional_record_resolution(monkeypatch, headless, record):
    world = MagicMock()
    # Preserve the model camera's configured size unless recording requests one.
    world.system_camera.resolution = [320, 240]
    compiler = motrixsim_compiler.MotrixSimSceneCompiler()
    monkeypatch.setattr(compiler, "build_world", lambda scene: world)

    model = compiler.compile(SceneCfg(), SimCfg(), render=RenderConfig(headless=headless, record=record))

    assert model is world.build.return_value
    target = motrixsim_compiler.mtx.msd.RenderTarget
    assert world.system_camera.render_target == (target.Image if headless else target.Window)
    assert world.system_camera.resolution == ([record.width, record.height] if record is not None else [320, 240])


def test_backend_headless_target_preserves_live_simulation(monkeypatch):
    # Use the real MSD/model API without starting a GPU render service. Camera
    # output must be configured before build, without changing the live model.
    render_app = MagicMock()
    monkeypatch.setattr(motrixsim_renderer, "RenderApp", lambda headless=False: render_app)
    config = RenderConfig(headless=True, record=RecordConfig(width=128, height=64))
    backend = MotrixSimBackend(SceneCfg(), SimCfg(), 1, render=config)
    live_data = backend._data
    live_model = backend._model
    renderer = backend.create_renderer(
        config,
        num_envs=1,
        render_spacing=1.0,
    )
    render_model = render_app.launch.call_args.args[0]
    assert render_model is live_model
    assert backend._model is live_model
    assert live_model.cameras.system_render_target == "image"
    assert backend._data is live_data
    renderer.render()
    render_app.sync.assert_called_once_with(data=live_data)
    renderer.close()


def test_headless_renderer_captures(monkeypatch):
    render_app = MagicMock()
    monkeypatch.setattr(motrixsim_renderer, "RenderApp", MagicMock(return_value=render_app))
    image = MagicMock()
    image.pixels = np.full((4, 6, 4), 255, dtype=np.uint8)
    render_app.system_camera.capture.return_value.take_image.return_value = image
    model = Mock()
    data = object()
    config = RenderConfig(headless=True, record=RecordConfig(width=128, height=64))

    renderer = motrixsim_renderer.MotrixSimRenderer(
        model,
        lambda: data,
        config,
        num_envs=2,
        render_spacing=1.0,
        system_camera=_system_camera(),
    )
    assert isinstance(renderer, SimRenderer)
    assert render_app.launch.call_args.args[0] is model
    assert render_app.launch.call_args.kwargs["batch"] == 2
    assert motrixsim_renderer.RenderApp.call_args.kwargs == {"headless": True}

    frame = renderer.capture()
    assert frame.shape == (4, 6, 3)
    assert frame.dtype == torch.uint8
    # The capture request rides the sync frame to the renderer and its map
    # callback only fires on a later submit's maintenance, so every capture
    # must block on the readback within the same call (issue #37).
    render_app.sync.assert_called_once_with(data=data, wait=True)

    render_app.sync.reset_mock()
    renderer.render()
    assert render_app.sync.call_args.kwargs == {"data": data}
