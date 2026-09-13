# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim renderer contract tests (issue #222): RenderApp wiring and modes."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import motrix_env_motrixsim.renderer as motrixsim_renderer
from motrix_env_core.config.scene import SystemCameraCfg
from motrix_env_core.sim.backend import RenderConfig, SimRenderer


def _system_camera() -> SystemCameraCfg:
    return SystemCameraCfg(
        lookat=(1.0, 2.0, 3.0),
        distance=6.0,
        elevation=-10.0,
        azimuth=180.0,
    )


def test_interactive_renderer_launches_with_system_camera(monkeypatch):
    render_app = MagicMock()
    monkeypatch.setattr(motrixsim_renderer, "RenderApp", lambda headless=False, fps=None: render_app)
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
    monkeypatch.setattr(motrixsim_renderer, "RenderApp", lambda headless=False, fps=None: render_app)
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
    monkeypatch.setattr(motrixsim_renderer, "RenderApp", lambda headless=False, fps=None: render_app)

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


def test_headless_renderer_configures_camera_resolution_and_captures(monkeypatch):
    render_app = MagicMock()
    monkeypatch.setattr(motrixsim_renderer, "RenderApp", MagicMock(return_value=render_app))
    image = MagicMock()
    image.pixels = np.full((4, 6, 4), 255, dtype=np.uint8)
    render_app.system_camera.capture.return_value.take_image.return_value = image
    model = Mock()
    data = object()
    config = RenderConfig(
        headless=True,
        path=Path("/tmp/video.mp4"),
        fps=20,
        num_frames=10,
        width=128,
        height=64,
        camera_lookat=(0.5, 0.5, 0.5),
    )

    renderer = motrixsim_renderer.MotrixSimRenderer(
        model,
        lambda: data,
        config,
        num_envs=2,
        render_spacing=1.0,
        system_camera=_system_camera(),
    )
    assert isinstance(renderer, SimRenderer)
    model.cameras.set_system_render_target.assert_called_once_with("image", 128, 64)
    assert render_app.launch.call_args.kwargs["batch"] == 2
    # Headless default is the render service's on-demand runner (no loop cap).
    assert motrixsim_renderer.RenderApp.call_args.kwargs == {"headless": True, "fps": None}

    frame = renderer.capture()
    assert frame.shape == (4, 6, 3)
    assert frame.dtype == np.uint8
    # The capture request rides the sync frame to the renderer and its map
    # callback only fires on a later submit's maintenance, so every capture
    # must block on the readback within the same call (issue #37).
    render_app.sync.assert_called_once_with(data=data, wait=True)

    render_app.sync.reset_mock()
    renderer.render()
    assert render_app.sync.call_args.kwargs == {"data": data}
