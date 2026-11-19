# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral rendering tests: renderer selection, pacing, and encoding.

Simulator-owned renderer internals are covered by each backend package
(``motrix-env-motrixsim`` ships the MotrixSim renderer contract tests).
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import torch

from motrix_env_core.renderer import RecordConfig, VideoRecorder, create_renderer
from motrix_env_core.sim.backend import RenderConfig, SimRenderer


def test_create_renderer_returns_none_without_config():
    env = Mock(spec=["create_renderer"])
    assert create_renderer(env, None) is None
    env.create_renderer.assert_not_called()


def test_create_renderer_returns_interactive_renderer_directly():
    env = Mock()
    renderer = create_renderer(env, RenderConfig())
    assert renderer is env.create_renderer.return_value
    env.create_renderer.assert_called_once_with(RenderConfig())


def test_create_renderer_wraps_headless_renderer_in_video_recorder():
    env = Mock()
    env.cfg.ctrl_dt = 0.05
    record = RecordConfig(path=Path("/tmp/v.mp4"), fps=10, num_frames=5)
    config = RenderConfig(headless=True, record=record)

    recorder = create_renderer(env, config)

    assert isinstance(recorder, VideoRecorder)
    env.create_renderer.assert_called_once_with(config)
    assert config.record is record


def test_headless_still_capture_needs_no_video_settings():
    env = Mock()
    config = RenderConfig(headless=True, record=RecordConfig(width=127, height=65))
    assert create_renderer(env, config) is env.create_renderer.return_value
    env.create_renderer.assert_called_once_with(config)


def test_render_config_rejects_windowed_video_recording(tmp_path):
    record = RecordConfig(path=tmp_path / "video.mp4", num_frames=5)
    with pytest.raises(ValueError, match="video recording requires headless=True"):
        RenderConfig(record=record)


def test_render_config_rejects_invalid_settings():
    with pytest.raises(ValueError, match="width and height must be positive"):
        RecordConfig(width=0)
    with pytest.raises(ValueError, match="fps must be positive"):
        RecordConfig(path=Path("/tmp/v.mp4"), fps=0, num_frames=5)
    with pytest.raises(ValueError, match="must be even"):
        RecordConfig(width=127, path=Path("/tmp/v.mp4"), fps=10, num_frames=5)
    with pytest.raises(ValueError, match="requires positive num_frames"):
        RecordConfig(path=Path("/tmp/v.mp4"), fps=10)


def test_video_recorder_paces_frames_and_closes_frame_source(tmp_path, monkeypatch):
    frames = MagicMock(spec=SimRenderer)
    frames.capture.return_value = torch.zeros((4, 4, 3), dtype=torch.uint8)
    writer = MagicMock()
    get_writer = MagicMock(return_value=writer)
    # renderer imports imageio lazily inside _write_frame, so patch the real
    # module attribute the function resolves on import. imageio is optional
    # (motrix-env-core[video]); skip cleanly on minimal core installs.
    imageio = pytest.importorskip("imageio.v2", reason="video recording requires motrix-env-core[video]")

    monkeypatch.setattr(imageio, "get_writer", get_writer)

    recorder = VideoRecorder(
        frames,
        control_dt=0.1,
        path=tmp_path / "video.mp4",
        fps=20,
        num_frames=3,
    )
    # Each control step is 0.1s at 20 fps: the first step emits two repeated
    # frames; the second captures once more for the remaining frame and the
    # recorder finishes.
    assert recorder.render() is True
    assert frames.capture.call_count == 1
    assert recorder.render() is False
    assert frames.capture.call_count == 2
    assert recorder.is_done
    recorder.close()
    recorder.close()
    frames.close.assert_called_once()
    writer.close.assert_called_once()
    get_writer.assert_called_once_with(
        str(tmp_path / "video.mp4"), fps=20, codec="libx264", pixelformat="yuv420p", macro_block_size=None
    )
    assert writer.append_data.call_count == 3
