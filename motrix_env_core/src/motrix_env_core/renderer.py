# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral rendering: renderer selection and video encoding pacing.

All simulator contact lives in backend-owned renderers (:class:`SimRenderer`
implementations created through the environments); this module only paces and
encodes frames.
"""

from pathlib import Path
from typing import Protocol

import numpy as np

from motrix_env_core.base import EnvCfg
from motrix_env_core.sim.backend import (
    RenderConfig,
    SimRenderer,
)

__all__ = [
    "RenderConfig",
    "RendererHost",
    "SimRenderer",
    "VideoRecorder",
    "create_renderer",
]


class RendererHost(Protocol):
    """Environment surface needed to create a renderer.

    Both ``ArrayEnv`` (NumPy manager/np frontends) and ``TorchEnv`` (which is
    only an ``ABEnv``) satisfy this; typing against the protocol keeps the
    torch call sites type-checked.
    """

    @property
    def cfg(self) -> EnvCfg: ...

    def create_renderer(self, config: RenderConfig) -> SimRenderer: ...


class VideoRecorder:
    """Encode frames from a backend headless renderer at fps-paced control steps."""

    def __init__(
        self,
        frames: SimRenderer,
        *,
        control_dt: float,
        path: str | Path,
        fps: int,
        num_frames: int,
    ):
        self._frames = frames
        self._path = Path(path)
        self._fps = fps
        self._num_frames = num_frames
        self._frames_written = 0
        self._simulation_steps = 0
        self._control_dt = float(control_dt)
        self._writer = None
        self._closed = False

    @property
    def is_done(self) -> bool:
        return self._frames_written >= self._num_frames

    def render(self) -> bool:
        if self.is_done:
            self.close()
            return False

        self._simulation_steps += 1
        target_frames = min(
            self._num_frames,
            int(self._simulation_steps * self._control_dt * self._fps + 1e-9),
        )
        if target_frames <= self._frames_written:
            return True

        frame = self._frames.capture()
        while self._frames_written < target_frames:
            self._write_frame(frame)
            self._frames_written += 1

        if self.is_done:
            self.close()
            return False
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            self._frames.close()

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            # imageio is an optional dependency (``motrix-env-core[video]``):
            # importing it lazily keeps this module importable in installs
            # that only run headless training without video recording.
            try:
                import imageio.v2 as imageio
            except ImportError as e:
                raise ImportError(
                    "VideoRecorder requires the optional video dependencies; "
                    "install them with `pip install motrix-env-core[video]`"
                ) from e

            self._path.parent.mkdir(parents=True, exist_ok=True)
            # macro_block_size=None keeps the configured frame size untouched;
            # the yuv420p pixel format still requires even dimensions, which
            # RenderConfig validates up front.
            self._writer = imageio.get_writer(
                str(self._path),
                fps=self._fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=None,
            )
        self._writer.append_data(frame)


def create_renderer(env: RendererHost, config: RenderConfig | None) -> SimRenderer | VideoRecorder | None:
    """Create the environment's renderer selected by ``config``.

    Interactive configs return the backend renderer directly; headless configs
    wrap it in a :class:`VideoRecorder` that paces frame capture against the
    control step and encodes the video. Both expose ``render`` / ``close``.
    """
    if config is None:
        return None
    renderer = env.create_renderer(config)
    if config.headless:
        assert config.path is not None and config.fps is not None and config.num_frames is not None
        return VideoRecorder(
            renderer,
            control_dt=env.cfg.ctrl_dt,
            path=config.path,
            fps=config.fps,
            num_frames=config.num_frames,
        )
    return renderer
