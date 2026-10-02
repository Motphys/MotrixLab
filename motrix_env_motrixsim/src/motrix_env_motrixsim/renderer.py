# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim-owned render pipeline: the frontend never touches render types."""

from collections.abc import Callable, Sequence

import motrixsim as mtx
import numpy as np
from motrixsim.render import RenderApp, RenderSettings

from motrix_env_core.config.scene import SystemCameraCfg
from motrix_env_core.sim.backend import RenderConfig, SimRenderer

# The live simulator data is read through this source at every frame: NumPy
# environments recreate their ``SceneData`` on re-initialization, while batched
# runtimes mutate one stable instance in place.
DataSource = Callable[[], mtx.SceneData]


class MotrixSimRenderer(SimRenderer):
    """MotrixSim renderer bound to one environment's live data.

    ``config.headless`` selects the mode: an interactive viewer window, or an
    offscreen system-camera frame source for video recording.
    """

    def __init__(
        self,
        model: mtx.SceneModel,
        data_source: DataSource,
        config: RenderConfig,
        *,
        num_envs: int,
        render_spacing: float,
        system_camera: SystemCameraCfg,
        follow_link: str | None = None,
    ):
        self._data_source = data_source
        self._headless = config.headless
        self._closed = False
        if config.headless:
            model.cameras.set_system_render_target("image", int(config.width), int(config.height))
        offsets = _render_layout(num_envs, render_spacing)
        # ``fps`` only affects headless RenderApp: ``None`` selects the render
        # service's on-demand runner — frames are rendered only when a sync
        # arrives and GPU readbacks are drained in-frame, so nothing renders
        # while idle and recording runs at full render speed (issue #37).
        # Windowed mode omits the knob and keeps the library default.
        render_kwargs = {"fps": None} if config.headless else {}
        self._render = RenderApp(headless=config.headless, **render_kwargs)
        self._render.launch(
            model,
            batch=num_envs,
            render_offset=offsets,
            render_settings=_render_settings(),
        )
        # The view is fixed at construction: config camera settings override
        # the scene's system-camera defaults in both modes. A follow target
        # replaces the lookat with the tracked link's live position on every
        # rendered frame (env row 0 plus its render offset).
        self._camera_distance = config.camera_distance if config.camera_distance is not None else system_camera.distance
        self._camera_elevation = (
            config.camera_elevation if config.camera_elevation is not None else system_camera.elevation
        )
        self._camera_azimuth = config.camera_azimuth if config.camera_azimuth is not None else system_camera.azimuth
        _set_system_camera_view(
            self._render,
            offsets,
            config.camera_lookat if config.camera_lookat is not None else system_camera.lookat,
            self._camera_distance,
            self._camera_elevation,
            self._camera_azimuth,
        )
        self._follow_offset = [float(v) for v in offsets[0]]
        self._follow_program = (
            model.compile_query({"follow": mtx.query.LinkPosition([follow_link])}).allocate(data_source())
            if follow_link is not None
            else None
        )
        self._sync_render_data = True
        self._render.system_camera.active = True

    def set_camera_view(self, lookat: Sequence[float], distance: float, elevation: float, azimuth: float) -> None:
        """Update the system camera view (windowed and headless)."""
        lookat = np.asarray(lookat, dtype=np.float64).reshape(-1)
        if lookat.shape != (3,):
            raise ValueError(f"lookat must contain 3 values, got {lookat!r}")
        self._render.system_camera.set_view(
            [float(v) for v in lookat],
            float(distance),
            float(elevation),
            float(azimuth),
        )

    def _update_follow_view(self, data: mtx.SceneData) -> None:
        position = np.asarray(self._follow_program.execute(data).values()[0])[0, 0]
        self._render.system_camera.set_view(
            [float(position[i]) + self._follow_offset[i] for i in range(3)],
            self._camera_distance,
            self._camera_elevation,
            self._camera_azimuth,
        )

    def render(self) -> None:
        if self._headless:
            data = self._data_source()
            if self._follow_program is not None:
                self._update_follow_view(data)
            self._render.sync(data=data)
            return
        if self._sync_render_data:
            data = self._data_source()
            if self._follow_program is not None:
                self._update_follow_view(data)
            self._render.sync(data=data)
        else:
            self._render.sync(data=None)
        if self._render.input.is_key_just_pressed("space"):
            self._sync_render_data = not self._sync_render_data
            self._render.system_camera.active = self._sync_render_data

    def capture(self) -> np.ndarray:
        if not self._headless:
            raise NotImplementedError(
                "Windowed renderers set no system render target; pass headless=True to capture frames."
            )
        if self._follow_program is not None:
            self._update_follow_view(self._data_source())
        # The capture request rides this frame's sync to the renderer; its map
        # callback only fires on a *later* submit's maintenance (issue #37), so
        # a blocking sync drains the service and guarantees the pixels exist
        # before readback.
        task = self._render.system_camera.capture()
        self._render.sync(data=self._data_source(), wait=True)
        image = task.take_image()
        if image is None:
            raise RuntimeError("system camera capture did not return an image")
        return _normalize_rgb_frame(image.pixels)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._render.__exit__(None, None, None)


def _render_layout(num_envs: int, render_spacing: float) -> list[list[float]]:
    cols = int(np.ceil(np.sqrt(num_envs)))
    offsets = []
    for i in range(num_envs):
        row = i // cols
        col = i % cols
        offsets.append([col * render_spacing, row * render_spacing, 0.0])
    return offsets


def _camera_lookat(offsets: list[list[float]], camera_lookat: Sequence[float] | None) -> list[float]:
    if camera_lookat is None:
        offsets_np = np.asarray(offsets, dtype=np.float64)
        return [
            float(np.mean(offsets_np[:, 0])),
            float(np.mean(offsets_np[:, 1])),
            0.75,
        ]
    lookat = np.asarray(camera_lookat, dtype=np.float64).reshape(-1)
    if lookat.shape != (3,):
        raise ValueError(f"camera_lookat must contain 3 values, got {camera_lookat!r}")
    return [float(lookat[0]), float(lookat[1]), float(lookat[2])]


def _set_system_camera_view(
    render: RenderApp,
    offsets: list[list[float]],
    lookat: Sequence[float] | None,
    distance: float,
    elevation: float,
    azimuth: float,
) -> None:
    render.system_camera.set_view(
        _camera_lookat(offsets, lookat),
        float(distance),
        float(elevation),
        float(azimuth),
    )


def _render_settings() -> RenderSettings:
    settings = RenderSettings.performance()
    settings.enable_shadow = True
    return settings


def _normalize_rgb_frame(pixels: np.ndarray) -> np.ndarray:
    frame = np.asarray(pixels)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 frame, got {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.ascontiguousarray(frame, dtype=np.uint8)
