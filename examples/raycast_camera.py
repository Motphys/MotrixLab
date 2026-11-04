# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Show a RaycastCamera and its hit points in the MotrixSim viewer.

The viewer shows the RaycastCamera depth image in an image widget, while the
scene camera is shown as a green sphere with a sparse set of 3D rays. Yellow
rays hit an obstacle; gray rays are misses. Hit points are shown as red spheres.
Misses are blue in the depth widget; nearer surfaces are brighter.

Run from the repository root::

    python examples/raycast_camera.py

Close the MotrixSim window (or press Escape) to exit.
"""

from __future__ import annotations

import time
from pathlib import Path

import motrixsim as mtx
import numpy as np
from motrixsim.render import Color, Layout, RenderApp, RenderSettings

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import SceneCfg, SceneObjsCfg, SceneSensorsCfg, SceneVisualCfg
from motrix_env_core.config.scene.light import LightCfg
from motrix_env_core.config.scene.sensor import PinholeCameraCfg, RaycastCameraSensorCfg
from motrix_env_core.config.scene.site import SiteCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_motrixsim.compiler import build_scene_model

SCENE_PATH = Path(__file__).with_name("assets") / "raycast_camera_scene.xml"
CAMERA_SITE = "camera_mount"
CAMERA_WIDTH = 48
CAMERA_HEIGHT = 27
CAMERA_FAR = 10.0
CAMERA_POSITION = np.array([0.0, -4.0, 1.6], dtype=np.float32)
CAMERA_ROTATION = np.array([-0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32)


@configclass
class ExampleObjects(SceneObjsCfg):
    camera_mount: SiteCfg = SiteCfg(
        parent_link_name="camera_mount",
        rotation=tuple(float(value) for value in CAMERA_ROTATION),
    )
    key_light: LightCfg = LightCfg(
        position=(-4.0, -4.0, 8.0),
        direction=(0.35, 0.55, -1.0),
        color=(1.0, 0.92, 0.78),
        illuminance=18_000.0,
        cast_shadows=True,
    )
    fill_light: LightCfg = LightCfg(
        position=(4.0, -1.0, 5.0),
        direction=(-0.4, 0.5, -0.8),
        color=(0.65, 0.78, 1.0),
        illuminance=6_000.0,
        cast_shadows=False,
    )


@configclass
class ExampleSensors(SceneSensorsCfg):
    camera: RaycastCameraSensorCfg = RaycastCameraSensorCfg(
        site=CAMERA_SITE,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        projection=PinholeCameraCfg.from_fov(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            horizontal_fov=70.0,
        ),
        near=0.05,
        far=CAMERA_FAR,
        outputs=("depth", "point"),
        exclude="parent_body",
    )


def _camera_rays() -> tuple[np.ndarray, np.ndarray]:
    """Return sparse local ray directions and their image coordinates."""
    projection = ExampleSensors().camera.projection
    assert isinstance(projection, PinholeCameraCfg)
    rows = np.arange(1, CAMERA_HEIGHT, 3, dtype=np.float32)
    cols = np.arange(1, CAMERA_WIDTH, 3, dtype=np.float32)
    u, v = np.meshgrid(cols, rows)
    directions = np.stack(
        ((u + 0.5 - projection.cx) / projection.fx, (v + 0.5 - projection.cy) / projection.fy, np.ones_like(u)),
        axis=-1,
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return directions.reshape(-1, 3), np.stack((v, u), axis=-1).reshape(-1, 2).astype(np.int64)


def _depth_image(depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Convert metric depth into an RGBA preview for the image widget."""
    valid = depth >= 0.0
    intensity = np.zeros(depth.shape, dtype=np.uint8)
    intensity[valid] = (255.0 * (1.0 - np.clip(depth[valid] / max_depth, 0.0, 1.0))).astype(np.uint8)
    pixels = np.full((*depth.shape, 4), 255, dtype=np.uint8)
    pixels[..., :3] = intensity[..., None]
    pixels[~valid, :3] = (24, 40, 80)
    return pixels


def _draw_camera_result(render: RenderApp, depth: np.ndarray, points: np.ndarray) -> None:
    """Draw a sparse 3D visualization of the latest camera image."""
    directions, pixels = _camera_rays()
    # The camera is rotated -90 degrees around X: local +Z points toward +Y.
    world_directions = np.stack((directions[:, 0], directions[:, 2], -directions[:, 1]), axis=-1)
    render.gizmos.draw_sphere(0.08, CAMERA_POSITION, color=Color.rgb(0.1, 1.0, 0.2))
    render.gizmos.draw_axes(CAMERA_POSITION, CAMERA_ROTATION.tolist(), 0.35)

    for direction, (row, col) in zip(world_directions, pixels):
        hit_depth = float(depth[row, col])
        if hit_depth >= 0.0:
            endpoint = points[row, col]
            color = Color.rgb(1.0, 0.85, 0.1)
            render.gizmos.draw_sphere(0.025, endpoint, color=Color.rgb(1.0, 0.1, 0.1))
        else:
            endpoint = CAMERA_POSITION + direction * CAMERA_FAR
            color = Color.rgb(0.45, 0.45, 0.45)
        render.gizmos.draw_ray(CAMERA_POSITION, endpoint - CAMERA_POSITION, color=color)


def main() -> None:
    if not hasattr(mtx.msd, "RaycastCameraSensor"):
        raise RuntimeError(
            "This example requires a MotrixSim build with RaycastCamera support; "
            "the installed motrixsim-core does not expose msd.RaycastCameraSensor."
        )
    scene = SceneCfg(
        file=SCENE_PATH,
        objs=ExampleObjects(),
        sensors=ExampleSensors(),
        visual=SceneVisualCfg(
            ambient_light_color=(0.22, 0.24, 0.30),
            ambient_light_brightness=900.0,
            head_light_color=(0.55, 0.62, 0.75),
            head_light_luminous_power=700.0,
            haze=(0.035, 0.045, 0.07, 1.0),
            tone_mapping="aces",
        ),
    )
    model = build_scene_model(scene, SimCfg(dt=0.01))
    data = mtx.SceneData(model, batch=[1])
    settings = RenderSettings.quality()
    settings.enable_shadow = True
    with RenderApp() as render:
        render.launch(
            model,
            batch=1,
            render_offset=[[0.0, 0.0, 0.0]],
            render_settings=settings,
        )
        preview = render.create_image(_depth_image(np.full((CAMERA_HEIGHT, CAMERA_WIDTH), -1.0), CAMERA_FAR))
        render.widgets.create_image_widget(
            preview,
            layout=Layout(right=16, top=16, width=384, height=216),
        )
        render.system_camera.set_view(
            lookat=[0.0, 2.0, 1.4],
            distance=10.0,
            elevation=-18.0,
            azimuth=180.0,
        )
        render.system_camera.active = True
        print("RaycastCamera viewer started")
        print("Lighting: warm key light, cool fill light, soft ambient light, and shadows.")
        print("Depth image widget: near is bright, far is dark, blue means miss.")
        print("Yellow rays hit geometry; gray rays miss; red spheres mark hit points.")
        print("Close the MotrixSim window or press Escape to exit.")

        while not render.is_closed:
            model.step(data)
            raw = model.get_sensor_value("camera", data)
            rays = np.asarray(raw[0], dtype=np.float32).reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 4)
            depth = rays[..., 0]
            points = rays[..., 1:4]
            preview.pixels = _depth_image(depth, CAMERA_FAR)
            _draw_camera_result(render, depth, points)
            render.sync(data=data)
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
