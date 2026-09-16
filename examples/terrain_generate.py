# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Demonstrate the procedural terrain generators.

Generates a showcase terrain for every primitive plus composed layouts, prints
height statistics, and saves grayscale PNG previews to ``--output``. With
``--render``, the composite terrain is additionally rendered offscreen through
the MotrixSim backend.

Usage::

    python examples/terrain_generate.py
    python examples/terrain_generate.py --output /tmp/previews --resolution 256
    python examples/terrain_generate.py --render

To use a generator in a scene, wrap it in a procedural height-field asset::

    from motrix_env_core.config.scene import ProceduralHFieldAssetCfg

    ProceduralHFieldAssetCfg(generator=MyTerrainGeneratorCfg(...), size=(16.0, 16.0), shape=(256, 256))
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    CompositeTerrainGeneratorCfg,
    DiscreteObstaclesTerrainGeneratorCfg,
    FlatTerrainGeneratorCfg,
    HFieldTerrainCfg,
    LightCfg,
    MaterialCfg,
    NoiseTerrainGeneratorCfg,
    ProceduralHFieldAssetCfg,
    PyramidSlopeTerrainGeneratorCfg,
    QuantizedTerrainGeneratorCfg,
    SceneAssetsCfg,
    SceneCfg,
    SceneObjsCfg,
    SceneVisualCfg,
    SkyboxCfg,
    StairsTerrainGeneratorCfg,
    TerrainGeneratorCfg,
    TerrainRegionCfg,
    TextureCfg,
    grid_terrain,
)

TERRAIN_SIZE = 16.0


def build_showcase() -> dict[str, TerrainGeneratorCfg]:
    """One representative terrain per primitive, plus composed layouts."""
    return {
        "flat": FlatTerrainGeneratorCfg(height=0.0, height_scale=0.3),
        "noise_rough": NoiseTerrainGeneratorCfg(seed=1, height_scale=0.08),
        "noise_smooth": NoiseTerrainGeneratorCfg(seed=1, height_scale=0.08, downsampled_scale=0.4),
        "noise_terraced": QuantizedTerrainGeneratorCfg(
            source=NoiseTerrainGeneratorCfg(seed=1, height_scale=0.08), levels=5, height_scale=0.08
        ),
        "stairs_ascending": StairsTerrainGeneratorCfg(
            axis="x", profile="ascending", step_count=8, step_height=0.1, height_scale=0.8
        ),
        "stairs_pyramid": StairsTerrainGeneratorCfg(
            axis="radial",
            profile="descending",
            step_height=0.14,
            step_width=0.3,
            platform_width=1.5,
            height_scale=1.6,
        ),
        "stairs_inverted_pyramid": StairsTerrainGeneratorCfg(
            axis="radial",
            profile="inverted_pyramid",
            step_height=0.14,
            step_width=0.3,
            platform_width=1.5,
            height_scale=1.6,
        ),
        "obstacles_choice": DiscreteObstaclesTerrainGeneratorCfg(
            seed=5,
            count=12,
            size_min=0.03,
            size_max=0.06,
            height=0.125,
            height_scale=0.25,
            height_mode="choice",
            platform_width=1.0,
        ),
        "slope_mound": PyramidSlopeTerrainGeneratorCfg(slope=0.1, platform_width=0.8, height_scale=0.9),
        "slope_pit": PyramidSlopeTerrainGeneratorCfg(slope=0.1, inverted=True, height_scale=0.9),
        "composite": composite_terrain(),
        "grid": grid_terrain(
            [
                [
                    StairsTerrainGeneratorCfg(
                        axis="radial", profile="descending", step_count=6, step_height=0.05, height_scale=0.3
                    ),
                    StairsTerrainGeneratorCfg(
                        axis="radial", profile="descending", step_count=8, step_height=0.08, height_scale=0.6
                    ),
                    StairsTerrainGeneratorCfg(
                        axis="radial", profile="descending", step_count=10, step_height=0.11, height_scale=1.0
                    ),
                ],
                [
                    StairsTerrainGeneratorCfg(
                        axis="radial", profile="descending", step_count=12, step_height=0.14, height_scale=1.6
                    ),
                    StairsTerrainGeneratorCfg(
                        axis="radial", profile="inverted_pyramid", step_count=12, step_height=0.14, height_scale=1.6
                    ),
                    StairsTerrainGeneratorCfg(
                        axis="radial", profile="inverted_pyramid", step_count=6, step_height=0.08, height_scale=0.5
                    ),
                ],
            ]
        ),
    }


def composite_terrain() -> CompositeTerrainGeneratorCfg:
    """A base rough field with stairs, a pit, obstacles, a slope, and a reset pad."""
    return CompositeTerrainGeneratorCfg(
        height_scale=1.7,
        base=QuantizedTerrainGeneratorCfg(
            source=NoiseTerrainGeneratorCfg(seed=11, height_scale=0.08), levels=5, height_scale=0.08
        ),
        regions=(
            TerrainRegionCfg(
                center=(0.28, 0.28),
                size=(0.44, 0.44),
                blend=0.05,
                generator=StairsTerrainGeneratorCfg(
                    axis="radial",
                    profile="descending",
                    step_count=12,
                    step_height=0.14,
                    step_width=0.3,
                    platform_width=1.5,
                    height_scale=1.6,
                ),
            ),
            TerrainRegionCfg(
                center=(0.28, 0.75),
                size=(0.4, 0.4),
                blend=0.05,
                generator=StairsTerrainGeneratorCfg(
                    axis="radial",
                    profile="inverted_pyramid",
                    step_count=12,
                    step_height=0.14,
                    step_width=0.3,
                    platform_width=1.5,
                    height_scale=1.6,
                ),
            ),
            TerrainRegionCfg(
                center=(0.75, 0.28),
                size=(0.44, 0.44),
                blend=0.05,
                generator=DiscreteObstaclesTerrainGeneratorCfg(
                    seed=5,
                    count=12,
                    size_min=0.03,
                    size_max=0.06,
                    height=0.125,
                    height_scale=0.25,
                    height_mode="choice",
                    platform_width=1.0,
                ),
            ),
            TerrainRegionCfg(
                center=(0.78, 0.78),
                size=(0.3, 0.3),
                blend=0.08,
                generator=PyramidSlopeTerrainGeneratorCfg(slope=0.15, platform_width=0.8, height_scale=0.9),
            ),
        ),
    )


def save_height_map(path: Path, heights: np.ndarray) -> None:
    """Save a normalized height field as an 8-bit grayscale PNG (stdlib only)."""
    lo, hi = float(heights.min()), float(heights.max())
    span = max(hi - lo, 1e-6)
    gray = ((heights - lo) / span * 255.0).astype(np.uint8)
    height, width = gray.shape
    raw = b"".join(b"\x00" + gray[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    idat = zlib.compress(raw)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("terrain_previews"), help="Preview output directory")
    parser.add_argument("--resolution", type=int, default=256, help="Height-field samples per axis")
    parser.add_argument("--render", action="store_true", help="Open the MotrixSim viewer for the composite terrain")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    shape = (args.resolution, args.resolution)
    for name, generator in build_showcase().items():
        generator.validate()
        heights = generator.generate((TERRAIN_SIZE, TERRAIN_SIZE), shape)
        physical = heights.astype(np.float64) * generator.height_scale
        print(
            f"{name:>24}: height {physical.min():+.3f} .. {physical.max():+.3f} m "
            f"({np.unique(np.round(physical, 3)).size} levels)"
        )
        save_height_map(args.output / f"{name}.png", heights)
    print(f"previews saved to {args.output}/")

    if args.render:
        view_terrain()


GROUND_TEXTURE = Path(__file__).parents[1] / "motrix_envs" / "src" / "motrix_envs" / "common" / "motphys-ground.png"


@configclass
class TerrainAssetsCfg(SceneAssetsCfg):
    skybox: SkyboxCfg = SkyboxCfg()
    tex_ground: TextureCfg = TextureCfg(file=GROUND_TEXTURE)
    mat_ground: MaterialCfg = MaterialCfg(texture="tex_ground", texture_repeat=(0.4, 0.4))
    terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
        generator=composite_terrain(), size=(TERRAIN_SIZE, TERRAIN_SIZE), shape=(256, 256)
    )


@configclass
class TerrainObjsCfg(SceneObjsCfg):
    floor: HFieldTerrainCfg = HFieldTerrainCfg(hfield="terrain", material="mat_ground")
    sun: LightCfg = LightCfg(color=(0.7, 0.7, 0.7), illuminance=10_000.0, cast_shadows=False)


def view_terrain() -> None:
    """Open the interactive viewer window showing the composite terrain."""
    import time

    import motrixsim as mtx
    from motrixsim.render import RenderApp, RenderClosedError, RenderSettings

    from motrix_env_motrixsim.compiler import build_scene_model

    model = build_scene_model(
        SceneCfg(
            assets=TerrainAssetsCfg(),
            objs=TerrainObjsCfg(),
            visual=SceneVisualCfg(ambient_light_color=(0.3, 0.3, 0.3), ambient_light_brightness=1_000.0),
        )
    )
    data = mtx.SceneData(model, batch=[1])
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
        # NOTE: system-camera elevation is inverted: negative elevation looks DOWN at the scene.
        renderer.system_camera.set_view(lookat=[0.0, 0.0, 0.3], distance=18.0, elevation=-50.0, azimuth=45.0)
        renderer.system_camera.active = True
        while not renderer.is_closed:
            renderer.sync(data=data)
            time.sleep(1.0 / 60.0)
    except RenderClosedError:
        pass
    finally:
        renderer.__exit__(None, None, None)


if __name__ == "__main__":
    main()
