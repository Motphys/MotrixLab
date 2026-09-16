# Procedural Terrain Generation

MotrixLab can declare procedural height-field terrain directly in the scene config: no
external heightmap files are needed — a **terrain generator** deterministically produces the
height data while the model is built. Rough ground, stairs, pyramid slopes, and discrete
obstacles each take a few lines of config, and a `seed` reproduces the exact terrain.

## Minimal example

Declare a procedural height-field asset (`ProceduralHFieldAssetCfg`) under
`SceneCfg.assets`, then mount it as the ground with `HFieldTerrainCfg` under `SceneCfg.objs`:

```python
from motrix_env_core.base import EnvCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    HFieldTerrainCfg,
    MaterialCfg,
    NoiseTerrainGeneratorCfg,
    ProceduralHFieldAssetCfg,
    SceneAssetsCfg,
    SceneCfg,
    SceneObjsCfg,
)


@configclass
class TerrainAssetsCfg(SceneAssetsCfg):
    mat_ground: MaterialCfg = MaterialCfg()
    terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
        generator=NoiseTerrainGeneratorCfg(seed=0, height_scale=0.1),
        size=(64.0, 64.0),   # full world-space X/Y width in meters
        shape=(320, 320),     # height-field resolution (rows, columns)
    )


@configclass
class TerrainObjsCfg(SceneObjsCfg):
    floor: HFieldTerrainCfg = HFieldTerrainCfg(hfield="terrain", material="mat_ground")


@configclass
class MyTaskEnvCfg(EnvCfg):
    scene: SceneCfg = SceneCfg(assets=TerrainAssetsCfg(), objs=TerrainObjsCfg())
```

Among the built-in environments, the rough-terrain tasks `go1-walk-rough`, `go2-walk-rough`,
and `anymalc-walk-rough` all generate terrain through the same
`NoiseTerrainGeneratorCfg(seed=0, height_scale=0.1, flip_y=True)`; preview it directly:

```bash
python scripts/view.py env=go1-walk-rough
```

## The generator contract

Every generator derives from `TerrainGeneratorCfg` and follows one contract:

- `generate(size, shape)` returns heights **normalized to [0, 1]**; the physical height is
  the normalized value times `height_scale` (meters). With `height_scale=0.1`, the whole
  terrain spans at most 0.1 m of elevation.
- The array is MuJoCo row-major: the first dimension (rows, `shape[0]`) maps to the hfield
  x axis and the second dimension (columns, `shape[1]`) to the y axis.
- `seed` drives all randomness; the same seed always yields the same terrain.
- Terrain is generated **once at model build time** and does not change during training; the
  compiler validates the returned array's shape and that values are finite.
- Each generator's `validate()` rejects configurations that exceed `height_scale` (for
  example a total stair climb taller than `height_scale`); the error message reports the
  exact bound.

## Built-in generators

Each generator is a `@configclass` whose fields are the complete set of knobs. The
screenshots below are rendered from the showcase configs in `examples/terrain_generate.py`.

### FlatTerrainGeneratorCfg — flat

A constant normalized height `height` (0.0 by default); typically the base of a composite
terrain or a reset pad.

### NoiseTerrainGeneratorCfg — noise

Per-cell heights uniformly sampled from `[0, 1)`. When `downsampled_scale` (meters) is set,
noise is generated on a coarse grid sampled at that pitch and bilinearly interpolated to the
output resolution, producing smooth rolling hills; without it you get per-cell speckle.

```{figure} /_static/images/tutorial/terrain/noise_rough.jpg
:alt: Screenshot of per-cell noise terrain with grainy bumps

Per-cell noise (`height_scale=0.08`): grainy bumps.
```

```{figure} /_static/images/tutorial/terrain/noise_smooth.jpg
:alt: Screenshot of downsampled noise terrain with smooth rolling hills

The same seed with `downsampled_scale=0.4`: bilinear interpolation turns it into smooth hills.
```

### QuantizedTerrainGeneratorCfg — terraces

Quantizes the normalized output of the `source` generator into `levels` discrete height
levels (including both 0 and 1), forming terraces. The wrapped generator's own
`height_scale` is ignored; the physical scale comes from this config's `height_scale`.

```{figure} /_static/images/tutorial/terrain/noise_terraced.jpg
:alt: Screenshot of quantized noise terrain with stepped terraces

Noise quantized into `levels=5` terraces.
```

### StairsTerrainGeneratorCfg — stairs

Regular stairs along one axis or radially from the center; the main fields:

| Field            | Meaning                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `axis`           | `"x"` / `"y"` linear along the axis; `"radial"` concentric square rings around the field center |
| `profile`        | stair arrangement: `ascending`, `descending`, `pyramid`, `inverted_pyramid`                    |
| `step_count`     | number of steps; when `step_width` is unset the steps evenly span the full axis                |
| `step_height`    | physical height of one step (m)                                                                |
| `step_width`     | optional fixed tread width (m); when set, `step_count` acts as the maximum level cap            |
| `platform_width` | central platform width (m) held at the profile's natural center level; 0 disables it            |
| `base_level`     | normalized base lift, giving pits headroom below the rim (hfield heights cannot be negative)    |

```{figure} /_static/images/tutorial/terrain/stairs_ascending.jpg
:alt: Screenshot of linear stairs rising along the x axis

`axis="x", profile="ascending"`: eight steps rising from low to high along x.
```

```{figure} /_static/images/tutorial/terrain/stairs_pyramid.jpg
:alt: Screenshot of radial descending stairs from a central platform

`axis="radial", profile="descending"`: a central platform descending outward step by step.
```

```{figure} /_static/images/tutorial/terrain/stairs_inverted_pyramid.jpg
:alt: Screenshot of radial inverted-pyramid stairs around a central pit

`axis="radial", profile="inverted_pyramid"`: a central pit with rings rising outward.
```

### DiscreteObstaclesTerrainGeneratorCfg — discrete obstacles

Randomly scatters rectangular bumps or pits over a flat base (normalized height 0.5). With
`height_mode="fixed"` every obstacle uses `height` (positive or negative); with
`height_mode="choice"` each obstacle draws from {+height, +height/2, −height/2, −height},
mixing bumps and pits. Obstacle extents are sampled per axis uniformly in
`size_min`–`size_max` as fractions of the field extent. Validation requires
`|height| <= 0.5 * height_scale`.

```{figure} /_static/images/tutorial/terrain/obstacles_choice.jpg
:alt: Screenshot of discrete-obstacles terrain with scattered bumps and pits

`height_mode="choice"`: a mix of randomly scattered bumps and pits.
```

### PyramidSlopeTerrainGeneratorCfg — pyramidal slope

Height rises linearly (`slope` is the rise per meter) with the Chebyshev distance to the
field boundary, peaking at the center of a square field; `inverted=True` flips it into a
central pit. The peak depends on the field size, known only at generation time — if it
exceeds `height_scale`, generation raises; increase `height_scale` or lower `slope`.

```{figure} /_static/images/tutorial/terrain/slope_compare.jpg
:alt: Pyramidal slope comparison: central mound on the left, inverted central pit on the right

Left: a central mound at `slope=0.1` with `platform_width=0.8`; right: `inverted=True`
flips it into a central pit. With gentle slopes (0.1 m of rise per meter) the relief is
hard to see from a top-down view — inspect slopes with a low-elevation camera.
```

## Composing terrains

`CompositeTerrainGeneratorCfg` stitches several generators into one field: `base` fills the
whole field first, then each `TerrainRegionCfg` in `regions` pastes a rectangular patch on
top. Key points:

- `center` and `size` are **fractions of the full field** (0–1), not meters.
- Regions are pasted in order; when they overlap, later regions overwrite earlier ones.
- Sub-generators are rescaled by `sub height_scale / composite height_scale` before pasting,
  so **physical heights are conserved** — a stair's `step_height` or a slope's `slope` keeps
  its physical size after composition.
- The composite `height_scale` must cover every sub-generator's physical peak; generation
  raises otherwise.
- `blend` (0–0.5) is the transition margin at region edges as a fraction of the region
  extent; edges flush with the field boundary do not blend.

```{figure} /_static/images/tutorial/terrain/composite.jpg
:alt: Screenshot of composite terrain: terraced noise base with stairs, a pit, obstacles, and a slope

Composite terrain: radial stairs, an inverted-pyramid pit, discrete obstacles, and a
pyramidal slope pasted over a terraced noise base.
```

`grid_terrain()` is a grid shortcut over composition — it lays generators out on a
rows × cols difficulty grid where every cell keeps its own generator, and `height_scale`
defaults to the largest sub-generator scale:

```python
from motrix_env_core.config.scene import (
    FlatTerrainGeneratorCfg,
    StairsTerrainGeneratorCfg,
    grid_terrain,
)

terrain = grid_terrain(
    [
        [FlatTerrainGeneratorCfg(), StairsTerrainGeneratorCfg(step_count=3, step_height=0.05)],
        [StairsTerrainGeneratorCfg(step_count=6, step_height=0.08), FlatTerrainGeneratorCfg()],
    ],
    blend=0.1,  # transition margin on every cell boundary, as a fraction of the cell extent
)
```

```{figure} /_static/images/tutorial/terrain/grid.jpg
:alt: Screenshot of grid terrain: 2x3 stair cells arranged by difficulty

`grid_terrain` arranges stairs of varying `step_count` and `step_height` into a difficulty grid.
```

## Using terrain in an environment

The established pattern in the built-in quadruped and humanoid tasks keeps the environment
logic identical to the flat task and swaps only the scene: the flat config uses
`FlatTerrainCfg` as the floor, and the rough variant inherits it, overriding only `scene` —
`assets` becomes the terrain-asset group and `objs.floor` becomes
`HFieldTerrainCfg(hfield="terrain")`:

```python
@registry.envcfg("my-robot-walk-rough")
@configclass
class MyRobotRoughEnvCfg(MyRobotFlatEnvCfg):
    scene: MySceneCfg = MySceneCfg(
        assets=TerrainAssetsCfg(),
        objs=StandardSceneObjsCfg(floor=HFieldTerrainCfg(hfield="terrain", material="mat_ground")),
    )
```

On terrain, resets and rewards are usually measured relative to the **local ground height**.
At runtime, sample the ground height under arbitrary (x, y) points through the
backend-neutral SimBackend interface:

```python
ground_height = env.sim.sample_terrain_height(
    env.cfg.ground_geom_name, env_ids, base_pos[:, None, :2]
)[:, 0]
```

At reset, `go1-walk-rough` samples the terrain near the spawn point and lifts the base above
the highest sample so the robot never spawns inside the ground; its body-height reward also
stays relative to the local terrain.

## Writing a custom generator

Derive from `TerrainGeneratorCfg`, implement `generate(size, shape)` returning normalized
`[0, 1]` heights, and the generator plugs into every composition mechanism:

```python
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import TerrainGeneratorCfg

import numpy as np


@configclass
class SinusoidalTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Sine waves along the x axis."""

    waves: int = 4

    def validate(self) -> None:
        super().validate()
        if self.waves < 1:
            raise ValueError(f"waves must be at least 1, got {self.waves}")

    def generate(self, size: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
        phase = np.linspace(0.0, self.waves * 2.0 * np.pi, shape[0])[:, None]
        heights = 0.5 + 0.5 * np.sin(phase)  # normalized to [0, 1]
        return np.broadcast_to(heights, shape).astype(np.float32)
```

## Previewing and debugging

`examples/terrain_generate.py` builds a showcase terrain for every built-in generator plus
composed layouts, prints height statistics, and saves normalized height maps as grayscale
PNGs; `--render` opens the composite terrain in the MotrixSim viewer:

```bash
python examples/terrain_generate.py                       # every generator, output to terrain_previews/
python examples/terrain_generate.py --resolution 256      # height-field resolution
python examples/terrain_generate.py --render              # open the viewer for the composite terrain
```

The screenshots on this page were rendered offscreen from that script's showcase configs.
