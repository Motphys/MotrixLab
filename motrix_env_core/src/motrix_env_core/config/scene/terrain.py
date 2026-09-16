# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Build-time procedural terrain generators: primitives, quantization, and composition.

All generators follow the `TerrainGeneratorCfg` contract: `generate(size, shape)`
returns normalized `[0, 1]` heights as a MuJoCo-row-major 2D array whose first
dimension (rows, ``shape[0]``) maps to the hfield x axis and second dimension
(columns, ``shape[1]``) maps to the y axis. Physical heights are normalized
values scaled by ``height_scale`` in meters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene._utils import Vec2
from motrix_env_core.config.scene.asset import TerrainGeneratorCfg

# Relative tolerance when comparing physical heights against height_scale bounds.
_HEIGHT_EPS = 1e-6

_STAIRS_AXES = ("x", "y", "radial")
_STAIRS_PROFILES = ("ascending", "descending", "pyramid", "inverted_pyramid")
_PYRAMID_PROFILES = ("pyramid", "inverted_pyramid")
_OBSTACLE_HEIGHT_MODES = ("fixed", "choice")


@configclass
class FlatTerrainGeneratorCfg(TerrainGeneratorCfg):
    """A flat terrain at a constant normalized height (0.0 by default)."""

    # Constant normalized height in [0, 1]; useful as a composition base or reset area.
    height: float = 0.0

    def validate(self) -> None:
        super().validate()
        if not np.isfinite(self.height) or not 0.0 <= self.height <= 1.0:
            raise ValueError(f"FlatTerrainGeneratorCfg.height must be finite and within [0, 1], got {self.height!r}")

    def generate(self, size: Vec2, shape: tuple[int, int]) -> np.ndarray:
        del size
        return np.full(shape, self.height, dtype=np.float32)


@configclass
class StairsTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Regular stairs along one field axis or radially from the center.

    Linear modes occupy the full field along the stair axis and are constant
    along the other axis. ``axis="x"`` varies along ``shape[0]`` rows (the
    hfield x axis); ``axis="y"`` varies along ``shape[1]`` columns (the hfield y
    axis). ``axis="radial"`` lays the steps as concentric square rings around
    the field center (Isaac Lab pyramid-stairs style): ``ascending`` rises
    outward from a low center (a central pit), ``descending`` falls outward
    from a central peak platform, and ``pyramid`` / ``inverted_pyramid`` place
    the extreme ring at mid radius.

    Profiles (step level as a function of position along the stairs):
    - ``ascending``: rises from 0 at the low-index edge to the top at the far edge.
    - ``descending``: starts at the top and descends to 0 at the far edge.
    - ``pyramid``: rises to a central peak and descends again (peak level is
      ``step_count // 2`` steps above the edges).
    - ``inverted_pyramid``: a central pit with the edges at the top (peak level
      is ``step_count // 2`` steps above the pit floor).
    """

    # Stair direction: "x" -> shape[0] rows, "y" -> shape[1] columns,
    # "radial" -> concentric rings around the field center.
    axis: str = "x"
    # Step arrangement profile.
    profile: str = "ascending"
    # Number of steps (each step spans 1/step_count of the field along the axis).
    step_count: int = 8
    # Physical height of a single step in meters.
    step_height: float = 0.05
    # Optional physical tread width in meters. When set, steps are laid out with this
    # fixed tread width (Isaac Lab ``step_width`` style) and ``step_count`` acts as the
    # maximum level cap; when None, ``step_count`` treads evenly span the field.
    step_width: float | None = None
    # Flat central zone width in meters (radial) or center band (linear), kept at the
    # profile's natural center level (platform for descending/pyramid, pit floor for
    # ascending/inverted). 0 disables the platform.
    platform_width: float = 0.0
    # Normalized base offset added to every cell. Pits need headroom below their rim
    # (hfield heights cannot be negative), so tiled layouts raise the base: e.g.
    # base_level=0.5 makes a mound span [0.5, 1.0] and a pit span [0.0, 0.5], keeping
    # tile boundaries at a seamless common level.
    base_level: float = 0.0

    def validate(self) -> None:
        super().validate()
        if self.axis not in _STAIRS_AXES:
            raise ValueError(f"StairsTerrainGeneratorCfg.axis must be one of {_STAIRS_AXES}, got {self.axis!r}")
        if self.profile not in _STAIRS_PROFILES:
            raise ValueError(
                f"StairsTerrainGeneratorCfg.profile must be one of {_STAIRS_PROFILES}, got {self.profile!r}"
            )
        if self.step_count < 2:
            raise ValueError(f"StairsTerrainGeneratorCfg.step_count must be at least 2, got {self.step_count}")
        if not np.isfinite(self.step_height) or self.step_height <= 0.0:
            raise ValueError(
                f"StairsTerrainGeneratorCfg.step_height must be finite and positive, got {self.step_height!r}"
            )
        max_level = (self.step_count - 1) // 2 if self.profile in _PYRAMID_PROFILES else self.step_count - 1
        if max_level * self.step_height > self.height_scale * (1.0 + _HEIGHT_EPS):
            raise ValueError(
                f"StairsTerrainGeneratorCfg requires max level ({max_level}) * step_height = "
                f"{max_level * self.step_height:.4f} m to fit within height_scale = "
                f"{self.height_scale:.4f} m"
            )
        if self.step_width is not None and (not np.isfinite(self.step_width) or self.step_width <= 0.0):
            raise ValueError(
                f"StairsTerrainGeneratorCfg.step_width must be finite and positive when set, got {self.step_width!r}"
            )
        if not np.isfinite(self.base_level) or not 0.0 <= self.base_level <= 1.0:
            raise ValueError(
                f"StairsTerrainGeneratorCfg.base_level must be finite and within [0, 1], got {self.base_level!r}"
            )
        if self.base_level + max_level * self.step_height / self.height_scale > 1.0 + _HEIGHT_EPS:
            raise ValueError(
                f"StairsTerrainGeneratorCfg base_level + stair climb = "
                f"{self.base_level + max_level * self.step_height / self.height_scale:.4f} "
                f"must fit within [0, 1]"
            )
        if not np.isfinite(self.platform_width) or self.platform_width < 0.0:
            raise ValueError(
                f"StairsTerrainGeneratorCfg.platform_width must be finite and non-negative, got {self.platform_width!r}"
            )

    def generate(self, size: Vec2, shape: tuple[int, int]) -> np.ndarray:
        steps = self.step_count
        step_span = self.step_height / self.height_scale
        base_level_frac = self.base_level
        extent = self._extent(size)
        if self.axis == "radial":
            rows = (np.arange(shape[0], dtype=np.float64) + 0.5) / shape[0] - 0.5
            cols = (np.arange(shape[1], dtype=np.float64) + 0.5) / shape[1] - 0.5
            # Chebyshev distance from the center, as a fraction of the field extent.
            radius = np.maximum(np.abs(rows)[:, None], np.abs(cols)[None, :])
            index = self._index_from_distance(radius, extent, steps)
            index = np.broadcast_to(index, shape)
        else:
            axis_length = shape[0] if self.axis == "x" else shape[1]
            pos_frac = (np.arange(axis_length, dtype=np.float64) + 0.5) / axis_length
            index = self._index_from_distance(pos_frac, extent, steps)
            index = np.broadcast_to(index[:, None] if self.axis == "x" else index[None, :], shape)
        if self.profile == "ascending":
            level = index
        elif self.profile == "descending":
            level = steps - 1 - index
        elif self.profile == "pyramid":
            level = np.minimum(index, steps - 1 - index)
        else:  # inverted_pyramid
            level = (steps - 1) // 2 - np.minimum(index, steps - 1 - index)
        level = self._apply_platform(level, size, shape)
        return (base_level_frac + level * step_span).astype(np.float32)

    def _extent(self, size: Vec2) -> float:
        """Physical extent the profile spans: the stair axis for linear modes, the
        shorter side for the radial mode (the largest inscribed square)."""
        if self.axis == "x":
            return size[0]
        if self.axis == "y":
            return size[1]
        return min(size)

    def _index_from_distance(self, distance: np.ndarray, extent: float, steps: int) -> np.ndarray:
        """Step index as a function of distance (fraction of extent) from the profile origin.

        Linear axes span ``steps`` treads over the full extent; the radial axis spans
        ``2 * steps`` (radius is half the extent). With ``step_width`` set, treads have
        a fixed physical width and cells inside the platform sit one level below the
        first outer ring so the platform edge is a real step.
        """
        platform_half_frac = (self.platform_width / 2.0) / extent
        if self.step_width is None:
            scale = 2.0 * steps if self.axis == "radial" else float(steps)
            index = distance * scale
        else:
            w_frac = self.step_width / extent
            inner = np.maximum(distance - platform_half_frac, 0.0)
            bonus = (distance > platform_half_frac) if self.platform_width > 0.0 else 0
            index = np.floor(inner / w_frac) + bonus
        return np.minimum(index.astype(np.int64), steps - 1)

    def _apply_platform(self, level: np.ndarray, size: Vec2, shape: tuple[int, int]) -> np.ndarray:
        """Flatten the central zone to the profile's natural center level."""
        if self.platform_width <= 0.0:
            return level
        platform_half_frac = (self.platform_width / 2.0) / self._extent(size)
        if self.axis == "radial":
            rows = (np.arange(shape[0], dtype=np.float64) + 0.5) / shape[0] - 0.5
            cols = (np.arange(shape[1], dtype=np.float64) + 0.5) / shape[1] - 0.5
            radius = np.maximum(np.abs(rows)[:, None], np.abs(cols)[None, :])
            mask = radius <= platform_half_frac
        else:
            axis_length = shape[0] if self.axis == "x" else shape[1]
            pos_frac = (np.arange(axis_length, dtype=np.float64) + 0.5) / axis_length
            mask_1d = np.abs(pos_frac - 0.5) <= platform_half_frac
            mask = mask_1d[:, None] if self.axis == "x" else mask_1d[None, :]
            mask = np.broadcast_to(mask, shape)
        return np.where(mask, level[tuple(np.array(shape) // 2)], level)


@configclass
class DiscreteObstaclesTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Randomly scattered rectangular bumps or pits on a flat base."""

    # Number of obstacles scattered over the field.
    count: int = 8
    # How obstacle heights are drawn: "fixed" uses ``height`` as-is for every
    # obstacle; "choice" randomly picks from {+height, +height/2, -height/2,
    # -height} per obstacle (Isaac Lab style mixed bumps and pits).
    height_mode: str = "fixed"
    # Signed obstacle height in meters: positive raises a bump, negative digs a pit.
    height: float = 0.1
    # Obstacle extent sampled uniformly in [size_min, size_max] per axis, as a
    # fraction of the full field extent.
    size_min: float = 0.05
    size_max: float = 0.15
    # Flat central zone width in meters kept at base level; 0 disables it.
    platform_width: float = 0.0

    def validate(self) -> None:
        super().validate()
        if self.height_mode not in _OBSTACLE_HEIGHT_MODES:
            raise ValueError(
                f"DiscreteObstaclesTerrainGeneratorCfg.height_mode must be one of "
                f"{_OBSTACLE_HEIGHT_MODES}, got {self.height_mode!r}"
            )
        if self.count < 0:
            raise ValueError(f"DiscreteObstaclesTerrainGeneratorCfg.count must be non-negative, got {self.count}")
        if not 0.0 < self.size_min <= self.size_max <= 1.0:
            raise ValueError(
                "DiscreteObstaclesTerrainGeneratorCfg requires 0 < size_min <= size_max <= 1, "
                f"got {(self.size_min, self.size_max)!r}"
            )
        if not np.isfinite(self.height) or abs(self.height) > 0.5 * self.height_scale * (1.0 + _HEIGHT_EPS):
            raise ValueError(
                f"DiscreteObstaclesTerrainGeneratorCfg requires |height| <= 0.5 * height_scale = "
                f"{0.5 * self.height_scale:.4f} m, got {self.height!r}"
            )
        if not np.isfinite(self.platform_width) or self.platform_width < 0.0:
            raise ValueError(
                f"DiscreteObstaclesTerrainGeneratorCfg.platform_width must be finite and "
                f"non-negative, got {self.platform_width!r}"
            )

    def generate(self, size: Vec2, shape: tuple[int, int]) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        heights = np.full(shape, 0.5, dtype=np.float64)
        rows, cols = shape
        magnitude = abs(self.height) / self.height_scale
        sign = 1.0 if self.height >= 0 else -1.0
        for _ in range(self.count):
            if self.height_mode == "choice":
                delta = float(rng.choice([-magnitude, -magnitude / 2.0, magnitude / 2.0, magnitude]))
            else:
                delta = sign * magnitude
            row_frac = rng.uniform(self.size_min, self.size_max)
            col_frac = rng.uniform(self.size_min, self.size_max)
            center_row = rng.uniform(row_frac / 2.0, 1.0 - row_frac / 2.0)
            center_col = rng.uniform(col_frac / 2.0, 1.0 - col_frac / 2.0)
            r0 = int(np.floor((center_row - row_frac / 2.0) * rows))
            r1 = int(np.ceil((center_row + row_frac / 2.0) * rows))
            c0 = int(np.floor((center_col - col_frac / 2.0) * cols))
            c1 = int(np.ceil((center_col + col_frac / 2.0) * cols))
            heights[r0 : max(r0 + 1, r1), c0 : max(c0 + 1, c1)] = 0.5 + delta
        if self.platform_width > 0.0:
            half_row = (self.platform_width / 2.0) / size[0]
            half_col = (self.platform_width / 2.0) / size[1]
            row_mask = np.abs((np.arange(rows, dtype=np.float64) + 0.5) / rows - 0.5) <= half_row
            col_mask = np.abs((np.arange(cols, dtype=np.float64) + 0.5) / cols - 0.5) <= half_col
            heights[np.ix_(row_mask, col_mask)] = 0.5
        return heights.astype(np.float32)


@configclass
class PyramidSlopeTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Smooth pyramidal slope rising linearly toward the field center.

    Elevation is ``slope`` (rise per meter) times the Chebyshev distance to the
    field boundary, so the peak sits at the center of a square field. With
    ``inverted`` the center dips instead (central pit). ``platform_width``
    flattens a central zone at the peak (or pit floor) level. Generation raises
    when the physical peak exceeds ``height_scale`` since the peak depends on
    the terrain size known only at generation time.
    """

    # Rise per meter of horizontal distance toward the center.
    slope: float = 0.2
    # False: central mound; True: central pit.
    inverted: bool = False
    # Flat central zone width in meters; 0 runs the slope to the center point.
    platform_width: float = 0.0

    def validate(self) -> None:
        super().validate()
        if not np.isfinite(self.slope) or self.slope <= 0.0:
            raise ValueError(f"PyramidSlopeTerrainGeneratorCfg.slope must be finite and positive, got {self.slope!r}")
        if not np.isfinite(self.platform_width) or self.platform_width < 0.0:
            raise ValueError(
                f"PyramidSlopeTerrainGeneratorCfg.platform_width must be finite and non-negative, "
                f"got {self.platform_width!r}"
            )

    def generate(self, size: Vec2, shape: tuple[int, int]) -> np.ndarray:
        half = (size[0] / 2.0, size[1] / 2.0)
        offset_x = (np.arange(shape[0], dtype=np.float64) + 0.5) / shape[0] * size[0] - half[0]
        offset_y = (np.arange(shape[1], dtype=np.float64) + 0.5) / shape[1] * size[1] - half[1]
        edge_distance = np.minimum((half[0] - np.abs(offset_x))[:, None], (half[1] - np.abs(offset_y))[None, :])
        heights = np.maximum(edge_distance - self.platform_width / 2.0, 0.0) * self.slope
        peak = float(heights.max())
        if self.inverted:
            heights = peak - heights
        normalized = heights / self.height_scale
        if normalized.max() > 1.0 + _HEIGHT_EPS:
            raise ValueError(
                f"PyramidSlopeTerrainGeneratorCfg peaks at {float(heights.max()):.4f} m which "
                f"exceeds height_scale = {self.height_scale:.4f} m; increase height_scale or "
                f"lower slope"
            )
        return normalized.astype(np.float32)


@configclass
class QuantizedTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Quantize another generator's output into discrete height levels (terraces).

    The wrapped generator's own ``height_scale`` is ignored: the source output is
    treated as normalized [0, 1] heights and the physical scale is this config's
    ``height_scale``.
    """

    # Generator whose normalized output is quantized.
    source: TerrainGeneratorCfg
    # Number of discrete levels, including both 0 and 1.
    levels: int = 8

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.source, TerrainGeneratorCfg):
            raise TypeError(
                "QuantizedTerrainGeneratorCfg.source must contain TerrainGeneratorCfg, "
                f"got {type(self.source).__name__}"
            )
        if self.levels < 2:
            raise ValueError(f"QuantizedTerrainGeneratorCfg.levels must be at least 2, got {self.levels}")
        self.source.validate()

    def generate(self, size: Vec2, shape: tuple[int, int]) -> np.ndarray:
        heights = np.asarray(self.source.generate(size, shape), dtype=np.float32)
        return (np.rint(heights * (self.levels - 1)) / (self.levels - 1)).astype(np.float32)


@configclass
class TerrainRegionCfg:
    """A rectangular generator patch inside a composite terrain.

    ``center`` and ``size`` are fractions of the full field: center coordinates
    in [0, 1] with (0, 0) at the low-index corner, sizes in (0, 1]. Index 0
    follows ``shape[0]`` rows (hfield x), index 1 follows ``shape[1]`` columns
    (hfield y).
    """

    # Generator sampled over the region rectangle.
    generator: TerrainGeneratorCfg
    # Region center as a fraction of the full field.
    center: Vec2 = (0.5, 0.5)
    # Region extent as a fraction of the full field.
    size: Vec2 = (1.0, 1.0)
    # Blend margin as a fraction of the region extent per axis (0 = hard seam).
    # Edges flush with the field boundary do not blend; gradients only appear on
    # interior seams.
    blend: float = 0.0

    def validate(self) -> None:
        if not isinstance(self.generator, TerrainGeneratorCfg):
            raise TypeError(
                f"TerrainRegionCfg.generator must contain TerrainGeneratorCfg, got {type(self.generator).__name__}"
            )
        for name, vec in (("center", self.center), ("size", self.size)):
            if len(vec) != 2 or any(not np.isfinite(value) for value in vec):
                raise ValueError(f"TerrainRegionCfg.{name} must contain two finite values, got {vec!r}")
        if any(not 0.0 <= value <= 1.0 for value in self.center):
            raise ValueError(f"TerrainRegionCfg.center must lie within [0, 1], got {self.center!r}")
        if any(not 0.0 < value <= 1.0 for value in self.size):
            raise ValueError(f"TerrainRegionCfg.size must lie within (0, 1], got {self.size!r}")
        if not np.isfinite(self.blend) or not 0.0 <= self.blend <= 0.5:
            raise ValueError(f"TerrainRegionCfg.blend must be finite and within [0, 0.5], got {self.blend!r}")
        self.generator.validate()


def _edge_ramp(count: int, flush_start: bool, flush_end: bool, width: float) -> np.ndarray:
    """Linear 0→1 ramp from each non-flush region edge, 1 inside the core."""
    ramp = np.ones(count, dtype=np.float32)
    if width > 0.0:
        pos = np.arange(count, dtype=np.float32)
        if not flush_start:
            ramp = np.minimum(ramp, pos / width)
        if not flush_end:
            ramp = np.minimum(ramp, (count - 1 - pos) / width)
        np.clip(ramp, 0.0, 1.0, out=ramp)
    return ramp


def _check_peak(name: str, heights: np.ndarray, generator: TerrainGeneratorCfg, height_scale: float) -> None:
    peak = float(np.max(heights)) * generator.height_scale
    if peak > height_scale * (1.0 + _HEIGHT_EPS):
        raise ValueError(
            f"{name} peaks at {peak:.4f} m which exceeds its composite height_scale = "
            f"{height_scale:.4f} m; increase the composite height_scale"
        )


@configclass
class CompositeTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Compose terrains: a full-field base with rectangular regions pasted on top.

    Regions are pasted in order; when regions overlap, later regions overwrite
    earlier ones inside the blend core. Sub-generators are rescaled by
    ``sub.height_scale / height_scale`` before pasting, so physical heights are
    conserved (e.g. a stair ``step_height`` stays the same physical size). The
    composite ``height_scale`` must cover every sub-generator's physical peak;
    ``generate()`` raises otherwise since the compiler does not check [0, 1].
    """

    # Generator filling the whole field first.
    base: TerrainGeneratorCfg
    # Regions pasted over the base in order.
    regions: tuple[TerrainRegionCfg, ...] = ()

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.base, TerrainGeneratorCfg):
            raise TypeError(
                f"CompositeTerrainGeneratorCfg.base must contain TerrainGeneratorCfg, got {type(self.base).__name__}"
            )
        self.base.validate()
        for region in self.regions:
            if not isinstance(region, TerrainRegionCfg):
                raise TypeError(
                    f"CompositeTerrainGeneratorCfg.regions must contain TerrainRegionCfg, got {type(region).__name__}"
                )
            region.validate()

    def generate(self, size: Vec2, shape: tuple[int, int]) -> np.ndarray:
        heights = np.asarray(self.base.generate(size, shape), dtype=np.float32)
        _check_peak("CompositeTerrainGeneratorCfg.base", heights, self.base, self.height_scale)
        heights = heights * (self.base.height_scale / self.height_scale)
        for index, region in enumerate(self.regions):
            heights = self._paste_region(heights, region, index, size)
        return heights

    def _paste_region(self, heights: np.ndarray, region: TerrainRegionCfg, index: int, size: Vec2) -> np.ndarray:
        nrow, ncol = heights.shape
        r0 = max(int(np.floor((region.center[0] - region.size[0] / 2.0) * nrow)), 0)
        r1 = min(int(np.ceil((region.center[0] + region.size[0] / 2.0) * nrow)), nrow)
        c0 = max(int(np.floor((region.center[1] - region.size[1] / 2.0) * ncol)), 0)
        c1 = min(int(np.ceil((region.center[1] + region.size[1] / 2.0) * ncol)), ncol)
        if r1 <= r0 or c1 <= c0:
            return heights
        region_shape = (r1 - r0, c1 - c0)
        region_size = (region.size[0] * size[0], region.size[1] * size[1])
        sub = np.asarray(region.generator.generate(region_size, region_shape), dtype=np.float32)
        _check_peak(f"CompositeTerrainGeneratorCfg.regions[{index}]", sub, region.generator, self.height_scale)
        scaled = sub * (region.generator.height_scale / self.height_scale)
        ramp_rows = _edge_ramp(region_shape[0], r0 == 0, r1 == nrow, region.blend * region_shape[0])
        ramp_cols = _edge_ramp(region_shape[1], c0 == 0, c1 == ncol, region.blend * region_shape[1])
        weight = ramp_rows[:, None] * ramp_cols[None, :]
        patch = heights[r0:r1, c0:c1]
        heights[r0:r1, c0:c1] = patch * (1.0 - weight) + scaled * weight
        return heights


def grid_terrain(
    cells: Sequence[Sequence[TerrainGeneratorCfg]],
    *,
    height_scale: float | None = None,
    blend: float = 0.0,
    base: TerrainGeneratorCfg | None = None,
) -> CompositeTerrainGeneratorCfg:
    """Lay generators out on a rows x cols grid as a composite terrain.

    ``cells`` is a rectangular nested sequence (rows of columns). Each cell keeps
    its own generator, so physical heights are conserved. ``height_scale``
    defaults to the largest sub-generator height_scale; ``blend`` is applied to
    every cell boundary. Difficulty gradients are expressed by arranging the
    cells; runtime difficulty switching is out of scope.
    """
    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    if rows == 0 or cols == 0:
        raise ValueError(f"grid_terrain requires a non-empty rectangular grid, got {rows}x{cols}")
    if any(len(row) != cols for row in cells):
        raise ValueError(f"grid_terrain requires every row to have {cols} cells")
    scale = height_scale
    if scale is None:
        scale = max(generator.height_scale for row in cells for generator in row)
    regions = tuple(
        TerrainRegionCfg(
            generator=generator,
            center=((i + 0.5) / rows, (j + 0.5) / cols),
            size=(1.0 / rows, 1.0 / cols),
            blend=blend,
        )
        for i, row in enumerate(cells)
        for j, generator in enumerate(row)
    )
    return CompositeTerrainGeneratorCfg(
        base=base if base is not None else FlatTerrainGeneratorCfg(height_scale=scale),
        regions=regions,
        height_scale=scale,
    )
