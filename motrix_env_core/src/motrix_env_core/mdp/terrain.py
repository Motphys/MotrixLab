# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Static terrain-height grid shared by manager reset and reward terms."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class HeightFieldGrid:
    """Static terrain grid for batched bilinear height lookup.

    ``enabled=False`` degenerates to the flat-ground ``constant`` height.
    Torch-side lookups (:func:`heightfield_lookup`)
    mirror the engine's hfield sample convention in float64:
    bilinear interpolation of the four surrounding samples after mapping
    world ``xy`` to fractional grid coordinates.

    Attributes:
        heights: ``torch.Tensor`` of shape ``(nrow, ncol)`` holding the
            height-field sample values (row = y 方向, column = x 方向)，
            单位与引擎导出的 hfield 数据一致；存储为 float64，
            插值直接在存储精度上进行。
        origin: 长度为 2 的 ``torch.Tensor``，网格首个采样点 ``(x, y)`` 的
            世界坐标，是 bilinear 插值前的平移原点。
        spacing: 长度为 2 的 ``torch.Tensor``，相邻采样点在世界坐标系下的
            间距 ``(dx, dy)``，用于把世界坐标映射到分数网格坐标。
        z0: 长度为 1 的 ``torch.Tensor``，叠加在插值结果之上的基准高度偏移。
        constant: 平地退化模式下的常数高度；仅在 ``enabled=False`` 时生效，
            对应平地 ground geom 的世界 z 坐标。
        enabled: 是否启用网格插值；``False`` 时高度查找直接返回
            ``constant``，忽略所有网格字段。
    """

    heights: torch.Tensor
    origin: torch.Tensor
    spacing: torch.Tensor
    z0: torch.Tensor
    constant: float
    enabled: bool


def heightfield_lookup(grid: HeightFieldGrid, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched bilinear height lookup in float64 over a height-field grid.

    ``x``/``y`` are ``(N,)`` world coordinates. ``grid`` is the batch
    context's kernel-data view of a :class:`HeightFieldGrid`; its fields
    arrive as zero-copy tensor views of the shared grid arrays.
    """
    if not grid.enabled:
        return torch.full((x.shape[0],), float(grid.constant), dtype=torch.float64)
    heights = grid.heights
    origin = grid.origin
    spacing = grid.spacing
    z0 = grid.z0
    # Grid fields are float64; mixed arithmetic promotes the float32
    # world coordinates, so the whole mapping stays in float64.
    fx = ((x - origin[0]) / spacing[0]).clamp(0.0, heights.shape[1] - 1.0 - 1e-6)
    fy = ((y - origin[1]) / spacing[1]).clamp(0.0, heights.shape[0] - 1.0 - 1e-6)
    col = fx.to(torch.long)
    row = fy.to(torch.long)
    tx = fx - col
    ty = fy - row
    h00 = heights[row, col]
    h01 = heights[row, col + 1]
    h10 = heights[row + 1, col]
    h11 = heights[row + 1, col + 1]
    top = h00 * (1.0 - tx) + h01 * tx
    bottom = h10 * (1.0 - tx) + h11 * tx
    return z0[0] + top * (1.0 - ty) + bottom * ty


__all__ = [
    "heightfield_lookup",
    "HeightFieldGrid",
]
