# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""In-kernel terrain-height lookup over a static height-field grid."""

import numpy as np
from numba import njit

from motrix_env_core.manager import SharedArray, kernel_data


@kernel_data
class HeightFieldGrid:
    """Static terrain grid for in-kernel bilinear height lookup.

    ``enabled=False`` degenerates to the flat-ground ``constant`` height.

    采样高度计算公式（见 :func:`heightfield_lookup`）::

        fx = (x - origin[0]) / spacing[0]
        fy = (y - origin[1]) / spacing[1]          # clamp 到 [0, n - 1 - eps]
        col, row = int(fx), int(fy); tx, ty = fx - col, fy - row
        top    = heights[row, col]     * (1 - tx) + heights[row, col + 1]       * tx
        bottom = heights[row + 1, col] * (1 - tx) + heights[row + 1, col + 1]   * tx
        h(x, y) = z0[0] + top * (1 - ty) + bottom * ty

    Attributes:
        heights: ``SharedArray`` of shape ``(nrow, ncol)`` holding the
            height-field sample values (row = y 方向, column = x 方向)，
            单位与引擎导出的 hfield 数据一致。
        origin: 长度为 2 的 ``SharedArray``，网格首个采样点 ``(x, y)`` 的
            世界坐标，是 bilinear 插值前的平移原点。
        spacing: 长度为 2 的 ``SharedArray``，相邻采样点在世界坐标系下的
            间距 ``(dx, dy)``，用于把世界坐标映射到分数网格坐标。
        z0: 长度为 1 的 ``SharedArray``，叠加在插值结果之上的基准高度偏移。
        constant: 平地退化模式下的常数高度；仅在 ``enabled=False`` 时生效，
            对应平地 ground geom 的世界 z 坐标。
        enabled: 是否启用网格插值；``False`` 时 ``heightfield_lookup``
            直接返回 ``constant``，忽略所有网格字段。
    """

    heights: SharedArray
    origin: SharedArray
    spacing: SharedArray
    z0: SharedArray
    constant: np.float32
    enabled: bool


@njit(inline="always")
def heightfield_lookup(grid: HeightFieldGrid, x: float, y: float) -> float:
    """Bilinear grid lookup matching the engine's hfield sample convention."""
    if not grid.enabled:
        return grid.constant
    nrow = grid.heights.shape[0]
    ncol = grid.heights.shape[1]
    fx = (x - grid.origin[0]) / grid.spacing[0]
    fy = (y - grid.origin[1]) / grid.spacing[1]
    fx = min(max(fx, 0.0), ncol - 1.0 - 1e-6)
    fy = min(max(fy, 0.0), nrow - 1.0 - 1e-6)
    col = int(fx)
    row = int(fy)
    tx = fx - col
    ty = fy - row
    h00 = grid.heights[row, col]
    h01 = grid.heights[row, col + 1]
    h10 = grid.heights[row + 1, col]
    h11 = grid.heights[row + 1, col + 1]
    top = h00 * (1.0 - tx) + h01 * tx
    bottom = h10 * (1.0 - tx) + h11 * tx
    return grid.z0[0] + top * (1.0 - ty) + bottom * ty


__all__ = [
    "HeightFieldGrid",
    "heightfield_lookup",
]
