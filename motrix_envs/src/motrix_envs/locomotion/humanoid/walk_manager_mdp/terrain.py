# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Walk-task builder for the shared in-kernel terrain grid."""

import torch

from motrix_env_core.mdp.terrain import HeightFieldGrid
from motrix_env_core.sim import SimModel


def ground_height_grid(model: SimModel, ground_geom: str) -> HeightFieldGrid:
    """Return the terrain grid shared by all terms.

    The height-field query registers its export only when the floor is an
    HField terrain; the backend builds that grid (float64 storage) once and
    every term shares the same immutable record. Flat presets degenerate to
    ``ground_geom``'s constant world z.
    """
    grid = model.others.get("ground_heightfield")
    if grid is not None:
        return grid
    spec = model.others["geoms"][ground_geom]
    return HeightFieldGrid(
        heights=torch.zeros((1, 1), dtype=torch.float64),
        origin=torch.zeros((2,), dtype=torch.float64),
        spacing=torch.ones((2,), dtype=torch.float64),
        z0=torch.zeros((1,), dtype=torch.float64),
        constant=float(spec.local_pose[2]),
        enabled=False,
    )
