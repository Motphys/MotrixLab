# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Walk-task builder for the shared in-kernel terrain grid."""

import torch

from motrix_env_core.mdp.terrain import HeightFieldGrid


def ground_height_grid(ctx, ground_geom: str) -> HeightFieldGrid:
    """Build the terrain grid from the build context's compiled model queries.

    Rough presets read the exported height-field grid; flat presets degenerate
    to the ground geom's constant world z.
    """
    cfg = ctx.cfg
    if cfg.ground_heightfield_geom is not None:
        hf = ctx.model.others["ground_heightfield"]
        return HeightFieldGrid(
            heights=hf["heights"].to(torch.float64).contiguous(),
            origin=hf["origin"].to(torch.float64).contiguous(),
            spacing=hf["spacing"].to(torch.float64).contiguous(),
            z0=hf["z0"].to(torch.float64).contiguous(),
            constant=0.0,
            enabled=True,
        )
    spec = ctx.model.others["geoms"][ground_geom]
    return HeightFieldGrid(
        heights=torch.zeros((1, 1), dtype=torch.float64),
        origin=torch.zeros((2,), dtype=torch.float64),
        spacing=torch.ones((2,), dtype=torch.float64),
        z0=torch.zeros((1,), dtype=torch.float64),
        constant=float(spec.local_pose[2]),
        enabled=False,
    )
