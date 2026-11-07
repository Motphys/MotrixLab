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
            heights=hf["heights"].contiguous(),
            origin=hf["origin"].contiguous(),
            spacing=hf["spacing"].contiguous(),
            z0=hf["z0"].contiguous(),
            constant=0.0,
            enabled=True,
        )
    spec = ctx.model.others["geoms"][ground_geom]
    return HeightFieldGrid(
        heights=torch.zeros((1, 1), dtype=torch.float32),
        origin=torch.zeros((2,), dtype=torch.float32),
        spacing=torch.ones((2,), dtype=torch.float32),
        z0=torch.zeros((1,), dtype=torch.float32),
        constant=float(spec.local_pose[2]),
        enabled=False,
    )
