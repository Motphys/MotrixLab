# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Walk-task builder for the shared in-kernel terrain grid."""

import numpy as np

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
            heights=hf["heights"],
            origin=hf["origin"],
            spacing=hf["spacing"],
            z0=hf["z0"],
            constant=np.float32(0.0),
            enabled=True,
        )
    spec = ctx.model.others["geoms"][ground_geom]
    return HeightFieldGrid(
        heights=np.zeros((1, 1), dtype=np.float32),
        origin=np.zeros((2,), dtype=np.float32),
        spacing=np.ones((2,), dtype=np.float32),
        z0=np.zeros((1,), dtype=np.float32),
        constant=np.float32(spec.local_pose[2]),
        enabled=False,
    )
