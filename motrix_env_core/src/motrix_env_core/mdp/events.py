# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable interval events for manager environments."""

from __future__ import annotations

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.numba.kernel_data import Map
from motrix_env_core.numba.manager.context import BuildContext, ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.manager.events import IntervalEvent, IntervalEventCfg
from motrix_env_core.sim.write import AddBodyLinearVelocityWrite


@dispatch
def random_velocity_kick(ctx: ManagerContext, writes: Map[np.ndarray], bounds: np.ndarray) -> None:
    delta = writes["delta"]
    for axis in range(3):
        delta[0, axis] = ctx.rand.uniform_range(bounds[axis, 0], bounds[axis, 1])


@configclass(kw_only=True)
class RandomVelocityKickCfg(IntervalEventCfg):
    """Sample a world XYZ velocity delta for one floating body's root."""

    body: str
    velocity_delta_x: tuple[float, float] = (0.0, 0.0)
    velocity_delta_y: tuple[float, float] = (0.0, 0.0)
    velocity_delta_z: tuple[float, float] = (0.0, 0.0)

    def __call__(self, ctx: BuildContext) -> IntervalEvent:
        del ctx
        ranges = []
        for axis, delta_range in (
            ("x", self.velocity_delta_x),
            ("y", self.velocity_delta_y),
            ("z", self.velocity_delta_z),
        ):
            if len(delta_range) != 2 or not all(np.isfinite(delta_range)) or delta_range[0] > delta_range[1]:
                raise ValueError(f"velocity_delta_{axis} must be a finite (low, high) pair with low <= high.")
            ranges.append(delta_range)
        bounds = np.asarray(ranges, dtype=np.float32)
        return IntervalEvent(random_velocity_kick, bounds, writes={"delta": AddBodyLinearVelocityWrite((self.body,))})
