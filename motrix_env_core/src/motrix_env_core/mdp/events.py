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
    velocity_delta_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    def __call__(self, ctx: BuildContext) -> IntervalEvent:
        del ctx
        bounds = np.asarray(self.velocity_delta_range, dtype=np.float32)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Velocity delta ranges must be three finite (low, high) pairs with low <= high.")
        return IntervalEvent(random_velocity_kick, bounds, writes={"delta": AddBodyLinearVelocityWrite((self.body,))})
