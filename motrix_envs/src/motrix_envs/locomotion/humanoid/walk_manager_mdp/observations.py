# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the humanoid velocity-tracking task."""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.numba.manager.context import BuildContext, ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.manager.observations import ObservationTermCfg, ObsTerm
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommand


@dispatch
def gait_phase_obs(ctx: ManagerContext, out: np.ndarray, offset: np.int64, size: np.int64) -> None:
    walk: WalkCommand = ctx.commands["walk"]
    for index in range(size):
        out[index] = walk.sin_cos[offset + index]


@configclass(kw_only=True)
class GaitPhaseObsCfg(ObservationTermCfg):
    """``sin``/``cos`` slice of the gait-phase clock.

    The command term's ``sin_cos`` lane layout is
    ``[sin_l, sin_r, cos_l, cos_r]``: offset 0 gives the two sin values and
    offset 2 the two cos values, matching the direct env's obs order.
    """

    offset: int = 0
    size: int = 2

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        return ObsTerm(self.size, gait_phase_obs, np.int64(self.offset), np.int64(self.size))
