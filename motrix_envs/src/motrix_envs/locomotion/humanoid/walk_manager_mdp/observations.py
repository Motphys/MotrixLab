# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the humanoid velocity-tracking task."""

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.observations import ObservationTermCfg, ObsTerm


def gait_phase_obs(ctx: ManagerContext, out: torch.Tensor, offset: np.int64, size: np.int64) -> None:
    walk = ctx.commands["walk"]
    out.copy_(walk.sin_cos[:, offset : offset + size])


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
