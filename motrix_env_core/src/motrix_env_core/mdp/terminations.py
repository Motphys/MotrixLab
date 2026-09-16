# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable termination terms for manager-based environments."""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, TerminationTerm, TerminationTermCfg
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim import GeomPairCollidingQuery


@dispatch
def colliding_termination(ctx: ManagerContext, colliding: np.ndarray) -> bool:
    return bool(colliding.any())


@configclass(kw_only=True)
class CollidingTerminationCfg(TerminationTermCfg):
    """Terminate when any of ``termination_geoms`` contacts ``ground_geom``.

    The collision query is passed as an argument; the dispatch receives the
    query's lane view (one bool per declared geom pair).
    """

    termination_geoms: tuple[str, ...] = ()
    ground_geom: str = ""

    def __call__(self, ctx: BuildContext) -> TerminationTerm:
        if not self.termination_geoms or not self.ground_geom:
            raise ValueError("CollidingTerminationCfg requires non-empty termination_geoms and ground_geom.")
        query = GeomPairCollidingQuery(pairs=tuple((name, self.ground_geom) for name in self.termination_geoms))
        return TerminationTerm(colliding_termination, query)


__all__ = [
    "CollidingTerminationCfg",
    "colliding_termination",
]
