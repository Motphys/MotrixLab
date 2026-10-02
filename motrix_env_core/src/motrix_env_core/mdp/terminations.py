# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable termination terms for manager-based environments."""

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, TerminationTerm, TerminationTermCfg
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim import BodyJointVelocityQuery, GeomPairCollidingQuery


@dispatch
def colliding_termination(ctx: ManagerContext, colliding: np.ndarray) -> bool:
    return bool(colliding.any())


@dispatch
def bad_dof_velocity_termination(ctx: ManagerContext, dof_vel: np.ndarray, threshold: np.float32) -> bool:
    error = 0.0
    finite = True
    for joint_id in range(dof_vel.shape[0]):
        velocity = dof_vel[joint_id]
        if math.isfinite(velocity):
            error = max(error, abs(velocity))
        else:
            finite = False
            error = math.inf
    ctx.metrics["dof_vel_abs_max"][0] = error
    return (not finite) or error > threshold


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


@configclass(kw_only=True)
class BadDofVelocityTerminationCfg(TerminationTermCfg):
    """Terminate when one body's joint speed magnitude exceeds ``threshold``.

    Training hygiene for harsh-contact tasks: a physics near-blowup leaves the
    lane with finite-but-absurd joint speeds; terminating resets the lane so
    those states do not keep feeding the learner. The threshold must stay far
    above any healthy gait speed.
    """

    body: str = "robot"
    threshold: float = 100.0

    def __call__(self, ctx: BuildContext) -> TerminationTerm:
        if self.threshold <= 0.0:
            raise ValueError(f"BadDofVelocityTerminationCfg.threshold must be positive, got {self.threshold!r}")
        body = ctx.model.bodies[self.body]
        return TerminationTerm(
            bad_dof_velocity_termination,
            BodyJointVelocityQuery(body=body.base_link_name),
            np.float32(self.threshold),
            metric_names=("dof_vel_abs_max",),
        )


__all__ = [
    "BadDofVelocityTerminationCfg",
    "CollidingTerminationCfg",
    "bad_dof_velocity_termination",
    "colliding_termination",
]
