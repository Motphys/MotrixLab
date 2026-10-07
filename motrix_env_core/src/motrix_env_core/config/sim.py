# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from math import isfinite

from motrix_env_core.config.decorate import configclass


@configclass
class SimCfg:
    """Physics simulation settings consumed by a scene compiler."""

    dt: float = 0.01
    solver_iterations: int | None = None
    solver_tolerance: float | None = None
    gravity: tuple[float, float, float] | None = None

    def validate(self) -> None:
        if isinstance(self.dt, bool) or not isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"sim.dt must be positive and finite, got {self.dt}")
        if self.solver_iterations is not None and (
            isinstance(self.solver_iterations, bool)
            or not isinstance(self.solver_iterations, int)
            or self.solver_iterations <= 0
        ):
            raise ValueError(f"sim.solver_iterations must be a positive integer, got {self.solver_iterations}")
        if self.solver_tolerance is not None and (not isfinite(self.solver_tolerance) or self.solver_tolerance <= 0.0):
            raise ValueError(f"sim.solver_tolerance must be positive and finite, got {self.solver_tolerance}")
        if self.gravity is not None and len(self.gravity) != 3:
            raise ValueError(f"sim.gravity must contain 3 values, got {self.gravity!r}")
