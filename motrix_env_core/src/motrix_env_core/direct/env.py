# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral direct-workflow frontend environment.

``DirectEnv`` consumes the SimBackend boundary (see
``wiki/design/direct-env-sim-backend.md``): the backend is exposed as
``self.sim`` and concrete environments compile their own model/read programs
and write programs against it. No concrete simulator type crosses this module.
"""

from __future__ import annotations

from typing import TypeVar

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.array.env import ArrayEnvState as ArrayEnvState
from motrix_env_core.base import EnvCfg
from motrix_env_core.config import configclass
from motrix_env_core.sim.backend import (
    RenderConfig,
    SimBackend,
    SimRenderer,
)


@configclass
class DirectEnvCfg(EnvCfg):
    """Base configuration for direct-workflow environments."""


DirectEnvCfgType = TypeVar("DirectEnvCfgType", bound=DirectEnvCfg)


class DirectEnv(ArrayEnv[DirectEnvCfgType]):
    """Direct-workflow frontend that exposes only the simulator backend.

    Concrete environments own their model/data query declarations and compile
    those programs in their own constructors.
    """

    def __init__(self, cfg: DirectEnvCfgType, num_envs: int = 1, backend: str | None = None):
        super().__init__(cfg, num_envs)
        from motrix_env_core.sim.registry import create_sim_backend, default_sim_backend_name

        factory = create_sim_backend(backend or default_sim_backend_name())
        # Construction compiles the scene inside the backend: no compiled
        # artifact crosses the boundary.
        self.sim: SimBackend = factory(cfg.scene, cfg.sim, num_envs)

    @property
    def num_dof_pos(self) -> int:
        return self.sim.num_dof_pos

    @property
    def num_dof_vel(self) -> int:
        return self.sim.num_dof_vel

    @property
    def num_actuators(self) -> int:
        return self.sim.num_actuators

    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        return self.sim.create_renderer(
            config,
            num_envs=self.num_envs,
            render_spacing=self.render_spacing,
            system_camera=self.cfg.scene.system_camera,
        )

    def physics_step(self) -> None:
        self.sim.step(self._cfg.sim_substeps)


__all__ = ["DirectEnv", "DirectEnvCfg"]
