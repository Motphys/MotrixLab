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

import torch

from motrix_env_core.array.env import ArrayEnv, EnvObs
from motrix_env_core.array.env import ArrayEnvState as ArrayEnvState
from motrix_env_core.base import EnvCfg
from motrix_env_core.config import configclass
from motrix_env_core.sim.backend import (
    RecordConfig,
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

    def __init__(
        self,
        cfg: DirectEnvCfgType,
        num_envs: int = 1,
        backend: str | None = None,
        *,
        headless: bool = False,
        record: RecordConfig | None = None,
    ):
        super().__init__(cfg, num_envs)
        from motrix_env_core.sim.registry import create_sim_backend, default_sim_backend_name

        factory = create_sim_backend(backend or default_sim_backend_name())
        if not factory.capabilities.env_creation:
            raise ValueError(
                f"Backend {factory.name!r} cannot create a DirectEnv or ManagerEnv. "
                f"{factory.capabilities.env_creation_unavailable_reason}"
            )
        # Construction compiles the scene inside the backend: no compiled
        # artifact crosses the boundary.
        render = RenderConfig(headless=headless, record=record)
        self.sim: SimBackend = factory(cfg.scene, cfg.sim, num_envs, render=render)
        self._reward = torch.zeros(num_envs)
        self._terminated = torch.zeros(num_envs, dtype=torch.bool)
        self._obs: EnvObs | None = None
        self._reward_terms: dict[str, torch.Tensor] = {}

    def _init_owned_state(self) -> None:
        self._obs_buffers = (self._zeros_obs(self.observation_spec), self._zeros_obs(self.observation_spec))
        self._obs_index = 0
        self._obs = self._obs_buffers[self._obs_index]
        self._reward.zero_()
        self._terminated.zero_()
        self._metrics.clear()
        self._reward_terms.clear()

    def _prev_physics_step(self) -> None:
        super()._prev_physics_step()
        # Metrics and reward terms are per-step outputs like reward and
        # termination: start every step from empty owner dicts so partial
        # task writes never leak the previous step's values. Tasks write
        # both through the projected state (``state.metrics.update(...)``,
        # ``state.reward_terms.update(...)``) without a manual clear. (The
        # manager frontend instead projects fixed-key views over term-owned
        # buffers that terms overwrite in place, so it clears nothing here.)
        self._metrics.clear()
        self._reward_terms.clear()

    def _pre_compute_observation(self) -> None:
        # Retain the starting observation through step return for collectors,
        # including same-device CPU views that do not copy the tensor.
        self._obs_index = 1 - self._obs_index
        self._obs = self._obs_buffers[self._obs_index]

    @property
    def state(self) -> ArrayEnvState:
        """Project the public state over the direct frontend's owned tensors."""
        return ArrayEnvState(
            obs=self._obs,
            reward=self._reward,
            terminated=self._terminated,
            truncated=self._truncated,
            episode_steps=self._episode_steps,
            reward_terms=self._reward_terms,
            metrics=self._metrics,
            final_obs=self._final_obs,
        )

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
        )

    def physics_step(self) -> None:
        self.sim.step(self._cfg.sim_substeps, result=self.engine_error_codes)


__all__ = ["DirectEnv", "DirectEnvCfg"]
