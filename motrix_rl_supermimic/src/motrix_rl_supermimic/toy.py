# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Toy single-joint task: the provider's demo and independent-acceptance env.

A damped joint tracking a cyclic per-frame target on a minimal in-memory
sim backend. It exists so the provider builds, tests, and trains without
any task package or robot constants; the TaskMDP heads are deliberately
trivial. Registered as ``supermimic-toy``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg
from motrix_env_core.sim.backend import SimBackend
from motrix_env_core.sim.model import ModelQuery, SimModel
from motrix_env_core.sim.read import PhysicsReadProgram
from motrix_env_core.sim.write import WriteProgram
from motrix_rl_supermimic.mbrl_env import MbrlEnvBase
from motrix_rl_supermimic.task_mdp import StateSpec

TARGETS = (0.0, 0.5, -0.5, 1.0)

SPEC = StateSpec(
    state_dim=15,  # canonical (p, quat, v, w, q, qd) with one joint
    action_dim=1,
    context_dim=0,
    observation_dim=4,
    critic_observation_dim=3,
    command_dim=1,
    state_groups=((3, 0.003), (4, 0.005), (3, 0.02), (3, 0.05), (1, 0.003), (1, 0.05)),
)


class ToyTaskMDP:
    """Single-joint tracking heads; frame indexes ``TARGETS`` cyclically."""

    spec = SPEC

    def observation(self, state, prev_action, context, frame):
        theta, theta_dot = state[..., 13:14], state[..., 14:15]
        return torch.cat((torch.sin(theta), torch.cos(theta), theta_dot, prev_action), -1)

    def add_observation_noise(self, observation):
        return observation + torch.randn_like(observation) * 1e-4

    def reward(self, state, frame, action, prev_action):
        theta = state[..., 13]
        target = torch.tensor(TARGETS, dtype=state.dtype, device=state.device)[frame.to(state.device) % len(TARGETS)]
        rate = ((action - prev_action) ** 2).sum(-1)
        return torch.exp(-((target - theta) ** 2) / 0.25) - 0.1 * rate

    def soft_termination(self, state, frame):
        return torch.sigmoid(state[..., 13].abs() - 1.5)

    def critic_observation(self, state, frame, context):
        theta = state[..., 13:14]
        target = torch.tensor(TARGETS, dtype=state.dtype, device=state.device)[frame.to(state.device) % len(TARGETS)][
            ..., None
        ]
        return torch.cat((state[..., 13:], target - theta), -1)

    def command(self, frame, horizon):
        target = torch.tensor(TARGETS, dtype=torch.float32, device=frame.device)[frame % len(TARGETS)]
        return target[..., None, None].expand(*frame.shape, horizon, 1)

    def normalize_context(self, context):
        return context


class _NullProgram:
    def execute(self, *_: Any, **__: Any) -> None:  # pragma: no cover - wiring only
        pass


class _ToyBackend(SimBackend):
    name = "supermimic-toy-sim"

    def __init__(self, scene: SceneCfg, sim: SimCfg, num_envs: int) -> None:
        super().__init__(scene, sim, num_envs)
        self.theta = np.zeros(num_envs, dtype=np.float32)
        self.theta_dot = np.zeros(num_envs, dtype=np.float32)
        self.reset_rows: list[np.ndarray] = []

    @property
    def num_dof_pos(self) -> int:
        return 1

    @property
    def num_dof_vel(self) -> int:
        return 1

    @property
    def num_actuators(self) -> int:
        return 1

    def step(self, substeps: int) -> None:
        self.theta += self.theta_dot * np.float32(substeps)

    @property
    def model_compiler(self):
        return self

    def compile(self, scene: SceneCfg, queries: Mapping[str, ModelQuery]) -> SimModel:  # pragma: no cover
        return SimModel(actuators=(), init_dof_pos=torch.zeros(0))

    def compile_reads(self, queries) -> PhysicsReadProgram:
        return _NullProgram()  # type: ignore[return-value]

    @property
    def write_compiler(self):
        return self

    def compile_writes(self, *_: Any, **__: Any) -> WriteProgram:
        return _NullProgram()  # type: ignore[return-value]


@configclass
class ToyEnvCfg(DirectEnvCfg):
    scene: SceneCfg = SceneCfg()


class ToyEnv(MbrlEnvBase):
    """Velocity-controlled joint: action -> theta_dot, frame advances each step."""

    @classmethod
    def build_task(cls, cfg, device):
        return ToyTaskMDP()

    def __init__(
        self,
        cfg: ToyEnvCfg | None = None,
        num_envs: int = 1,
        backend: str | None = None,
        compile_task: bool = False,
    ):
        super().__init__(
            cfg or ToyEnvCfg(),
            ToyTaskMDP(),
            num_envs,
            backend or "supermimic-toy-sim",
            compile_task=compile_task,
        )

    def state(self) -> torch.Tensor:
        state = torch.zeros(self.num_envs, 15)
        state[:, 13] = torch.from_numpy(self.sim.theta)
        state[:, 14] = torch.from_numpy(self.sim.theta_dot)
        return state

    def _write_action(self, actions: torch.Tensor) -> None:
        self.sim.theta_dot = actions[:, 0].numpy().copy()

    def _refresh_sim_state(self) -> None:
        self.frame = (self.frame + 1) % len(TARGETS)

    def _reset_rows(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.numpy()
        self.sim.reset_rows.append(ids.copy())
        self.sim.theta[ids] = 0.0
        self.sim.theta_dot[ids] = 0.0
        self.frame[ids] = 0

    def _termination(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., 13].abs() > 2.0


_registered = False


def register() -> None:
    """Register the toy backend and environment exactly once."""
    global _registered
    if _registered:
        return
    _registered = True
    from motrix_env_core.sim.registry import register_sim_backend

    register_sim_backend("supermimic-toy-sim", lambda: _ToyBackend)
    registry.envcfg("supermimic-toy")(ToyEnvCfg)
    registry.env("supermimic-toy")(ToyEnv)
