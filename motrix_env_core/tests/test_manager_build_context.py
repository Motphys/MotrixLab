# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Numerical term construction consumes static declarations, not runtime terms."""

from dataclasses import dataclass

import gymnasium as gym
import torch

from motrix_env_core.manager.actions import ActionState, ActionTerm
from motrix_env_core.manager.commands import CommandTerm
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.env import ManagerBasedEnvCfg
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.mdp.observations import ActionsObsCfg, CommandObsCfg
from motrix_env_core.sim.model import SimModel


class MutableAction(ActionTerm):
    def __init__(self, width: int) -> None:
        super().__init__(
            gym.spaces.Box(-1.0, 1.0, (width,)), ActionState(torch.zeros(1, 2, width), torch.zeros(1, dtype=torch.long))
        )
        self.offset = torch.zeros(1, width)

    def _process(self, actions: torch.Tensor) -> torch.Tensor:
        return actions + self.offset

    def _reset(self, ids: torch.Tensor) -> None:
        pass


@dataclass(frozen=True, slots=True)
class Goal(CommandTerm):
    def update(self, ctx: ManagerContext) -> None:
        pass

    def advance(self, ctx: ManagerContext) -> None:
        pass

    def reset(self, ctx: ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        pass


def test_host_action_and_command_terms_build_observation_widths_without_environment() -> None:
    cfg = ManagerBasedEnvCfg()
    ctx = BuildContext(
        cfg=cfg,
        num_envs=1,
        seed=17,
        actions=Map({"residual": MutableAction(7)}),
        commands=Map({"reference": Goal(torch.zeros(1, 11))}),
    )
    assert ActionsObsCfg(action_name="residual")(ctx).size == 7
    assert CommandObsCfg(command_name="reference")(ctx).size == 11
    assert ctx.cfg is cfg
    assert (ctx.num_envs, ctx.seed) == (1, 17)
    # Concrete task-owned host data remains available without a second schema.
    ctx.actions["residual"].offset.fill_(0.25)
    torch.testing.assert_close(ctx.actions["residual"].process(torch.zeros(1, 7)), torch.full((1, 7), 0.25))
    independent = MutableAction(7)
    torch.testing.assert_close(independent.offset, torch.zeros(1, 7))
    assert independent.state.action_queue.data_ptr() != ctx.actions["residual"].state.action_queue.data_ptr()


def test_explicit_model_is_borrowed() -> None:
    ctx = BuildContext(
        cfg=ManagerBasedEnvCfg(),
        num_envs=1,
        seed=17,
        actions=Map({}),
        commands=Map({}),
    )
    assert ctx.model is None
    model = SimModel(actuators=(), init_dof_pos=torch.zeros(0))
    resolved = BuildContext(
        cfg=ctx.cfg,
        num_envs=ctx.num_envs,
        seed=ctx.seed,
        actions=ctx.actions,
        commands=ctx.commands,
        model=model,
    )
    assert resolved.model is model
