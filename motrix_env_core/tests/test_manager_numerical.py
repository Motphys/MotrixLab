# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Standalone numerical terms bind caller-owned batches without a ManagerEnv."""

import pytest
import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.observations import ObsTerm
from motrix_env_core.manager.random import RandomSource
from motrix_env_core.manager.resolver import ManagerTermResolver
from motrix_env_core.manager.rewards import RewardTerm
from motrix_env_core.manager.terminations import TerminationTerm
from motrix_env_core.sim.read import JointPositionQuery


def _context(values: torch.Tensor) -> ManagerContext:
    count = values.shape[0]
    return ManagerContext(
        rand=RandomSource(),
        env_id=torch.arange(count),
        actions=Map({}),
        commands=Map({}),
        metrics=Map({}),
        sim=Map({"position": values}),
        dt=0.02,
        sim_reset_requested=torch.zeros(count, dtype=torch.bool),
        terminated=torch.zeros(count, dtype=torch.bool),
    )


def _reward(ctx: ManagerContext, position: torch.Tensor, scale: float) -> torch.Tensor:
    return position.square().sum(dim=-1) * scale


def _observation(ctx: ManagerContext, out: torch.Tensor, position: torch.Tensor) -> None:
    out.copy_(position * 2)


def _soft_termination(ctx: ManagerContext, position: torch.Tensor) -> torch.Tensor:
    return position[:, 0].sigmoid()


@pytest.mark.parametrize("compiled", [False, True])
def test_resolved_terms_rebind_query_inputs_and_observation_batch_to_each_context(compiled: bool) -> None:
    query = JointPositionQuery(joints=("joint",))
    resolver = ManagerTermResolver({query: "position"})
    reward = resolver.resolve_reward("energy", 0, RewardTerm(_reward, query, 3.0))
    observation = resolver.resolve_observation("policy", "position", ObsTerm(1, _observation, query))
    invoke_reward = torch.compile(reward.invoke, backend="eager", fullgraph=True) if compiled else reward.invoke
    invoke_observation = (
        torch.compile(observation.invoke, backend="eager", fullgraph=True) if compiled else observation.invoke
    )
    first = _context(torch.tensor([[1.0], [2.0]]))
    second = _context(torch.tensor([[4.0], [5.0], [6.0]]))

    torch.testing.assert_close(invoke_reward(first), torch.tensor([3.0, 12.0]))
    torch.testing.assert_close(invoke_reward(second), torch.tensor([48.0, 75.0, 108.0]))
    first_out = torch.empty(2, 1)
    invoke_observation(first, first_out)
    torch.testing.assert_close(first_out, torch.tensor([[2.0], [4.0]]))
    second_out = torch.empty(3, 1)
    invoke_observation(second, second_out)
    torch.testing.assert_close(second_out, torch.tensor([[8.0], [10.0], [12.0]]))


def test_standalone_soft_termination_preserves_probabilities_and_gradients() -> None:
    query = JointPositionQuery(joints=("joint",))
    call = ManagerTermResolver({query: "position"}).resolve_termination(
        "failure", 0, TerminationTerm(_soft_termination, query)
    )
    position = torch.tensor([[0.0], [0.0]], requires_grad=True)
    probability = call.invoke(_context(position))

    torch.testing.assert_close(probability, torch.full((2,), 0.5))
    probability.sum().backward()
    torch.testing.assert_close(position.grad, torch.full((2, 1), 0.25))


def test_term_resolver_rejects_an_undeclared_query() -> None:
    query = JointPositionQuery(joints=("joint",))
    with pytest.raises(ValueError, match="no declared input binding"):
        ManagerTermResolver().resolve_reward("energy", 0, RewardTerm(_reward, query, 3.0))


def test_term_resolver_rejects_missing_context_annotation() -> None:
    def untyped(ctx) -> torch.Tensor:
        return torch.ones(1)

    with pytest.raises(TypeError, match="annotate its first parameter as ManagerContext"):
        ManagerTermResolver().resolve_termination("failure", 0, TerminationTerm(untyped))


def test_term_resolver_rejects_wrong_observation_output_annotation() -> None:
    def wrong_output(ctx: ManagerContext, out: float) -> None:
        pass

    with pytest.raises(TypeError, match="first two parameters"):
        ManagerTermResolver().resolve_observation("policy", "position", ObsTerm(1, wrong_output))


def test_term_resolver_rejects_missing_dispatch_argument() -> None:
    with pytest.raises(TypeError, match="positional args"):
        ManagerTermResolver().resolve_reward("energy", 0, RewardTerm(_reward, 3.0))
