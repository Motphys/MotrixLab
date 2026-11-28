# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import torch
from motrix_rl_appo.vtrace import compute_vtrace_targets, impact_ratio


def test_off_policy_hand_computation_retains_pg_rho() -> None:
    # target/mu = .5,2. Values [1,2], endpoint 4, gamma=.5.
    # delta=[.5,5], c=[.5,1]; v=[2.75,7].
    # A=[.5*(1+.5*7-1), 5+.5*4-2] = [1.75,5].
    target = torch.tensor([[0.5], [2.0]]).log()
    zeros = torch.zeros(2, 1)
    result = compute_vtrace_targets(
        torch.tensor([[1.0], [5.0]]),
        torch.full((2, 1), 0.5),
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([4.0]),
        target,
        zeros,
    )
    torch.testing.assert_close(result.values, torch.tensor([[2.75], [7.0]]))
    torch.testing.assert_close(result.advantages, torch.tensor([[1.75], [5.0]]))
    # At pi=target, cap2 yields R=1 for both samples, NOT rhoPG.
    torch.testing.assert_close(impact_ratio(target, zeros, target), torch.ones_like(zeros))
    extreme = torch.tensor([0.1]).log()
    torch.testing.assert_close(impact_ratio(extreme, torch.zeros(1), extreme), torch.tensor([0.2]))


def test_done_cuts_trace_but_endpoint_bootstraps_nonterminal() -> None:
    rewards = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    discounts = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    result = compute_vtrace_targets(
        rewards, discounts, torch.zeros(2, 2), torch.tensor([5.0, 7.0]), torch.zeros(2, 2), torch.zeros(2, 2)
    )
    torch.testing.assert_close(result.values, torch.tensor([[2.0, 12.0], [8.0, 10.0]]))
    assert not result.values.requires_grad


def test_policy_gradient_cap_is_independent_from_value_cap() -> None:
    result = compute_vtrace_targets(
        torch.tensor([[4.0]]),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1),
        torch.tensor([[2.0]]).log(),
        torch.zeros(1, 1),
        rho_clip=1.0,
        pg_rho_clip=0.25,
    )
    torch.testing.assert_close(result.values, torch.tensor([[4.0]]))
    torch.testing.assert_close(result.advantages, torch.tensor([[1.0]]))


def test_timeout_reward_is_already_bootstrapped_and_stops_trace() -> None:
    result = compute_vtrace_targets(
        torch.tensor([[2.0 + 0.9 * 4.0], [100.0]]),
        torch.tensor([[0.0], [0.9]]),
        torch.zeros(2, 1),
        torch.tensor([50.0]),
        torch.zeros(2, 1),
        torch.zeros(2, 1),
    )
    torch.testing.assert_close(result.values[0], torch.tensor([5.6]))
