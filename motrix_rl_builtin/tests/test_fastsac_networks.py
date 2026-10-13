# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC actor output and gradient contracts."""

import pytest
import torch
from motrix_rl_builtin.fastsac.networks import Actor


@pytest.mark.parametrize("use_tanh", [True, False])
def test_actor_preserves_gaussian_mean_and_its_gradient(use_tanh):
    """Regression: a hard mean clamp must not kill gradients outside its bounds."""
    actor = Actor(2, 2, 16, 2.0, -5.0, use_tanh=use_tanh)
    expected_mean = torch.tensor([4.0, -4.0])
    with torch.no_grad():
        actor.fc_mu.weight.zero_()
        actor.fc_mu.bias.copy_(expected_mean)

    action, mean, _ = actor(torch.zeros(3, 2))
    torch.testing.assert_close(mean, expected_mean.expand(3, 2))
    expected_action = expected_mean.tanh() if use_tanh else expected_mean
    torch.testing.assert_close(action, expected_action.expand(3, 2))

    mean.sum().backward()
    torch.testing.assert_close(actor.fc_mu.bias.grad, torch.full((2,), 3.0))
