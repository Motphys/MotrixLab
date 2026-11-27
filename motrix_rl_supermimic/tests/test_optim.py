# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Optimizer selection must preserve the requested algorithm on CPU."""

import copy

import pytest
import torch
from motrix_rl_supermimic.learner import _make_optimizer
from motrix_rl_supermimic.optim import MuonAdam
from torch import nn


def _mlp() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 2))


@pytest.mark.parametrize("name", ["adam", "muon"])
def test_selected_optimizer_updates_parameters_and_restores_state(name: str) -> None:
    model = _mlp()
    optimizer = _make_optimizer(name, model, [model], lr=1e-3, device="cpu")
    if name == "muon":
        assert isinstance(optimizer, MuonAdam)
        assert isinstance(optimizer.muon, torch.optim.Muon)
    else:
        assert isinstance(optimizer, torch.optim.Adam)

    before = [parameter.detach().clone() for parameter in model.parameters()]
    inputs = torch.randn(3, 4)
    model(inputs).square().mean().backward()
    optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

    restored_model = copy.deepcopy(model)
    restored = _make_optimizer(name, restored_model, [restored_model], lr=1e-3, device="cpu")
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for current_model, current_optimizer in ((model, optimizer), (restored_model, restored)):
        current_optimizer.zero_grad(set_to_none=True)
        current_model(inputs).square().mean().backward()
        current_optimizer.step()
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_unknown_optimizer_is_rejected() -> None:
    model = _mlp()
    with pytest.raises(ValueError, match="Unknown optimizer"):
        _make_optimizer("muonn", model, [model], lr=1e-3)
