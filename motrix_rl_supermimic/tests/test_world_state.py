# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned state composition preserves capture order and differentiable dynamics."""

from collections.abc import Mapping

import pytest
import torch
from motrix_rl_supermimic.networks import WorldModel, run_mlp
from motrix_rl_supermimic.wm_graph import FlattenFullLoss
from motrix_rl_supermimic.world_state import EuclideanState, StateStatistics, WorldStateManager, WorldStateTerm


class SimEuclidean(EuclideanState):
    def capture(self, sim: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
        out.copy_(sim["state"])


class ParticleCodec(WorldStateTerm):
    """Small task with passive drift and different raw/feature widths."""

    size = 2
    feature_dim = 3
    prediction_dim = 2

    def capture(self, sim: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
        out.copy_(sim["particle"])

    def features(self, state: torch.Tensor) -> torch.Tensor:
        return torch.cat((state, state[..., 1:].square()), -1)

    def delta(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return next_state - state

    def _targets(self, state: torch.Tensor, delta: torch.Tensor, structured: bool, dt: float) -> torch.Tensor:
        if not structured:
            return delta
        return torch.cat((delta[..., :1] - state[..., 1:] * dt, delta[..., 1:]), -1)

    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return self._targets(state, self.delta(state, next_state), structured, dt)

    def statistics(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> StateStatistics:
        delta = self.delta(state, next_state)
        return StateStatistics(self.features(state), self._targets(state, delta, structured, dt), delta)

    def integrate(
        self, state: torch.Tensor, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        drift = state[..., 1:] * dt if structured else torch.zeros_like(state[..., 1:])
        return torch.cat((state[..., :1] + prediction[..., :1] + drift, state[..., 1:] + prediction[..., 1:]), -1)

    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return self.delta(truth, pred)

    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        return torch.full((2,), 0.01, device=device)


def composed_state() -> WorldStateManager:
    return WorldStateManager(
        {
            "agent": SimEuclidean(2, error_scale=0.1),
            "object": SimEuclidean(3, error_scale=0.2),
        }
    )


@pytest.mark.parametrize("structured", [False, True])
def test_composed_euclidean_model_features_dynamics_and_loss_vjp(structured: bool) -> None:
    world = composed_state()
    assert world.slices == {"agent": slice(0, 2), "object": slice(2, 5)}
    assert world.size == world.feature_dim == world.prediction_dim == 5
    torch.testing.assert_close(world.error_floor("cpu"), torch.tensor([0.1, 0.1, 0.2, 0.2, 0.2]))
    model = WorldModel(world, context_dim=1, hidden=(8,), structured=structured, action_dim=2)
    state = torch.randn(3, 5, requires_grad=True)
    action, context = torch.randn(3, 2), torch.randn(3, 1)
    torch.testing.assert_close(model.inputs(state, action, context), torch.cat((state, action, context), -1))
    increment = torch.randn(5) * 0.1
    with torch.no_grad():
        model.out_norm.mean.copy_(increment)
    predicted = model.step(state, action, context)
    torch.testing.assert_close(predicted, state + increment)
    torch.testing.assert_close(model.delta(state, predicted), increment.expand_as(state))
    torch.testing.assert_close(model.targets(state, predicted), increment.expand_as(state))
    torch.testing.assert_close(torch.autograd.grad(predicted.sum(), state)[0], torch.ones_like(state))
    targets = torch.randn(3, 2, 5)
    actions, contexts = torch.randn(3, 2, 2), torch.randn(3, 2, 1)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    floor = world.error_floor("cpu")
    kernel = FlattenFullLoss(model, floor, 2, 3, normalize_context=lambda value: value)
    kernel.sync_norms(floor)
    current, predictions = state, []
    for step in range(2):
        current = model.step(current, actions[:, step], contexts[:, step])
        predictions.append(current)
    error = model.state_error(torch.stack(predictions, 1), targets)
    expected = ((error / torch.maximum(model.inc_norm.std(), floor)).square().mean(-1) * mask).sum() / 3
    actual = kernel(state, actions, contexts, targets, mask)
    torch.testing.assert_close(actual, expected)
    variables = (state, *model.parameters())
    reference_vjp = torch.autograd.grad(expected, variables)
    actual_vjp = torch.autograd.grad(actual, variables)
    for actual_grad, expected_grad in zip(actual_vjp, reference_vjp, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_duplicate_sim_view_keys_are_rejected() -> None:
    """Two terms claiming the same sim view key never produced a consistent capture."""

    class SharedKeyState(EuclideanState):
        def capture(self, sim: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
            out.copy_(sim["state"])

        def views(self, state: torch.Tensor) -> Mapping[str, torch.Tensor]:
            return {"state": state}

    combined = WorldStateManager({"first": SharedKeyState(4), "second": SharedKeyState(4)})
    with pytest.raises(ValueError, match="duplicate sim view 'state'"):
        combined.views(torch.zeros(2, 8))


@pytest.mark.parametrize("structured", [False, True])
def test_mixed_state_model_step_matches_flat_path_and_compiles(structured: bool) -> None:
    particle = ParticleCodec()
    world = WorldStateManager({"particle": particle, "object": SimEuclidean(3)})
    model = WorldModel(world, context_dim=1, hidden=(8,), structured=structured, action_dim=2)
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 0.01)
        model.out_norm.mean.normal_(0, 0.01)
    state = torch.randn(2, world.size)
    state.requires_grad_()
    action = torch.randn(2, 2, requires_grad=True)
    context = torch.randn(2, 1)
    inputs = model.inputs(state, action, context)
    prediction = run_mlp(model.net, model.in_norm(inputs)) * model.out_norm.std() + model.out_norm.mean
    expected = world.integrate(state, prediction, structured=structured, dt=model.dt)
    actual = torch.compile(model.step, backend="aot_eager", fullgraph=True)(state, action, context)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == state.shape
    weight = torch.randn_like(actual)
    variables = (state, action, *model.parameters())
    expected_gradients = torch.autograd.grad(expected, variables, weight)
    actual_gradients = torch.autograd.grad(actual, variables, weight)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("structured", [False, True])
def test_combined_world_model_statistics_preserve_normalizer_updates(structured: bool) -> None:
    import copy

    particle = ParticleCodec()
    world = WorldStateManager({"particle": particle, "object": SimEuclidean(3)})
    model = WorldModel(world, context_dim=1, hidden=(8,), structured=structured, action_dim=2)
    expected = copy.deepcopy(model)
    # Two updates protect ordered accumulation, not just one batch's moments.
    for _ in range(2):
        state = torch.randn(2, 3, world.size)
        next_state = world.integrate(state, torch.randn(2, 3, world.prediction_dim) * 0.01, structured=structured)
        action, context = torch.randn(2, 3, 2), torch.randn(2, 3, 1)
        expected.in_norm.update(expected.inputs(state, action, context))
        expected.out_norm.update(expected.targets(state, next_state))
        expected.inc_norm.update(expected.delta(state, next_state))
        model.update_norms(state, next_state, action, context)
    for name in ("in_norm", "out_norm", "inc_norm"):
        actual_buffers = dict(model.get_submodule(name).named_buffers())
        expected_buffers = dict(expected.get_submodule(name).named_buffers())
        for key, value in actual_buffers.items():
            torch.testing.assert_close(value, expected_buffers[key])
