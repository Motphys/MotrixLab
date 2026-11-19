# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned state composition preserves capture order and differentiable dynamics."""

import pytest
import torch
from motrix_rl_supermimic.networks import WorldModel
from motrix_rl_supermimic.robot_state import CanonicalRobotState
from motrix_rl_supermimic.wm_graph import FlattenFullLoss
from motrix_rl_supermimic.world_state import EuclideanState, WorldState, WorldStateTerm

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.math.rotation import so3_exp


def capture(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.sim["state"])


def composed_state() -> WorldState:
    return WorldState(
        {
            "agent": WorldStateTerm(capture, EuclideanState(2, error_scale=0.1)),
            "object": WorldStateTerm(capture, EuclideanState(3, error_scale=0.2)),
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
    torch.testing.assert_close(model.increments(state, predicted), increment.expand_as(state))
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


@pytest.mark.parametrize("structured", [False, True])
def test_robot_codec_roundtrip_and_composition_preserve_values_and_vjp(structured: bool) -> None:
    robot = CanonicalRobotState(2)
    world = WorldState({"robot": WorldStateTerm(capture, robot), "object": WorldStateTerm(capture, EuclideanState(3))})
    state = robot.pack(
        (
            torch.randn(4, 3),
            so3_exp(torch.randn(4, 3) * 0.2),
            torch.randn(4, 3),
            torch.randn(4, 3),
            torch.randn(4, 2),
            torch.randn(4, 2),
        )
    ).requires_grad_()
    prediction = (torch.randn(4, robot.prediction_dim) * 0.05).requires_grad_()
    next_state = robot.integrate(state, prediction, structured=structured, dt=0.04)
    torch.testing.assert_close(robot.targets(state, next_state, structured=structured, dt=0.04), prediction)
    torch.testing.assert_close(robot.pack(robot.unpack(state)), state)
    object_state = torch.randn(4, 3, requires_grad=True)
    object_delta = torch.randn(4, 3, requires_grad=True)
    composed = world.integrate(
        torch.cat((state, object_state), -1),
        torch.cat((prediction, object_delta), -1),
        structured=structured,
        dt=0.04,
    )
    expected = torch.cat((next_state, object_state + object_delta), -1)
    torch.testing.assert_close(composed, expected)
    torch.testing.assert_close(
        world.features(torch.cat((state, object_state), -1)), torch.cat((robot.features(state), object_state), -1)
    )
    weight = torch.randn_like(composed)
    variables = (state, prediction, object_state, object_delta)
    expected_vjp = torch.autograd.grad(expected, variables, weight, retain_graph=True)
    actual_vjp = torch.autograd.grad(composed, variables, weight)
    for actual, reference in zip(actual_vjp, expected_vjp, strict=True):
        torch.testing.assert_close(actual, reference)
