# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned state composition preserves capture order and differentiable dynamics."""

import pytest
import torch
from motrix_rl_supermimic.networks import WorldModel, run_mlp
from motrix_rl_supermimic.robot_state import CanonicalRobotState, RobotState
from motrix_rl_supermimic.wm_graph import FlattenFullLoss
from motrix_rl_supermimic.world_state import EuclideanState, WorldState

from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.math.rotation import so3_exp


class SimEuclidean(EuclideanState):
    def capture(self, sim: Map[torch.Tensor], out: torch.Tensor) -> None:
        out.copy_(sim["state"])


class SimRobot(CanonicalRobotState):
    def capture(self, sim: Map[torch.Tensor], out: torch.Tensor) -> None:
        out.copy_(sim["state"])


def composed_state() -> WorldState:
    return WorldState(
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
def test_named_robot_state_compiles_with_dynamics_values_and_gradients(structured: bool) -> None:
    robot = SimRobot(2)
    state = robot.pack(
        RobotState(
            position=torch.randn(2, 3),
            rotation=so3_exp(torch.randn(2, 3) * 0.2),
            linear_velocity=torch.randn(2, 3),
            angular_velocity=torch.randn(2, 3),
            joint_position=torch.randn(2, 2),
            joint_velocity=torch.randn(2, 2),
        )
    ).requires_grad_()
    prediction = (torch.randn(2, robot.prediction_dim) * 0.05).requires_grad_()

    def evaluate(state: torch.Tensor, prediction: torch.Tensor) -> tuple[RobotState, torch.Tensor]:
        parts = robot.unpack(state)
        next_state = robot.integrate(robot.pack(parts), prediction, structured=structured)
        return parts, next_state

    expected_parts, expected = evaluate(state, prediction)
    actual_parts, actual = torch.compile(evaluate, backend="aot_eager", fullgraph=True)(state, prediction)
    assert isinstance(actual_parts, RobotState)
    torch.testing.assert_close(actual_parts.position, expected_parts.position)
    torch.testing.assert_close(actual_parts.rotation, expected_parts.rotation)
    torch.testing.assert_close(actual_parts.linear_velocity, expected_parts.linear_velocity)
    torch.testing.assert_close(actual_parts.angular_velocity, expected_parts.angular_velocity)
    torch.testing.assert_close(actual_parts.joint_position, expected_parts.joint_position)
    torch.testing.assert_close(actual_parts.joint_velocity, expected_parts.joint_velocity)
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(actual)
    expected_vjp = torch.autograd.grad(expected, (state, prediction), weight)
    actual_vjp = torch.autograd.grad(actual, (state, prediction), weight)
    for actual_gradient, expected_gradient in zip(actual_vjp, expected_vjp, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("structured", [False, True])
def test_robot_codec_roundtrip_and_composition_preserve_values_and_vjp(structured: bool) -> None:
    robot = SimRobot(2)
    world = WorldState({"robot": robot, "object": SimEuclidean(3)})
    state = robot.pack(
        RobotState(
            position=torch.randn(4, 3),
            rotation=so3_exp(torch.randn(4, 3) * 0.2),
            linear_velocity=torch.randn(4, 3),
            angular_velocity=torch.randn(4, 3),
            joint_position=torch.randn(4, 2),
            joint_velocity=torch.randn(4, 2),
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


@pytest.mark.parametrize("structured", [False, True])
def test_prepared_mixed_state_model_step_matches_flat_path_and_compiles(structured: bool) -> None:
    robot = SimRobot(2)
    world = WorldState({"robot": robot, "object": SimEuclidean(3)})
    model = WorldModel(world, context_dim=1, hidden=(8,), structured=structured, action_dim=2)
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 0.01)
        model.out_norm.mean.normal_(0, 0.01)
    state = torch.randn(2, world.size)
    state[:, 3:7] = torch.tensor([0.1, 0.2, 0.3, 0.9])
    state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
    state.requires_grad_()
    action = torch.randn(2, 2, requires_grad=True)
    context = torch.randn(2, 1)
    inputs = model.inputs(state, action, context)
    prediction = run_mlp(model.net, model.in_norm(inputs)) * model.out_norm.std() + model.out_norm.mean
    expected = world.integrate(state, prediction, structured=structured, dt=model.dt)
    actual = torch.compile(model.step, backend="aot_eager", fullgraph=True)(state, action, context)
    torch.testing.assert_close(actual, expected)
    # Output remains a canonical flat snapshot, including a unit root quaternion.
    torch.testing.assert_close(actual[:, 3:7].norm(dim=-1), torch.ones(2))
    weight = torch.randn_like(actual)
    variables = (state, action, *model.parameters())
    expected_gradients = torch.autograd.grad(expected, variables, weight)
    actual_gradients = torch.autograd.grad(actual, variables, weight)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("structured", [False, True])
def test_combined_world_model_statistics_preserve_normalizer_updates(structured: bool) -> None:
    import copy

    robot = SimRobot(2)
    world = WorldState({"robot": robot, "object": SimEuclidean(3)})
    model = WorldModel(world, context_dim=1, hidden=(8,), structured=structured, action_dim=2)
    expected = copy.deepcopy(model)
    # Two updates protect ordered accumulation, not just one batch's moments.
    for _ in range(2):
        state = torch.randn(2, 3, world.size)
        state[..., 3:7] = torch.tensor([0.1, 0.2, 0.3, 0.9])
        state[..., 3:7] /= state[..., 3:7].norm(dim=-1, keepdim=True)
        next_state = world.integrate(state, torch.randn(2, 3, world.prediction_dim) * 0.01, structured=structured)
        action, context = torch.randn(2, 3, 2), torch.randn(2, 3, 1)
        expected.in_norm.update(expected.inputs(state, action, context))
        expected.out_norm.update(expected.targets(state, next_state))
        expected.inc_norm.update(expected.increments(state, next_state))
        model.update_norms(state, next_state, action, context)
    for name in ("in_norm", "out_norm", "inc_norm"):
        actual_buffers = dict(model.get_submodule(name).named_buffers())
        expected_buffers = dict(expected.get_submodule(name).named_buffers())
        for key, value in actual_buffers.items():
            torch.testing.assert_close(value, expected_buffers[key])
