# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Imagination selects existing Manager numerical calls, not lifecycle hooks."""

from dataclasses import dataclass, replace

import pytest
import torch
from gymnasium.spaces import Box
from motrix_rl_supermimic.imagination import ImaginationBindings, ImaginationInputs, ImaginationRuntime, ImaginationTerm
from motrix_rl_supermimic.manager_task import ManagerTaskMDP, resolve_manager_task
from motrix_rl_supermimic.task_mdp import StateSpec
from motrix_rl_supermimic.world_context import WorldContext, WorldContextTerm
from motrix_rl_supermimic.world_state import EuclideanState, WorldState, WorldStateTerm

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.observations import ObsTerm
from motrix_env_core.manager.rewards import RewardTerm
from motrix_env_core.manager.terminations import TerminationTerm
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec

SPEC = StateSpec(state_dim=4, context_dim=1)


@dataclass(frozen=True, slots=True)
class Reference:
    frame: torch.Tensor
    context: torch.Tensor
    curriculum: torch.Tensor

    def update(self, ctx: ManagerContext) -> None:
        raise AssertionError("imagination executed command update")

    def advance(self, ctx: ManagerContext) -> None:
        raise AssertionError("imagination executed command advance")

    def reset(self, ctx: ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        raise AssertionError("imagination executed reset")


def capture_context(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.commands["reference"].context)


def capture_state(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.sim["state"])


def policy(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(torch.cat((ctx.sim["state"][:, :1], ctx.actions["action"].current()), -1))


def value(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.sim["state"][:, :1] + ctx.commands["reference"].context)


def reward(ctx: ManagerContext) -> torch.Tensor:
    action = ctx.actions["action"]
    return ctx.sim["state"].square().sum(-1) - (action.current() - action.previous()).square().sum(-1)


def soft(ctx: ManagerContext) -> torch.Tensor:
    return ctx.sim["state"][:, 0].sigmoid()


def goal(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.commands["reference"].frame[:, None].expand(-1, 3))


class FixtureImagination(ImaginationTerm):
    def bind(self, inputs: ImaginationInputs) -> ImaginationBindings:
        command = Reference(inputs.frame, inputs.context["reference"], torch.ones(3))
        return ImaginationBindings(commands=Map({"reference": command}), sim=Map({"state": inputs.state["state"]}))

    def bind_reference(self, frame: torch.Tensor) -> ImaginationBindings:
        command = Reference(frame, torch.empty(frame.numel(), 0), torch.ones(3))
        return ImaginationBindings(commands=Map({"reference": command}), sim=Map({}))

    def valid_starts(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        return torch.ones_like(frame, dtype=torch.bool)


def make_task(imagination: ImaginationTerm | None = None) -> ManagerTaskMDP:
    return resolve_manager_task(
        SPEC,
        0.02,
        3.0,
        FixtureImagination() if imagination is None else imagination,
        ObsTerm(2, policy),
        ObsTerm(1, value),
        RewardTerm(reward),
        TerminationTerm(soft),
        world_state=WorldState({"state": WorldStateTerm(capture_state, EuclideanState(SPEC.state_dim))}),
        action_space=Box(-1.0, 1.0, (1,)),
        action_slices={"action": slice(0, 1)},
        reference_action=lambda: None,
        goal=ObsTerm(3, goal),
        goal_frames=3,
        world_context=WorldContext({"reference": WorldContextTerm(1, capture_context, (0.0,), (1.0,))}),
        observation_spec=ObservationSpec(
            groups={
                "policy": ObservationGroupSpec(Box(-float("inf"), float("inf"), (2,))),
                "value": ObservationGroupSpec(Box(-float("inf"), float("inf"), (1,))),
                "goal": ObservationGroupSpec(Box(-float("inf"), float("inf"), (3,))),
            },
        ),
    )


def test_runtime_slices_named_physical_inputs_and_preserves_all_gradients() -> None:
    class NamedImagination(FixtureImagination):
        def bind(self, inputs: ImaginationInputs) -> ImaginationBindings:
            assert inputs.frame.shape == (6,)
            assert tuple(inputs.state) == ("agent", "object")
            assert tuple(inputs.context) == ("terrain", "goal")
            assert tuple(inputs.actions) == ("legs", "arms")
            return ImaginationBindings(
                commands=Map({}),
                sim=Map(
                    {
                        "agent": inputs.state["agent"],
                        "object": inputs.state["object"],
                        "terrain": inputs.context["terrain"],
                        "goal": inputs.context["goal"],
                    }
                ),
            )

    world_state = WorldState(
        {
            "agent": WorldStateTerm(capture_state, EuclideanState(2)),
            "object": WorldStateTerm(capture_state, EuclideanState(3)),
        }
    )
    world_context = WorldContext(
        {
            "terrain": WorldContextTerm(1, capture_context, (0.0,), (1.0,)),
            "goal": WorldContextTerm(2, capture_context, (1.0, 2.0), (2.0, 3.0)),
        }
    )
    runtime = ImaginationRuntime(
        NamedImagination(), world_state, world_context, {"legs": slice(0, 2), "arms": slice(2, 3)}, 0.06
    )
    state = torch.randn(2, 3, 5, requires_grad=True)
    context = torch.randn(2, 3, 3, requires_grad=True)
    action = torch.randn(2, 3, 3, requires_grad=True)
    previous = torch.randn(2, 3, 3, requires_grad=True)
    ctx = runtime.bind(torch.arange(6).reshape(2, 3), state=state, context=context, action=action, previous=previous)
    torch.testing.assert_close(ctx.sim["agent"], state.reshape(6, 5)[:, :2])
    torch.testing.assert_close(ctx.sim["object"], state.reshape(6, 5)[:, 2:])
    torch.testing.assert_close(ctx.sim["terrain"], context.reshape(6, 3)[:, :1])
    torch.testing.assert_close(ctx.sim["goal"], context.reshape(6, 3)[:, 1:])
    assert ctx.sim["agent"].data_ptr() == state.data_ptr()
    assert ctx.sim["object"].data_ptr() == state.reshape(6, 5)[:, 2:].data_ptr()
    assert ctx.sim["terrain"].data_ptr() == context.data_ptr()
    assert ctx.sim["goal"].data_ptr() == context.reshape(6, 3)[:, 1:].data_ptr()
    for name, part in runtime.action_slices.items():
        torch.testing.assert_close(ctx.actions[name].current(), action.reshape(6, 3)[:, part])
        torch.testing.assert_close(ctx.actions[name].previous(), previous.reshape(6, 3)[:, part])
    loss = sum(value.sum() for value in ctx.sim.values())
    loss += sum((term.current() * 2 + term.previous() * 3).sum() for term in ctx.actions.values())
    gradients = torch.autograd.grad(loss, (state, context, action, previous))
    for gradient, variable, scale in zip(gradients, (state, context, action, previous), (1, 1, 2, 3), strict=True):
        torch.testing.assert_close(gradient, torch.full_like(variable, scale))
    assert ctx.dt == 0.06
    assert len(ctx.metrics) == 0
    assert not ctx.terminated.any() and not ctx.sim_reset_requested.any()


def test_configured_observation_noise_is_shared_once_by_real_and_imagined_dispatch() -> None:
    from motrix_env_core.manager.resolver import ManagerTermResolver

    mdp = make_task()
    state, frame = torch.randn(3, SPEC.state_dim), torch.arange(3)
    previous, context = torch.ones(3, 1), torch.zeros(3, 1)
    ctx = make_task().binding.bind(frame, state=state, action=previous, previous=previous, context=context)
    clean = mdp.observation(mdp.binding.bind(frame, state=state, action=previous, previous=previous, context=context))
    rng = torch.random.get_rng_state()

    def noisy_policy(ctx: ManagerContext, out: torch.Tensor) -> None:
        policy(ctx, out)
        out.add_(2)

    call = ManagerTermResolver().resolve_observation("policy", "noisy", ObsTerm(2, noisy_policy))
    out = torch.empty_like(clean)
    call.invoke(ctx, out)
    torch.testing.assert_close(out, clean + 2)
    mdp = replace(mdp, policy_call=call)
    predicted = state.clone().requires_grad_(True)
    torch.testing.assert_close(
        mdp.observation(mdp.binding.bind(frame, state=predicted, action=previous, previous=previous, context=context)),
        out,
    )
    gradient = torch.autograd.grad(
        mdp.observation(
            mdp.binding.bind(frame, state=predicted, action=previous, previous=previous, context=context)
        ).sum(),
        predicted,
    )[0]
    expected_gradient = torch.zeros_like(predicted)
    expected_gradient[:, 0] = 1
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


def test_resolved_numerical_calls_share_real_dispatch_and_reward_units() -> None:
    mdp = make_task()
    assert mdp.policy_call.function is policy
    assert mdp.reward_call.function is reward
    assert mdp.value_call.function is value
    assert mdp.soft_call.function is soft
    state, action, previous = torch.randn(4, SPEC.state_dim), torch.randn(4, 1), torch.randn(4, 1)
    context, frame = torch.randn(4, 1), torch.arange(4)
    real = make_task().binding.bind(frame, state=state, action=action, previous=previous, context=context)
    torch.testing.assert_close(mdp.reward(real), mdp.reward_call.invoke(real).float() * 0.06)
    # Observation action input is the previously applied action by contract.
    observed = mdp.binding.bind(frame, state=state, action=previous, previous=previous, context=context)
    policy_out = torch.empty(4, 2)
    mdp.policy_call.invoke(observed, policy_out)
    torch.testing.assert_close(mdp.observation(observed), policy_out)
    zero = torch.zeros_like(action)
    critic_ctx = mdp.binding.bind(frame, state=state, action=zero, previous=zero, context=context)
    value_out = torch.empty(4, 1)
    mdp.value_call.invoke(real, value_out)
    torch.testing.assert_close(mdp.critic_observation(critic_ctx), value_out)
    torch.testing.assert_close(mdp.soft_termination(real), mdp.soft_call.invoke(real))


def test_heads_share_supplied_context_and_reward_soft_gradients_without_rebinding() -> None:
    class NoBinding(FixtureImagination):
        def bind(self, inputs: ImaginationInputs) -> ImaginationBindings:
            raise AssertionError("a numerical head rebound its supplied context")

    mdp = make_task(NoBinding())
    state = torch.randn(4, SPEC.state_dim, requires_grad=True)
    action = torch.randn(4, mdp.action_dim, requires_grad=True)
    previous = torch.randn_like(action)
    xi, frame = torch.randn_like(action), torch.arange(4)
    ctx = make_task().binding.bind(frame, state=state, action=action, previous=previous, context=xi)
    torch.testing.assert_close(mdp.observation(ctx), torch.cat((state[:, :1], action), -1))
    torch.testing.assert_close(mdp.critic_observation(ctx), state[:, :1] + xi)
    torch.testing.assert_close(mdp.goal(ctx)[:, 0], frame.float())
    rewards, probability = mdp.reward(ctx), mdp.soft_termination(ctx)
    torch.testing.assert_close(rewards, (state.square().sum(-1) - (action - previous).square().sum(-1)) * 0.06)
    gs, ga = torch.autograd.grad((rewards * (1 - probability)).sum(), (state, action))
    expected_state = state * (0.12 * (1 - probability))[:, None]
    expected_state[:, 0] -= rewards * probability * (1 - probability)
    torch.testing.assert_close(gs, expected_state)
    torch.testing.assert_close(ga, -0.12 * (action - previous) * (1 - probability)[:, None])


def test_reward_output_is_float32_with_gradients_from_float64_term() -> None:
    mdp = make_task()
    state = torch.randn(2, SPEC.state_dim, dtype=torch.float64, requires_grad=True)
    action = torch.zeros(2, mdp.action_dim, dtype=torch.float64)
    result = mdp.reward(
        mdp.binding.bind(
            torch.arange(2),
            state=state,
            action=action,
            previous=action,
            context=state.new_zeros((2, mdp.spec.context_dim)),
        )
    )
    assert result.dtype == torch.float32
    torch.testing.assert_close(torch.autograd.grad(result.sum(), state)[0], state * 0.12)


@pytest.mark.parametrize("n", [1, 15, 37])
def test_arbitrary_predicted_batch_has_reward_and_observation_gradients(n: int) -> None:
    mdp = make_task()
    state, action = torch.randn(n, SPEC.state_dim, requires_grad=True), torch.randn(n, 1, requires_grad=True)
    frame, previous, context = torch.arange(n), torch.zeros(n, 1), torch.ones(n, 1)
    result = mdp.reward(
        mdp.binding.bind(
            frame,
            state=state,
            action=action,
            previous=previous,
            context=state.new_zeros((*frame.shape, mdp.spec.context_dim)),
        )
    )
    gs, ga = torch.autograd.grad(result.sum(), (state, action))
    torch.testing.assert_close(gs, state * 0.12)
    torch.testing.assert_close(ga, -action * 0.12)
    observation = mdp.observation(
        mdp.binding.bind(frame, state=state, action=previous, previous=previous, context=context)
    )
    assert observation.shape == (n, 2)
    expected = torch.zeros_like(state)
    expected[:, 0] = 1
    torch.testing.assert_close(torch.autograd.grad(observation.sum(), state)[0], expected)
    assert mdp.goal(mdp.binding.bind_reference(frame)).shape == (n, 3)


@pytest.mark.parametrize("shape", [(), (4,), (3, 5), (2, 3, 4)])
def test_binder_flattens_horizon_batches_and_caller_restores_outputs(shape: tuple[int, ...]) -> None:
    mdp = make_task()
    state = torch.randn(*shape, SPEC.state_dim, requires_grad=True)
    action = torch.randn(*shape, mdp.action_dim, requires_grad=True)
    previous, context = torch.zeros_like(action), torch.randn(*shape, SPEC.context_dim)
    frame = torch.zeros(shape, dtype=torch.long)
    shared = mdp.binding.bind(frame, state=state, action=action, previous=previous, context=context)
    observed = mdp.binding.bind(frame, state=state, action=previous, previous=previous, context=context)
    flat_rewards, flat_probability = mdp.reward(shared), mdp.soft_termination(shared)
    flat_observation, flat_critic = mdp.observation(observed), mdp.critic_observation(shared)
    flat_goals = mdp.goal(mdp.binding.bind_reference(frame))
    assert flat_rewards.shape == flat_probability.shape == (frame.numel(),)
    assert flat_observation.shape == (frame.numel(), mdp.observation_dim)
    assert flat_critic.shape == (frame.numel(), mdp.critic_observation_dim)
    assert flat_goals.shape == (frame.numel(), 3)
    rewards, probability = flat_rewards.reshape(shape), flat_probability.reshape(shape)
    observation = flat_observation.reshape(*shape, mdp.observation_dim)
    critic = flat_critic.reshape(*shape, mdp.critic_observation_dim)
    goals = flat_goals.reshape(*shape, 3)
    torch.testing.assert_close(rewards, (state.square().sum(-1) - action.square().sum(-1)) * 0.06)
    torch.testing.assert_close(critic, state[..., :1] + context)
    torch.testing.assert_close(probability, state[..., 0].sigmoid())
    soft_gradient = torch.autograd.grad(probability.sum(), state, retain_graph=True)[0]
    expected_soft_gradient = torch.zeros_like(state)
    expected_soft_gradient[..., 0] = probability * (1 - probability)
    torch.testing.assert_close(soft_gradient, expected_soft_gradient)
    torch.testing.assert_close(goals[..., 0], frame.float())
    gs, ga = torch.autograd.grad(rewards.sum(), (state, action))
    torch.testing.assert_close(gs, 0.12 * state)
    torch.testing.assert_close(ga, -0.12 * action)
    expected = torch.zeros_like(state)
    expected[..., 0] = 1
    torch.testing.assert_close(torch.autograd.grad(observation.sum(), state)[0], expected)


def test_explicit_input_binding_drives_same_numerical_dispatch_values() -> None:
    class ShiftedImagination(FixtureImagination):
        def bind(self, inputs: ImaginationInputs) -> ImaginationBindings:
            bindings = super().bind(inputs)
            return replace(bindings, sim=Map({"state": inputs.state["state"] + 2}))

    mdp = make_task(ShiftedImagination())
    state, frame = torch.zeros(2, SPEC.state_dim), torch.arange(2)
    action = torch.zeros(2, 1)
    context = mdp.binding.bind(frame, state=state, action=action, previous=action, context=torch.zeros(2, 1))
    expected = mdp.reward_call.invoke(context).to(torch.float32) * 0.06
    torch.testing.assert_close(mdp.reward(context), expected)
    torch.testing.assert_close(expected, torch.full((2,), SPEC.state_dim * 4 * 0.06))
    assert mdp.reward_call.function is reward


@pytest.mark.parametrize("shape", [(), (4,), (3, 5)])
def test_goal_uses_flat_frames_without_constructing_physical_inputs(shape: tuple[int, ...]) -> None:
    seen = []

    class FrameOnlyImagination(FixtureImagination):
        def bind_reference(self, frame: torch.Tensor) -> ImaginationBindings:
            seen.append(tuple(frame.shape))
            bindings = super().bind_reference(frame)
            assert len(bindings.sim) == 0
            return bindings

    mdp = make_task(FrameOnlyImagination())
    frame = torch.full(shape, 2, dtype=torch.long)
    ctx = mdp.binding.bind_reference(frame)
    assert len(ctx.sim) == len(ctx.actions) == len(ctx.metrics) == 0
    assert ctx.env_id.shape == ctx.terminated.shape == (frame.numel(),)
    assert ctx.sim_reset_requested.shape == (frame.numel(), 1)
    flat = mdp.goal(ctx)
    assert flat.shape == (frame.numel(), mdp.observation_spec["goal"].space.shape[0])
    goals = flat.reshape(*shape, mdp.observation_spec["goal"].space.shape[0])
    assert seen == [(frame.numel(),)]
    assert goals.shape == (*shape, mdp.observation_spec["goal"].space.shape[0])
    torch.testing.assert_close(goals[..., 0], frame.float())


def test_missing_predicted_input_reports_its_declared_key() -> None:
    class MissingImagination(FixtureImagination):
        def bind(self, inputs: ImaginationInputs) -> ImaginationBindings:
            return replace(super().bind(inputs), sim=Map({}))

    mdp = make_task(MissingImagination())
    with pytest.raises(KeyError, match="state"):
        mdp.reward(
            mdp.binding.bind(
                torch.arange(3),
                state=torch.zeros(3, SPEC.state_dim),
                action=torch.zeros(3, 1),
                previous=torch.zeros(3, 1),
                context=torch.zeros(3, mdp.spec.context_dim),
            )
        )


def test_imagined_evaluation_does_not_advance_real_command_or_curriculum() -> None:
    state, frame = torch.randn(3, SPEC.state_dim), torch.arange(3)
    action, context = torch.zeros(3, 1), torch.ones(3, 1)
    real = make_task().binding.bind(frame, state=state, action=action, previous=action, context=context)
    reference = real.commands["reference"]
    before = [
        tensor.clone()
        for tensor in (
            reference.frame,
            reference.context,
            reference.curriculum,
            real.actions["action"].action_queue,
            real.sim_reset_requested,
        )
    ]

    # Deliberately borrow the real command objects: even then selecting the
    # numerical program must never schedule lifecycle hooks or mutate metadata.
    class BorrowedImagination(FixtureImagination):
        def bind(self, inputs: ImaginationInputs) -> ImaginationBindings:
            return ImaginationBindings(commands=real.commands, sim=real.sim)

    mdp = make_task(BorrowedImagination())
    ctx = mdp.binding.bind(frame, state=state, action=action, previous=action, context=context)
    for _ in range(3):
        mdp.reward(ctx)
        mdp.observation(ctx)
        mdp.critic_observation(ctx)
        mdp.soft_termination(ctx)
        mdp.goal(ctx)
    for old, actual in zip(
        before,
        (
            reference.frame,
            reference.context,
            reference.curriculum,
            real.actions["action"].action_queue,
            real.sim_reset_requested,
        ),
        strict=True,
    ):
        torch.testing.assert_close(old, actual)
