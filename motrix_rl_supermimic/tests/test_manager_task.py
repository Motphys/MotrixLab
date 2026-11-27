# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Imagination selects existing Manager numerical calls, not lifecycle hooks."""

from collections.abc import Callable, Mapping
from dataclasses import replace

import gymnasium as gym
import pytest
import torch
from motrix_rl_supermimic.env import SuperMimicEnvCfg
from motrix_rl_supermimic.imagination import ImaginationCfg, ImaginationContextFactory, ImaginationTerm
from motrix_rl_supermimic.imagined_task import ImaginedTask
from motrix_rl_supermimic.world_context import WorldContext, WorldContextTerm, WorldContextTermCfg
from motrix_rl_supermimic.world_state import EuclideanState, WorldStateManager, WorldStateTerm, WorldStateTermCfg

from motrix_env_core.config import configclass
from motrix_env_core.manager.actions import ActionCfg, ActionState, ActionTerm
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.observations import ManagerObservationGroupCfg, ObservationTermCfg, ObsTerm
from motrix_env_core.manager.rewards import RewardTerm, RewardTermCfg
from motrix_env_core.manager.terminations import TerminationTerm, TerminationTermCfg


def capture_context(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.commands["reference"].context)


class SimStateTerm(EuclideanState):
    def capture(self, sim: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
        out.copy_(sim["state"])

    def views(self, state: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"state": state}


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
    def goals(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.ones(frame.numel(), 3)

    def valid_starts(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        return torch.ones_like(frame, dtype=torch.bool)


class SimViewTerm(SimStateTerm):
    def views(self, state: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {type(self).__name__.removesuffix("Term").lower(): state}


@configclass
class _FixtureImaginationCfg(ImaginationCfg):
    def __call__(self, ctx: BuildContext) -> ImaginationTerm:
        return FixtureImagination()


@configclass
class _FixtureStateCfg(WorldStateTermCfg):
    term: WorldStateTerm = None

    def __call__(self, ctx: BuildContext) -> WorldStateTerm:
        return self.term


@configclass
class _ReferenceContextCfg(WorldContextTermCfg):
    def __call__(self, ctx: BuildContext) -> WorldContextTerm:
        return WorldContextTerm(1, capture_context, (0.0,), (1.0,))


@configclass(kw_only=True)
class _ObsCfg(ObservationTermCfg):
    size: int
    fn: Callable

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        return ObsTerm(self.size, self.fn)


@configclass
class _RewardCfg(RewardTermCfg):
    def __call__(self, ctx: BuildContext) -> RewardTerm:
        return RewardTerm(reward)


@configclass
class _SoftCfg(TerminationTermCfg):
    def __call__(self, ctx: BuildContext) -> TerminationTerm:
        return TerminationTerm(soft)


@configclass
class _OneDofActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, actuators: tuple | None) -> ActionTerm:
        return _OneDofAction(ctx.num_envs)


class _OneDofAction(ActionTerm):
    def __init__(self, num_envs: int) -> None:
        super().__init__(
            gym.spaces.Box(-1.0, 1.0, (1,)),
            ActionState(torch.zeros(num_envs, 2, 1), torch.zeros(1, dtype=torch.long)),
        )

    def _process(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def _reset(self, env_ids: torch.Tensor) -> None:
        pass


@configclass
class _PolicyGroupCfg(ManagerObservationGroupCfg):
    task: ObservationTermCfg = _ObsCfg(size=2, fn=policy)


@configclass
class _ValueGroupCfg(ManagerObservationGroupCfg):
    task: ObservationTermCfg = _ObsCfg(size=1, fn=value)


@configclass
class _GoalGroupCfg(ManagerObservationGroupCfg):
    task: ObservationTermCfg = _ObsCfg(size=3, fn=goal)


@configclass
class FixtureTaskCfg(SuperMimicEnvCfg):
    world_model_state: dict[str, WorldStateTermCfg] = {"state": _FixtureStateCfg(term=None)}
    world_model_context: dict[str, WorldContextTermCfg] = {"reference": _ReferenceContextCfg()}
    imagination: ImaginationCfg = _FixtureImaginationCfg()
    soft_termination: TerminationTermCfg = _SoftCfg()
    goal_frames: int = 3
    ctrl_dt: float = 0.02

    def __post_init__(self) -> None:
        self.actions = {"action": _OneDofActionCfg()}
        self.observations = {
            "policy": _PolicyGroupCfg(),
            "value": _ValueGroupCfg(),
            "goal": _GoalGroupCfg(),
        }
        self.rewards = {"task": _RewardCfg(weight=3.0)}


def make_task(world_state: WorldStateManager | None = None) -> ImaginedTask:
    state = SimStateTerm(4) if world_state is None else world_state.terms["state"]
    cfg = replace(FixtureTaskCfg(), world_model_state={"state": _FixtureStateCfg(term=state)})
    return ImaginedTask(cfg.for_imagined(), "cpu")


def test_runtime_slices_named_physical_inputs_and_preserves_all_gradients() -> None:
    class AgentTerm(SimViewTerm):
        pass

    class ObjectTerm(SimViewTerm):
        pass

    world_state = WorldStateManager(
        {
            "agent": AgentTerm(2),
            "object": ObjectTerm(3),
        }
    )
    world_context = WorldContext(
        {
            "terrain": WorldContextTerm(1, capture_context, (0.0,), (1.0,)),
            "goal": WorldContextTerm(2, capture_context, (1.0, 2.0), (2.0, 3.0)),
        }
    )
    runtime = ImaginationContextFactory(
        FixtureImagination(), world_state, world_context, {"legs": slice(0, 2), "arms": slice(2, 3)}, 0.06, "reference"
    )
    state = torch.randn(2, 3, 5, requires_grad=True)
    context = torch.randn(2, 3, 3, requires_grad=True)
    action = torch.randn(2, 3, 3, requires_grad=True)
    previous = torch.randn(2, 3, 3, requires_grad=True)
    ctx = runtime.context(torch.arange(6).reshape(2, 3), state=state, xi=context, action=action, previous=previous)
    torch.testing.assert_close(ctx.sim["agent"], state.reshape(6, 5)[:, :2])
    torch.testing.assert_close(ctx.sim["object"], state.reshape(6, 5)[:, 2:])
    reference = ctx.commands["reference"]
    torch.testing.assert_close(reference.context, context.reshape(6, 3))
    torch.testing.assert_close(reference.physical_state, state.reshape(6, 5))
    torch.testing.assert_close(reference.frame, torch.arange(6))
    torch.testing.assert_close(reference.command, torch.ones(6, 3))
    assert ctx.sim["agent"].data_ptr() == state.data_ptr()
    assert ctx.sim["object"].data_ptr() == state.reshape(6, 5)[:, 2:].data_ptr()
    assert reference.context.data_ptr() == context.data_ptr()
    for name, part in runtime.action_slices.items():
        torch.testing.assert_close(ctx.actions[name].current(), action.reshape(6, 3)[:, part])
        torch.testing.assert_close(ctx.actions[name].previous(), previous.reshape(6, 3)[:, part])
    loss = sum(value.sum() for value in ctx.sim.values())
    loss += ctx.commands["reference"].context.sum()
    loss += sum((term.current() * 2 + term.previous() * 3).sum() for term in ctx.actions.values())
    gradients = torch.autograd.grad(loss, (state, context, action, previous))
    for gradient, variable, scale in zip(gradients, (state, context, action, previous), (1, 1, 2, 3), strict=True):
        torch.testing.assert_close(gradient, torch.full_like(variable, scale))
    assert ctx.dt == 0.06
    assert len(ctx.metrics) == 0
    assert not ctx.terminated.any() and not ctx.sim_reset_requested.any()


def test_configured_observation_noise_is_shared_once_by_real_and_imagined_dispatch() -> None:
    from motrix_env_core.manager.resolver import ManagerTermResolver

    task = make_task()
    state, frame = torch.randn(3, 4), torch.arange(3)
    previous, context = torch.ones(3, 1), torch.zeros(3, 1)
    ctx = make_task().contexts.context(frame, state=state, action=previous, previous=previous, xi=context)
    clean = task.heads.observation(
        task.contexts.context(frame, state=state, action=previous, previous=previous, xi=context)
    )
    rng = torch.random.get_rng_state()

    def noisy_policy(ctx: ManagerContext, out: torch.Tensor) -> None:
        policy(ctx, out)
        out.add_(2)

    call = ManagerTermResolver().resolve_observation("policy", "noisy", ObsTerm(2, noisy_policy))
    out = torch.empty_like(clean)
    call.invoke(ctx, out)
    torch.testing.assert_close(out, clean + 2)
    task.heads.policy_call = call
    predicted = state.clone().requires_grad_(True)
    torch.testing.assert_close(
        task.heads.observation(
            task.contexts.context(frame, state=predicted, action=previous, previous=previous, xi=context)
        ),
        out,
    )
    gradient = torch.autograd.grad(
        task.heads.observation(
            task.contexts.context(frame, state=predicted, action=previous, previous=previous, xi=context)
        ).sum(),
        predicted,
    )[0]
    expected_gradient = torch.zeros_like(predicted)
    expected_gradient[:, 0] = 1
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


def test_resolved_numerical_calls_share_real_dispatch_and_reward_units() -> None:
    task = make_task()
    # The resolved calls wrap the assembled group dispatch, so heads and raw
    # call invocations share one code path; the behavioral checks below
    # cover the shared-dispatch contract.
    state, action, previous = torch.randn(4, 4), torch.randn(4, 1), torch.randn(4, 1)
    context, frame = torch.randn(4, 1), torch.arange(4)
    real = make_task().contexts.context(frame, state=state, action=action, previous=previous, xi=context)
    torch.testing.assert_close(task.heads.reward(real), task.heads.reward_call.invoke(real).float() * task.heads.dt)
    # Observation action input is the previously applied action by contract.
    observed = task.contexts.context(frame, state=state, action=previous, previous=previous, xi=context)
    policy_out = torch.empty(4, 2)
    task.heads.policy_call.invoke(observed, policy_out)
    torch.testing.assert_close(task.heads.observation(observed), policy_out)
    zero = torch.zeros_like(action)
    critic_ctx = task.contexts.context(frame, state=state, action=zero, previous=zero, xi=context)
    value_out = torch.empty(4, 1)
    task.heads.value_call.invoke(real, value_out)
    torch.testing.assert_close(task.heads.critic_observation(critic_ctx), value_out)
    torch.testing.assert_close(task.heads.soft_termination(real), task.heads.soft_call.invoke(real))


def test_heads_share_supplied_context_and_reward_soft_gradients() -> None:
    task = make_task()
    state = torch.randn(4, 4, requires_grad=True)
    action = torch.randn(4, task.heads.action_dim, requires_grad=True)
    previous = torch.randn_like(action)
    xi, frame = torch.randn_like(action), torch.arange(4)
    ctx = make_task().contexts.context(frame, state=state, action=action, previous=previous, xi=xi)
    torch.testing.assert_close(task.heads.observation(ctx), torch.cat((state[:, :1], action), -1))
    torch.testing.assert_close(task.heads.critic_observation(ctx), state[:, :1] + xi)
    torch.testing.assert_close(task.heads.goal(ctx)[:, 0], frame.float())
    rewards, probability = task.heads.reward(ctx), task.heads.soft_termination(ctx)
    torch.testing.assert_close(rewards, (state.square().sum(-1) - (action - previous).square().sum(-1)) * 0.06)
    gs, ga = torch.autograd.grad((rewards * (1 - probability)).sum(), (state, action))
    expected_state = state * (0.12 * (1 - probability))[:, None]
    expected_state[:, 0] -= rewards * probability * (1 - probability)
    torch.testing.assert_close(gs, expected_state)
    torch.testing.assert_close(ga, -0.12 * (action - previous) * (1 - probability)[:, None])


def test_reward_output_is_float32_with_gradients_from_float64_term() -> None:
    task = make_task()
    state = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    action = torch.zeros(2, task.heads.action_dim, dtype=torch.float64)
    result = task.heads.reward(
        task.contexts.context(
            torch.arange(2),
            state=state,
            action=action,
            previous=action,
            xi=state.new_zeros((2, task.world_context.size)),
        )
    )
    assert result.dtype == torch.float32
    torch.testing.assert_close(torch.autograd.grad(result.sum(), state)[0], state * 0.12)


@pytest.mark.parametrize("n", [1, 15, 37])
def test_arbitrary_predicted_batch_has_reward_and_observation_gradients(n: int) -> None:
    task = make_task()
    state, action = torch.randn(n, 4, requires_grad=True), torch.randn(n, 1, requires_grad=True)
    frame, previous, context = torch.arange(n), torch.zeros(n, 1), torch.ones(n, 1)
    result = task.heads.reward(
        task.contexts.context(
            frame,
            state=state,
            action=action,
            previous=previous,
            xi=state.new_zeros((*frame.shape, task.world_context.size)),
        )
    )
    gs, ga = torch.autograd.grad(result.sum(), (state, action))
    torch.testing.assert_close(gs, state * 0.12)
    torch.testing.assert_close(ga, -action * 0.12)
    observation = task.heads.observation(
        task.contexts.context(frame, state=state, action=previous, previous=previous, xi=context)
    )
    assert observation.shape == (n, 2)
    expected = torch.zeros_like(state)
    expected[:, 0] = 1
    torch.testing.assert_close(torch.autograd.grad(observation.sum(), state)[0], expected)
    assert task.heads.goal(task.contexts.reference_context(frame)).shape == (n, 3)


@pytest.mark.parametrize("shape", [(), (4,), (3, 5), (2, 3, 4)])
def test_binder_flattens_horizon_batches_and_caller_restores_outputs(shape: tuple[int, ...]) -> None:
    task = make_task()
    state = torch.randn(*shape, 4, requires_grad=True)
    action = torch.randn(*shape, task.heads.action_dim, requires_grad=True)
    previous, context = torch.zeros_like(action), torch.randn(*shape, 1)
    frame = torch.zeros(shape, dtype=torch.long)
    shared = task.contexts.context(frame, state=state, action=action, previous=previous, xi=context)
    observed = task.contexts.context(frame, state=state, action=previous, previous=previous, xi=context)
    flat_rewards, flat_probability = task.heads.reward(shared), task.heads.soft_termination(shared)
    flat_observation, flat_critic = task.heads.observation(observed), task.heads.critic_observation(shared)
    flat_goals = task.heads.goal(task.contexts.reference_context(frame))
    assert flat_rewards.shape == flat_probability.shape == (frame.numel(),)
    assert flat_observation.shape == (frame.numel(), task.heads.observation_dim)
    assert flat_critic.shape == (frame.numel(), task.heads.critic_observation_dim)
    assert flat_goals.shape == (frame.numel(), 3)
    rewards, probability = flat_rewards.reshape(shape), flat_probability.reshape(shape)
    observation = flat_observation.reshape(*shape, task.heads.observation_dim)
    critic = flat_critic.reshape(*shape, task.heads.critic_observation_dim)
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
    class ShiftedState(SimStateTerm):
        def views(self, state: torch.Tensor) -> Mapping[str, torch.Tensor]:
            return {"state": state + 2}

    task = make_task(world_state=WorldStateManager({"state": ShiftedState(4)}))
    state, frame = torch.zeros(2, 4), torch.arange(2)
    action = torch.zeros(2, 1)
    context = task.contexts.context(frame, state=state, action=action, previous=action, xi=torch.zeros(2, 1))
    expected = task.heads.reward_call.invoke(context).to(torch.float32) * task.heads.dt
    torch.testing.assert_close(task.heads.reward(context), expected)
    torch.testing.assert_close(expected, torch.full((2,), 4 * 4 * 0.06))


def test_contexts_use_configured_reference_command_name() -> None:
    cfg = replace(
        FixtureTaskCfg(),
        reference_command="timeline",
        world_model_state={"state": _FixtureStateCfg(term=SimStateTerm(4))},
    )
    task = ImaginedTask(cfg.for_imagined(), "cpu")
    frame = torch.tensor([1, 2])
    state = torch.randn(2, task.world_state.size, requires_grad=True)
    action = torch.zeros(2, task.heads.action_dim)
    xi = torch.randn(2, task.world_context.size, requires_grad=True)

    physical = task.contexts.context(frame, state=state, action=action, previous=action, xi=xi)
    reference_only = task.contexts.reference_context(frame)
    for ctx in (physical, reference_only):
        assert tuple(ctx.commands) == ("timeline",)
        torch.testing.assert_close(ctx.commands["timeline"].frame, frame)
        torch.testing.assert_close(ctx.commands["timeline"].command, torch.ones(2, 3))
    torch.testing.assert_close(physical.commands["timeline"].context, xi)
    torch.testing.assert_close(physical.commands["timeline"].physical_state, state)


@pytest.mark.parametrize("shape", [(), (4,), (3, 5)])
def test_goal_uses_flat_frames_without_constructing_physical_inputs(shape: tuple[int, ...]) -> None:
    task = make_task()
    frame = torch.full(shape, 2, dtype=torch.long)
    ctx = task.contexts.reference_context(frame)
    assert len(ctx.sim) == len(ctx.actions) == len(ctx.metrics) == 0
    assert ctx.env_id.shape == ctx.terminated.shape == (frame.numel(),)
    assert ctx.sim_reset_requested.shape == (frame.numel(), 1)
    flat = task.heads.goal(ctx)
    assert flat.shape == (frame.numel(), task.heads.observation_spec["goal"].space.shape[0])
    goals = flat.reshape(*shape, task.heads.observation_spec["goal"].space.shape[0])
    assert goals.shape == (*shape, task.heads.observation_spec["goal"].space.shape[0])
    torch.testing.assert_close(goals[..., 0], frame.float())


def test_missing_predicted_input_reports_its_declared_key() -> None:
    class SilentState(EuclideanState):
        def capture(self, sim: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
            out.copy_(sim["state"])

    task = make_task(world_state=WorldStateManager({"state": SilentState(4)}))
    with pytest.raises(KeyError, match="state"):
        task.heads.reward(
            task.contexts.context(
                torch.arange(3),
                state=torch.zeros(3, 4),
                action=torch.zeros(3, 1),
                previous=torch.zeros(3, 1),
                xi=torch.zeros(3, task.world_context.size),
            )
        )


def test_imagined_evaluation_does_not_mutate_the_supplied_context() -> None:
    state, frame = torch.randn(3, 4), torch.arange(3)
    action, context = torch.zeros(3, 1), torch.ones(3, 1)
    task = make_task()
    ctx = task.contexts.context(frame, state=state, action=action, previous=action, xi=context)
    reference = ctx.commands["reference"]
    watched = (
        reference.frame,
        reference.context,
        reference.command,
        reference.physical_state,
        ctx.actions["action"].action_queue,
        ctx.sim_reset_requested,
    )
    before = [tensor.clone() for tensor in watched]
    # Repeated numerical evaluation must never mutate the supplied context:
    # the reference view carries no lifecycle hooks by construction.
    for _ in range(3):
        task.heads.reward(ctx)
        task.heads.observation(ctx)
        task.heads.critic_observation(ctx)
        task.heads.soft_termination(ctx)
        task.heads.goal(ctx)
    for old, actual in zip(before, watched, strict=True):
        torch.testing.assert_close(old, actual)
