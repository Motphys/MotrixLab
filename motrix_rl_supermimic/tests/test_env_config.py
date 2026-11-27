# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MBRL configs select resolved Manager numerical terms without physics."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from _helpers import ReferenceContextCfg, make_toy
from _supermimic_toy import TARGETS, ToyEnvCfg
from motrix_rl_supermimic.env import SuperMimicEnv, SuperMimicEnvCfg
from motrix_rl_supermimic.imagined_task import ImaginedTask

from motrix_env_core.config import configclass
from motrix_env_core.manager.commands import CommandCfg, CommandTerm
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.env import ManagerEnv
from motrix_env_core.manager.observations import ObservationTermCfg, ObsTerm
from motrix_env_core.manager.rewards import RewardTerm, RewardTermCfg
from motrix_env_core.sim.read import DofPositionQuery


def make_cfg() -> SuperMimicEnvCfg:
    return ToyEnvCfg()


@pytest.mark.parametrize("goal_frames,imagined", [(0, False), (-1, False), (True, True), (1.5, True)])
def test_goal_window_requires_positive_integer_before_build(goal_frames, imagined: bool) -> None:
    cfg = ToyEnvCfg(goal_frames=goal_frames)
    with pytest.raises(ValueError, match="goal_frames must be a positive integer"):
        if imagined:
            ImaginedTask(cfg.for_imagined(), "cpu")
        else:
            make_toy(cfg=cfg)


@pytest.mark.parametrize("shape", [(2,), (3, 2), (2, 3, 2)])
def test_affine_context_parameters_broadcast_with_gradients(shape: tuple[int, ...]) -> None:
    cfg = ToyEnvCfg(
        world_model_context={"reference": ReferenceContextCfg(size=2, center=(1.0, -3.0), scale=(2.0, 4.0))}
    )
    task = ImaginedTask(cfg.for_imagined(), "cpu")
    context = torch.randn(shape, requires_grad=True)
    normalized = task.heads.normalize_context(context)
    torch.testing.assert_close(normalized[..., 0], (context[..., 0] - 1) / 2)
    torch.testing.assert_close(normalized[..., 1], (context[..., 1] + 3) / 4)
    torch.testing.assert_close(
        torch.autograd.grad(normalized.sum(), context)[0], context.new_tensor([0.5, 0.25]).expand_as(context)
    )
    assert task.world_context.center.dtype == task.world_context.scale.dtype == torch.float32
    assert task.world_context.center.device == task.world_context.scale.device == torch.device("cpu")


@pytest.mark.parametrize(
    ("offset", "scale", "message"),
    [
        ((), (), "lengths must equal its size"),
        ((0.0,), (1.0, 1.0), "lengths must equal its size"),
        ((0.0, 0.0), (1.0,), "lengths must equal its size"),
        ((float("nan"), 0.0), (1.0, 1.0), "finite center"),
        ((float("inf"), 0.0), (1.0, 1.0), "finite center"),
        ((0.0, 0.0), (0.0, 1.0), "finite positive scale"),
        ((0.0, 0.0), (-1.0, 1.0), "finite positive scale"),
        ((0.0, 0.0), (float("nan"), 1.0), "finite positive scale"),
        ((0.0, 0.0), (float("inf"), 1.0), "finite positive scale"),
    ],
)
def test_context_parameters_are_validated_at_assembly(offset, scale, message: str) -> None:
    cfg = ToyEnvCfg(world_model_context={"reference": ReferenceContextCfg(size=2, center=offset, scale=scale)})
    with pytest.raises(ValueError, match=message):
        ImaginedTask(cfg.for_imagined(), "cpu")


def test_empty_context_uses_empty_affine_parameters() -> None:
    task = ImaginedTask(ToyEnvCfg().for_imagined(), "cpu")
    context = torch.empty(2, 3, 0, requires_grad=True)
    result = task.heads.normalize_context(context)
    assert result.shape == context.shape
    assert task.world_context.size == 0
    assert task.world_context.slices == {}
    assert task.world_context.center.shape == task.world_context.scale.shape == (0,)
    assert torch.autograd.grad(result.sum(), context)[0].shape == context.shape


def test_imagined_command_override_is_used_without_sharing_runtime_state() -> None:
    original = ToyEnvCfg()
    original_command_cfg = original.commands["reference"]

    @configclass(kw_only=True)
    class ScaledCommandCfg(CommandCfg):
        def __call__(self, ctx: BuildContext) -> CommandTerm:
            command = original_command_cfg(ctx)
            return replace(command, targets=command.targets * 2)

    @configclass
    class ImaginedCommandEnvCfg(ToyEnvCfg):
        def for_imagined(self) -> ToyEnvCfg:
            cfg = deepcopy(self)
            cfg.commands = {"reference": ScaledCommandCfg()}
            return cfg

    cfg = ImaginedCommandEnvCfg()
    first = ImaginedTask(cfg.for_imagined(), "cpu")
    second = ImaginedTask(cfg.for_imagined(), "cpu")
    frame = torch.arange(4)
    imagined = ImaginedTask(original.for_imagined(), "cpu")
    torch.testing.assert_close(
        first.heads.goal(first.contexts.reference_context(frame)),
        imagined.heads.goal(imagined.contexts.reference_context(frame)) * 2,
    )
    state = torch.zeros(4, 2)
    action, context = torch.zeros(4, first.heads.action_dim), torch.zeros(4, 0)
    first_command = first.contexts.context(
        frame.clone(), state=state, action=action, previous=action, xi=context
    ).commands["reference"]
    second_command = second.contexts.context(
        frame.clone(), state=state, action=action, previous=action, xi=context
    ).commands["reference"]
    first_command.frame.fill_(7)
    torch.testing.assert_close(second_command.frame, frame)
    assert type(cfg.commands["reference"]) is type(original_command_cfg)


def test_configured_goal_term_is_shared_by_real_and_imagined_observations() -> None:
    def offset_goal(ctx: ManagerContext, out: torch.Tensor, width: int) -> None:
        out.copy_(ctx.commands["reference"].frame[:, None].float().expand(-1, width) + 7)

    @configclass
    class OffsetGoalCfg(ObservationTermCfg):
        def __call__(self, ctx: BuildContext) -> ObsTerm:
            return ObsTerm(ctx.cfg.goal_frames, offset_goal, ctx.cfg.goal_frames)

    cfg = ToyEnvCfg(goal_frames=3)
    cfg.observations["goal"] = {"offset": OffsetGoalCfg()}
    env = make_toy(num_envs=2, cfg=cfg)
    env.initialize()
    state = env.state
    task = ImaginedTask(cfg.for_imagined(), "cpu")
    frame = env.command_terms["reference"].frame.clone()
    torch.testing.assert_close(state.obs["goal"], task.heads.goal(task.contexts.reference_context(frame)))
    torch.testing.assert_close(
        task.heads.goal(task.contexts.reference_context(torch.tensor([[1, 2], [3, 4]]))).reshape(2, 2, -1)[..., 0],
        torch.tensor([[8.0, 9.0], [10.0, 11.0]]),
    )
    torch.testing.assert_close(env.command_terms["reference"].frame, frame)


def test_task_program_builds_without_physics_and_exposes_extensions_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("learner constructed a simulator")

    monkeypatch.setattr(ManagerEnv, "__init__", forbidden)
    task = ImaginedTask(make_cfg().for_imagined(), "cpu")
    assert task.reference_action() is None
    state = torch.zeros(5, 2, requires_grad=True)
    result = task.heads.reward(
        task.contexts.context(
            torch.arange(5),
            state=state,
            action=torch.ones(5, 1),
            previous=torch.zeros(5, 1),
            xi=state.new_zeros((5, task.world_context.size)),
        )
    )
    assert result.shape == (5,)
    assert torch.isfinite(torch.autograd.grad(result.sum(), state)[0]).all()


@pytest.mark.parametrize("field", ["world_model_state", "imagination", "soft_termination"])
def test_required_program_declarations_are_reported_clearly(field: str) -> None:
    with pytest.raises(ValueError, match="world_model_state, imagination and soft_termination"):
        ImaginedTask(replace(make_cfg(), **{field: None}).for_imagined(), "cpu")


def test_imagined_program_assembles_existing_multiple_observation_and_reward_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("analytic task constructed a simulator")

    def observe(ctx: ManagerContext, out: torch.Tensor, scale: float) -> None:
        out.copy_(ctx.sim["position"] * scale)

    def reward(ctx: ManagerContext, scale: float) -> torch.Tensor:
        return ctx.sim["position"][:, 0] * scale

    @configclass(kw_only=True)
    class Observation(ObservationTermCfg):
        scale: float

        def __call__(self, ctx: BuildContext) -> ObsTerm:
            return ObsTerm(1, observe, self.scale)

    @configclass(kw_only=True)
    class Reward(RewardTermCfg):
        scale: float

        def __call__(self, ctx: BuildContext) -> RewardTerm:
            return RewardTerm(reward, self.scale)

    cfg = make_cfg()
    cfg.observations = {
        "policy": {"first": Observation(scale=2.0), "second": Observation(scale=-3.0)},
        "value": {"privileged": Observation(scale=4.0)},
        "goal": cfg.observations["goal"],
    }
    cfg.rewards = {"first": Reward(weight=0.5, scale=2.0), "second": Reward(weight=-2.0, scale=3.0)}
    monkeypatch.setattr(ManagerEnv, "__init__", forbidden)
    task = ImaginedTask(cfg.for_imagined(), "cpu")
    state = torch.zeros(3, 2, requires_grad=True)
    with torch.no_grad():
        state[:, 0] = torch.tensor([0.1, 0.2, 0.3])
    frame, previous, context = torch.arange(3), torch.zeros(3, 1), torch.zeros(3, 0)
    actual_obs = task.heads.observation(
        task.contexts.context(frame, state=state, action=previous, previous=previous, xi=context)
    )
    torch.testing.assert_close(actual_obs, torch.cat((state[:, :1] * 2, state[:, :1] * -3), -1))
    torch.testing.assert_close(
        task.heads.critic_observation(
            task.contexts.context(
                frame,
                state=state,
                action=state.new_zeros((*frame.shape, task.heads.action_dim)),
                previous=state.new_zeros((*frame.shape, task.heads.action_dim)),
                xi=context,
            )
        ),
        state[:, :1] * 4,
    )
    actual_reward = task.heads.reward(
        task.contexts.context(
            frame,
            state=state,
            action=previous,
            previous=previous,
            xi=state.new_zeros((*frame.shape, task.world_context.size)),
        )
    )
    torch.testing.assert_close(actual_reward, state[:, 0] * (2 * 0.5 - 3 * 2) * cfg.ctrl_dt)
    expected_gradient = torch.zeros_like(state)
    expected_gradient[:, 0] = -5 * cfg.ctrl_dt
    torch.testing.assert_close(torch.autograd.grad(actual_reward.sum(), state)[0], expected_gradient)


def query_reward(ctx: ManagerContext, predicted: torch.Tensor) -> torch.Tensor:
    return predicted[:, 0].square()


@configclass(kw_only=True)
class QueryReward(RewardTermCfg):
    def __call__(self, ctx: BuildContext) -> RewardTerm:
        return RewardTerm(query_reward, DofPositionQuery())


def test_imagined_terms_reject_simulator_query_arguments() -> None:
    cfg = make_cfg()
    cfg.rewards = {"query": QueryReward(weight=1.0)}
    with pytest.raises(ValueError, match="declared input binding"):
        ImaginedTask(cfg.for_imagined(), "cpu")


def test_real_goal_observation_does_not_build_imagined_task(monkeypatch) -> None:
    from motrix_rl_supermimic.imagined_task import ImaginedTask

    def forbidden(*args, **kwargs):
        pytest.fail("Real command access must not construct an imagined task")

    monkeypatch.setattr(ImaginedTask, "__init__", forbidden)

    env = make_toy(num_envs=2, cfg=ToyEnvCfg(goal_frames=4))
    env.initialize()
    env.command_terms["reference"].frame.fill_(1)
    env.step(torch.zeros(2, 1))
    target = torch.tensor(TARGETS[2 % len(TARGETS)])
    torch.testing.assert_close(env.state.obs["goal"], target.expand(2, 4))


def test_base_env_requires_task_configuration() -> None:
    with pytest.raises(TypeError, match="inheriting SuperMimicEnvCfg"):
        SuperMimicEnv()
