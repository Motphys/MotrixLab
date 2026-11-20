# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""ManagerEnv lifecycle and term-registry contracts on the Torch executor.

The fixtures build a physics-less manager env (``physics_step`` is a no-op):
action and command terms encode all observable state, so reset semantics,
observation lifetimes, perf phases, and term validation can be asserted
without a simulator backend.
"""

from dataclasses import dataclass, fields, replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.base import EnvCfg  # noqa: E402
from motrix_env_core.config import configclass  # noqa: E402
from motrix_env_core.config.scene import SceneCfg  # noqa: E402
from motrix_env_core.manager import (  # noqa: E402
    ActionCfg,
    ActionState,
    ActionTerm,
    CommandCfg,
    CommandTerm,
    ManagerActionsCfg,
    ManagerBasedEnvCfg,
    ManagerCommandsCfg,
    ManagerContext,
    ManagerEnv,
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ManagerRewardsCfg,
    ManagerTerminationsCfg,
    ObservationTermCfg,
    ObsTerm,
    RewardTerm,
    RewardTermCfg,
    TerminationTerm,
    TerminationTermCfg,
    metric,
)
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.observations import create_observation_groups
from motrix_env_core.mdp.observations import add_uniform_noise


@dataclass(frozen=True, slots=True)
class _TestActionState(ActionState):
    reset_flags: torch.Tensor
    source: torch.Tensor


@dataclass(frozen=True, slots=True)
class _SecondActionState(ActionState):
    reset_flags: torch.Tensor
    source: torch.Tensor


class _TestActionTerm(ActionTerm):
    def __init__(self, action_space: gym.spaces.Box, state: _TestActionState) -> None:
        super().__init__(action_space, state)

    def _process(self, actions: torch.Tensor) -> None:
        self.state.source.copy_(actions)

    def _reset(self, env_ids: torch.Tensor) -> None:
        self.state.reset_flags.fill_(False)
        self.state.reset_flags[env_ids] = True


class _SecondActionTerm(ActionTerm):
    def __init__(self, action_space: gym.spaces.Box, state: _SecondActionState) -> None:
        super().__init__(action_space, state)

    def _process(self, actions: torch.Tensor) -> None:
        self.state.source.copy_(actions)

    def _reset(self, env_ids: torch.Tensor) -> None:
        self.state.reset_flags.fill_(False)
        self.state.reset_flags[env_ids] = True


def _test_action_space(_: ManagerEnv, __: np.ndarray | None) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)


def _second_action_space(_: ManagerEnv, __: np.ndarray | None) -> gym.spaces.Box:
    return gym.spaces.Box(
        np.asarray([-2.0, -3.0], dtype=np.float32),
        np.asarray([2.0, 3.0], dtype=np.float32),
    )


@dataclass(frozen=True, slots=True)
class _ObservationParams:
    offset: torch.Tensor
    scale: float


def _injected_observation(ctx: ManagerContext, out: torch.Tensor, params: _ObservationParams) -> None:
    action = ctx.actions["test"]
    out[:, 0] = action.source[:, 0] * params.scale + params.offset[0]
    out[:, 1] = ctx.commands["counter"].command[:, 0]


@configclass(kw_only=True)
class _InjectedObservationCfg(ObservationTermCfg):
    scale: float

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(
            2,
            _injected_observation,
            _ObservationParams(torch.from_numpy(np.asarray([1.0], dtype=np.float32)), float(self.scale)),
        )


def _derived_observation(ctx: ManagerContext, out: torch.Tensor) -> None:
    out[:, 0] = ctx.commands["counter"].double[:, 0]


@configclass(kw_only=True)
class _DerivedObservationCfg(ObservationTermCfg):
    def size(self, env: ManagerEnv) -> int:
        del env
        return 1

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(1, _derived_observation)


def _lane_observation(ctx: ManagerContext, out: torch.Tensor) -> None:
    out[:, 0] = ctx.env_id


@configclass(kw_only=True)
class _LaneObservationCfg(ObservationTermCfg):
    def size(self, env: ManagerEnv) -> int:
        del env
        return 1

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(1, _lane_observation)


@configclass
class _LanePolicyObsCfg(ManagerObservationGroupCfg):
    lane: _LaneObservationCfg = _LaneObservationCfg()


@configclass
class _LaneObservationsCfg(ManagerObservationsCfg):
    policy: _LanePolicyObsCfg = _LanePolicyObsCfg()


def _injected_reward(ctx: ManagerContext):
    return ctx.actions["test"].source[:, 0]


@configclass(kw_only=True)
class _InjectedRewardCfg(RewardTermCfg):
    def __call__(self, env: ManagerEnv) -> RewardTerm:
        del env
        return RewardTerm(_injected_reward)


def _injected_termination(ctx: ManagerContext, threshold) -> bool:
    value = ctx.actions["test"].source[:, 0]
    ctx.metrics["source_at_termination"][:, 0] = value
    return value >= threshold


@configclass(kw_only=True)
class _InjectedTerminationCfg(TerminationTermCfg):
    threshold: float

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        return TerminationTerm(
            _injected_termination,
            float(self.threshold),
            metric_names=("source_at_termination",),
        )


@configclass(kw_only=True)
class _TestActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, actuator_indices: np.ndarray | None) -> ActionTerm:
        return _TestActionTerm(
            _test_action_space(ctx, actuator_indices),
            _TestActionState(
                torch.from_numpy(np.zeros((ctx.num_envs, 2, 1), dtype=np.float32)),
                torch.from_numpy(np.zeros(1, dtype=np.int64)),
                torch.from_numpy(np.zeros((ctx.num_envs, 1), dtype=bool)),
                torch.from_numpy(np.zeros((ctx.num_envs, 1), dtype=np.float32)),
            ),
        )


@configclass(kw_only=True)
class _SecondActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] = ()

    def __call__(self, ctx: BuildContext, actuator_indices: np.ndarray | None) -> ActionTerm:
        return _SecondActionTerm(
            _second_action_space(ctx, actuator_indices),
            _SecondActionState(
                torch.from_numpy(np.zeros((ctx.num_envs, 2, 2), dtype=np.float32)),
                torch.from_numpy(np.zeros(1, dtype=np.int64)),
                torch.from_numpy(np.zeros((ctx.num_envs, 1), dtype=bool)),
                torch.from_numpy(np.zeros((ctx.num_envs, 2), dtype=np.float32)),
            ),
        )


@configclass(kw_only=True)
class _CounterCommandCfg(CommandCfg):
    def __call__(self, ctx: BuildContext) -> CommandTerm:
        shape = (ctx.num_envs, 1)
        return _CounterCommand(
            double=torch.zeros(shape, dtype=torch.float32),
            command=torch.zeros(shape, dtype=torch.float32),
        )


@dataclass(frozen=True, slots=True)
class _CounterCommand(CommandTerm):
    double: torch.Tensor = metric()

    def update(self: "_CounterCommand", ctx: ManagerContext) -> None:
        self.double[:, 0] = 2.0 * ctx.actions["test"].source[:, 0]

    def advance(self: "_CounterCommand", ctx: ManagerContext) -> None:
        self.command[:, 0] += ctx.dt

    def reset(self: "_CounterCommand", ctx: ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        self.command[env_ids, 0] = -1.0


@configclass
class _ManagerPolicyObsCfg(ManagerObservationGroupCfg):
    injected: _InjectedObservationCfg = _InjectedObservationCfg(scale=3.0)
    derived: _DerivedObservationCfg = _DerivedObservationCfg()
    lane: _LaneObservationCfg = _LaneObservationCfg()


@configclass
class _ManagerObservationsCfg(ManagerObservationsCfg):
    policy: _ManagerPolicyObsCfg = _ManagerPolicyObsCfg()


@configclass
class _ManagerRewardsCfg(ManagerRewardsCfg):
    source: _InjectedRewardCfg = _InjectedRewardCfg(weight=2.0)


@configclass
class _ManagerTerminationsCfg(ManagerTerminationsCfg):
    limit: _InjectedTerminationCfg = _InjectedTerminationCfg(threshold=0.5)


def _manager_groups(*, scale: float = 3.0, threshold: float = 0.5, reward_weight: float = 2.0) -> ManagerBasedEnvCfg:
    return ManagerBasedEnvCfg(
        observations=_ManagerObservationsCfg(
            policy=_ManagerPolicyObsCfg(injected=_InjectedObservationCfg(scale=scale)),
        ),
        rewards=_ManagerRewardsCfg(source=_InjectedRewardCfg(weight=reward_weight)),
        terminations=_ManagerTerminationsCfg(limit=_InjectedTerminationCfg(threshold=threshold)),
    )


@configclass
class _ManagerCommandsCfg(ManagerCommandsCfg):
    counter: _CounterCommandCfg = _CounterCommandCfg()


@configclass
class _ManagerActionsCfg(ManagerActionsCfg):
    test: _TestActionCfg = _TestActionCfg()


@configclass
class _ManagerEnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = SceneCfg()
    commands: _ManagerCommandsCfg = _ManagerCommandsCfg()
    actions: _ManagerActionsCfg = _ManagerActionsCfg()
    observations: _ManagerObservationsCfg = _manager_groups().observations
    rewards: _ManagerRewardsCfg = _manager_groups().rewards
    terminations: _ManagerTerminationsCfg = _manager_groups().terminations


class _ManagerEnv(ManagerEnv[EnvCfg]):
    def __init__(
        self,
        num_envs: int = 2,
        rand_seed: int = 1,
        max_episode_seconds: float | None = None,
    ):
        cfg = _ManagerEnvCfg(max_episode_seconds=max_episode_seconds)
        super().__init__(cfg, num_envs, seed=rand_seed)

    def physics_step(self) -> None:
        pass


def _counter_command(env: _ManagerEnv) -> _CounterCommand:
    command = env.command_terms["counter"]
    assert isinstance(command, _CounterCommand)
    return command


@pytest.mark.parametrize("policy_h,value_h", [(1, 1), (3, 1), (2, 3)])
def test_history_frames_time_major_and_partial_episode_reset(policy_h: int, value_h: int) -> None:
    cfg = _ManagerEnvCfg(
        observations={
            "policy": _ManagerPolicyObsCfg(history_length=policy_h),
            "value": _LanePolicyObsCfg(history_length=value_h),
        }
    )
    env = ManagerEnv(cfg, num_envs=2)
    env.physics_step = lambda: None
    state = env.init_state()
    assert env.observation_spec.policy.space.shape == (4 * policy_h,)
    assert env.observation_spec.value.space.shape == (value_h,)
    assert env.observation_spec.policy.frame_space.shape == (4,)
    frames = [env.state.obs.policy[:, -4:].clone()]
    torch.testing.assert_close(state.obs.policy.reshape(2, policy_h, 4), frames[0][:, None].expand(-1, policy_h, -1))
    for action in (0.1, 0.2, 0.3):
        state = env.step(torch.full((2, 1), action))
        frames.append(env.state.obs.policy[:, -4:].clone())
        expected = ([frames[0]] * policy_h + frames[1:])[-policy_h:]
        torch.testing.assert_close(state.obs.policy.reshape(2, policy_h, 4), torch.stack(expected, dim=1))
    untouched = state.obs.policy[1].clone()
    frame_untouched = env.state.obs.policy[:, -4:][1].clone()
    env.reset(torch.tensor([0]))
    env.reset_observation(env.state, env_ids=torch.tensor([0]))
    current = env.state.obs.policy[:, -4:]
    torch.testing.assert_close(env.state.obs.policy[0].reshape(policy_h, 4), current[0].expand(policy_h, -1))
    torch.testing.assert_close(env.state.obs.policy[1], untouched)
    torch.testing.assert_close(current[1], frame_untouched)


@pytest.mark.parametrize("history_length", [1, 3])
def test_explicit_sim_only_reset_waits_for_next_control_step(history_length: int) -> None:
    cfg = _ManagerEnvCfg(observations={"policy": _ManagerPolicyObsCfg(history_length=history_length)})
    env = ManagerEnv(cfg, num_envs=2)
    env.physics_step = lambda: None
    env.init_state()
    state = env.step(torch.full((2, 1), 0.1))
    published = state.obs.policy.clone()
    steps = state.episode_steps.clone()
    env.reset(torch.empty(0, dtype=torch.long), sim_reset_ids=torch.tensor([1]))
    assert env.command_terms["counter"].command[1, 0] == -1.0
    torch.testing.assert_close(env.state.obs.policy, published)
    torch.testing.assert_close(env.state.episode_steps, steps)
    state = env.step(torch.full((2, 1), 0.2))
    window = state.obs.policy.reshape(2, history_length, 4)
    torch.testing.assert_close(window[:, :-1], published.reshape(2, history_length, 4)[:, 1:])
    torch.testing.assert_close(state.episode_steps, steps + 1)
    assert window[1, -1, 1] == pytest.approx(-1.0 + env.cfg.ctrl_dt)
    assert window[1, -1, 0] == pytest.approx(1.6)
    # Explicit whole-batch publication is an additional frame, not a cache read.
    snapshot = window.clone()
    env.compute_observation(state)
    updated = state.obs.policy.reshape(2, history_length, 4)
    torch.testing.assert_close(updated[:, :-1], snapshot[:, 1:])
    torch.testing.assert_close(updated[:, -1], snapshot[:, -1])


def test_history_final_preview_owned_frames_and_sim_only_reset() -> None:
    cfg = _ManagerEnvCfg(
        max_episode_seconds=0.03,
        observations={"policy": _ManagerPolicyObsCfg(history_length=3)},
    )
    env = ManagerEnv(cfg, num_envs=2)
    env.physics_step = lambda: None
    state = env.init_state()
    state = env.step(torch.full((2, 1), 0.1))
    prefix = state.obs.policy.reshape(2, 3, 4).clone()
    # Request a simulator-only reset during the real control step. It changes
    # the endpoint command but must retain this episode's historical prefix.
    evaluate = env._task_program.evaluate_kernel

    def request_sim_reset(weights, buffers, outputs):
        evaluate(weights, buffers, outputs)
        env._sim_reset_requested[1] = True

    env._task_program = replace(env._task_program, evaluate_kernel=request_sim_reset)
    state.episode_steps[0] = 2
    state = env.step(torch.full((2, 1), 0.2))
    final = state.final_obs
    assert final is not None
    final_frame = final.obs.policy[:, -4:]
    torch.testing.assert_close(final.env_ids, torch.tensor([0]))
    expected = torch.cat((prefix[[0], 1:], final_frame[:, None]), dim=1)
    torch.testing.assert_close(final.obs.policy.reshape(1, 3, 4), expected)
    frame = env.state.obs.policy[:, -4:]
    assert state.episode_steps[1] == 2
    assert frame[1, 1] == -1.0
    torch.testing.assert_close(state.obs.policy[0].reshape(3, 4), frame[0].expand(3, -1))
    torch.testing.assert_close(state.obs.policy[1].reshape(3, 4)[:-1], prefix[1, 1:])
    saved = final_frame.clone()
    state.episode_steps.zero_()
    env.step(torch.full((2, 1), 0.1))
    assert env.state.final_obs is None
    torch.testing.assert_close(final_frame, saved)
    env.reset(torch.tensor([1]))
    assert env.state.final_obs is None


@pytest.mark.parametrize("history_length", [1, 3])
def test_named_goal_group_history_reset_and_final_observations(history_length: int) -> None:
    @configclass
    class _NamedObservationsCfg(ManagerObservationsCfg):
        policy: _LanePolicyObsCfg = _LanePolicyObsCfg()
        goal: _ManagerPolicyObsCfg = _ManagerPolicyObsCfg(history_length=history_length)

    cfg = _ManagerEnvCfg(max_episode_seconds=0.03, observations=_NamedObservationsCfg())
    env = ManagerEnv(cfg, num_envs=2)
    env.physics_step = lambda: None
    initial = env.init_state()
    assert set(initial.obs.groups) == {"policy", "goal"}
    assert env.observation_spec["goal"].space.shape == (4 * history_length,)
    assert initial.obs.value is None
    torch.testing.assert_close(
        initial.obs["goal"].reshape(2, history_length, 4),
        initial.obs["goal"][:, -4:][:, None].expand(-1, history_length, -1),
    )
    state = env.step(torch.full((2, 1), 0.1))
    prefix = state.obs["goal"].reshape(2, history_length, 4).clone()
    state.episode_steps[0] = 2
    state = env.step(torch.full((2, 1), 0.2))
    final = state.final_obs
    assert final is not None
    torch.testing.assert_close(final.env_ids, torch.tensor([0]))
    expected_endpoint = torch.tensor([[1.6, -1.0 + 2 * cfg.ctrl_dt, 0.4, 0.0]])
    expected_final = torch.cat((prefix[[0], 1:], expected_endpoint[:, None]), dim=1)
    torch.testing.assert_close(final.obs["goal"].reshape(1, history_length, 4), expected_final)
    goal = state.obs["goal"].reshape(2, history_length, 4)
    torch.testing.assert_close(goal[0], goal[0, -1:].expand(history_length, -1))
    torch.testing.assert_close(goal[1, :-1], prefix[1, 1:])
    saved_final = final.obs["goal"].clone()
    untouched = state.obs["goal"][1].clone()
    env.reset(torch.tensor([0]))
    env.reset_observation(env.state, torch.tensor([0]))
    torch.testing.assert_close(env.state.obs["goal"][1], untouched)
    env.step(torch.full((2, 1), 0.1))
    torch.testing.assert_close(final.obs["goal"], saved_final)


def test_action_and_command_terms_are_environment_owned() -> None:
    env = _ManagerEnv(num_envs=3)

    assert isinstance(env.action_terms["test"], ActionTerm)
    assert env.action_terms is env._action_terms
    assert env.action_terms["test"].state.reset_flags.shape == (3, 1)
    assert _counter_command(env) is env.command_terms["counter"]
    build = env.build_context()
    assert build.num_envs == 3 and build.seed == 1
    assert build.commands["counter"] is env.command_terms["counter"]
    assert build.actions["test"] is env.action_terms["test"]


def test_same_cfg_class_instances_keep_independent_runtime_state() -> None:
    """Cfg-produced state must be freshly allocated per environment instance.

    The framework consumes cfg values by reference (no defensive clone), so
    two environments built from the same cfg class must not share any mutable
    per-env buffer: their action queues and command state stay distinct
    objects and interleaved stepping only reflects each environment's own
    actions and resets.
    """
    first = _ManagerEnv(num_envs=3)
    second = _ManagerEnv(num_envs=3)
    first.init_state()
    second.init_state()

    first_queue = first.action_terms["test"].state.action_queue
    second_queue = second.action_terms["test"].state.action_queue
    assert first_queue.data_ptr() != second_queue.data_ptr()
    assert _counter_command(first).command.data_ptr() != _counter_command(second).command.data_ptr()
    assert _counter_command(first).double.data_ptr() != _counter_command(second).double.data_ptr()

    dt = float(first.cfg.ctrl_dt)
    first_state = first.step(torch.full((3, 1), 0.1, dtype=torch.float32))
    second_state = second.step(torch.full((3, 1), -0.1, dtype=torch.float32))
    first.step(torch.full((3, 1), 0.2, dtype=torch.float32))
    second.step(torch.full((3, 1), -0.2, dtype=torch.float32))
    del first_state, second_state
    first.reset(torch.tensor([0], dtype=torch.long))
    second.reset(torch.tensor([0], dtype=torch.long))

    np.testing.assert_allclose(_counter_command(first).double.numpy()[:, 0], [0.4, 0.4, 0.4])
    np.testing.assert_allclose(_counter_command(second).double.numpy()[:, 0], [-0.4, -0.4, -0.4])
    np.testing.assert_allclose(_counter_command(first).command.numpy()[:, 0], [-1.0, -1.0 + 2 * dt, -1.0 + 2 * dt])
    np.testing.assert_allclose(_counter_command(second).command.numpy()[:, 0], [-1.0, -1.0 + 2 * dt, -1.0 + 2 * dt])


def test_reset_rejects_numpy_ids_and_non_integer_dtype() -> None:
    """The manager frontend's public reset currency is uniformly torch."""
    env = _ManagerEnv(num_envs=3)
    env.init_state()

    with pytest.raises(TypeError, match="torch.tensor"):
        env.reset(np.asarray([0]))
    with pytest.raises(TypeError, match="torch.tensor"):
        env.reset(torch.tensor([0]), sim_reset_ids=np.asarray([1]))
    with pytest.raises(TypeError, match="integer dtype"):
        env.reset(torch.tensor([0.0]))


def test_command_rematerialization_reuses_transition_kernel_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _ManagerEnv(num_envs=3)
    state = env.init_state()
    read_count = 0
    refresh_sim_reads = env._refresh_sim_reads

    def record_read() -> None:
        nonlocal read_count
        read_count += 1
        refresh_sim_reads()

    monkeypatch.setattr(env, "_refresh_sim_reads", record_read)

    env.compute_transition(state)

    assert read_count == 1


def test_episode_reset_reuses_transition_kernel_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _ManagerEnv(num_envs=3)
    env.init_state()
    read_count = 0
    refresh_sim_reads = env._refresh_sim_reads

    def record_read() -> None:
        nonlocal read_count
        read_count += 1
        refresh_sim_reads()

    monkeypatch.setattr(env, "_refresh_sim_reads", record_read)

    state = env.step(torch.ones((env.num_envs, 1), dtype=torch.float32))

    np.testing.assert_array_equal(state.terminated, True)
    assert read_count == 1


def test_manager_cfg_accepts_dict_groups_and_empty_commands() -> None:
    cfg = ManagerBasedEnvCfg(
        scene=SceneCfg(),
        actions={"test": _TestActionCfg()},
        commands={},
        observations={"policy": {"lane": _LaneObservationCfg()}},
    )
    env = ManagerEnv(cfg, num_envs=2)

    state = env.init_state()

    assert env.command_cfgs == {}
    assert env.command_terms == {}
    assert tuple(env.action_cfgs) == ("test",)
    assert tuple(env.observation_groups["policy"].terms)[0].name == "lane"
    np.testing.assert_array_equal(state.obs.policy[:, 0], [0.0, 1.0])


def test_multiple_action_terms_concatenate_spaces_and_receive_ordered_slices() -> None:
    def _multiple_action_observation(ctx: ManagerContext, out: torch.Tensor) -> None:
        first = ctx.actions["test"]
        second = ctx.actions["second"]
        out[:, 0] = first.source[:, 0]
        out[:, 1:] = second.source

    @configclass(kw_only=True)
    class _MultipleActionObservationCfg(ObservationTermCfg):
        def size(self, env: ManagerEnv) -> int:
            del env
            return 3

        def __call__(self, env: ManagerEnv) -> ObsTerm:
            del env
            return ObsTerm(3, _multiple_action_observation)

    @configclass
    class _MultipleActionPolicyCfg(ManagerObservationGroupCfg):
        actions: _MultipleActionObservationCfg = _MultipleActionObservationCfg()

    @configclass
    class _MultipleActionObservationsCfg(ManagerObservationsCfg):
        policy: _MultipleActionPolicyCfg = _MultipleActionPolicyCfg()

    @configclass
    class _MultipleActionsCfg(ManagerActionsCfg):
        test: _TestActionCfg = _TestActionCfg()
        second: _SecondActionCfg = _SecondActionCfg()

    cfg = _ManagerEnvCfg(actions=_MultipleActionsCfg(), observations=_MultipleActionObservationsCfg())
    env = ManagerEnv(cfg, num_envs=2)
    state = env.init_state()
    actions = torch.tensor([[0.25, 1.0, 2.0], [0.5, -1.0, -2.0]], dtype=torch.float32)

    env.apply_action(actions, state)

    assert tuple(env.action_terms) == ("test", "second")
    assert env.action_slices == {"test": slice(0, 1), "second": slice(1, 3)}
    assert env.action_space.shape == (3,)
    np.testing.assert_array_equal(env.action_space.low, [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(env.action_terms["test"].state.source, actions[:, :1])
    np.testing.assert_array_equal(env.action_terms["second"].state.source, actions[:, 1:])
    env._refresh_sim_reads()
    env._task_program.observe_kernel(env.observation_manager.frame_output)
    np.testing.assert_array_equal(state.obs.policy, actions)
    assert [(term.name, term.input_slice) for term in env.manager_layout.actions] == [
        ("test", slice(0, 1)),
        ("second", slice(1, 3)),
    ]

    state.terminated[:] = torch.tensor([False, True])
    env._reset_done_envs()
    np.testing.assert_array_equal(env.action_terms["test"].state.reset_flags[:, 0], [False, True])
    np.testing.assert_array_equal(env.action_terms["second"].state.reset_flags[:, 0], [False, True])


def test_multiple_action_terms_validate_total_action_shape() -> None:
    env = _ManagerEnv(num_envs=2)
    state = env.init_state()

    with pytest.raises(ValueError, match=r"Expected action shape \(2, 1\), got \(2, 2\)"):
        env.apply_action(torch.zeros((2, 2), dtype=torch.float32), state)


@pytest.mark.parametrize("seed", [None, 7])
def test_manager_construction_does_not_reseed_or_consume_default_rng(seed: int | None) -> None:
    torch.manual_seed(193)
    torch.rand(5)
    before = torch.get_rng_state().clone()
    created = ManagerEnv(_ManagerEnvCfg(), seed=seed)
    assert torch.equal(before, torch.get_rng_state())
    sample = created._random.uniform((16,))
    torch.set_rng_state(before)
    torch.testing.assert_close(sample, torch.rand(16), rtol=0, atol=0)


def test_managers_share_training_seeded_default_stream() -> None:
    torch.manual_seed(7)
    first, second = _ManagerEnv(rand_seed=7), _ManagerEnv(rand_seed=8)
    draws = [first._random.uniform((3, 2)), torch.rand(13), second._random.uniform((3, 2))]
    torch.manual_seed(7)
    for actual, shape in zip(draws, ((3, 2), (13,), (3, 2)), strict=True):
        torch.testing.assert_close(actual, torch.rand(shape), rtol=0, atol=0)


def test_step_perf_records_standard_and_manager_phases() -> None:
    env = _ManagerEnv()
    state = env.init_state()
    assert state.episode_steps.shape == (env.num_envs,)
    assert set(state.reward_terms) == {"source"}
    actions = torch.zeros((env.num_envs, 1), dtype=torch.float32)

    env.step(actions)
    assert env.perf.snapshot() == ()

    env.perf.enable()
    env.step(actions)

    step = env.perf.snapshot()[0]
    assert [child.name for child in step.children] == [
        "apply_action",
        "physics",
        "transition",
        "reset",
        "observation",
    ]
    assert [child.name for child in step.child("transition").children] == [
        "read_inputs",
        "evaluate",
    ]
    assert [child.name for child in step.child("transition").child("read_inputs").children] == ["sim_read"]
    assert [child.name for child in step.child("reset").children] == ["done_mask"]
    assert [child.name for child in step.child("observation").children] == ["observe"]


def test_step_preserves_previous_observation_with_alternating_preallocated_buffers() -> None:
    env = _ManagerEnv()
    initial = env.init_state()
    initial_policy = initial.obs.policy
    initial_snapshot = initial_policy.clone()

    first = env.step(torch.full((env.num_envs, 1), 0.25, dtype=torch.float32))

    np.testing.assert_array_equal(initial_policy, initial_snapshot)
    assert initial_policy.data_ptr() != first.obs.policy.data_ptr()
    first_policy = first.obs.policy
    first_snapshot = first_policy.clone()

    second = env.step(torch.full((env.num_envs, 1), 0.5, dtype=torch.float32))

    np.testing.assert_array_equal(first_policy, first_snapshot)
    assert first_policy.data_ptr() != second.obs.policy.data_ptr()
    assert initial_policy.data_ptr() == second.obs.policy.data_ptr()


def test_manager_context_is_injected_once_and_reused_across_all_term_kinds() -> None:
    env = _ManagerEnv()
    state = env.init_state()

    action = env.action_terms["test"]
    assert isinstance(action, ActionTerm)
    command = _counter_command(env)
    assert env._compiled_manager_program is not None
    context = env._compiled_manager_program.context
    assert context.actions["test"] is action.state
    assert context.commands["counter"] is command
    assert context.actions["test"].action_queue is action.state.action_queue
    assert context.sim is not None and context.dt == float(env.cfg.ctrl_dt)
    assert [term.output_slice for term in env.manager_layout.observations["policy"].terms] == [
        slice(0, 2),
        slice(2, 3),
        slice(3, 4),
    ]

    action.state.source[:, 0] = torch.tensor([0.25, 0.75])
    env.compute_transition(state)
    # Exercise dispatch independently of the cached normal-publication lifecycle.
    env._task_program.observe_kernel(env.observation_manager.frame_output)

    assert env.action_terms["test"] is action
    assert env.command_terms["counter"] is command
    np.testing.assert_allclose(state.obs.policy[:, 0], [1.75, 3.25])
    # advance added one ctrl_dt to the reset command value of -1.0.
    np.testing.assert_allclose(state.obs.policy[:, 1], -1.0 + env.cfg.ctrl_dt)
    np.testing.assert_allclose(state.obs.policy[:, 2], [0.5, 1.5])
    np.testing.assert_allclose(state.obs.policy[:, 3], [0.0, 1.0])
    np.testing.assert_allclose(state.reward, [0.005, 0.015])
    np.testing.assert_array_equal(state.terminated, [False, True])
    np.testing.assert_allclose(env.metrics["source_at_termination"][:, 0], [0.25, 0.75])
    np.testing.assert_array_equal(state.metrics["limit"], [False, True])
    np.testing.assert_allclose(np.ravel(state.metrics["double"]), [0.5, 1.5])
    np.testing.assert_allclose(state.reward_terms["source"], [0.005, 0.015])
    assert env._compiled_manager_program is not None


def test_reset_keeps_command_metric_alias_for_direct_context() -> None:
    env = _ManagerEnv(num_envs=2)
    state = env.init_state()
    command = _counter_command(env)
    command.command[:] = 3.0
    state.terminated[:] = torch.tensor([True, False])

    env._reset_done_envs()

    assert command.double.data_ptr() == env.metrics["double"].data_ptr()
    np.testing.assert_allclose(command.command[:, 0], [-1.0, 3.0])
    np.testing.assert_allclose(state.metrics["double"], [0.0, 0.0])


def test_read_plan_refreshes_sim_views_in_place_without_rebuilding_context() -> None:
    env = _ManagerEnv(num_envs=3)
    env.init_state()
    assert env._compiled_manager_program is not None
    program = env._compiled_manager_program
    context = program.context
    sim_views = {key: value for key, value in context.sim.items()}

    program.read_plan.read(env)
    program.read_plan.read(env, torch.tensor([0, 2], dtype=torch.int64))

    assert program.context is context
    assert all(context.sim[key] is view for key, view in sim_views.items())


def test_command_evaluation_updates_state_only_in_evaluate_kernel() -> None:
    env = _ManagerEnv()
    state = env.init_state()
    assert env._compiled_manager_program is not None
    action = env.action_terms["test"]
    assert isinstance(action, ActionTerm)
    action.state.source[:, 0] = torch.tensor([1.0, 2.0])
    derived = _counter_command(env).double
    derived.fill_(-1.0)

    env._refresh_sim_reads()
    env._task_program.observe_kernel(env.observation_manager.frame_output)
    np.testing.assert_array_equal(derived[:, 0], [-1.0, -1.0])
    np.testing.assert_array_equal(state.obs.policy[:, 2], [-1.0, -1.0])

    env._execute_evaluate_kernel()
    np.testing.assert_allclose(derived[:, 0], [2.0, 4.0])
    env._task_program.observe_kernel(env.observation_manager.frame_output)
    np.testing.assert_allclose(state.obs.policy[:, 2], [2.0, 4.0])


def test_command_term_updates_lifecycle_and_selected_reset_ids() -> None:
    env = _ManagerEnv(num_envs=3)
    state = env.init_state()
    command = _counter_command(env).command
    np.testing.assert_array_equal(command, -1.0)

    command[:] = torch.tensor([[1.0], [2.0], [3.0]])
    action = env.action_terms["test"]
    state.terminated[:] = torch.tensor([False, True, False])
    env._reset_done_envs()

    np.testing.assert_array_equal(command[:, 0], [1.0, -1.0, 3.0])
    np.testing.assert_array_equal(np.flatnonzero(action.state.reset_flags[:, 0]), [1])
    assert tuple(term_field.name for term_field in fields(env.cfg.commands)) == ("counter",)


@pytest.mark.parametrize("compiled", [False, True])
def test_reset_passes_compact_episode_reset_in_selected_lane_order(
    compiled: bool, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    # Keep these fullgraph variants independent of other manager schedules.
    torch._dynamo.reset()
    request.addfinalizer(torch._dynamo.reset)
    compile_torch = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: (
            compile_torch(fn, backend="eager", fullgraph=True, dynamic=kwargs.get("dynamic")) if compiled else fn
        ),
    )

    @dataclass(frozen=True)
    class EpisodeCommand(_CounterCommand):
        def reset(self, ctx: ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
            self.command[env_ids, 0] = torch.where(episode_reset, -1.0, -2.0)

    @configclass
    class EpisodeCommandCfg(CommandCfg):
        def __call__(self, ctx: BuildContext) -> CommandTerm:
            return EpisodeCommand(command=torch.zeros(ctx.num_envs, 1), double=torch.zeros(ctx.num_envs, 1))

    cfg = _ManagerEnvCfg(commands={"counter": EpisodeCommandCfg()})
    env = ManagerEnv(cfg, num_envs=4)
    env.init_state()
    command = env.command_terms["counter"].command
    command[:, 0] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    env.state.episode_steps[:] = torch.tensor([3, 4, 5, 6])

    env.reset(torch.tensor([3, 0]), sim_reset_ids=torch.tensor([2]))
    torch.testing.assert_close(command[:, 0], torch.tensor([-1.0, 20.0, -2.0, -1.0]))
    torch.testing.assert_close(env.state.episode_steps, torch.tensor([0, 4, 5, 0]))

    # A second pass reuses the schedule, but must not retain its previous mask.
    env.reset(torch.tensor([2]), sim_reset_ids=torch.tensor([0, 1]))
    torch.testing.assert_close(command[:, 0], torch.tensor([-2.0, -2.0, -1.0, -1.0]))

    # Overlapping public inputs reset the shared lane once as an episode.
    env.reset(torch.tensor([1]), sim_reset_ids=torch.tensor([1, 3]))
    torch.testing.assert_close(command[:, 0], torch.tensor([-2.0, -1.0, -1.0, -2.0]))


def test_done_envs_reset_before_observation_and_preserve_transition_outputs() -> None:
    env = _ManagerEnv(num_envs=3)
    state = env.init_state()
    state = env.step(torch.tensor([[0.25], [0.75], [0.25]], dtype=torch.float32))

    np.testing.assert_allclose(state.obs.policy[[0, 2], 0], 1.75)
    np.testing.assert_array_equal(state.obs.policy[1], [3.25, -1.0, 1.5, 1.0])
    np.testing.assert_allclose(state.reward, [0.005, 0.015, 0.005])
    np.testing.assert_array_equal(state.terminated, [False, True, False])
    assert env._compiled_manager_program is not None


def test_truncated_envs_reset_before_observation() -> None:
    env = _ManagerEnv(num_envs=2, max_episode_seconds=0.01)

    state = env.step(torch.full((2, 1), 0.25, dtype=torch.float32))

    np.testing.assert_array_equal(state.truncated, True)
    np.testing.assert_array_equal(state.episode_steps, 0)
    np.testing.assert_array_equal(state.obs.policy[:, 0], 1.75)
    np.testing.assert_allclose(state.reward, 0.005)


@pytest.mark.parametrize("with_value", [False, True])
def test_final_observations_capture_command_before_reset_and_survive_steps(with_value: bool) -> None:
    cfg = _ManagerEnvCfg(max_episode_seconds=0.03)
    if with_value:
        cfg.observations = {"policy": _ManagerPolicyObsCfg(), "value": _LanePolicyObsCfg()}
    env = ManagerEnv(cfg, num_envs=3)
    # No physics is needed: action/command terms encode observable state.
    env.physics_step = lambda: None
    initial = env.init_state()
    assert initial.final_obs is None
    initial.episode_steps[:] = torch.tensor([2, 0, 2])
    env.command_terms["counter"].command[:] = 2.0
    state = env.step(torch.full((3, 1), 0.25, dtype=torch.float32))
    final = state.final_obs
    assert final is not None
    assert isinstance(final.env_ids, torch.Tensor)
    np.testing.assert_array_equal(final.env_ids, [0, 2])
    # Final observations capture the post-transition (post-advance) command.
    np.testing.assert_allclose(final.obs.policy[:, 1], 2.0 + env.cfg.ctrl_dt)
    np.testing.assert_allclose(state.obs.policy[[0, 2], 1], -1.0)
    np.testing.assert_allclose(state.obs.policy[1, 1], 2.0 + env.cfg.ctrl_dt)
    if with_value:
        np.testing.assert_array_equal(final.obs.value[:, 0], [0, 2])
    else:
        assert final.obs.value is None
        assert final.obs.value_or_policy is final.obs.policy
    assert final.obs.policy.data_ptr() != state.obs.policy.data_ptr()
    saved = final.obs.policy.clone()
    next_state = env.step(torch.full((3, 1), 0.25, dtype=torch.float32))
    assert next_state.final_obs is None
    assert state.final_obs is final
    env.step(torch.full((3, 1), 0.25, dtype=torch.float32))
    np.testing.assert_array_equal(final.obs.policy, saved)


def test_final_observation_preview_consumes_shared_default_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep this lifecycle test cheap while exercising the real compiled
    # observation schedule, not a replacement sampler.
    compile_torch = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: compile_torch(fn, backend="eager", fullgraph=True, dynamic=kwargs.get("dynamic")),
    )

    def noisy(ctx: ManagerContext, out: torch.Tensor) -> None:
        out.zero_()
        add_uniform_noise(ctx, out, 0.25)

    @configclass
    class NoisyCfg(ObservationTermCfg):
        def __call__(self, ctx: BuildContext) -> ObsTerm:
            return ObsTerm(1, noisy)

    cfg = _ManagerEnvCfg(observations={"policy": {"noise": NoisyCfg()}})
    torch.manual_seed(31)
    expected_frames = [torch.rand(2, 1) * 0.5 - 0.25 for _ in range(5)]
    expected_task = torch.rand(8)
    expected_after = torch.get_rng_state().clone()
    torch.manual_seed(31)
    first, second = ManagerEnv(cfg, num_envs=2, seed=31), ManagerEnv(cfg, num_envs=2, seed=31)
    for env in (first, second):
        env.physics_step = lambda: None
        env.init_state()
    torch.testing.assert_close(first.state.obs.policy, expected_frames[0])
    torch.testing.assert_close(second.state.obs.policy, expected_frames[1])
    published = first.state.obs.policy.clone()
    before_preview = torch.get_rng_state().clone()
    final = first._capture_final_obs(torch.tensor([0], dtype=torch.long))
    assert not torch.equal(before_preview, torch.get_rng_state())
    torch.testing.assert_close(final.obs.policy, expected_frames[2][[0]])
    saved_final = final.obs.policy.clone()
    torch.testing.assert_close(first.state.obs.policy, published)
    first.compute_observation(first.state)
    torch.testing.assert_close(first.state.obs.policy, expected_frames[3])
    # Preview changes the next draw even for another environment instance.
    second.compute_observation(second.state)
    torch.testing.assert_close(second.state.obs.policy, expected_frames[4])
    torch.testing.assert_close(final.obs.policy, saved_final)
    torch.testing.assert_close(first._random.uniform((8,)), expected_task)
    assert torch.equal(expected_after, torch.get_rng_state())


def test_post_reset_observation_does_not_run_command_evaluation() -> None:
    env = _ManagerEnv(num_envs=3)
    state = env.init_state()
    assert env._compiled_manager_program is not None
    action = env.action_terms["test"]
    assert isinstance(action, ActionTerm)
    action.state.source[:, 0] = torch.tensor([0.25, 0.5, 0.75])
    state.terminated[:] = torch.tensor([False, True, False])
    env._refresh_sim_reads()
    env._reset_done_envs()
    env.reset_observation(state, env_ids=torch.tensor([1]))

    np.testing.assert_array_equal(state.obs.policy[:, 2], 0.0)
    env._refresh_sim_reads()
    env._execute_evaluate_kernel()
    env._task_program.observe_kernel(env.observation_manager.frame_output)

    np.testing.assert_allclose(state.obs.policy[:, 2], [0.5, 1.0, 1.5])


def test_duplicate_termination_config_type_is_rejected() -> None:
    @configclass
    class DuplicateTerminationsCfg(ManagerTerminationsCfg):
        first: _InjectedTerminationCfg = _InjectedTerminationCfg(threshold=0.5)
        second: _InjectedTerminationCfg = _InjectedTerminationCfg(threshold=0.75)

    cfg = _ManagerEnvCfg(terminations=DuplicateTerminationsCfg())
    with pytest.raises(ValueError, match="each configured termination type must be unique"):
        ManagerEnv(cfg)


def test_shared_arrays_in_terms_are_writable() -> None:
    env = _ManagerEnv(num_envs=2)
    injected = env.observation_groups["policy"].terms[0].term

    assert isinstance(injected, ObsTerm)
    assert injected.args
    assert torch.is_tensor(injected.args[0].offset)


def test_observation_term_must_be_a_term_with_instance_compute() -> None:
    class _InvalidObservation:
        def compute(self, env_id: int, out: torch.Tensor) -> None:
            out[0] = env_id

    @configclass(kw_only=True)
    class _InvalidObservationCfg(ObservationTermCfg):
        def size(self, env: ManagerEnv) -> int:
            del env
            return 1

        def __call__(self, env: ManagerEnv) -> ObsTerm:
            del env
            return _InvalidObservation()

    @configclass
    class _InvalidPolicyCfg(ManagerObservationGroupCfg):
        invalid: _InvalidObservationCfg = _InvalidObservationCfg()

    @configclass
    class _InvalidObservationsCfg(ManagerObservationsCfg):
        policy: _InvalidPolicyCfg = _InvalidPolicyCfg()

    env = _ManagerEnv(num_envs=1)
    cfg = ManagerBasedEnvCfg(observations=_InvalidObservationsCfg())
    with pytest.raises(TypeError, match="must return ObsTerm"):
        create_observation_groups(cfg, env.build_context)

    def _static_observation(ctx: ManagerContext, out: torch.Tensor) -> None:
        out[:, 0] = ctx.env_id

    @configclass(kw_only=True)
    class _StaticObservationCfg(ObservationTermCfg):
        def size(self, env: ManagerEnv) -> int:
            del env
            return 1

        def __call__(self, env: ManagerEnv) -> ObsTerm:
            del env
            return ObsTerm(1, _static_observation)

    @configclass
    class _StaticPolicyCfg(ManagerObservationGroupCfg):
        static: _StaticObservationCfg = _StaticObservationCfg()

    @configclass
    class _StaticObservationsCfg(ManagerObservationsCfg):
        policy: _StaticPolicyCfg = _StaticPolicyCfg()

    cfg = ManagerBasedEnvCfg(observations=_StaticObservationsCfg())
    static_groups = create_observation_groups(cfg, env.build_context)
    assert static_groups["policy"].terms[0].term.dispatch is _static_observation


def test_dispatch_term_parameter_names_are_not_constrained() -> None:
    def _renamed_reward(context: ManagerContext):
        return context.actions["test"].source[:, 0] * 2.0

    @configclass(kw_only=True)
    class _RenamedRewardCfg(RewardTermCfg):
        def __call__(self, env: ManagerEnv) -> RewardTerm:
            del env
            return RewardTerm(_renamed_reward)

    @configclass
    class _RenamedRewardsCfg(ManagerRewardsCfg):
        renamed: _RenamedRewardCfg = _RenamedRewardCfg(weight=1.0)

    env = ManagerEnv(_ManagerEnvCfg(rewards=_RenamedRewardsCfg()), num_envs=2)
    state = env.init_state()
    actions = torch.full((env.num_envs, 1), 0.5, dtype=torch.float32)
    env.apply_action(actions, state)
    env.compute_transition(state)

    np.testing.assert_allclose(state.reward, 2.0 * 0.5 * env.cfg.ctrl_dt)
