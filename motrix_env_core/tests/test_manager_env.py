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
from motrix_env_core.manager.dispatch import dispatch  # noqa: E402
from motrix_env_core.manager.observations import create_observation_groups


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

    def __call__(self, env: ManagerEnv, actuator_indices: np.ndarray | None) -> ActionTerm:
        return _TestActionTerm(
            _test_action_space(env, actuator_indices),
            _TestActionState(
                torch.from_numpy(np.zeros((env.num_envs, 2, 1), dtype=np.float32)),
                torch.from_numpy(np.zeros(1, dtype=np.int64)),
                torch.from_numpy(np.zeros((env.num_envs, 1), dtype=bool)),
                torch.from_numpy(np.zeros((env.num_envs, 1), dtype=np.float32)),
            ),
        )


@configclass(kw_only=True)
class _SecondActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] = ()

    def __call__(self, env: ManagerEnv, actuator_indices: np.ndarray | None) -> ActionTerm:
        return _SecondActionTerm(
            _second_action_space(env, actuator_indices),
            _SecondActionState(
                torch.from_numpy(np.zeros((env.num_envs, 2, 2), dtype=np.float32)),
                torch.from_numpy(np.zeros(1, dtype=np.int64)),
                torch.from_numpy(np.zeros((env.num_envs, 1), dtype=bool)),
                torch.from_numpy(np.zeros((env.num_envs, 2), dtype=np.float32)),
            ),
        )


@configclass(kw_only=True)
class _CounterCommandCfg(CommandCfg):
    def __call__(self, env: ManagerEnv) -> "_CounterCommand":
        shape = (env.num_envs, 1)
        return _CounterCommand(
            double=torch.zeros(shape, dtype=torch.float32),
            command=torch.zeros(shape, dtype=torch.float32),
        )


@dataclass(frozen=True, slots=True)
class _CounterCommand(CommandTerm):
    double: torch.Tensor = metric()

    @staticmethod
    @dispatch
    def update(receiver, ctx: ManagerContext) -> None:
        receiver.double[:, 0] = 2.0 * ctx.actions["test"].source[:, 0]

    def reset(self, ctx) -> None:
        self.command[ctx.env_ids] = -1.0

    @staticmethod
    @dispatch
    def advance(receiver, ctx: ManagerContext) -> None:
        receiver.command[:, 0] += ctx.dt

    @staticmethod
    @dispatch
    def reset_env(receiver, ctx: ManagerContext) -> None:
        receiver.command[:, 0] = -1.0


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


def test_action_and_command_terms_are_environment_owned() -> None:
    env = _ManagerEnv(num_envs=3)

    assert isinstance(env.action_terms["test"], ActionTerm)
    assert env.action_terms is env._action_terms
    assert env.action_terms["test"].state.reset_flags.shape == (3, 1)
    assert _counter_command(env) is env.command_terms["counter"]


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
    @dispatch
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
    env._execute_observe_kernel()
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


def test_rand_value_owns_seeded_per_environment_states() -> None:
    first = _ManagerEnv(num_envs=3, rand_seed=7)._rand.state
    reproduced = _ManagerEnv(num_envs=3, rand_seed=7)._rand.state
    different = _ManagerEnv(num_envs=3, rand_seed=8)._rand.state

    # The state is a signed-int64 tensor view over the per-lane uint64 bits.
    assert tuple(first.shape) == (3, 1)
    assert first.dtype == torch.int64
    np.testing.assert_array_equal(first.numpy().view(np.uint64), reproduced.numpy().view(np.uint64))
    assert not np.array_equal(first.numpy(), different.numpy())


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
        "command_on_transition",
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
    env.compute_observation(state)

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

    assert command.double.data_ptr() == torch.from_numpy(env.metrics["double"]).data_ptr()
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
    program.read_plan.read(env, np.asarray([0, 2], dtype=np.int64))

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
    env._execute_observe_kernel()
    np.testing.assert_array_equal(derived[:, 0], [-1.0, -1.0])
    np.testing.assert_array_equal(state.obs.policy[:, 2], [-1.0, -1.0])

    env._execute_evaluate_kernel()
    np.testing.assert_allclose(derived[:, 0], [2.0, 4.0])
    env._execute_observe_kernel()
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
    initial.episode_steps[:] = [2, 0, 2]
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


def test_final_observation_noise_uses_shared_observation_rng() -> None:
    env = _ManagerEnv()
    env.init_state()
    rng_before = env._rand.state.clone()
    observe = env._task_program.observe_kernel

    def noisy_observe(outputs):
        observe(outputs)
        env._rand.state[:] += np.uint64(1)
        outputs.policy_obs[:, 0] = env._rand.state[:, 0].to(torch.float32)

    env._task_program = replace(env._task_program, observe_kernel=noisy_observe)
    final = env._capture_final_obs(np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(
        env._rand.state.numpy().view(np.uint64), rng_before.numpy().view(np.uint64) + np.uint64(1)
    )
    saved_final = final.obs.policy.clone()
    np.testing.assert_array_equal(
        final.obs.policy[:, 0],
        (rng_before[[0], 0] + 1).to(torch.float32).numpy(),
    )
    env._execute_observe_kernel()
    np.testing.assert_array_equal(env._rand.state, rng_before + np.uint64(2))
    np.testing.assert_array_equal(final.obs.policy, saved_final)


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
    env.compute_observation(state)

    np.testing.assert_array_equal(state.obs.policy[:, 2], 0.0)
    env._refresh_sim_reads()
    env._execute_evaluate_kernel()
    env._execute_observe_kernel()

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
        create_observation_groups(cfg, env)

    @dispatch
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
    static_groups = create_observation_groups(cfg, env)
    assert static_groups["policy"].terms[0].term.dispatch is _static_observation


def test_dispatch_term_parameter_names_are_not_constrained() -> None:
    @dispatch
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
