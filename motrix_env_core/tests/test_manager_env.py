# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""ManagerEnv lifecycle and term-registry contracts on the Torch executor.

The fixtures build a physics-less manager env (``physics_step`` is a no-op):
action and command terms encode all observable state, so reset semantics,
observation lifetimes, perf phases, and term validation can be asserted
without a simulator backend.
"""

from dataclasses import fields, replace

import gymnasium as gym
import numpy as np
import pytest
import torch

import motrix_env_core.manager.env as manager_env_module
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
    SharedArray,
    TerminationTerm,
    TerminationTermCfg,
    kernel_data,
    metric,
)
from motrix_env_core.manager.dispatch import dispatch  # noqa: E402
from motrix_env_core.manager.observations import create_observation_groups
from motrix_env_core.manager.rewards import create_reward_terms
from motrix_env_core.manager.terminations import TerminationManager


@kernel_data
class _TestActionState(ActionState):
    reset_flags: np.ndarray
    source: np.ndarray


@kernel_data
class _SecondActionState(ActionState):
    reset_flags: np.ndarray
    source: np.ndarray


class _TestActionTerm(ActionTerm):
    def __init__(self, action_space: gym.spaces.Box, state: _TestActionState) -> None:
        super().__init__(action_space, state)

    def _process(self, actions: np.ndarray) -> None:
        self.state.source[...] = actions

    def _reset(self, env_ids: np.ndarray) -> None:
        self.state.reset_flags.fill(False)
        self.state.reset_flags[env_ids] = True


class _SecondActionTerm(ActionTerm):
    def __init__(self, action_space: gym.spaces.Box, state: _SecondActionState) -> None:
        super().__init__(action_space, state)

    def _process(self, actions: np.ndarray) -> None:
        self.state.source[...] = actions

    def _reset(self, env_ids: np.ndarray) -> None:
        self.state.reset_flags.fill(False)
        self.state.reset_flags[env_ids] = True


def _test_action_space(_: ManagerEnv, __: np.ndarray | None) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)


def _second_action_space(_: ManagerEnv, __: np.ndarray | None) -> gym.spaces.Box:
    return gym.spaces.Box(
        np.asarray([-2.0, -3.0], dtype=np.float32),
        np.asarray([2.0, 3.0], dtype=np.float32),
    )


@kernel_data
class _ObservationParams:
    offset: SharedArray
    scale: np.float32


@dispatch
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
            _ObservationParams(np.asarray([1.0], dtype=np.float32), np.float32(self.scale)),
        )


@dispatch
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


@dispatch
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


@dispatch
def _injected_reward(ctx: ManagerContext):
    return ctx.actions["test"].source[:, 0]


@configclass(kw_only=True)
class _InjectedRewardCfg(RewardTermCfg):
    def __call__(self, env: ManagerEnv) -> RewardTerm:
        del env
        return RewardTerm(_injected_reward)


@dispatch
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
            np.float32(self.threshold),
            metric_names=("source_at_termination",),
        )


@configclass(kw_only=True)
class _TestActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] = ()

    def __call__(self, env: ManagerEnv, actuator_indices: np.ndarray | None) -> ActionTerm:
        return _TestActionTerm(
            _test_action_space(env, actuator_indices),
            _TestActionState(
                np.zeros((env.num_envs, 2, 1), dtype=np.float32),
                np.zeros(1, dtype=np.int64),
                np.zeros((env.num_envs, 1), dtype=bool),
                np.zeros((env.num_envs, 1), dtype=np.float32),
            ),
        )


@configclass(kw_only=True)
class _SecondActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] = ()

    def __call__(self, env: ManagerEnv, actuator_indices: np.ndarray | None) -> ActionTerm:
        return _SecondActionTerm(
            _second_action_space(env, actuator_indices),
            _SecondActionState(
                np.zeros((env.num_envs, 2, 2), dtype=np.float32),
                np.zeros(1, dtype=np.int64),
                np.zeros((env.num_envs, 1), dtype=bool),
                np.zeros((env.num_envs, 2), dtype=np.float32),
            ),
        )


@configclass(kw_only=True)
class _CounterCommandCfg(CommandCfg):
    def __call__(self, env: ManagerEnv) -> "_CounterCommand":
        shape = (env.num_envs, 1)
        return _CounterCommand(
            double=np.zeros(shape, dtype=np.float32),
            command=np.zeros(shape, dtype=np.float32),
        )


@kernel_data
class _CounterCommand(CommandTerm):
    double: np.ndarray = metric()

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

    state = env.step(np.ones((env.num_envs, 1), dtype=np.float32))

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
    actions = np.asarray([[0.25, 1.0, 2.0], [0.5, -1.0, -2.0]], dtype=np.float32)

    env.apply_action(actions, state)

    assert tuple(env.action_terms) == ("test", "second")
    assert env.action_slices == {"test": slice(0, 1), "second": slice(1, 3)}
    assert env.action_space.shape == (3,)
    np.testing.assert_array_equal(env.action_space.low, [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(env.action_terms["test"].state.source, actions[:, :1])
    np.testing.assert_array_equal(env.action_terms["second"].state.source, actions[:, 1:])
    env._refresh_sim_reads()
    env._execute_observe_kernel(env._kernel_inputs)
    np.testing.assert_array_equal(state.obs.policy, actions)
    assert [(term.name, term.input_slice) for term in env.manager_layout.actions] == [
        ("test", slice(0, 1)),
        ("second", slice(1, 3)),
    ]

    state.terminated[:] = [False, True]
    env._reset_done_envs()
    np.testing.assert_array_equal(env.action_terms["test"].state.reset_flags[:, 0], [False, True])
    np.testing.assert_array_equal(env.action_terms["second"].state.reset_flags[:, 0], [False, True])


def test_multiple_action_terms_validate_total_action_shape() -> None:
    env = _ManagerEnv(num_envs=2)
    state = env.init_state()

    with pytest.raises(ValueError, match=r"Expected action shape \(2, 1\), got \(2, 2\)"):
        env.apply_action(np.zeros((2, 2), dtype=np.float32), state)


def test_rand_value_owns_seeded_per_environment_states() -> None:
    first = _ManagerEnv(num_envs=3, rand_seed=7)._rand.state
    reproduced = _ManagerEnv(num_envs=3, rand_seed=7)._rand.state
    different = _ManagerEnv(num_envs=3, rand_seed=8)._rand.state

    assert first.shape == (3, 1)
    assert first.dtype == np.uint64
    np.testing.assert_array_equal(first, reproduced)
    assert not np.array_equal(first, different)


def test_step_perf_records_standard_and_manager_phases() -> None:
    env = _ManagerEnv()
    state = env.init_state()
    assert state.episode_steps.shape == (env.num_envs,)
    assert set(state.reward_terms) == {"source"}
    actions = np.zeros((env.num_envs, 1), dtype=np.float32)

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
    initial_snapshot = initial_policy.copy()

    first = env.step(np.full((env.num_envs, 1), 0.25, dtype=np.float32))

    np.testing.assert_array_equal(initial_policy, initial_snapshot)
    assert not np.shares_memory(initial_policy, first.obs.policy)
    first_policy = first.obs.policy
    first_snapshot = first_policy.copy()

    second = env.step(np.full((env.num_envs, 1), 0.5, dtype=np.float32))

    np.testing.assert_array_equal(first_policy, first_snapshot)
    assert not np.shares_memory(first_policy, second.obs.policy)
    assert np.shares_memory(initial_policy, second.obs.policy)


def test_manager_context_is_injected_once_and_reused_across_all_term_kinds() -> None:
    env = _ManagerEnv()
    state = env.init_state()

    action = env.action_terms["test"]
    assert isinstance(action, ActionTerm)
    command = _counter_command(env)
    context_sources = [slot for slot in env.manager_layout.inputs if "manager_context" in slot.source]
    scopes = {slot.source: slot.scope.value for slot in context_sources}
    assert scopes["manager_context.actions.test.action_queue"] == "per_env"
    assert scopes["manager_context.actions.test.action_ptr"] == "shared"
    assert [term.output_slice for term in env.manager_layout.observations["policy"].terms] == [
        slice(0, 2),
        slice(2, 3),
        slice(3, 4),
    ]

    action.state.source[:, 0] = [0.25, 0.75]
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

    warmup = env.warmup()
    assert warmup.signatures


def test_read_plan_uses_compile_time_flat_inputs_without_runtime_flattening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _ManagerEnv(num_envs=3)
    env.init_state()
    assert env._compiled_manager_program is not None
    read_plan = env._compiled_manager_program.read_plan
    inputs = read_plan.read(env)

    def fail_runtime_flatten(value: object) -> tuple[object, ...]:
        del value
        raise AssertionError("flatten_kernel_data() must not run after manager compilation")

    monkeypatch.setattr(manager_env_module, "flatten_kernel_data", fail_runtime_flatten)

    assert read_plan.read(env) is inputs
    assert read_plan.read(env, np.asarray([0, 2], dtype=np.int64)) is inputs


def test_command_evaluation_updates_state_only_in_evaluate_kernel() -> None:
    env = _ManagerEnv()
    state = env.init_state()
    assert env._compiled_manager_program is not None
    action = env.action_terms["test"]
    assert isinstance(action, ActionTerm)
    action.state.source[:, 0] = [1.0, 2.0]
    derived = _counter_command(env).double
    derived.fill(-1.0)

    env._refresh_sim_reads()
    inputs = env._kernel_inputs
    env._execute_observe_kernel(inputs)
    np.testing.assert_array_equal(derived[:, 0], [-1.0, -1.0])
    np.testing.assert_array_equal(state.obs.policy[:, 2], [-1.0, -1.0])

    env._execute_evaluate_kernel(inputs)
    np.testing.assert_allclose(derived[:, 0], [2.0, 4.0])
    env._execute_observe_kernel(inputs)
    np.testing.assert_allclose(state.obs.policy[:, 2], [2.0, 4.0])


def test_command_term_updates_lifecycle_and_selected_reset_ids() -> None:
    env = _ManagerEnv(num_envs=3)
    state = env.init_state()
    command = _counter_command(env).command
    np.testing.assert_array_equal(command, -1.0)

    command[:] = [[1.0], [2.0], [3.0]]
    action = env.action_terms["test"]
    state.terminated[:] = [False, True, False]
    env._reset_done_envs()

    np.testing.assert_array_equal(command[:, 0], [1.0, -1.0, 3.0])
    np.testing.assert_array_equal(np.flatnonzero(action.state.reset_flags[:, 0]), [1])
    assert tuple(term_field.name for term_field in fields(env.cfg.commands)) == ("counter",)


def test_done_envs_reset_before_observation_and_preserve_transition_outputs() -> None:
    env = _ManagerEnv(num_envs=3)
    state = env.init_state()
    state = env.step(np.asarray([[0.25], [0.75], [0.25]], dtype=np.float32))

    np.testing.assert_allclose(state.obs.policy[[0, 2], 0], 1.75)
    np.testing.assert_array_equal(state.obs.policy[1], [3.25, -1.0, 1.5, 1.0])
    np.testing.assert_allclose(state.reward, [0.005, 0.015, 0.005])
    np.testing.assert_array_equal(state.terminated, [False, True, False])
    assert env._compiled_manager_program is not None


def test_truncated_envs_reset_before_observation() -> None:
    env = _ManagerEnv(num_envs=2, max_episode_seconds=0.01)

    state = env.step(np.asarray([[0.25], [0.25]], dtype=np.float32))

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
    state = env.step(np.full((3, 1), 0.25, dtype=np.float32))
    final = state.final_obs
    assert final is not None
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
    assert not np.shares_memory(final.obs.policy, state.obs.policy)
    saved = final.obs.policy.copy()
    next_state = env.step(np.full((3, 1), 0.25, dtype=np.float32))
    assert next_state.final_obs is None
    assert state.final_obs is final
    env.step(np.full((3, 1), 0.25, dtype=np.float32))
    np.testing.assert_array_equal(final.obs.policy, saved)


def test_final_observation_noise_uses_shared_observation_rng() -> None:
    env = _ManagerEnv()
    env.init_state()
    rng_before = env._rand.state.copy()
    observe = env._task_program.observe_kernel

    def noisy_observe(inputs, outputs):
        observe(inputs, outputs)
        env._rand.state[:] += np.uint64(1)
        outputs.policy_obs[:, 0] = env._rand.state[:, 0].astype(np.float32)

    env._task_program = replace(env._task_program, observe_kernel=noisy_observe)
    final = env._capture_final_obs(np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(env._rand.state, rng_before + np.uint64(1))
    saved_final = final.obs.policy.copy()
    np.testing.assert_array_equal(final.obs.policy[:, 0], (rng_before[[0], 0] + np.uint64(1)).astype(np.float32))
    env._execute_observe_kernel(env._kernel_inputs)
    np.testing.assert_array_equal(env._rand.state, rng_before + np.uint64(2))
    np.testing.assert_array_equal(final.obs.policy, saved_final)


def test_post_reset_observation_does_not_run_command_evaluation() -> None:
    env = _ManagerEnv(num_envs=3)
    state = env.init_state()
    assert env._compiled_manager_program is not None
    action = env.action_terms["test"]
    assert isinstance(action, ActionTerm)
    action.state.source[:, 0] = [0.25, 0.5, 0.75]
    state.terminated[:] = [False, True, False]
    env._refresh_sim_reads()
    env._reset_done_envs()
    env.compute_observation(state)

    np.testing.assert_array_equal(state.obs.policy[:, 2], 0.0)
    env._refresh_sim_reads()
    env._execute_evaluate_kernel(env._kernel_inputs)
    env._execute_observe_kernel(env._kernel_inputs)

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
    assert injected.args[0].offset.flags.writeable


def test_observation_term_must_be_kernel_data_with_instance_compute() -> None:
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


def test_reward_term_requires_dispatch_descriptor() -> None:
    class _InvalidReward:
        pass

    @configclass(kw_only=True)
    class _InvalidRewardCfg(RewardTermCfg):
        def __call__(self, env: ManagerEnv) -> RewardTerm:
            del env
            return _InvalidReward()

    env = _ManagerEnv(num_envs=1)
    with pytest.raises(TypeError, match="must return RewardTerm"):
        create_reward_terms({"invalid": _InvalidRewardCfg(weight=1.0)}, env)

    def _undecorated_reward(ctx: ManagerContext) -> float:
        del ctx
        return 1.0

    @configclass(kw_only=True)
    class _UndecoratedRewardCfg(RewardTermCfg):
        def __call__(self, env: ManagerEnv) -> RewardTerm:
            del env
            return RewardTerm(_undecorated_reward)

    with pytest.raises(TypeError, match="must be decorated with @dispatch"):
        create_reward_terms({"undecorated": _UndecoratedRewardCfg(weight=1.0)}, env)


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
    actions = np.full((env.num_envs, 1), 0.5, dtype=np.float32)
    env.apply_action(actions, state)
    env.compute_transition(state)

    np.testing.assert_allclose(state.reward, 2.0 * 0.5 * env.cfg.ctrl_dt)


def test_termination_term_requires_dispatch_descriptor() -> None:
    class _InvalidTermination:
        pass

    @configclass(kw_only=True)
    class _InvalidTerminationCfg(TerminationTermCfg):
        def __call__(self, env: ManagerEnv) -> TerminationTerm:
            del env
            return _InvalidTermination()

    env = _ManagerEnv(num_envs=1)
    with pytest.raises(TypeError, match="must return TerminationTerm"):
        TerminationManager({"invalid": _InvalidTerminationCfg()}, env)

    def _undecorated_termination(ctx: ManagerContext) -> bool:
        del ctx
        return False

    @configclass(kw_only=True)
    class _UndecoratedTerminationCfg(TerminationTermCfg):
        def __call__(self, env: ManagerEnv) -> TerminationTerm:
            del env
            return TerminationTerm(_undecorated_termination)

    with pytest.raises(TypeError, match="must be decorated with @dispatch"):
        TerminationManager({"undecorated": _UndecoratedTerminationCfg()}, env)
