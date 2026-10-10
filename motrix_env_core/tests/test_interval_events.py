# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Interval-event lifecycle and compiled dispatch contracts on an in-memory backend."""

import numpy as np
import pytest
from test_manager_sim_backend import (
    _FakeBackend,
    _FakeManagerCfg,
    _FakeReadProgram,
    _FakeWriteCompiler,
    _FakeWriteProgram,
)

from motrix_env_core.config import configclass
from motrix_env_core.mdp.events import RandomVelocityKickCfg
from motrix_env_core.numba.kernel_data import Map
from motrix_env_core.numba.manager.context import BuildContext, ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.manager.env import ManagerEnv
from motrix_env_core.numba.manager.events import IntervalEvent, IntervalEventCfg
from motrix_env_core.numba.manager.rewards import RewardTerm, RewardTermCfg
from motrix_env_core.numba.manager.terminations import TerminationTerm, TerminationTermCfg
from motrix_env_core.sim.registry import register_sim_backend
from motrix_env_core.sim.write import AddBodyLinearVelocityWrite


class _EventWriteProgram(_FakeWriteProgram):
    def __init__(self, backend, writes, reset):
        super().__init__(backend, writes, reset)
        self._event_writes = {
            name: write for name, write in writes.items() if isinstance(write, AddBodyLinearVelocityWrite)
        }
        for name, write in self._event_writes.items():
            assert write.bodies == ("robot",)
            self._buffers[name] = np.zeros((backend.num_envs, 1, 3), dtype=np.float32)

    def execute(self, env_ids=None):
        super().execute(env_ids)
        ids = np.arange(self._backend.num_envs, dtype=np.int64) if env_ids is None else env_ids
        if self._reset:
            self._backend.body_velocity[ids] = 0.0
        for name in self._event_writes:
            delta = self.buffer(name)[ids].copy()
            self._backend.event_writes.append((ids.copy(), delta))
            self._backend.body_velocity[ids] += delta[:, 0]


class _EventWriteCompiler(_FakeWriteCompiler):
    def compile(self, writes, *, reset=False, forward_kinematics=True):
        if any(isinstance(write, AddBodyLinearVelocityWrite) for write in writes.values()):
            self._backend.event_compile_options.append((reset, forward_kinematics))
        return _EventWriteProgram(self._backend, writes, reset)


class _EventReadProgram(_FakeReadProgram):
    def execute(self, env_ids=None):
        self._backend.read_calls.append(None if env_ids is None else env_ids.copy())
        super().execute(env_ids)


class _EventBackend(_FakeBackend):
    name = "fake-interval-events"

    def __init__(self, scene, sim, num_envs):
        super().__init__(scene, sim, num_envs)
        self.body_velocity = np.zeros((num_envs, 3), dtype=np.float32)
        self.event_writes = []
        self.event_compile_options = []
        self.read_calls = []
        self.physics_velocities = []
        self._write_compiler = _EventWriteCompiler(self)

    def compile_reads(self, queries):
        return _EventReadProgram(self, queries)

    def step(self, substeps):
        self.step_calls.append(substeps)
        self.physics_velocities.append(self.body_velocity.copy())
        self.dof_pos += self.body_velocity[:, :2] * np.float32(self._sim_dt * substeps)


register_sim_backend("fake-interval-events", lambda: _EventBackend)


@dispatch
def _selected_row_event(ctx: ManagerContext, writes: Map[np.ndarray]) -> None:
    writes["delta"][0, 0] = ctx.env_id + 1
    writes["delta"][0, 1] = ctx.sim["dof_pos"][1]
    writes["delta"][0, 2] = 0.0


@configclass(kw_only=True)
class _SelectedRowEventCfg(IntervalEventCfg):
    def __call__(self, ctx: BuildContext) -> IntervalEvent:
        del ctx
        return IntervalEvent(_selected_row_event, writes={"delta": AddBodyLinearVelocityWrite(("robot",))})


@dispatch
def _position_reward(ctx: ManagerContext) -> float:
    return ctx.sim["dof_pos"][0]


@configclass(kw_only=True)
class _PositionRewardCfg(RewardTermCfg):
    weight: float = 1.0

    def __call__(self, env):
        del env
        return RewardTerm(_position_reward)


@dispatch
def _position_termination(ctx: ManagerContext, threshold: float) -> bool:
    return ctx.sim["dof_pos"][0] > threshold


@configclass(kw_only=True)
class _PositionTerminationCfg(TerminationTermCfg):
    threshold: float

    def __call__(self, env):
        del env
        return TerminationTerm(_position_termination, self.threshold)


def _kick(interval=(0.1, 0.3), delta=((-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0))):
    return RandomVelocityKickCfg(interval_range_s=interval, body="robot", velocity_delta_range=delta)


def _make_env(events, *, seed=17, rewards=None, terminations=None):
    cfg = _FakeManagerCfg(interval_events=events)
    if rewards is not None:
        cfg.rewards = rewards
    if terminations is not None:
        cfg.terminations = terminations
    env = ManagerEnv(cfg, num_envs=3, backend="fake-interval-events", seed=seed)
    # The fake physics uses the configured timestep, not a task-specific constant.
    env.sim._sim_dt = cfg.ctrl_dt / cfg.sim_substeps
    env.init_state()
    return env


@pytest.mark.parametrize("interval", [(0.0, 1.0), (-1.0, 1.0), (2.0, 1.0), (1.0, np.inf), (np.nan, 1.0)])
def test_interval_bounds_are_validated_at_configuration_boundary(interval):
    with pytest.raises(ValueError, match="intervals"):
        _make_env({"kick": _kick(interval=interval)})


@pytest.mark.parametrize(
    "bounds",
    [((1.0, -1.0), (0.0, 0.0), (0.0, 0.0)), ((0.0, np.inf), (0.0, 0.0), (0.0, 0.0))],
)
def test_velocity_delta_bounds_are_validated_at_configuration_boundary(bounds):
    with pytest.raises(ValueError, match="Velocity delta ranges"):
        _make_env({"kick": _kick(delta=bounds)})


def test_interval_scheduling_dispatches_only_due_rows_and_resamples_their_timers():
    env = _make_env({"selected": _SelectedRowEventCfg(interval_range_s=(0.5, 0.5))})
    runtime = env.interval_event_manager.events["selected"]
    runtime.remaining_s[:] = [0.25, 0.75, 0.125]
    runtime.buffers[0][:] = 99.0
    rng_before = env._rand.state.copy()
    reads_before = len(env.sim.read_calls)

    env.interval_event_manager.apply(0.25)

    assert env.sim.event_compile_options == [(False, False)]
    assert len(env.sim.event_writes) == 1
    ids, deltas = env.sim.event_writes[0]
    np.testing.assert_array_equal(ids, [0, 2])
    np.testing.assert_array_equal(deltas[:, 0], [[1.0, -0.5, 0.0], [3.0, -0.5, 0.0]])
    np.testing.assert_array_equal(runtime.buffers[0][1], [[99.0, 99.0, 99.0]])
    np.testing.assert_array_equal(env.sim.body_velocity[1], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(runtime.remaining_s, [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(env._rand.state[1], rng_before[1])
    assert np.all(env._rand.state[[0, 2]] != rng_before[[0, 2]])
    assert len(env.sim.read_calls) == reads_before


def test_no_due_event_skips_kernel_write_and_read(monkeypatch):
    env = _make_env({"kick": _kick(interval=(1.0, 1.0))})
    runtime = env.interval_event_manager.events["kick"]
    runtime.remaining_s[:] = [0.5, 0.75, 1.0]
    rng_before = env._rand.state.copy()
    buffers_before = runtime.buffers[0].copy()

    def unexpected_call(*args, **kwargs):
        pytest.fail("An event with no due rows must not execute a callback, write, or read")

    monkeypatch.setattr(runtime, "kernel", unexpected_call)
    monkeypatch.setattr(runtime.program, "execute", unexpected_call)
    monkeypatch.setattr(env.sim_data, "execute", unexpected_call)
    env.interval_event_manager.apply(0.25)

    np.testing.assert_array_equal(runtime.remaining_s, [0.25, 0.5, 0.75])
    np.testing.assert_array_equal(runtime.buffers[0], buffers_before)
    np.testing.assert_array_equal(env._rand.state, rng_before)
    assert not env.sim.event_writes


def test_partial_episode_reset_resamples_only_episode_rows_not_sim_only_rows():
    env = _make_env({"kick": _kick(interval=(0.5, 0.75))})
    runtime = env.interval_event_manager.events["kick"]
    runtime.remaining_s[:] = [0.1, 0.2, 0.3]
    runtime.buffers[0][:] = 42.0
    rng_before = env._rand.state.copy()
    env.sim.reset_calls.clear()
    env._state.terminated[:] = [False, True, False]
    env._state.episode_steps[:] = [5, 6, 7]
    env._sim_reset_requested[:, 0] = [False, True, True]

    env._reset_done_envs()

    np.testing.assert_array_equal(runtime.remaining_s[[0, 2]], np.float32([0.1, 0.3]))
    assert 0.5 <= runtime.remaining_s[1] <= 0.75
    np.testing.assert_array_equal(env._rand.state[[0, 2]], rng_before[[0, 2]])
    assert env._rand.state[1, 0] != rng_before[1, 0]
    np.testing.assert_array_equal(runtime.buffers[0], np.full((3, 1, 3), 42.0, dtype=np.float32))
    np.testing.assert_array_equal(env._state.episode_steps, [5, 0, 7])
    assert len(env.sim.reset_calls) == 1
    np.testing.assert_array_equal(env.sim.reset_calls[0][0], [1, 2])
    assert not env.sim.event_writes


def test_random_kicks_and_timers_are_seeded_and_lane_independent():
    envs = [_make_env({"kick": _kick()}, seed=seed) for seed in (123, 123, 124)]
    first, second, different = envs
    first_event = first.interval_event_manager.events["kick"]
    np.testing.assert_array_equal(first_event.remaining_s, second.interval_event_manager.events["kick"].remaining_s)
    assert not np.array_equal(first_event.remaining_s, different.interval_event_manager.events["kick"].remaining_s)
    assert np.unique(first_event.remaining_s).size == 3

    for _ in range(8):
        for env in envs:
            env.interval_event_manager.apply(0.125)
        np.testing.assert_array_equal(first_event.remaining_s, second.interval_event_manager.events["kick"].remaining_s)
        np.testing.assert_array_equal(first.sim.body_velocity, second.sim.body_velocity)
        np.testing.assert_array_equal(first._rand.state, second._rand.state)

    assert len(first.sim.event_writes) == len(second.sim.event_writes) > 1
    for (ids, delta), (other_ids, other_delta) in zip(first.sim.event_writes, second.sim.event_writes):
        np.testing.assert_array_equal(ids, other_ids)
        np.testing.assert_array_equal(delta, other_delta)
        assert np.all(delta[:, 0] >= [-2.0, -3.0, -4.0])
        assert np.all(delta[:, 0] <= [2.0, 3.0, 4.0])
    assert not np.array_equal(first.sim.body_velocity, different.sim.body_velocity)


def test_multiple_events_on_same_body_accumulate_without_replacing_velocity():
    env = _make_env(
        {
            "first": _kick(interval=(0.5, 0.5), delta=((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))),
            "second": _kick(interval=(0.75, 0.75), delta=((4.0, 4.0), (-2.0, -2.0), (1.0, 1.0))),
        }
    )
    env.sim.body_velocity[:] = [10.0, 20.0, 30.0]
    env.interval_event_manager.events["first"].remaining_s[:] = [0.1, 1.0, 0.1]
    env.interval_event_manager.events["second"].remaining_s[:] = [0.1, 0.1, 1.0]

    env.interval_event_manager.apply(0.1)

    np.testing.assert_array_equal(env.sim.body_velocity, [[15.0, 20.0, 34.0], [14.0, 18.0, 31.0], [11.0, 22.0, 33.0]])
    assert len(env.sim.event_writes) == 2
    np.testing.assert_array_equal(env.sim.event_writes[0][0], [0, 2])
    np.testing.assert_array_equal(env.sim.event_writes[1][0], [0, 1])


def test_warmup_and_compile_preserve_event_rng_timers_and_write_buffers():
    env = _make_env({"kick": _kick()})
    control = _make_env({"kick": _kick()})
    runtime = env.interval_event_manager.events["kick"]
    rng_before = env._rand.state.copy()
    timers_before = runtime.remaining_s.copy()
    runtime.buffers[0][:] = 37.0
    buffers_before = runtime.buffers[0].copy()

    result = env.warmup()
    env.compile()

    assert result.signatures
    assert runtime.kernel.nopython_signatures
    np.testing.assert_array_equal(env._rand.state, rng_before)
    np.testing.assert_array_equal(runtime.remaining_s, timers_before)
    np.testing.assert_array_equal(runtime.buffers[0], buffers_before)
    assert not env.sim.event_writes
    assert not env.sim.step_calls
    for candidate in (env, control):
        candidate.interval_event_manager.apply(0.3)
    np.testing.assert_array_equal(env.sim.body_velocity, control.sim.body_velocity)
    np.testing.assert_array_equal(runtime.remaining_s, control.interval_event_manager.events["kick"].remaining_s)


def test_events_run_before_physics_and_feed_reward_observation_and_termination():
    cfg_event = _kick(interval=(1.0, 1.0), delta=((2.0, 2.0), (3.0, 3.0), (0.0, 0.0)))
    # One kicked lane terminates; another survives to expose its post-physics observation.
    probe = _FakeManagerCfg()
    env = _make_env(
        {"kick": cfg_event},
        rewards={"position": _PositionRewardCfg()},
        terminations={"position": _PositionTerminationCfg(threshold=0.5 + probe.ctrl_dt)},
    )
    env.sim.dof_pos[2, 0] = 0.0
    env._refresh_sim_reads()
    env.sim.read_calls.clear()
    env.interval_event_manager.events["kick"].remaining_s[:] = [0.0, 1.0, 0.0]

    state = env.step(np.zeros((3, 1), dtype=np.float32))

    np.testing.assert_array_equal(env.sim.physics_velocities[0], [[2.0, 3.0, 0.0], [0.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    expected_position = np.asarray([0.5 + 2 * env.cfg.ctrl_dt, 0.5, 2 * env.cfg.ctrl_dt], dtype=np.float32)
    np.testing.assert_allclose(state.reward, expected_position * env.cfg.ctrl_dt)
    np.testing.assert_array_equal(state.terminated, [True, False, False])
    np.testing.assert_array_equal(state.truncated, [False, False, False])
    np.testing.assert_allclose(state.obs.policy[0], [0.5, -0.5])
    np.testing.assert_allclose(state.obs.policy[2], [2 * env.cfg.ctrl_dt, -0.5 + 3 * env.cfg.ctrl_dt])
    np.testing.assert_array_equal(state.episode_steps, [0, 1, 1])
    # One normal transition read and one selected reset read, never an event read.
    assert len(env.sim.read_calls) == 2
    assert env.sim.read_calls[0] is None
    np.testing.assert_array_equal(env.sim.read_calls[1], [0])
