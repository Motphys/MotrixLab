# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the ArrayEnv hierarchy and final-observation lifecycle."""

import gymnasium as gym
import numpy as np
import pytest

from motrix_env_core.array.env import ArrayEnv, ArrayEnvState, FinalObs, NpObs
from motrix_env_core.base import EnvCfg, ObsSpace
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.numba.manager.env import ManagerEnv
from motrix_env_core.registry import _infer_sim_backend
from motrix_env_core.sim.backend import RenderConfig, SimRenderer


class _CounterEnv(ArrayEnv):
    """Exercise the frontend lifecycle without constructing a simulator backend."""

    def __init__(self, *, with_value: bool = False):
        super().__init__(EnvCfg(scene=SceneCfg(), ctrl_dt=1.0, max_episode_seconds=3.0), num_envs=4)
        policy_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        value_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32) if with_value else None
        self._observation_space = ObsSpace(policy=policy_space, value=value_space)
        self._action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.counters = np.zeros(self.num_envs, dtype=np.float32)
        self.next_terminated = np.zeros(self.num_envs, dtype=bool)
        self.reset_rows: list[np.ndarray] = []
        self.physics_steps = 0
        self.observation_calls = 0

    @property
    def observation_space(self) -> ObsSpace:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def reset(self, env_ids: np.ndarray) -> None:
        self.reset_rows.append(env_ids.copy())
        self.counters[env_ids] = 0.0

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
        return state

    def physics_step(self) -> None:
        self.physics_steps += 1
        self.counters += 1.0

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        state.reward[:] = self.counters
        state.terminated[:] = self.next_terminated
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        self.observation_calls += 1
        state.obs.policy[:, 0] = self.counters
        if state.obs.value is not None:
            state.obs.value[:, 0] = self.counters
            state.obs.value[:, 1] = -self.counters
        return state

    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        raise NotImplementedError("The counter environment has no simulator or renderer")


class _FinalCounterEnv(_CounterEnv):
    def __init__(self, *, with_value: bool = False):
        super().__init__(with_value=with_value)
        self.capture_rows: list[np.ndarray] = []
        self.capture_episode_steps: list[np.ndarray] = []
        self.capture_rewards: list[np.ndarray] = []

    def _capture_final_obs(self, env_ids: np.ndarray) -> FinalObs:
        self.capture_rows.append(env_ids.copy())
        self.capture_episode_steps.append(self.state.episode_steps[env_ids].copy())
        self.capture_rewards.append(self.state.reward[env_ids].copy())
        policy = self.counters[env_ids, None].copy()
        value = np.concatenate((policy, -policy), axis=1) if self.has_value_observation else None
        return FinalObs(env_ids=env_ids.copy(), obs=NpObs(policy=policy, value=value))


def _step(env: _CounterEnv) -> ArrayEnvState:
    return env.step(np.zeros((env.num_envs, 1), dtype=np.float32))


def test_np_env_inherits_array_env() -> None:
    assert issubclass(DirectEnv, ArrayEnv)


def test_manager_env_is_array_env_sibling_of_np_env() -> None:
    assert issubclass(ManagerEnv, ArrayEnv)
    assert not issubclass(ManagerEnv, DirectEnv)


def test_registry_infers_np_backend_for_manager_env() -> None:
    assert _infer_sim_backend(ManagerEnv) == "np"
    assert _infer_sim_backend(DirectEnv) == "np"


@pytest.mark.parametrize("with_value", [False, True])
def test_final_obs_absent_on_init_and_nontruncated_step(with_value: bool) -> None:
    env = _FinalCounterEnv(with_value=with_value)
    initial = env.init_state()

    assert initial.final_obs is None
    np.testing.assert_array_equal(initial.obs.policy, np.zeros((4, 1)))
    np.testing.assert_array_equal(env.reset_rows[0], np.arange(4))
    assert env.physics_steps == 0
    assert env.observation_calls == 1
    assert env.capture_rows == []

    state = _step(env)

    assert state.final_obs is None
    assert not state.truncated.any()
    assert not state.terminated.any()
    np.testing.assert_array_equal(state.episode_steps, np.ones(4))
    np.testing.assert_array_equal(state.obs.policy, np.ones((4, 1)))
    assert env.physics_steps == 1
    assert env.observation_calls == 2
    assert len(env.reset_rows) == 1
    assert env.capture_rows == []


@pytest.mark.parametrize("with_value", [False, True])
def test_final_obs_contains_only_truncated_rows_before_auto_reset(with_value: bool) -> None:
    env = _FinalCounterEnv(with_value=with_value)
    env.init_state()
    env.state.episode_steps[:] = [2, 0, 2, 0]
    env.counters[:] = [10, 20, 30, 40]
    env.next_terminated[1] = True

    state = _step(env)

    final = state.final_obs
    assert isinstance(final, FinalObs)
    assert isinstance(final.obs, NpObs)
    np.testing.assert_array_equal(state.truncated, [True, False, True, False])
    np.testing.assert_array_equal(state.terminated, [False, True, False, False])
    np.testing.assert_array_equal(final.env_ids, [0, 2])
    np.testing.assert_array_equal(final.obs.policy, [[11], [31]])
    np.testing.assert_array_equal(env.capture_rows[0], [0, 2])
    np.testing.assert_array_equal(env.capture_episode_steps[0], [3, 3])
    np.testing.assert_array_equal(env.capture_rewards[0], [11, 31])
    np.testing.assert_array_equal(env.reset_rows[-1], [0, 1, 2])
    np.testing.assert_array_equal(state.episode_steps, [0, 0, 0, 1])
    np.testing.assert_array_equal(state.obs.policy, [[0], [0], [0], [41]])
    assert env.observation_calls == 2
    assert not np.shares_memory(final.obs.policy, env.counters)
    assert not np.shares_memory(final.obs.policy, state.obs.policy)
    if with_value:
        np.testing.assert_array_equal(final.obs.value, [[11, -11], [31, -31]])
        np.testing.assert_array_equal(state.obs.value, [[0, 0], [0, 0], [0, 0], [41, -41]])
        assert final.obs.value_or_policy is final.obs.value
        assert not np.shares_memory(final.obs.value, state.obs.value)
    else:
        assert final.obs.value is None
        assert state.obs.value is None
        assert final.obs.value_or_policy is final.obs.policy


@pytest.mark.parametrize("with_value", [False, True])
def test_retained_final_obs_survives_later_steps_and_init_state(with_value: bool) -> None:
    env = _FinalCounterEnv(with_value=with_value)
    env.init_state()
    env.state.episode_steps[:] = [2, 0, 2, 0]
    env.counters[:] = [10, 20, 30, 40]
    retained = _step(env)
    final = retained.final_obs
    assert final is not None

    current = _step(env)
    assert current is not retained
    assert current.final_obs is None
    assert retained.final_obs is final
    assert len(env.capture_rows) == 1

    # A subsequent capture must not overwrite the first transition's snapshot.
    env.state.episode_steps[1] = 2
    later = _step(env)
    assert later.final_obs is not None
    assert later.final_obs is not final
    np.testing.assert_array_equal(later.final_obs.env_ids, [1, 3])
    np.testing.assert_array_equal(later.final_obs.obs.policy, [[23], [43]])

    reinitialized = env.init_state()
    assert reinitialized.final_obs is None
    assert retained.final_obs is final
    np.testing.assert_array_equal(final.env_ids, [0, 2])
    np.testing.assert_array_equal(final.obs.policy, [[11], [31]])
    if with_value:
        np.testing.assert_array_equal(final.obs.value, [[11, -11], [31, -31]])
    else:
        assert final.obs.value is None


def test_default_final_obs_hook_is_unsupported_on_truncation() -> None:
    env = _CounterEnv()
    env.init_state()
    env.state.episode_steps[2] = 2
    env.counters[2] = 10.0

    state = _step(env)

    assert state.final_obs is None
    np.testing.assert_array_equal(state.truncated, [False, False, True, False])
    np.testing.assert_array_equal(env.reset_rows[-1], [2])
    np.testing.assert_array_equal(state.obs.policy, [[1], [1], [0], [1]])
    np.testing.assert_array_equal(state.episode_steps, [1, 1, 0, 1])
