# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Host accessors for manager-owned raw-action history."""

import gymnasium as gym
import numpy as np
import pytest

from motrix_env_core.mdp.action import JointPositionActionState, JointPositionActionTerm
from motrix_env_core.numba.kernel_data import canonicalize_kernel_data, kernel_data
from motrix_env_core.numba.manager.actions import ActionState


@kernel_data
class _ActionState(ActionState):
    scale: float


@pytest.mark.parametrize("ptr", [0, 1, 2])
def test_action_state_accessors_return_history_views(ptr: int) -> None:
    queue = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    state = canonicalize_kernel_data(
        _ActionState(action_queue=queue, action_ptr=np.asarray([ptr], dtype=np.int64), scale=1.0),
        context="action state test",
    )
    history = state.action_queue.copy()
    pointer = state.action_ptr.copy()

    current = state.current()
    previous = state.previous()

    np.testing.assert_array_equal(current, history[:, ptr])
    np.testing.assert_array_equal(previous, history[:, (ptr - 1) % 3])
    assert current.shape == previous.shape == (2, 3)
    assert np.shares_memory(current, state.action_queue)
    assert np.shares_memory(previous, state.action_queue)
    np.testing.assert_array_equal(state.action_queue, history)
    np.testing.assert_array_equal(state.action_ptr, pointer)

    current[0, 0] = -1.0
    assert state.action_queue[0, ptr, 0] == -1.0


def test_joint_position_action_applies_per_lane_delay() -> None:
    state = JointPositionActionState(
        action_queue=np.asarray(
            [
                [[10.0], [20.0], [30.0], [40.0]],
                [[1.0], [2.0], [3.0], [4.0]],
            ],
            dtype=np.float32,
        ),
        action_ptr=np.asarray([2], dtype=np.int64),
        default_angles=np.asarray([100.0], dtype=np.float32),
        joint_lower=np.asarray([-100.0], dtype=np.float32),
        joint_upper=np.asarray([100.0], dtype=np.float32),
        action_scales=np.asarray([2.0], dtype=np.float32),
        delay_steps=np.asarray([1, 2], dtype=np.int64),
        delay_lo=0,
        delay_hi=2,
    )
    term = JointPositionActionTerm(gym.spaces.Box(-np.inf, np.inf, shape=(1,)), state)

    np.testing.assert_array_equal(term._process(np.asarray([[0.0], [0.0]], dtype=np.float32)), [[140.0], [102.0]])


def test_joint_position_action_resamples_delay_on_reset(monkeypatch) -> None:
    state = JointPositionActionState(
        action_queue=np.zeros((3, 2, 1), dtype=np.float32),
        action_ptr=np.asarray([0], dtype=np.int64),
        default_angles=np.zeros(1, dtype=np.float32),
        joint_lower=np.zeros(1, dtype=np.float32),
        joint_upper=np.ones(1, dtype=np.float32),
        action_scales=np.ones(1, dtype=np.float32),
        delay_steps=np.zeros(3, dtype=np.int64),
        delay_lo=1,
        delay_hi=3,
    )
    term = JointPositionActionTerm(gym.spaces.Box(-np.inf, np.inf, shape=(1,)), state)
    monkeypatch.setattr(np.random, "randint", lambda lo, hi, size: np.full(size, lo + 1, dtype=np.int64))

    term._reset(np.asarray([0, 2], dtype=np.int64))

    np.testing.assert_array_equal(term.state.delay_steps, [2, 0, 2])


def test_action_state_accessors_follow_cursor_changes() -> None:
    state = ActionState(
        action_queue=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        action_ptr=np.zeros(1, dtype=np.int64),
    )
    initial_current = state.current().copy()
    initial_previous = state.previous().copy()
    state.action_ptr[0] = 1

    np.testing.assert_array_equal(state.current(), initial_previous)
    np.testing.assert_array_equal(state.previous(), initial_current)
