# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Host accessors for manager-owned raw-action history."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.manager.actions import ActionState
from motrix_env_core.manager.kernel_data import canonicalize_kernel_data
from motrix_env_core.mdp.action import JointPositionActionState, JointPositionActionTerm


@dataclass(frozen=True, slots=True)
class _ActionState(ActionState):
    scale: float


@pytest.mark.parametrize("ptr", [0, 1, 2])
def test_action_state_accessors_return_history_views(ptr: int) -> None:
    queue = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    state = canonicalize_kernel_data(
        _ActionState(
            action_queue=torch.from_numpy(queue),
            action_ptr=torch.from_numpy(np.asarray([ptr], dtype=np.int64)),
            scale=1.0,
        ),
        context="action state test",
    )
    history = state.action_queue.numpy().copy()
    pointer = state.action_ptr.numpy().copy()

    current = state.current()
    previous = state.previous()

    np.testing.assert_array_equal(current.numpy(), history[:, ptr])
    np.testing.assert_array_equal(previous.numpy(), history[:, (ptr - 1) % 3])
    assert tuple(current.shape) == tuple(previous.shape) == (2, 3)
    assert current.data_ptr() == state.action_queue[:, ptr, :].data_ptr()
    np.testing.assert_array_equal(state.action_queue.numpy(), history)
    np.testing.assert_array_equal(state.action_ptr.numpy(), pointer)

    current[0, 0] = -1.0
    assert state.action_queue[0, ptr, 0] == -1.0


def test_joint_position_action_applies_per_lane_delay() -> None:
    state = JointPositionActionState(
        action_queue=torch.from_numpy(
            np.asarray(
                [
                    [[10.0], [20.0], [30.0], [40.0]],
                    [[1.0], [2.0], [3.0], [4.0]],
                ],
                dtype=np.float32,
            )
        ),
        action_ptr=torch.from_numpy(np.asarray([2], dtype=np.int64)),
        default_angles=torch.from_numpy(np.asarray([100.0], dtype=np.float32)),
        joint_lower=torch.from_numpy(np.asarray([-100.0], dtype=np.float32)),
        joint_upper=torch.from_numpy(np.asarray([100.0], dtype=np.float32)),
        action_scales=torch.from_numpy(np.asarray([2.0], dtype=np.float32)),
        delay_steps=torch.from_numpy(np.asarray([1, 2], dtype=np.int64)),
        delay_lo=0,
        delay_hi=2,
    )
    term = JointPositionActionTerm(gym.spaces.Box(-np.inf, np.inf, shape=(1,)), state)

    np.testing.assert_array_equal(term._process(torch.zeros(2, 1, dtype=torch.float32)).numpy(), [[140.0], [102.0]])


def test_joint_position_action_resamples_delay_on_reset(monkeypatch) -> None:
    state = JointPositionActionState(
        action_queue=torch.zeros((3, 2, 1), dtype=torch.float32),
        action_ptr=torch.zeros(1, dtype=torch.int64),
        default_angles=torch.zeros(1, dtype=torch.float32),
        joint_lower=torch.zeros(1, dtype=torch.float32),
        joint_upper=torch.ones(1, dtype=torch.float32),
        action_scales=torch.ones(1, dtype=torch.float32),
        delay_steps=torch.zeros(3, dtype=torch.int64),
        delay_lo=1,
        delay_hi=3,
    )
    term = JointPositionActionTerm(gym.spaces.Box(-np.inf, np.inf, shape=(1,)), state)
    monkeypatch.setattr(np.random, "randint", lambda lo, hi, size: np.full(size, lo + 1, dtype=np.int64))

    term._reset(np.asarray([0, 2], dtype=np.int64))

    np.testing.assert_array_equal(term.state.delay_steps.numpy(), [2, 0, 2])


def test_action_state_accessors_follow_cursor_changes() -> None:
    state = ActionState(
        action_queue=torch.from_numpy(np.arange(12, dtype=np.float32).reshape(2, 2, 3)),
        action_ptr=torch.zeros(1, dtype=torch.int64),
    )
    initial_current = state.current().clone()
    initial_previous = state.previous().clone()
    state.action_ptr[0] = 1

    np.testing.assert_array_equal(state.current().numpy(), initial_previous.numpy())
    np.testing.assert_array_equal(state.previous().numpy(), initial_current.numpy())
