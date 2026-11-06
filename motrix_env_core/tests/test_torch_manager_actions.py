# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Tensor action processing agrees with the native manager's history lifecycle."""

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.mdp.action import JointPositionActionState, JointPositionActionTerm
from motrix_env_core.torch_manager.actions import process_joint_position_actions, reset_action_history


def _make_term(width: int, delayed: bool) -> JointPositionActionTerm:
    num_envs, num_actions = width + 1, 3
    # Nonzero history catches off-by-one reads before fresh actions fill the ring.
    queue = np.arange(num_envs * width * num_actions, dtype=np.float32).reshape(num_envs, width, num_actions) / 8
    state = JointPositionActionState(
        action_queue=torch.from_numpy(queue),
        action_ptr=torch.from_numpy(np.asarray([width - 1], dtype=np.int64)),
        default_angles=torch.from_numpy(np.asarray([0.25, -2.0, 0.75], dtype=np.float32)),
        joint_lower=torch.from_numpy(np.full(num_actions, -np.inf, dtype=np.float32)),
        joint_upper=torch.from_numpy(np.full(num_actions, np.inf, dtype=np.float32)),
        action_scales=torch.from_numpy(np.asarray([0.5, 2.0, 0.0], dtype=np.float32)),
        delay_steps=torch.from_numpy(
            np.arange(num_envs, dtype=np.int64) % width if delayed else np.zeros(num_envs, dtype=np.int64)
        ),
        # Keep native reset resampling deterministic without mocking its path.
        delay_lo=width - 1 if delayed else 0,
        delay_hi=width - 1 if delayed else 0,
    )
    return JointPositionActionTerm(gym.spaces.Box(-np.inf, np.inf, shape=(num_actions,)), state)


def _exercise_process(process: Callable, *, width: int, delayed: bool) -> None:
    term = _make_term(width, delayed)
    state = term.state
    queue = state.action_queue.clone()
    ptr = state.action_ptr.clone()
    delays = state.delay_steps.clone()
    scales = state.action_scales.clone()
    defaults = state.default_angles.clone()
    rng = np.random.default_rng(17)
    for _ in range(2 * width + 1):
        raw = rng.uniform(-5, 5, size=tuple(state.current().shape)).astype(np.float32)
        actions = torch.from_numpy(raw.copy())
        queue_before, ptr_before = queue.clone(), ptr.clone()
        targets, new_queue, new_ptr, current, previous = process(queue, ptr, actions, delays, scales, defaults)
        expected = term.process(torch.from_numpy(raw))
        torch.testing.assert_close(targets, expected)
        torch.testing.assert_close(new_queue, state.action_queue, rtol=0, atol=0)
        torch.testing.assert_close(new_ptr, state.action_ptr, rtol=0, atol=0)
        torch.testing.assert_close(current, state.current(), rtol=0, atol=0)
        torch.testing.assert_close(previous, state.previous(), rtol=0, atol=0)
        torch.testing.assert_close(queue, queue_before, rtol=0, atol=0)
        torch.testing.assert_close(ptr, ptr_before, rtol=0, atol=0)
        torch.testing.assert_close(actions, torch.from_numpy(raw), rtol=0, atol=0)
        assert new_ptr.shape == (1,) and new_ptr.dtype == torch.int64
        queue, ptr = new_queue, new_ptr


@pytest.mark.parametrize("width", [2, 5])
@pytest.mark.parametrize("delayed", [False, True])
def test_joint_position_process_matches_native(width: int, delayed: bool) -> None:
    _exercise_process(process_joint_position_actions, width=width, delayed=delayed)


def _exercise_reset(reset: Callable, process: Callable) -> None:
    term = _make_term(4, True)
    state = term.state
    queue = state.action_queue.clone()
    ptr = state.action_ptr.clone()
    for mask in (
        np.asarray([False, True, False, True, False]),
        np.zeros(5, dtype=bool),
        np.ones(5, dtype=bool),
    ):
        before = queue.clone()
        pointer_before = ptr.clone()
        queue_after = reset(queue, torch.from_numpy(mask))
        term.reset(np.flatnonzero(mask))
        torch.testing.assert_close(queue_after, state.action_queue, rtol=0, atol=0)
        torch.testing.assert_close(queue, before, rtol=0, atol=0)
        torch.testing.assert_close(ptr, pointer_before, rtol=0, atol=0)
        torch.testing.assert_close(ptr, state.action_ptr, rtol=0, atol=0)
        # Reset randomization belongs to the host, not the pure history helper.
        delays = state.delay_steps.clone()
        raw = np.full((5, 3), 3.5, dtype=np.float32)
        targets, queue, ptr, current, previous = process(
            queue_after,
            ptr,
            torch.from_numpy(raw),
            delays,
            state.action_scales,
            state.default_angles,
        )
        torch.testing.assert_close(targets, term.process(torch.from_numpy(raw)))
        torch.testing.assert_close(queue, state.action_queue, rtol=0, atol=0)
        torch.testing.assert_close(current, state.current(), rtol=0, atol=0)
        torch.testing.assert_close(previous, state.previous(), rtol=0, atol=0)


def test_reset_action_history_matches_native_selected_lanes() -> None:
    _exercise_reset(reset_action_history, process_joint_position_actions)


@pytest.mark.slow
def test_action_helpers_compile_fullgraph_cpu() -> None:
    process = torch.compile(process_joint_position_actions, backend="inductor", fullgraph=True)
    reset = torch.compile(reset_action_history, backend="inductor", fullgraph=True)
    _exercise_process(process, width=4, delayed=False)
    _exercise_process(process, width=4, delayed=True)
    _exercise_reset(reset, process)


def test_batch_action_views_read_each_lanes_own_slot() -> None:
    """The shared (1,) cursor must gather every lane's row, not lane 0's.

    Regression: a (1,1,A) gather index silently broadcast lane 0's action to
    all lanes, corrupting the per-lane action-echo observation and the
    action-rate reward.
    """
    from motrix_env_core.torch_manager.runtime import _BatchActions

    num_envs, width, num_actions = 8, 4, 3
    queue = torch.arange(num_envs * width * num_actions, dtype=torch.float32).reshape(num_envs, width, num_actions)
    for ptr in (0, 1, width - 1):
        namespace = _FixedActionNamespace(queue, torch.tensor([ptr], dtype=torch.int64))
        actions = _BatchActions(namespace)
        torch.testing.assert_close(actions.current(), queue[:, ptr, :])
        previous = queue[:, (ptr - 1) % width, :]
        torch.testing.assert_close(actions.previous(), previous)


class _FixedActionNamespace:
    def __init__(self, queue: torch.Tensor, ptr: torch.Tensor) -> None:
        self.action_queue = queue
        self.action_ptr = ptr
