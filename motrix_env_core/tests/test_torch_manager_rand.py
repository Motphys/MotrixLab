# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from motrix_env_core.manager.rand import initialize_rand_states, next_uniform
from motrix_env_core.torch_manager.rand import draw_uniform


def _oracle(states: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    samples = np.empty((len(states), count), dtype=np.float32)
    updated = states.copy()
    for row, state in enumerate(updated):
        state_ref = np.array([state], dtype=np.uint64)
        for column in range(count):
            samples[row, column] = next_uniform(state_ref)
        updated[row] = state_ref[0]
    return samples, updated


def _states() -> torch.Tensor:
    boundaries = np.array([0, 1, 2**64 - 1, 2**63, 2**63 - 1], dtype=np.uint64)
    seeded = initialize_rand_states(11, 12345).ravel()
    bits = np.concatenate((boundaries, seeded))
    return torch.from_numpy(bits.view(np.int64))[:, None]


@pytest.mark.parametrize("count", [1, 7, 67])
def test_draw_uniform_matches_host_oracle(count: int) -> None:
    state = _states()
    original = state.clone()
    expected_samples, expected_state = _oracle(state.numpy().ravel().view(np.uint64), count)
    samples, updated = draw_uniform(state, count)
    np.testing.assert_array_equal(samples.numpy().view(np.uint32), expected_samples.view(np.uint32))
    np.testing.assert_array_equal(updated.numpy().ravel().view(np.uint64), expected_state)
    torch.testing.assert_close(state, original)
    following_samples, following_state = draw_uniform(updated, count)
    expected_samples, expected_state = _oracle(expected_state, count)
    np.testing.assert_array_equal(following_samples.numpy().view(np.uint32), expected_samples.view(np.uint32))
    np.testing.assert_array_equal(following_state.numpy().ravel().view(np.uint64), expected_state)


def test_draw_uniform_zero_count_preserves_state() -> None:
    state = _states()
    samples, updated = draw_uniform(state, 0)
    assert samples.shape == (state.shape[0], 0)
    assert samples.dtype is torch.float32
    torch.testing.assert_close(updated, state)


@pytest.mark.slow
@pytest.mark.parametrize("count", [0, 7, 67])
def test_draw_uniform_compiles_with_fullgraph_inductor(count: int) -> None:
    state = _states()
    compiled = torch.compile(draw_uniform, backend="inductor", fullgraph=True)
    actual_samples, actual_state = compiled(state, count)
    expected_samples, expected_state = _oracle(state.numpy().ravel().view(np.uint64), count)
    np.testing.assert_array_equal(actual_samples.numpy().view(np.uint32), expected_samples.view(np.uint32))
    np.testing.assert_array_equal(actual_state.numpy().ravel().view(np.uint64), expected_state)
