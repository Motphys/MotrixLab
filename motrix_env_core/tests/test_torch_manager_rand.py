# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from motrix_env_core.manager.rand import draw_uniform, initialize_rand_states


def _oracle(states: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    samples = torch.empty((states.shape[0], count), dtype=torch.float32)
    updated = states.clone()
    for row in range(states.shape[0]):
        row_samples, row_state = draw_uniform(updated[row : row + 1], count)
        samples[row] = row_samples[0]
        updated[row : row + 1] = row_state
    return samples, updated


def _states() -> torch.Tensor:
    boundaries = np.array([0, 1, 2**64 - 1, 2**63, 2**63 - 1], dtype=np.uint64)
    seeded = initialize_rand_states(11, 12345).ravel()
    bits = torch.from_numpy(np.concatenate((boundaries, seeded.numpy().view(np.uint64))))
    return bits.view(torch.int64)[:, None]


@pytest.mark.parametrize("count", [1, 7, 67])
def test_draw_uniform_matches_host_oracle(count: int) -> None:
    state = _states()
    original = state.clone()
    expected_samples, expected_state = _oracle(state, count)
    samples, updated = draw_uniform(state, count)
    torch.testing.assert_close(samples, expected_samples, rtol=0, atol=0)
    torch.testing.assert_close(updated, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(state, original)
    following_samples, following_state = draw_uniform(updated, count)
    expected_samples, expected_state = _oracle(expected_state, count)
    torch.testing.assert_close(following_samples, expected_samples, rtol=0, atol=0)
    torch.testing.assert_close(following_state, expected_state, rtol=0, atol=0)


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
    expected_samples, expected_state = _oracle(state, count)
    torch.testing.assert_close(actual_samples, expected_samples, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
