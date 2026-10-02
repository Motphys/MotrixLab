# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pure NumPy sampler invariants, completion semantics and level isolation."""

import numpy as np
import pytest

from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain_sampling import TerrainSamplingState


def test_no_evidence_preserves_column_encoded_type_proportions():
    state = TerrainSamplingState(num_cols=10, num_levels=10)
    probabilities = state.probabilities()
    np.testing.assert_allclose(probabilities, state.base_probabilities)
    # Duplicate columns encode type proportions, rather than uniform type names.
    type_ids = np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7])
    type_probabilities = np.bincount(type_ids, weights=probabilities)
    np.testing.assert_allclose(type_probabilities, np.bincount(type_ids) / type_ids.size)
    np.testing.assert_array_equal(state.failure_ema, 0.5)
    np.testing.assert_array_equal(state.evidence_counts, 0)


def test_update_counts_active_completed_episodes_and_timeout_successes():
    state = TerrainSamplingState(3, 2, ema_alpha=0.5)
    state.update(
        cols=np.array([[0], [0], [1], [2]]),
        levels=np.array([[1], [1], [0], [0]]),
        terminated=np.array([True, False, True, False]),
        active=np.array([[True], [True], [False], [True]]),
    )
    np.testing.assert_array_equal(state.evidence_counts, [[0, 2], [0, 0], [1, 0]])
    assert state.failure_ema[0, 1] == pytest.approx(0.5)
    assert state.failure_ema[1, 0] == pytest.approx(0.5)
    assert state.failure_ema[2, 0] == pytest.approx(0.25)


def test_repeated_indices_match_identical_scalar_ema_and_ignore_batch_order():
    state = TerrainSamplingState(2, 2, ema_alpha=0.1)
    state.update(np.zeros(7, dtype=int), np.ones(7, dtype=int), np.ones(7, dtype=bool), np.ones(7, dtype=bool))
    assert state.failure_ema[0, 1] == pytest.approx(1.0 - 0.5 * 0.9**7)
    assert state.evidence_counts[0, 1] == 7

    cols = np.array([0, 1, 0, 1, 0, 1])
    levels = np.array([0, 1, 0, 1, 0, 0])
    failed = np.array([True, False, False, True, True, False])
    first, second = TerrainSamplingState(2, 2), TerrainSamplingState(2, 2)
    first.update(cols, levels, failed, np.ones(6, dtype=bool))
    permutation = np.array([5, 2, 4, 1, 0, 3])
    second.update(cols[permutation], levels[permutation], failed[permutation], np.ones(6, dtype=bool))
    np.testing.assert_array_equal(first.evidence_counts, second.evidence_counts)
    np.testing.assert_array_equal(first.failure_ema, second.failure_ema)


def test_empty_and_inactive_initial_resets_do_not_change_evidence():
    state = TerrainSamplingState(2, 3)
    state.update(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=bool), np.array([], dtype=bool))
    state.update(np.array([0, 1]), np.array([1, 2]), np.ones(2, dtype=bool), np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(state.evidence_counts, 0)
    np.testing.assert_array_equal(state.failure_ema, 0.5)
    np.testing.assert_allclose(state.probabilities(), state.base_probabilities)


def test_subthreshold_evidence_is_neutral():
    state = TerrainSamplingState(2, 2, min_count=4, ema_alpha=1.0)
    state.update(np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.ones(3, dtype=bool), np.ones(3, dtype=bool))
    np.testing.assert_allclose(state.probabilities(), state.base_probabilities)
    state.update(np.array([0]), np.array([0]), np.array([True]), np.array([True]))
    assert state.probabilities()[0] < state.probabilities()[1]


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.7, 1.0])
@pytest.mark.parametrize("multiplier", [1.0, 2.0, 4.0])
def test_normalization_floor_and_relative_cap_after_normalization(fraction, multiplier):
    state = TerrainSamplingState(11, 3, adaptive_fraction=fraction, max_multiplier=multiplier, min_count=1)
    rng = np.random.default_rng(41)
    for _ in range(12):
        state.failure_ema[:] = rng.uniform(0.0, 1.0, state.failure_ema.shape)
        state.evidence_counts[:] = rng.integers(0, 3, state.evidence_counts.shape)
        probabilities = state.probabilities()
        assert np.all(np.isfinite(probabilities))
        assert probabilities.sum() == pytest.approx(1.0)
        assert np.all(probabilities >= (1.0 - fraction) * state.base_probabilities - 1e-15)
        assert np.all(probabilities <= (1.0 + fraction * (multiplier - 1.0)) * state.base_probabilities + 1e-15)
        assert np.all(probabilities <= multiplier * state.base_probabilities + 1e-15)


def test_frontier_is_preferred_over_mastered_or_impossible_saturated_cells():
    state = TerrainSamplingState(3, 1, ema_alpha=1.0, min_count=2)
    state.update(
        np.repeat(np.arange(3), 2),
        np.zeros(6, dtype=int),
        np.array([False, False, False, True, True, True]),
        np.ones(6, dtype=bool),
    )
    probabilities = state.probabilities()
    assert probabilities[1] > probabilities[0]
    assert probabilities[1] > probabilities[2]
    assert probabilities[0] == pytest.approx(probabilities[2])
    assert probabilities[0] >= 0.75 * state.base_probabilities[0]
    assert probabilities[1] <= 2.0 * state.base_probabilities[1]


def test_levels_remain_isolated_and_unseen_rows_do_not_mask_saturation():
    state = TerrainSamplingState(2, 3, ema_alpha=1.0, min_count=2)
    state.update(np.array([0, 0]), np.array([0, 0]), np.array([False, True]), np.ones(2, dtype=bool))
    frontier = state.failure_ema[0, 0]
    # A much larger all-failure high-level batch must not overwrite low-level evidence.
    state.update(np.zeros(100, dtype=int), np.ones(100, dtype=int), np.ones(100, dtype=bool), np.ones(100, dtype=bool))
    assert state.failure_ema[0, 0] == frontier
    assert state.failure_ema[0, 1] == 1.0
    assert state.evidence_counts[0, 2] == 0
    np.testing.assert_allclose(state.probabilities(), state.base_probabilities)
    state.update(np.ones(2, dtype=int), np.ones(2, dtype=int), np.ones(2, dtype=bool), np.ones(2, dtype=bool))
    assert state.probabilities()[0] > state.probabilities()[1]
    assert state.evidence_counts[1, 0] == 0
    assert state.evidence_counts[1, 2] == 0


def test_seeded_resample_returns_only_reset_bindings_and_empty_consumes_no_draws():
    state = TerrainSamplingState(4, 2)
    rng, same_rng = np.random.default_rng(19), np.random.default_rng(19)
    reset_ids = np.array([1, 4, 7])
    columns = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    before = columns.copy()
    chosen = state.resample(reset_ids, rng)
    np.testing.assert_array_equal(chosen, state.resample(reset_ids, same_rng))
    assert chosen.shape == reset_ids.shape
    assert np.all((chosen >= 0) & (chosen < state.num_cols))
    columns[reset_ids] = chosen
    untouched = np.array([0, 2, 3, 5, 6])
    np.testing.assert_array_equal(columns[untouched], before[untouched])
    assert state.resample(np.array([], dtype=int), rng).size == 0
    np.testing.assert_array_equal(rng.random(5), same_rng.random(5))
    np.testing.assert_array_equal(state.evidence_counts, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_cols": 0},
        {"num_levels": 0},
        {"adaptive_fraction": -0.1},
        {"adaptive_fraction": 1.1},
        {"max_multiplier": 0.9},
        {"max_multiplier": np.inf},
        {"ema_alpha": 0.0},
        {"ema_alpha": 1.1},
        {"min_count": 0},
    ],
)
def test_invalid_public_sampling_parameters(kwargs):
    parameters = {"num_cols": 3, "num_levels": 4, **kwargs}
    with pytest.raises(ValueError):
        TerrainSamplingState(**parameters)
