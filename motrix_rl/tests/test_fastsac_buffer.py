# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contract tests for the FastSAC replay buffer.

The buffer shares observation storage across time steps (``next_obs`` of step
``t`` is the stored observation of step ``t + 1``). These tests verify sampled
transitions against an independently computed ground truth: n-step returns,
effective step counts, terminal flags and next observations must match the
original transition sequence even after the ring has wrapped several times.
"""

from unittest import mock

import pytest
import torch

from motrix_rl.fastsac.buffer import SimpleReplayBuffer

N_ENV = 3
N_OBS = 4
N_ACT = 2
N_CRITIC_OBS = 5
BUFFER_SIZE = 6
GAMMA = 0.97


def _fill(rb, times, dones_by_t, truncs_by_t, rewards=None):
    """Feed transitions t=0..times-1 with obs[t] scalar-encoded along dim 0."""
    for t in range(times):
        obs = torch.full((N_ENV, N_OBS), float(t))
        critic_obs = torch.full((N_ENV, N_CRITIC_OBS), float(100 + t))
        actions = torch.full((N_ENV, N_ACT), float(t))
        rew = torch.full((N_ENV,), float(t)) if rewards is None else rewards[t]
        dones = torch.full((N_ENV,), dones_by_t[t], dtype=torch.long)
        truncs = torch.full((N_ENV,), truncs_by_t[t], dtype=torch.long)
        rb.extend(obs, critic_obs, actions, rew, dones, truncs)
    return times


def _expected_n_step(t0, last, n_steps, dones_by_t, truncs_by_t):
    """Ground truth for a window starting at ``t0`` (mirrors SAC semantics).

    ``last`` is the index of the newest complete transition (the buffer lags
    one step behind ingestion because a transition's next observation is the
    following step's stored observation). Rewards are accumulated until the
    first done (exclusive shift: the reward leading into a terminal step
    counts) and capped at that newest transition; ``final`` additionally stops
    at the first truncation.
    """
    max_off = min(n_steps - 1, last - t0)
    first_done = next((k for k in range(n_steps) if k <= max_off and dones_by_t[t0 + k]), n_steps - 1)
    first_trunc = next((k for k in range(n_steps) if k <= max_off and truncs_by_t[t0 + k]), n_steps - 1)
    final = min(first_done, first_trunc, max_off)
    ret, eff = 0.0, 0
    for k in range(n_steps):
        if k > max_off:
            break
        if k > 0 and dones_by_t[t0 + k - 1]:
            break
        ret += (GAMMA**k) * float(t0 + k)
        eff += 1
    return ret, eff, final


def _sample_all(rb, batch_size):
    """Sample with a deterministic randint returning the full index range."""
    with mock.patch.object(
        torch,
        "randint",
        side_effect=lambda low, high, size, device=None: (
            torch.arange(low, high)[: size[1]].repeat(size[0], 1).to(device)
        ),
    ):
        return rb.sample(batch_size)


def test_observations_stored_once_per_timestep():
    rb = SimpleReplayBuffer(N_ENV, BUFFER_SIZE, N_OBS, N_ACT, N_CRITIC_OBS, device="cpu")
    assert rb.observations.shape == (N_ENV, BUFFER_SIZE + 1, N_OBS)
    assert rb.critic_observations.shape == (N_ENV, BUFFER_SIZE + 1, N_CRITIC_OBS)
    assert not hasattr(rb, "next_observations") and not hasattr(rb, "next_critic_observations")


def test_sample_rejects_empty_buffer_and_keeps_effective_n_steps_integer():
    empty = SimpleReplayBuffer(N_ENV, BUFFER_SIZE, N_OBS, N_ACT, N_CRITIC_OBS, device="cpu")
    with pytest.raises(RuntimeError):
        empty.sample(4)
    # with a single ingested step there is no complete transition yet either
    # (its next observation is the following step's stored observation); this
    # previously let the n-step branch fabricate next_obs from slot garbage
    for n_steps in (1, 3):
        single = SimpleReplayBuffer(
            N_ENV, BUFFER_SIZE, N_OBS, N_ACT, N_CRITIC_OBS, n_steps=n_steps, gamma=GAMMA, device="cpu"
        )
        _fill(single, 1, [0], [0])
        with pytest.raises(RuntimeError):
            single.sample(4)
    # the n-step branch must return the same dtype as the 1-step branch
    for n_steps in (1, 3):
        rb = SimpleReplayBuffer(
            N_ENV, BUFFER_SIZE, N_OBS, N_ACT, N_CRITIC_OBS, n_steps=n_steps, gamma=GAMMA, device="cpu"
        )
        _fill(rb, BUFFER_SIZE + 2, [0] * (BUFFER_SIZE + 2), [0] * (BUFFER_SIZE + 2))
        batch = _sample_all(rb, rb.num_stored)
        assert batch["effective_n_steps"].dtype == torch.long


def test_one_step_samples_after_wrap():
    rb = SimpleReplayBuffer(N_ENV, BUFFER_SIZE, N_OBS, N_ACT, N_CRITIC_OBS, gamma=GAMMA, device="cpu")
    total = _fill(rb, BUFFER_SIZE * 3 + 2, [0] * (BUFFER_SIZE * 3 + 2), [0] * (BUFFER_SIZE * 3 + 2))
    batch = _sample_all(rb, rb.num_stored)
    for row in range(batch["obs"].shape[0]):
        t0 = int(batch["obs"][row, 0])
        # the newest transition (t = total - 1) waits for its successor obs
        assert 0 <= t0 <= total - 2
        assert int(batch["next_obs"][row, 0]) == t0 + 1
        assert int(batch["next_critic_obs"][row, 0]) == 100 + t0 + 1
        assert float(batch["rewards"][row]) == t0
        assert batch["effective_n_steps"][row] == 1


@pytest.mark.parametrize("n_steps", [2, 3])
def test_n_step_matches_ground_truth_with_terminations(n_steps):
    total = BUFFER_SIZE * 2 + 4
    dones = [1 if t % 5 == 4 else 0 for t in range(total)]
    truncs = [1 if t % 7 == 6 else 0 for t in range(total)]
    rb = SimpleReplayBuffer(N_ENV, BUFFER_SIZE, N_OBS, N_ACT, N_CRITIC_OBS, n_steps=n_steps, gamma=GAMMA, device="cpu")
    _fill(rb, total, dones, truncs)
    batch = _sample_all(rb, rb.num_stored)
    flat = batch["obs"].shape[0]
    for row in range(flat):
        t0 = int(batch["obs"][row, 0])
        ret, eff, final = _expected_n_step(t0, total - 2, n_steps, dones, truncs)
        assert batch["rewards"][row] == pytest.approx(ret, abs=1e-5), f"t0={t0}"
        assert int(batch["effective_n_steps"][row]) == eff, f"t0={t0}"
        assert int(batch["next_obs"][row, 0]) == t0 + final + 1, f"t0={t0}"
        assert int(batch["dones"][row]) == dones[t0 + final], f"t0={t0}"
        assert int(batch["truncations"][row]) == truncs[t0 + final], f"t0={t0}"


def test_n_step_window_clamped_at_newest_transition():
    # Full ring, no dones/truncs: windows starting near the newest transition
    # must not read stale slots from a cycle earlier.
    total = BUFFER_SIZE * 4
    rb = SimpleReplayBuffer(N_ENV, BUFFER_SIZE, N_OBS, N_ACT, N_CRITIC_OBS, n_steps=3, gamma=GAMMA, device="cpu")
    _fill(rb, total, [0] * total, [0] * total)
    batch = _sample_all(rb, rb.num_stored)
    for row in range(batch["obs"].shape[0]):
        t0 = int(batch["obs"][row, 0])
        ret, eff, final = _expected_n_step(t0, total - 2, 3, [0] * total, [0] * total)
        assert batch["rewards"][row] == pytest.approx(ret, abs=1e-5)
        assert int(batch["next_obs"][row, 0]) == t0 + final + 1
        assert t0 + final + 1 <= total - 1


def test_partial_buffer_n_step_only_full_windows():
    rb = SimpleReplayBuffer(N_ENV, BUFFER_SIZE, N_OBS, N_ACT, N_CRITIC_OBS, n_steps=3, gamma=GAMMA, device="cpu")
    total = 4  # < buffer_size: only full windows within complete transitions
    _fill(rb, total, [0] * total, [0] * total)
    batch = _sample_all(rb, max(1, total - 3))
    for row in range(batch["obs"].shape[0]):
        t0 = int(batch["obs"][row, 0])
        assert t0 <= total - 3
        assert int(batch["effective_n_steps"][row]) == 3
