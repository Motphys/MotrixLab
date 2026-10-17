# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Terminal-observation bootstrapping contracts.

Time-limit-truncated transitions must bootstrap from the stored TERMINAL
(pre-reset) observation, not the successor slot's auto-reset observation —
the reset state's value is unrelated to the episode that timed out. These
tests pin the replay terminal-observation and non-consuming adapter contracts.
"""

import multiprocessing as mp
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from fastsac_async_mocks import make_mock_learner
from motrix_rl_builtin.fastsac.async_impl.learner import DrainStaging
from motrix_rl_builtin.fastsac.async_impl.transport.ring import (
    FinalObsPayload,
    FinalObsStorage,
    SharedTransitionRing,
)
from motrix_rl_builtin.fastsac.buffer import SimpleReplayBuffer
from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap

from motrix_env_core.array.env import ArrayEnvState, FinalObs, NpObs


def test_numpy_adapter_reads_final_result_without_consuming_environment_state():
    final = FinalObs(np.array([0]), NpObs(np.array([[7.0]], dtype=np.float32)))
    state = ArrayEnvState(
        obs=NpObs(np.zeros((1, 1), dtype=np.float32)),
        reward=np.zeros(1, dtype=np.float32),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.ones(1, dtype=bool),
        episode_steps=np.zeros(1, dtype=np.uint64),
        final_obs=final,
    )
    initial = state.replace(final_obs=None)
    env = SimpleNamespace(
        num_envs=1,
        action_space=gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
        init_state=lambda: initial,
        step=lambda actions: state,
    )
    wrapped = FastSacNpEnvWrap(env, torch.device("cpu"))
    obs, critic_obs = wrapped.reset()
    torch.testing.assert_close(obs, torch.zeros(1, 1))
    torch.testing.assert_close(critic_obs, obs)
    result = wrapped.step(torch.zeros(1, 1))
    assert result.final_obs is final
    assert result.info == {"Reward": {}, "metrics": {}}
    assert state.final_obs is final
    assert result.final_obs.obs.value_or_policy is final.obs.policy
    wrapped.reset()
    assert result.final_obs is final
    assert state.final_obs is final


N_ENV = 2
N_OBS = 3
N_ACT = 1
N_CRITIC_OBS = 4


def _fill_with_truncations(rb, times, trunc_ts, env=0):
    """Feed scalar-encoded transitions; ``env`` truncates at each t in trunc_ts."""
    for t in range(times):
        trunc = torch.zeros(N_ENV, dtype=torch.long)
        if t in trunc_ts:
            trunc[env] = 1
        rb.extend(
            torch.full((N_ENV, N_OBS), float(t)),
            torch.full((N_ENV, N_CRITIC_OBS), float(100 + t)),
            torch.zeros(N_ENV, N_ACT),
            torch.zeros(N_ENV),
            torch.zeros(N_ENV, dtype=torch.long),
            trunc,
        )
        if trunc.any():
            rb.write_final(
                torch.nonzero(trunc).flatten(),
                t,
                torch.full((1, N_OBS), 1000.0 + t),
                torch.full((1, N_CRITIC_OBS), 2000.0 + t),
            )


def test_truncated_transitions_sample_terminal_observation():
    """Sampled truncated rows must return the recorded terminal observation;
    all other rows keep the successor slot's observation."""
    rb = SimpleReplayBuffer(N_ENV, 16, N_OBS, N_ACT, N_CRITIC_OBS, device="cpu", final_obs_bootstrap=True)
    _fill_with_truncations(rb, 10, trunc_ts={3, 8})

    torch.manual_seed(0)
    for _ in range(20):
        batch = rb.sample(batch_size=32)
        truncated = batch["truncations"] > 0
        if truncated.any():
            assert torch.all(batch["next_obs"][truncated][:, 0] >= 1000.0)
            assert torch.all(batch["next_critic_obs"][truncated][:, 0] >= 2000.0)
        continuing = batch["truncations"] == 0
        if continuing.any():
            assert torch.all(batch["next_obs"][continuing][:, 0] < 100.0)


def test_final_obs_survives_ring_wrap_and_row_reuse():
    """After enough truncations to reuse final rows and wrap the ring, every
    live truncated transition still resolves to ITS OWN terminal observation."""
    buffer_size = 16
    max_ep = 4
    rb = SimpleReplayBuffer(1, buffer_size, 1, 1, 1, device="cpu", final_obs_bootstrap=True)
    total = 40
    for t in range(total):
        truncated = 1 if (t + 1) % max_ep == 0 else 0
        rb.extend(
            torch.tensor([[float(t)]]),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([truncated], dtype=torch.long),
        )
        if truncated:
            rb.write_final(torch.tensor([0]), t, torch.tensor([[5000.0 + t]]), torch.zeros(1, 1))

    # Sampleable window: t in [ptr-1-num_stored, ptr-2]; expected truncations
    # are the multiples of max_ep within it.
    t_min = rb.ptr - 1 - rb.num_stored
    t_max = rb.ptr - 2
    expected = {t for t in range(t_min, t_max + 1) if (t + 1) % max_ep == 0}
    seen = set()
    for _ in range(80):
        batch = rb.sample(batch_size=16)
        for row in batch["next_obs"][batch["truncations"] > 0]:
            t = int(row.item() - 5000.0)
            assert t in expected, row
            seen.add(t)
    assert seen == expected


def test_buffer_without_final_obs_bootstrap_unchanged():
    """final_obs_bootstrap=False keeps the pre-feature memory and sample shape."""
    rb = SimpleReplayBuffer(1, 8, 1, 1, 1, device="cpu")
    assert rb._final_slot is None
    for t in range(10):
        rb.extend(
            torch.tensor([[float(t)]]),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )
    batch = rb.sample(batch_size=4)
    assert batch["next_obs"].shape == (4, 1)


def _ring(capacity=4, num_envs=3):
    return SharedTransitionRing(capacity, num_envs, 2, 3, 1, final_obs_store=FinalObsStorage(capacity, num_envs, 2, 3))


def _push(ring, tag, ids):
    n = ring.num_envs
    payload = None
    if ids:
        payload = FinalObsPayload(
            torch.tensor(ids), torch.full((len(ids), 2), float(tag)), torch.full((len(ids), 3), float(tag + 100))
        )
    return ring.push(
        torch.full((n, 2), float(tag)),
        torch.full((n, 3), float(tag)),
        torch.zeros(n, 1),
        torch.zeros(n),
        torch.zeros(n, dtype=torch.long),
        torch.ones(n, dtype=torch.long),
        final_obs=payload,
    )


def test_sparse_capacity_backpressure_is_atomic_and_acknowledgement_bound():
    ring = _ring(capacity=16)
    batches = ring.final_obs_store.capacity // ring.num_envs
    ids = list(range(ring.num_envs))
    for tag in range(1, batches + 1):
        assert not ring.is_full()
        assert _push(ring, tag, ids)
    count, fields = ring.read_span()
    assert count == batches
    payloads = ring.read_final_span(count)
    assert ring.size() < ring.capacity  # transition slots remain available
    assert ring.is_full()
    assert not _push(ring, batches + 1, [0])
    assert ring.write_completed_count == batches
    torch.testing.assert_close(fields[0][:, 0, 0], torch.arange(1, batches + 1, dtype=torch.float32))
    # Reading/copying payloads does not release the rows; only acknowledgement does.
    assert ring.is_full()
    ring.commit_reads(1)
    assert not ring.is_full()
    assert _push(ring, batches + 1, ids)
    # Owned payloads remain valid after the acknowledged sparse rows are reused.
    for tag, payload in enumerate(payloads, start=1):
        torch.testing.assert_close(payload.obs, torch.full((ring.num_envs, 2), float(tag)))
        torch.testing.assert_close(payload.critic_obs, torch.full((ring.num_envs, 3), float(tag + 100)))
    assert ring.read_issued_count == 1


def test_sparse_pre_step_gate_reserves_rows_for_every_environment():
    ring = _ring(capacity=16)
    used = ring.final_obs_store.capacity - ring.num_envs + 1
    for tag in range(used):
        assert not ring.is_full()
        assert _push(ring, tag, [0])
    store = ring.final_obs_store
    assert ring.size() < ring.capacity
    assert store.can_write_rows(1, ring.read_completed_count)
    assert not store.can_write_rows(ring.num_envs, ring.read_completed_count)
    assert ring.is_full()  # a sparse next payload might fit, but a full one would not
    count, _ = ring.read_span()
    ring.read_final_span(count)
    assert ring.is_full()
    ring.commit_reads(1)
    assert store.can_write_rows(ring.num_envs, ring.read_completed_count)
    assert not ring.is_full()
    assert _push(ring, used, list(range(ring.num_envs)))


def test_slot_and_sparse_row_wrap_keep_payload_alignment():
    ring = _ring(capacity=3)
    for tag in range(20):
        ids = [] if tag % 4 == 0 else list(range(tag % 3 + 1))
        assert _push(ring, tag, ids)
        count, fields = ring.read_span()
        assert count == 1
        payload = ring.read_final_span(count)[0]
        assert float(fields[0][0, 0, 0]) == tag
        if ids:
            assert payload.env_ids.tolist() == ids
            torch.testing.assert_close(payload.obs, torch.full((len(ids), 2), float(tag)))
            torch.testing.assert_close(payload.critic_obs, torch.full((len(ids), 3), float(tag + 100)))
        else:
            assert payload is None
        ring.commit_reads(1)


def test_main_ring_full_leaves_sparse_payload_unpublished():
    ring = _ring(capacity=1)
    assert _push(ring, 1, [])
    assert not _push(ring, 2, [0, 1, 2])
    assert ring.read_final_span(1) == [None]
    ring.commit_reads(1)
    assert _push(ring, 2, [0, 1, 2])
    assert ring.read_final_span(1)[0].env_ids.tolist() == [0, 1, 2]


def _producer(ring, count):
    torch.set_num_threads(1)
    deadline = time.monotonic() + 20
    for tag in range(count):
        ids = list(range(tag % ring.num_envs + 1))
        while not _push(ring, tag, ids):
            if time.monotonic() > deadline:
                raise TimeoutError("slot payload producer failed to make progress")
            time.sleep(0.0001)


@pytest.mark.integration
def test_spawned_producer_and_slow_consumer_preserve_slot_payloads():
    ring = _ring(capacity=5)
    count = 100
    process = mp.get_context("spawn").Process(target=_producer, args=(ring, count))
    process.start()
    deadline = time.monotonic() + 25
    consumed = 0
    try:
        while consumed < count:
            n, fields = ring.read_span()
            if not n:
                assert time.monotonic() < deadline
                assert process.exitcode in (None, 0)
                time.sleep(0.0001)
                continue
            payloads = ring.read_final_span(n)
            time.sleep(0.001)  # allow producer to try reusing unacknowledged rows
            for offset, payload in enumerate(payloads):
                tag = consumed + offset
                assert float(fields[0][offset, 0, 0]) == tag
                assert payload.env_ids.tolist() == list(range(tag % ring.num_envs + 1))
                assert payload.obs.eq(tag).all()
                assert payload.critic_obs.eq(tag + 100).all()
            ring.commit_reads(n)
            consumed += n
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def _buffer():
    return SimpleReplayBuffer(2, 3, 1, 1, 1, device="cpu", final_obs_bootstrap=True)


def _batch(count, first=0):
    obs = torch.arange(first, first + count, dtype=torch.float32)[:, None, None].expand(count, 2, 1)
    flags = torch.ones(count, 2, dtype=torch.long)
    return obs, obs, obs, obs[..., 0], torch.zeros_like(flags), flags


def _payload(tag):
    rows = torch.full((2, 1), float(tag))
    return SimpleNamespace(env_ids=torch.arange(2), obs=rows, critic_obs=rows + 100)


def _assert_live_rows(rb, tags):
    for absolute, tag in tags.items():
        indices = rb._final_slot[:, absolute % rb._cap].long()
        torch.testing.assert_close(rb._final_obs_table[indices], torch.full((2, 1), float(tag)))
        torch.testing.assert_close(rb._final_critic_table[indices], torch.full((2, 1), float(tag + 100)))
    live = [row for owners in rb._final_owners for row in owners.values()]
    assert len(live) == len(set(live))
    assert not set(live).intersection(rb._final_free)
    assert len(live) + len(rb._final_free) == rb._final_capacity


def test_every_step_truncation_grows_and_recycles_without_overwriting_live_rows():
    rb = _buffer()
    for step in range(20):
        rb.extend_batch(*_batch(1, step), final_payloads=[_payload(step)])
        _assert_live_rows(rb, {t: t for t in range(max(0, step - rb._cap + 1), step + 1)})
    assert rb._final_capacity == rb.n_env * rb._cap


def test_final_payload_batch_larger_than_replay_retains_latest_slots():
    rb = _buffer()
    rb.extend_batch(*_batch(10), final_payloads=[_payload(t) for t in range(10)])
    assert rb.ptr == 10
    _assert_live_rows(rb, {t: t for t in range(6, 10)})


def test_overwrite_without_final_releases_rows_and_clears_mapping():
    rb = _buffer()
    rb.extend_batch(*_batch(4), final_payloads=[_payload(t) for t in range(4)])
    rb.extend_batch(*_batch(2, 4))
    assert rb._final_slot[:, :2].eq(-1).all()
    _assert_live_rows(rb, {2: 2, 3: 3})


@pytest.mark.parametrize("n_steps", [1, 3])
def test_sample_uses_final_rows_after_pool_growth(n_steps):
    rb = _buffer()
    rb.n_steps = n_steps
    rb.extend_batch(*_batch(4), final_payloads=[_payload(99) for _ in range(4)])
    sample = rb.sample(32)
    torch.testing.assert_close(sample["next_obs"], torch.full_like(sample["next_obs"], 99))
    torch.testing.assert_close(sample["next_critic_obs"], torch.full_like(sample["next_critic_obs"], 199))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_final_payload_ingested_and_released_during_warmup(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ring = SharedTransitionRing(4, 2, 1, 1, 1, final_obs_store=FinalObsStorage(4, 2, 1, 1))
    learner = make_mock_learner([ring], [])
    learner.agent.device = torch.device(device)
    learner.agent.rb = SimpleReplayBuffer(2, 4, 1, 1, 1, device=device, final_obs_bootstrap=True)
    learner.agent.window_metrics = SimpleNamespace(add=lambda *args, **kwargs: None)
    learner._learning_starts = 1000
    learner._last_train_gstep = 0
    learner.ddp_rank = None
    learner.agent.cfg = SimpleNamespace(num_updates=4)
    learner.async_options.utd_mode = "strict"
    if device == "cuda":
        learner._staging = DrainStaging(learner.agent.rb, 4, torch.device(device))
    rows = torch.full((2, 1), 99.0)
    final = FinalObsPayload(torch.arange(2), rows, rows + 1)
    flags = torch.ones(2, dtype=torch.long)
    for step in range(12):
        obs = torch.full((2, 1), float(step))
        assert ring.push(obs, obs, obs, torch.zeros(2), torch.zeros_like(flags), flags, final_obs=final)
        assert learner.drain() == 1
        assert learner.maybe_train(step + 1) is False
        assert ring.size() == 0
    learner.wait_ingest()
    batch = learner.agent.rb.sample(16)
    torch.testing.assert_close(batch["next_obs"], torch.full_like(batch["next_obs"], 99))
    torch.testing.assert_close(batch["next_critic_obs"], torch.full_like(batch["next_critic_obs"], 100))
