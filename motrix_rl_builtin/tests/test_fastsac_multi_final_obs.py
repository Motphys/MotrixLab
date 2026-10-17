# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Async collector shards preserve terminal rows on the rank-local timeline."""

from types import SimpleNamespace

import pytest
import torch
from fastsac_async_mocks import make_mock_learner
from motrix_rl_builtin.fastsac.async_impl.learner import DrainStaging
from motrix_rl_builtin.fastsac.async_impl.transport import IpcTransitionRing, RingCursors
from motrix_rl_builtin.fastsac.async_impl.transport.ring import (
    FinalObsPayload,
    FinalObsStorage,
    SharedTransitionRing,
)
from motrix_rl_builtin.fastsac.buffer import SimpleReplayBuffer


def _ring(num_envs, capacity=5, *, ipc=False):
    final = FinalObsStorage(capacity, num_envs, 2, 3)
    if ipc:
        slots = torch.zeros(capacity, num_envs, 2 + 3 + 1 + 3, device="cuda")
        return IpcTransitionRing(RingCursors(), slots, capacity, num_envs, 2, 3, 1, final_obs_store=final)
    return SharedTransitionRing(capacity, num_envs, 2, 3, 1, final_obs_store=final)


def _rows(generation, env_ids, terminal=False):
    tags = generation * 100 + torch.tensor(env_ids, dtype=torch.float32)
    if terminal:
        tags = tags + 10000
    policy = torch.stack((tags, tags + 0.5), dim=1)
    critic = torch.stack((tags + 100000, tags + 200000, tags + 300000), dim=1)
    return policy, critic


def _push(ring, generation, offset, ids):
    obs, critic = _rows(generation, list(range(offset, offset + ring.num_envs)))
    truncated = torch.zeros(ring.num_envs, dtype=torch.long)
    payload = None
    if ids:
        truncated[ids] = 1
        final_obs, final_critic = _rows(generation, [offset + i for i in ids], terminal=True)
        payload = FinalObsPayload(torch.tensor(ids), final_obs, final_critic)
    return ring.push(
        obs,
        critic,
        torch.zeros(ring.num_envs, 1),
        torch.ones(ring.num_envs),
        torch.zeros_like(truncated),
        truncated,
        final_obs=payload,
    )


def _learner(rings, *, device="cpu", n_steps=1, buffer_size=8, staging=False):
    learner = make_mock_learner(rings, [])
    learner.agent.device = torch.device(device)
    learner.agent.rb = SimpleReplayBuffer(
        sum(r.num_envs for r in rings),
        buffer_size,
        2,
        1,
        3,
        n_steps=n_steps,
        device=device,
        final_obs_bootstrap=True,
    )
    learner.agent.window_metrics = SimpleNamespace(add=lambda *args, **kwargs: None)
    # IDs address this rank's replay, not global DDP environment IDs.
    learner.ddp_rank = 2
    if staging:
        learner._staging = DrainStaging(
            learner.agent.rb, learner.async_options.max_ingest_per_iter, learner.agent.device
        )
    return learner


def _assert_final(rb, generation, env_ids):
    slot = generation % rb._cap
    actual_ids = torch.nonzero(rb._final_slot[:, slot] >= 0).flatten().cpu().tolist()
    assert actual_ids == sorted(env_ids)
    if env_ids:
        rows = rb._final_slot[env_ids, slot].long()
        policy, critic = _rows(generation, env_ids, terminal=True)
        torch.testing.assert_close(rb._final_obs_table[rows].cpu(), policy)
        torch.testing.assert_close(rb._final_critic_table[rows].cpu(), critic)


def test_multi_collector_sparse_ids_and_empty_generations_remain_aligned():
    # Unequal shards, including a payload-less middle ring, pin cumulative
    # num_envs offsets rather than ring_index * first_shard_size.
    rings = [_ring(2), _ring(3), _ring(1)]
    learner = _learner(rings)
    calls = []

    def capture(*fields, final_payloads=None):
        calls.append((fields, final_payloads))

    learner.agent.rb = SimpleNamespace(extend_batch=capture)
    plans = [([1, 0], [0]), ([], []), ([0], []), ([], [0])]
    for generation, (first_ids, last_ids) in enumerate(plans):
        assert _push(rings[0], generation, 0, first_ids)
        assert _push(rings[1], generation, 2, [])
        assert _push(rings[2], generation, 5, last_ids)
    assert learner.drain() == 4
    assert len(calls) == 1
    fields, payloads = calls[0]
    assert len(payloads) == 4
    expected_ids = [[1, 0, 5], [], [0], [5]]
    for generation, ids in enumerate(expected_ids):
        obs, critic = _rows(generation, list(range(6)))
        torch.testing.assert_close(fields[0][generation], obs)
        torch.testing.assert_close(fields[1][generation], critic)
        payload = payloads[generation]
        if not ids:
            assert payload is None
        else:
            assert payload.env_ids.tolist() == ids
            final_obs, final_critic = _rows(generation, ids, terminal=True)
            torch.testing.assert_close(payload.obs, final_obs)
            torch.testing.assert_close(payload.critic_obs, final_critic)
    assert all(r.read_completed_count == 4 for r in rings)


def test_staggered_collectors_backpressure_and_wrap_preserve_final_rows():
    fast, slow = _ring(2, capacity=5), _ring(3, capacity=4)
    learner = _learner([fast, slow], buffer_size=5)
    for first in range(0, 24, 3):
        for generation in range(first, first + 3):
            assert _push(fast, generation, 0, [0, 1])
        # Sparse final storage backpressures before the main transition ring.
        assert fast.size() == 3 < fast.capacity
        assert fast.is_full()
        assert not _push(fast, first + 3, 0, [0, 1])
        assert learner.drain() == 0
        assert fast.read_completed_count == first
        assert slow.read_completed_count == first
        for generation in range(first, first + 3):
            ids = [2, 0] if generation % 2 == 0 else []
            assert _push(slow, generation, 2, ids)
            # Physical ring boundaries may split ready generations into runs.
            while learner.agent.rb.ptr <= generation:
                assert learner.drain() > 0
            learner.wait_ingest()
            assert learner.agent.rb.ptr == generation + 1
            _assert_final(learner.agent.rb, generation, [0, 1] + ([4, 2] if ids else []))
        assert not fast.is_full()
        assert fast.read_completed_count == slow.read_completed_count == first + 3
        # All retained rows must still belong to their generation after both
        # transition/sparse transport wraps and replay-slot reuse.
        for generation in range(max(0, first + 3 - learner.agent.rb._cap), first + 3):
            ids = [0, 1, 4, 2] if generation % 2 == 0 else [0, 1]
            _assert_final(learner.agent.rb, generation, ids)


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize(
    "path",
    [
        "cpu",
        *[
            pytest.param(path, marks=[pytest.mark.integration, pytest.mark.slow])
            for path in ("cuda-staged", "cuda-mixed", "cuda-ipc")
        ],
    ],
)
def test_multi_collector_bootstraps_at_each_envs_final_boundary(n_steps, path):
    if path != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    has_ipc = path in ("cuda-mixed", "cuda-ipc")
    rings = [_ring(2, ipc=path == "cuda-ipc"), _ring(3, ipc=has_ipc)]
    learner = _learner(
        rings,
        device="cuda" if path != "cpu" else "cpu",
        n_steps=n_steps,
        staging=path == "cuda-staged",
    )
    # Environment 1 times out at t=1; environment 4 at t=2. Other
    # environments continue, and auto-reset successors differ from final rows.
    for generation in range(5):
        assert _push(rings[0], generation, 0, [1] if generation == 1 else [])
        assert _push(rings[1], generation, 2, [2] if generation == 2 else [])
        if has_ipc:
            torch.cuda.synchronize()  # publish IPC producer's completed event
        assert learner.drain() == 1
    learner.wait_ingest()
    _assert_final(learner.agent.rb, 1, [1])
    _assert_final(learner.agent.rb, 2, [4])
    torch.manual_seed(7)
    sample = learner.agent.rb.sample(128)
    starts = (sample["obs"][:, 0] // 100).long().cpu()
    env_ids = (sample["obs"][:, 0] % 100).long().cpu()
    saw_terminal = set()
    saw_continuing = False
    for row, (start, env_id) in enumerate(zip(starts.tolist(), env_ids.tolist())):
        boundary = {1: 1, 4: 2}.get(env_id)
        terminal = boundary is not None and start <= boundary < start + n_steps
        if terminal:
            end = boundary
            saw_terminal.add(env_id)
            expected_obs, expected_critic = _rows(end, [env_id], terminal=True)
            assert sample["truncations"][row].item() == 1
            assert sample["dones"][row].item() == 0
        else:
            end = min(start + n_steps, 4)
            expected_obs, expected_critic = _rows(end, [env_id])
            assert sample["truncations"][row].item() == 0
            saw_continuing = True
        torch.testing.assert_close(sample["next_obs"][row].cpu(), expected_obs[0])
        torch.testing.assert_close(sample["next_critic_obs"][row].cpu(), expected_critic[0])
    assert saw_terminal == {1, 4}
    assert saw_continuing
    if has_ipc:
        torch.cuda.synchronize()
    assert all(r.read_completed_count == 5 for r in rings)
