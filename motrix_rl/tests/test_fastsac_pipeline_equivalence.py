# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-equivalence tests: 2c/2l must be indistinguishable from 1c/1l.

Three layers, in decreasing strictness:

1. ring/generation-merge: the same transitions fed through two shard rings
   (multi-collector path) must land in the replay buffer byte-identically to
   the single-ring path — env-major order, n-step adjacency included;
2. normalizer statistics: per-rank update + cross-rank stat sync, cycled,
   must always equal one global normalizer that saw ALL the data (this is
   the exact code path that produced NaN on the cluster);
3. DDP update averaging: two ranks on half batches + gradient AVG must match
   one rank on the full batch (CPU gloo, few steps) — covered by
   test_fastsac_ddp_equivalence.py, which needs real process spawning.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from fastsac_async_mocks import make_mock_learner

from motrix_rl.fastsac.async_impl.learner import Learner
from motrix_rl.fastsac.async_impl.transport import SharedTransitionRing
from motrix_rl.fastsac.buffer import EmpiricalNormalization, SimpleReplayBuffer

_OBS, _COBS, _ACT = 7, 5, 3


def _batch(envs: int, tag: float, seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(envs, _OBS, generator=g) + tag
    critic_obs = torch.randn(envs, _COBS, generator=g)
    actions = torch.randn(envs, _ACT, generator=g)
    rewards = torch.randn(envs, generator=g)
    dones = (torch.rand(envs, generator=g) < 0.1).long()
    truncations = (torch.rand(envs, generator=g) < 0.1).long()
    return obs, critic_obs, actions, rewards, dones, truncations


_make_learner = make_mock_learner


# --------------------------------------------------------------------- layer 1
def test_generation_merge_matches_single_ring_byte_for_byte() -> None:
    """Same 8 batches: one ring drained directly vs two shard rings merged."""
    envs, shards, batches = 8, 2, 8
    whole = [_batch(envs, tag=float(t), seed=t) for t in range(batches)]
    shard = lambda fields, r: tuple(f[r * (envs // shards) : (r + 1) * (envs // shards)] for f in fields)  # noqa: E731

    single_extends: list = []
    single = _make_learner([SharedTransitionRing(16, envs, _OBS, _COBS, _ACT)], single_extends)
    for fields in whole:
        assert single.rings[0].push(*fields)
    single.drain()

    dual_extends: list = []
    rings = [SharedTransitionRing(16, envs // shards, _OBS, _COBS, _ACT) for _ in range(shards)]
    dual = _make_learner(rings, dual_extends)
    # generation k: both collectors push their shard of batch k (env blocks
    # concatenate in collector order, matching the single-ring env order)
    for fields in whole:
        for r in range(shards):
            assert rings[r].push(*shard(fields, r))
    while dual.drain() == 0:
        pass

    assert len(single_extends) == len(dual_extends)
    for a, b in zip(single_extends, dual_extends):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
    # read cursors must advance by exactly the number of drained slots — a
    # desync here stalls the generation merge permanently (the learner waits
    # for a generation whose slots it already consumed)
    assert single.rings[0].read_idx == batches
    assert all(r.read_idx == batches for r in rings)


def test_generation_merge_preserves_n_step_adjacency() -> None:
    """Per-env trajectories in the merged buffer match the single-ring path."""
    envs, shards, batches = 8, 2, 6

    def make_rb():
        return SimpleReplayBuffer(n_env=envs, buffer_size=64, n_obs=_OBS, n_act=_ACT, n_critic_obs=_COBS, device="cpu")

    single_rb, dual_rb = make_rb(), make_rb()
    single = _make_learner([SharedTransitionRing(16, envs, _OBS, _COBS, _ACT)], None)
    single.agent.rb = single_rb
    rings = [SharedTransitionRing(16, envs // shards, _OBS, _COBS, _ACT) for _ in range(shards)]
    dual = _make_learner(rings, None)
    dual.agent.rb = dual_rb

    whole = [_batch(envs, tag=float(t), seed=100 + t) for t in range(batches)]
    for fields in whole:
        assert single.rings[0].push(*fields)
        for r in range(shards):
            assert rings[r].push(*tuple(f[r * (envs // shards) : (r + 1) * (envs // shards)] for f in fields))
    single.drain()
    while dual.drain() == 0:
        pass

    for name in ("observations", "actions", "rewards", "dones", "truncations"):
        torch.testing.assert_close(getattr(single_rb, name), getattr(dual_rb, name))
    assert single_rb.num_stored == dual_rb.num_stored


# --------------------------------------------------------------------- layer 2
def _fake_dist_sum(flats: list[torch.Tensor]):
    """Simulate a SUM all-reduce/broadcast over the given per-rank tensors."""

    class _Dist:
        def all_reduce(self, tensor, op=None):  # noqa: ARG002
            total = sum(flats)
            tensor.copy_(total)
            return tensor

        def broadcast(self, tensor, src=0):  # noqa: ARG002
            tensor.copy_(flats[0])
            return tensor

    return _Dist()


def test_normalizer_sync_cycles_equal_global_reference() -> None:
    """Cycles of (per-shard update -> cross-rank sync) == one global normalizer."""
    torch.manual_seed(0)
    dims, cycles, batch = 9, 50, 512
    ranks = [EmpiricalNormalization(dims, torch.device("cpu")) for _ in range(2)]
    reference = EmpiricalNormalization(dims, torch.device("cpu"))

    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(world_size=2, obs_normalizer=ranks[0], critic_obs_normalizer=ranks[1])
    learner.ddp_rank = 0

    for cycle in range(cycles):
        shards = [torch.randn(batch, dims) * (3.0 + cycle * 0.1) + cycle for _ in ranks]
        for norm, shard in zip(ranks, shards):
            if not norm.local_enabled:
                norm.seed_local_accumulators()
            norm.update(shard)
        reference.update(torch.cat(shards))

        # simulate the collective: both ranks run the SAME sync math on the
        # SUM of their sufficient statistics
        import torch.distributed as dist

        packs = [norm.local_sufficient_stats_flat() for norm in ranks]

        fake = _fake_dist_sum(packs)
        orig_all_reduce, orig_broadcast = dist.all_reduce, dist.broadcast
        dist.all_reduce, dist.broadcast = fake.all_reduce, fake.broadcast
        try:
            # run the real sync math on both ranks (each sees the SUM)
            for norm in ranks:
                learner.agent.obs_normalizer, learner.agent.critic_obs_normalizer = norm, None
                learner._sync_normalizer_stats()
        finally:
            dist.all_reduce, dist.broadcast = orig_all_reduce, orig_broadcast
            learner.agent.obs_normalizer, learner.agent.critic_obs_normalizer = ranks[0], ranks[1]

        for norm in ranks:
            assert torch.isfinite(norm._std).all(), f"cycle {cycle}: std went NaN"
            # float32 publics vs the reference's float32 incremental accumulation drift
            # over dozens of cycles; the merged path keeps locals in float64
            torch.testing.assert_close(norm._mean, reference._mean, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(norm._std, reference._std, rtol=1e-2, atol=1e-2)
        assert ranks[0].count == reference.count

    # zero-count start (the first-publish state) must stay finite too: both
    # ranks at count=0 -> merged n=0 -> the sync leaves init stats untouched.
    import torch.distributed as dist

    fresh = [EmpiricalNormalization(dims, torch.device("cpu")) for _ in range(2)]
    learner.agent.obs_normalizer, learner.agent.critic_obs_normalizer = fresh[0], fresh[1]
    zero = torch.zeros(1 + 2 * dims, dtype=torch.float64)
    orig = dist.all_reduce
    dist.all_reduce = lambda tensor, op=None: tensor.copy_(zero)  # noqa: ARG005
    try:
        learner._sync_normalizer_stats()
    finally:
        dist.all_reduce = orig
    assert torch.isfinite(fresh[0]._std).all() and fresh[0].count == 0
