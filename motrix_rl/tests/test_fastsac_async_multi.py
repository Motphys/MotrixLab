# Copyright Motphys Technology Co., Ltd. 2025, 2026

"""Multi-collector topology unit tests: control aggregation, env sharding,
strict-UTD accounting across sharded slots, stats aggregation and NUMA
helper selection. Process-level behavior is covered by end-to-end training."""

import math
from types import SimpleNamespace

import pytest
import torch
from fastsac_async_mocks import make_mock_learner

from motrix_rl.fastsac.async_impl.learner import Learner
from motrix_rl.fastsac.async_impl.numa import parse_cpulist, select_collector_cpus
from motrix_rl.fastsac.async_impl.stats import aggregate_collector_stats
from motrix_rl.fastsac.async_impl.topology import (
    CollectorInfo,
    LearnerInfo,
    TrainerTopology,
    resolve_learner_devices,
    resolve_trainer_topology,
    ring_transport_is_ipc,
    split_num_envs,
)
from motrix_rl.fastsac.async_impl.train import Trainer
from motrix_rl.fastsac.async_impl.transport import Control, SharedTransitionRing


def test_control_aggregates_per_collector_steps() -> None:
    control = Control(num_collectors=3)
    control.inc_collector_steps(0)
    control.inc_collector_steps(1)
    control.inc_collector_steps(1)
    control.inc_collector_steps(2)

    assert control.collector_steps == 4
    assert control.collector_steps_at(0) == 1
    assert control.collector_steps_at(1) == 2
    assert control.collector_steps_at(2) == 1


def test_control_resume_restarts_each_collector_from_checkpointed_iteration() -> None:
    control = Control(num_collectors=2)
    control.resume_collector_steps(5)

    # v is in full-batch equivalents: each collector's own counter restarts at v
    assert (control.collector_steps_at(0), control.collector_steps_at(1)) == (5, 5)
    assert control.collector_steps == 10


def test_single_collector_control_matches_previous_single_counter() -> None:
    control = Control()
    control.resume_collector_steps(7)
    control.inc_collector_steps(0)

    assert control.collector_steps == 8


def test_split_num_envs_requires_even_division() -> None:
    assert split_num_envs(2048, 2) == [1024, 1024]
    assert split_num_envs(2048, 1) == [2048]
    with pytest.raises(ValueError, match="divide evenly"):
        split_num_envs(2048, 3)
    with pytest.raises(ValueError, match="num_collectors must be >= 1"):
        split_num_envs(2048, 0)


def _strict_learner(num_updates: int) -> Learner:
    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(cfg=SimpleNamespace(num_updates=num_updates))
    learner.async_options = SimpleNamespace(utd_mode="strict")
    learner.control = SimpleNamespace(collector_steps=0, num_collectors=1)
    return learner


def _push_batch(ring, tag: float) -> None:
    n = ring.num_envs
    obs = torch.full((n, ring.obs.shape[-1]), tag)
    actions = torch.zeros(n, ring.actions.shape[-1])
    rewards = torch.zeros(n)
    dones = torch.zeros(n, dtype=torch.long)
    pushed = ring.push(obs, obs, actions, rewards, dones, dones)
    assert pushed


def test_drain_merges_shard_generations_into_full_batches() -> None:
    rings = [SharedTransitionRing(4, 4, 3, 3, 2) for _ in range(2)]
    extends: list = []
    learner = make_mock_learner(rings, extends)

    # only collector 0 produced generation 0: nothing is ingestible yet
    _push_batch(rings[0], 1.0)
    assert learner.drain() == 0
    assert extends == []

    # collector 1 catches up: generation 0 merges into one full 8-env batch
    _push_batch(rings[1], 2.0)
    assert learner.drain() == 1
    assert len(extends) == 1
    merged_obs = extends[0][0]
    # batched runs are slot-major: (count, num_envs, dim)
    assert merged_obs.shape == (1, 8, 3)
    # env blocks concatenate in collector order
    assert merged_obs[0, :4].eq(1.0).all()
    assert merged_obs[0, 4:].eq(2.0).all()
    # both read cursors advanced past the consumed generation
    assert all(ring.read_idx == 1 for ring in rings)


def test_drain_holds_generation_until_all_rings_deliver() -> None:
    rings = [SharedTransitionRing(4, 4, 3, 3, 2) for _ in range(2)]
    extends: list = []
    learner = make_mock_learner(rings, extends)

    _push_batch(rings[0], 1.0)
    _push_batch(rings[0], 1.5)
    _push_batch(rings[1], 2.0)
    assert learner.drain() == 1
    # generation 1 waits for collector 1's second slot
    assert learner._pending.next_gen == 1
    _push_batch(rings[1], 2.5)
    assert learner.drain() == 1
    assert all(ring.read_idx == 2 for ring in rings)


def test_single_ring_drain_batches_multiple_slots() -> None:
    ring = SharedTransitionRing(4, 4, 3, 3, 2)
    extends: list = []
    learner = make_mock_learner([ring], extends)

    for tag in (1.0, 2.0, 3.0):
        _push_batch(ring, tag)
    assert learner.drain() == 3
    # one extend_batch with all three batches stacked along the slot axis
    assert len(extends) == 1
    assert extends[0][0].shape == (3, 4, 3)
    assert extends[0][0][0].eq(1.0).all()
    assert extends[0][0][2].eq(3.0).all()
    assert ring.read_idx == 3


def _snapshot(collector_id: int, ret: float, episodes: int, lag: int, collect_ms: float) -> dict:
    return {
        "collector_id": collector_id,
        "return": ret,
        "ep_len": 100.0,
        "episodes": episodes,
        "reward_terms": {"alive": 0.5},
        "env_metrics": {},
        "policy_lag": lag,
        "timing_ms": {"collect": collect_ms},
    }


def test_aggregate_collector_stats_merges_snapshots() -> None:
    per_collector = {0: _snapshot(0, 10.0, 4, 1, 8.0), 1: _snapshot(1, 20.0, 6, 3, 12.0)}
    stats = aggregate_collector_stats(per_collector)

    assert stats["return"] == pytest.approx(15.0)
    assert stats["episodes"] == 10
    assert stats["policy_lag"] == 3
    assert stats["timing_ms"]["collect"] == pytest.approx(10.0)


def test_aggregate_collector_stats_skips_missing_collectors() -> None:
    stats = aggregate_collector_stats({0: _snapshot(0, 10.0, 4, 1, 8.0), 1: None})

    assert stats["return"] == pytest.approx(10.0)
    assert stats["episodes"] == 4


def test_aggregate_collector_stats_defaults_without_snapshots() -> None:
    stats = aggregate_collector_stats({0: None, 1: None})

    assert math.isnan(stats["return"])
    assert stats["episodes"] == 0
    assert stats["policy_lag"] == 0


def test_extend_batch_matches_repeated_extend_including_wraparound() -> None:
    """extend_batch writes exactly what successive extend calls would, including
    a run that wraps the circular buffer end — n-step time adjacency depends on it."""
    from motrix_rl.fastsac.buffer import SimpleReplayBuffer

    def make():
        return SimpleReplayBuffer(n_env=4, buffer_size=5, n_obs=3, n_act=2, n_critic_obs=3, device="cpu")

    def batch(tag: float, count: int):
        obs = torch.stack([torch.full((4, 3), tag) for _ in range(count)], dim=0)
        actions = torch.zeros(count, 4, 2)
        rewards = torch.tensor([[float(tag + j)] * 4 for j in range(count)])
        flags = torch.zeros(count, 4, dtype=torch.long)
        return obs, obs.clone(), actions, rewards, flags, flags.clone()

    stepwise, batched = make(), make()
    tags = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # runs past buffer_size -> wraps
    i = 0
    while i < len(tags):
        count = 2 if i + 1 < len(tags) else 1
        b = batch(0.0, count)  # obs tag overwritten per column below
        obs = torch.stack([torch.full((4, 3), tags[i + j]) for j in range(count)], dim=0)
        rewards = torch.tensor([[float(i + j)] * 4 for j in range(count)])
        batched.extend_batch(obs, obs.clone(), b[2], rewards, b[4], b[5])
        for j in range(count):
            single = batch(tags[i + j], 1)
            single = (single[0], single[1], single[2], torch.full((4, 1), float(i + j)), single[4], single[5])
            stepwise.extend(*(f[:, 0] for f in single))
        i += count

    for name in ("observations", "actions", "rewards", "dones"):
        torch.testing.assert_close(getattr(stepwise, name), getattr(batched, name))
    assert batched.ptr == stepwise.ptr
    assert batched.num_stored == 5


def test_parse_cpulist() -> None:
    assert parse_cpulist("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert parse_cpulist(" 5 ") == [5]
    assert parse_cpulist("") == []


def test_select_collector_cpus_chunks_and_shares() -> None:
    base = [0, 1, 2, 3, 4, 5]

    # without cpus_per_collector every collector shares the full set
    assert select_collector_cpus(base, 0, None) == base
    assert select_collector_cpus(base, 1, None) == base

    assert select_collector_cpus(base, 0, 3) == [0, 1, 2]
    assert select_collector_cpus(base, 1, 3) == [3, 4, 5]
    # over-subscription is a config error, not a silent empty binding
    with pytest.raises(ValueError, match="no CPUs"):
        select_collector_cpus(base, 2, 4)

    with pytest.raises(ValueError, match="positive"):
        select_collector_cpus(base, 0, 0)
    with pytest.raises(ValueError, match="empty"):
        select_collector_cpus([], 0, None)


def test_topology_pairs_collectors_with_owning_learner(monkeypatch) -> None:
    from motrix_rl.fastsac.async_impl import numa

    monkeypatch.setattr(numa, "available_numa_nodes", lambda: [0, 1])
    monkeypatch.setattr(numa, "gpu_numa_node", lambda index: {0: 0, 1: 1}.get(index))
    cuda = lambda n: torch.device("cuda", n)  # noqa: E731
    opts = SimpleNamespace(transition_ipc="auto", weight_ipc="auto", weight_ipc_min_bytes=0)

    # 2 learners x 2 collectors: each collector sits on its owner's GPU-local
    # node — the pair never straddles a NUMA boundary
    topo = resolve_trainer_topology(
        8,
        4,
        2,
        [cuda(0), cuda(1)],
        [cuda(0), cuda(0), cuda(1), cuda(1)],
        cuda(0),
        opts,
        actor_param_numel=0,
    )
    assert [learner.numa_node for learner in topo.learners] == [0, 1]
    assert [collector.numa_node for collector in topo.collectors] == [0, 0, 1, 1]
    assert topo.env_shards == [2, 2, 2, 2]
    assert topo.collector_owner(3) == 1
    assert [collector.ring_ipc for collector in topo.collectors] == [True, True, True, True]

    # multi-collector single learner: every collector follows the one learner
    topo = resolve_trainer_topology(8, 4, 1, [], [cuda(1)] * 4, cuda(1), opts, actor_param_numel=0)
    assert [learner.numa_node for learner in topo.learners] == [1]
    assert [collector.numa_node for collector in topo.collectors] == [1, 1, 1, 1]


def test_topology_chunks_cpus_by_node_local_ordinal(monkeypatch) -> None:
    """Each node's collectors chunk that node's CPU list from offset 0.

    A global collector index against a per-node base would offset rank>=1
    collectors past their own node's CPUs (asymmetric bindings, or a
    "leaves no CPUs" abort once the offset exceeds the node's list).
    """
    from motrix_rl.fastsac.async_impl import numa

    node_cpus = {0: list(range(8)), 1: list(range(100, 108))}
    monkeypatch.setattr(numa, "available_numa_nodes", lambda: [0, 1])
    monkeypatch.setattr(numa, "gpu_numa_node", lambda index: {0: 0, 1: 1}.get(index))
    monkeypatch.setattr(numa, "numa_node_cpus", lambda node: node_cpus[node])
    cuda = lambda n: torch.device("cuda", n)  # noqa: E731
    opts = SimpleNamespace(transition_ipc="auto", weight_ipc="auto", weight_ipc_min_bytes=0)

    topo = resolve_trainer_topology(
        8,
        4,
        2,
        [cuda(0), cuda(1)],
        [cuda(0), cuda(0), cuda(1), cuda(1)],
        cuda(0),
        opts,
        actor_param_numel=0,
        cpus_per_collector=4,
    )

    # node 0's collectors take node 0's CPUs from offset 0; node 1's
    # collectors take node 1's CPUs from offset 0 (NOT from offset 2*4)
    assert [c.cpus for c in topo.collectors] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [100, 101, 102, 103],
        [104, 105, 106, 107],
    ]


def test_topology_unbound_without_gpu_locality(monkeypatch) -> None:
    from motrix_rl.fastsac.async_impl import numa

    cuda = torch.device("cuda", 0)
    cpu = torch.device("cpu")
    opts = SimpleNamespace(transition_ipc="auto", weight_ipc="auto", weight_ipc_min_bytes=0)

    # single-node host: no binding anywhere
    monkeypatch.setattr(numa, "available_numa_nodes", lambda: [0])
    topo = resolve_trainer_topology(2, 2, 1, [], [cuda, cpu], cuda, opts, actor_param_numel=0)
    assert [learner.numa_node for learner in topo.learners] == [None]
    assert [collector.numa_node for collector in topo.collectors] == [None, None]

    # multi-node host, CPU learner: no binding anywhere
    monkeypatch.setattr(numa, "available_numa_nodes", lambda: [0, 1])
    topo = resolve_trainer_topology(2, 2, 1, [], [cpu, cpu], cpu, opts, actor_param_numel=0)
    assert [learner.numa_node for learner in topo.learners] == [None]
    assert [collector.numa_node for collector in topo.collectors] == [None, None]

    # multi-node host, unknown GPU locality: no binding anywhere
    monkeypatch.setattr(numa, "gpu_numa_node", lambda index: None)
    topo = resolve_trainer_topology(2, 2, 1, [], [cuda, cuda], cuda, opts, actor_param_numel=0)
    assert [learner.numa_node for learner in topo.learners] == [None]
    assert [collector.numa_node for collector in topo.collectors] == [None, None]

    # index-less cuda means device 0
    monkeypatch.setattr(numa, "gpu_numa_node", lambda index: {0: 0}.get(index))
    topo = resolve_trainer_topology(2, 2, 1, [], [cuda, cuda], torch.device("cuda"), opts, actor_param_numel=0)
    assert [learner.numa_node for learner in topo.learners] == [0]


def test_topology_rejects_bad_counts() -> None:
    opts = SimpleNamespace(transition_ipc="auto", weight_ipc="auto", weight_ipc_min_bytes=0)
    with pytest.raises(ValueError, match="divide evenly"):
        resolve_trainer_topology(3, 3, 2, [], [], torch.device("cpu"), opts, actor_param_numel=0)
    with pytest.raises(ValueError, match="invalid worker counts"):
        resolve_trainer_topology(4, 0, 1, [], [], torch.device("cpu"), opts, actor_param_numel=0)
    with pytest.raises(ValueError, match="collector devices"):
        resolve_trainer_topology(4, 2, 1, [], [torch.device("cpu")], torch.device("cpu"), opts, actor_param_numel=0)


def test_ring_slice_partitions_collectors_by_ownership() -> None:
    rings = list(range(6))
    topo = TrainerTopology(
        learners=[LearnerInfo(rank=r, device=torch.device("cpu"), numa_node=None, cpus=[]) for r in range(3)],
        collectors=[CollectorInfo(i, i // 2, 1, torch.device("cpu"), None, [], False, False) for i in range(6)],
    )

    # collector i belongs to learner i // k, so each rank drains a contiguous block
    assert topo.ring_slice_for_rank(rings, 0) == [0, 1]
    assert topo.ring_slice_for_rank(rings, 1) == [2, 3]
    assert topo.ring_slice_for_rank(rings, 2) == [4, 5]
    # 1:1 topology: one ring per learner
    two = TrainerTopology(
        learners=[
            LearnerInfo(rank=0, device=torch.device("cpu"), numa_node=None, cpus=[]),
            LearnerInfo(rank=1, device=torch.device("cpu"), numa_node=None, cpus=[]),
        ],
        collectors=[CollectorInfo(i, i, 1, torch.device("cpu"), None, [], False, False) for i in range(2)],
    )
    assert two.ring_slice_for_rank(rings, 1) == [1]
    # single learner owns everything
    one = TrainerTopology(
        learners=[LearnerInfo(rank=0, device=torch.device("cpu"), numa_node=None, cpus=[])],
        collectors=[CollectorInfo(i, 0, 1, torch.device("cpu"), None, [], False, False) for i in range(6)],
    )
    assert one.ring_slice_for_rank(rings, 0) == rings


def _lockstep_learner(num_updates: int, utd_mode: str, last_gstep: int = 5) -> tuple[Learner, list]:
    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(
        cfg=SimpleNamespace(num_updates=num_updates),
        rb=SimpleNamespace(num_stored=1),
        device=torch.device("cpu"),
    )
    learner.async_options = SimpleNamespace(utd_mode=utd_mode, idle_sleep_s=0.0)
    learner.control = SimpleNamespace(collector_steps=100, num_collectors=2, stop=False)
    learner.ddp_rank = 0
    learner._learning_starts = 10
    learner._staging = None
    learner._last_train_gstep = last_gstep
    learner._last_publish_ms = 0.0
    learner.weights = []
    learner.agent.rb.num_stored = 10
    calls: list = []
    learner.agent.update = lambda n: calls.append(n) or {}
    learner.publish_if_due = lambda: None
    return learner, calls


def test_lockstep_strict_due_scales_with_global_step_delta(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "broadcast", lambda tensor, src=0: tensor)
    learner, calls = _lockstep_learner(num_updates=4, utd_mode="strict")

    assert learner.maybe_train(8) is not None  # delta 3
    assert sum(calls) == 12 and all(c == 4 for c in calls)
    assert learner.maybe_train(8) is None  # no new global batches -> no update
    assert sum(calls) == 12
    assert learner.maybe_train(9) is not None
    assert sum(calls) == 16


def test_lockstep_learner_bound_runs_base_on_any_progress(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "broadcast", lambda tensor, src=0: tensor)
    learner, calls = _lockstep_learner(num_updates=4, utd_mode="learner_bound")

    assert learner.maybe_train(6) is not None
    assert sum(calls) == 4
    assert learner.maybe_train(6) is None
    assert sum(calls) == 4


def test_lockstep_gated_before_learning_starts(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "broadcast", lambda tensor, src=0: tensor)
    learner, calls = _lockstep_learner(num_updates=4, utd_mode="strict")
    learner.control.collector_steps = 19  # < learning_starts(10) * num_collectors(2)

    assert learner.maybe_train(8) is None
    assert calls == []


def _ipc_opts(mode: str = "auto") -> SimpleNamespace:
    return SimpleNamespace(transition_ipc=mode)


def test_ring_transport_ipc_colocated_collectors() -> None:
    """Collectors co-located with their owning learner's GPU get IPC rings per rank."""
    cuda = lambda n: torch.device("cuda", n)  # noqa: E731

    # multi-learner: each rank's collectors infer on its GPU -> all IPC
    flags = ring_transport_is_ipc(_ipc_opts(), [cuda(0), cuda(1)], [cuda(0), cuda(0), cuda(1), cuda(1)], 4, 2, cuda(0))
    assert flags == [True, True, True, True]

    # a cross-GPU collector drags its whole rank back to host rings
    flags = ring_transport_is_ipc(_ipc_opts(), [cuda(0), cuda(1)], [cuda(0), cuda(1), cuda(1), cuda(1)], 4, 2, cuda(0))
    assert flags == [False, False, True, True]

    # single learner, multi collector, all on its GPU -> IPC
    flags = ring_transport_is_ipc(_ipc_opts(), [], [cuda(0)] * 4, 4, 1, cuda(0))
    assert flags == [True] * 4

    # cpu collector inference -> host rings
    flags = ring_transport_is_ipc(_ipc_opts(), [], [torch.device("cpu")] * 2, 2, 1, cuda(0))
    assert flags == [False, False]


def test_normalizer_stat_sync_merges_shards(monkeypatch) -> None:
    """_sync_normalizer_stats merges per-rank (n, Sum, SumSq) into global stats."""
    import torch.distributed as dist

    from motrix_rl.fastsac.buffer import EmpiricalNormalization

    def make_norm(seed: int, data: torch.Tensor) -> EmpiricalNormalization:
        norm = EmpiricalNormalization(data.shape[1:], torch.device("cpu"))
        torch.manual_seed(seed)
        norm.update(data)
        return norm

    a = make_norm(0, torch.randn(1000, 5) * 2.0 + 1.0)
    b = make_norm(1, torch.randn(300, 5) * 0.5 - 2.0)

    # the other rank's flat sufficient statistics (seeded from its publics)
    b.seed_local_accumulators()
    other = b.local_sufficient_stats_flat()

    def fake_all_reduce(flat, op=None):  # noqa: ARG001
        flat += other

    a_count, a_mean, a_var = a.count.item(), a._mean.clone(), a._var.clone()

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(world_size=2, obs_normalizer=a, critic_obs_normalizer=None)
    learner._sync_normalizer_stats()

    # global empirical stats over both shards (Chan merge of sufficient stats)
    n = a_count + b.count.item()
    global_mean = (a_count * a_mean.squeeze(0) + b.count * b._mean.squeeze(0)) / n
    torch.testing.assert_close(a._mean.squeeze(0), global_mean)
    global_var = (
        a_count * a_var.squeeze(0)
        + b.count * b._var.squeeze(0)
        + a_count * b.count / n * (a_mean.squeeze(0) - b._mean.squeeze(0)) ** 2
    ) / n
    torch.testing.assert_close(a._var.squeeze(0), global_var)
    assert a.count == n


def test_normalizer_stat_sync_zero_count_is_noop(monkeypatch) -> None:
    """A never-updated normalizer (count=0) must contribute zeros, not NaN."""
    import torch.distributed as dist

    from motrix_rl.fastsac.buffer import EmpiricalNormalization

    fresh = EmpiricalNormalization(5, torch.device("cpu"))  # count=0
    other = EmpiricalNormalization(5, torch.device("cpu"))
    other.update(torch.randn(300, 5))

    other.seed_local_accumulators()
    other_flat = other.local_sufficient_stats_flat()
    monkeypatch.setattr(dist, "all_reduce", lambda flat, op=None: flat.__iadd__(other_flat))
    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(world_size=2, obs_normalizer=fresh, critic_obs_normalizer=None)
    learner._sync_normalizer_stats()

    assert torch.isfinite(fresh._mean).all() and torch.isfinite(fresh._std).all()
    torch.testing.assert_close(fresh._mean.squeeze(0), other._mean.squeeze(0))
    torch.testing.assert_close(fresh._var.squeeze(0), other._var.squeeze(0))
    assert fresh.count == 300


def _trainer_with_device(device: str | None) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer._rlcfg = SimpleNamespace(device=device)
    return trainer


def test_resolve_learner_devices_expands_generic_cuda(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)

    devices = resolve_learner_devices(None, 2, torch.device("cuda"))
    assert [str(d) for d in devices] == ["cuda:0", "cuda:1"]

    devices = resolve_learner_devices(["cuda:3", "cuda:1"], 2, torch.device("cuda"))
    assert [str(d) for d in devices] == ["cuda:3", "cuda:1"]


def test_resolve_learner_devices_rejects_bad_configs(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)

    with pytest.raises(ValueError, match="exactly num_learners"):
        resolve_learner_devices(["cuda:0"], 2, torch.device("cuda"))
    with pytest.raises(ValueError, match="more than once"):
        resolve_learner_devices(["cuda:0", "cuda:0"], 2, torch.device("cuda"))
    with pytest.raises(ValueError, match="does not exist"):
        resolve_learner_devices(["cuda:0", "cuda:5"], 2, torch.device("cuda"))
    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_learner_devices(None, 2, torch.device("cpu"))
