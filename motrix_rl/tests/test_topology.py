# Copyright Motphys Technology Co., Ltd. 2025, 2026

"""Tests for generic asynchronous worker topology resolution."""

import pytest
import torch

from motrix_rl import numa as system_numa
from motrix_rl.topology import (
    AsyncTrainerTopology,
    CollectorInfo,
    LearnerInfo,
    _allocate_worker_cpus,
    resolve_async_trainer_topology,
    resolve_collector_inference_devices,
    resolve_learner_devices,
    resolve_learner_numa_node,
    split_num_envs,
)


def test_split_num_envs_requires_even_division() -> None:
    assert split_num_envs(2048, 2) == [1024, 1024]
    assert split_num_envs(2048, 1) == [2048]
    with pytest.raises(ValueError, match="divide evenly"):
        split_num_envs(2048, 3)
    with pytest.raises(ValueError, match="num_collectors must be >= 1"):
        split_num_envs(2048, 0)


def test_allocate_worker_cpus_consumes_shared_domain() -> None:
    available = [0, 1, 2, 3, 4, 5]

    assert _allocate_worker_cpus(available, 3) == (0, 1, 2)
    assert available == [3, 4, 5]
    assert _allocate_worker_cpus(available, 3) == (3, 4, 5)
    assert available == []
    with pytest.raises(ValueError, match="empty"):
        _allocate_worker_cpus(available, 1)
    with pytest.raises(ValueError, match="positive"):
        _allocate_worker_cpus([0], 0)
    with pytest.raises(ValueError, match="empty"):
        _allocate_worker_cpus([], None)


def test_allocate_worker_cpus_shares_domain_when_unlimited() -> None:
    available = [0, 1, 2]
    assert _allocate_worker_cpus(available, None) == (0, 1, 2)
    assert available == [0, 1, 2]


def test_topology_connects_collectors_to_local_learners(monkeypatch) -> None:
    monkeypatch.setattr(system_numa, "available_numa_nodes", lambda: [0, 1])
    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: {0: 0, 1: 1}.get(index))
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    cuda = lambda n: torch.device("cuda", n)  # noqa: E731

    topo = resolve_async_trainer_topology(
        8,
        4,
        2,
        ["cuda:0", "cuda:1"],
        "cuda",
        cuda(0),
    )
    assert [str(learner.device) for learner in topo.learners] == ["cuda:0", "cuda:1"]
    assert [str(collector.infer_device) for collector in topo.collectors] == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
    assert [learner.numa_node for learner in topo.learners] == [0, 1]
    assert [collector.numa_node for collector in topo.collectors] == [0, 0, 1, 1]
    assert topo.env_shards == (2, 2, 2, 2)
    assert [channel.consumer_id for channel in topo.transition_channels] == [
        0,
        0,
        1,
        1,
    ]
    assert [channel.producer_id for channel in topo.weight_channels] == [
        0,
        0,
        1,
        1,
    ]

    topo = resolve_async_trainer_topology(8, 4, 1, None, "cuda", cuda(1))
    assert [str(collector.infer_device) for collector in topo.collectors] == ["cuda:1"] * 4
    assert [learner.numa_node for learner in topo.learners] == [1]
    assert [collector.numa_node for collector in topo.collectors] == [1, 1, 1, 1]


def test_topology_allocates_unique_cpus_across_learners_sharing_domain(monkeypatch) -> None:
    node_cpus = {0: list(range(8)), 1: list(range(100, 108))}
    monkeypatch.setattr(system_numa, "available_numa_nodes", lambda: [0, 1])
    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: {0: 0, 1: 1}.get(index))
    monkeypatch.setattr(system_numa, "numa_node_cpus", lambda node: node_cpus[node])
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(8)) | set(range(100, 108)))
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    cuda = lambda n: torch.device("cuda", n)  # noqa: E731

    topo = resolve_async_trainer_topology(
        8,
        4,
        2,
        ["cuda:0", "cuda:1"],
        "cuda",
        cuda(0),
        cpus_per_collector=4,
    )
    assert [collector.cpus for collector in topo.collectors] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (100, 101, 102, 103),
        (104, 105, 106, 107),
    ]


def test_topology_shares_cpu_domain_between_learners(monkeypatch) -> None:
    monkeypatch.setattr(system_numa, "available_numa_nodes", lambda: [0, 1])
    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: 0)
    monkeypatch.setattr(system_numa, "numa_node_cpus", lambda node: list(range(8)))
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(8)))
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)

    topo = resolve_async_trainer_topology(
        8,
        4,
        2,
        ["cuda:0", "cuda:1"],
        "cpu",
        torch.device("cuda:0"),
        cpus_per_collector=2,
    )

    assert [collector.cpus for collector in topo.collectors] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    assert [channel.consumer_id for channel in topo.transition_channels].count(0) == 2
    assert [channel.consumer_id for channel in topo.transition_channels].count(1) == 2


def test_topology_intersects_numa_cpus_with_inherited_affinity(monkeypatch) -> None:
    monkeypatch.setattr(system_numa, "available_numa_nodes", lambda: [0, 1])
    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: 0)
    monkeypatch.setattr(system_numa, "numa_node_cpus", lambda node: list(range(8)))
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: {0, 1, 4, 5})

    topology = resolve_async_trainer_topology(
        4,
        2,
        1,
        None,
        "cuda",
        torch.device("cuda:0"),
        cpus_per_collector=2,
    )

    assert [collector.cpus for collector in topology.collectors] == [(0, 1), (4, 5)]


def test_topology_unbound_without_gpu_locality(monkeypatch) -> None:
    cuda = torch.device("cuda", 0)
    cpu = torch.device("cpu")
    monkeypatch.setattr(system_numa, "available_numa_nodes", lambda: [0])
    topo = resolve_async_trainer_topology(2, 2, 1, None, "cpu", cuda)
    assert [learner.numa_node for learner in topo.learners] == [None]
    assert [collector.numa_node for collector in topo.collectors] == [None, None]

    monkeypatch.setattr(system_numa, "available_numa_nodes", lambda: [0, 1])
    topo = resolve_async_trainer_topology(2, 2, 1, None, "cpu", cpu)
    assert [learner.numa_node for learner in topo.learners] == [None]
    assert [collector.numa_node for collector in topo.collectors] == [None, None]

    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: None)
    topo = resolve_async_trainer_topology(2, 2, 1, None, "cuda", cuda)
    assert [learner.numa_node for learner in topo.learners] == [None]
    assert [collector.numa_node for collector in topo.collectors] == [None, None]

    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: {0: 0}.get(index))
    topo = resolve_async_trainer_topology(2, 2, 1, None, "cuda", torch.device("cuda"))
    assert [learner.numa_node for learner in topo.learners] == [0]


@pytest.mark.parametrize(("hip_version", "runtime"), [(None, "cuda"), ("6.3", "hip")])
def test_resolve_learner_numa_node_passes_runtime(monkeypatch, hip_version, runtime) -> None:
    seen = []
    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: seen.append((index, runtime)) or 2)
    assert resolve_learner_numa_node(torch.device("cuda:1"), multi_node=True) == 2
    assert seen == [(1, runtime)]
    assert resolve_learner_numa_node(torch.device("cpu"), multi_node=True) is None
    assert resolve_learner_numa_node(torch.device("cuda:1"), multi_node=False) is None
    assert seen == [(1, runtime)]


def test_topology_rejects_bad_counts() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        resolve_async_trainer_topology(3, 3, 2, None, "cpu", torch.device("cpu"))
    with pytest.raises(ValueError, match="invalid worker counts"):
        resolve_async_trainer_topology(4, 0, 1, None, "cpu", torch.device("cpu"))
    with pytest.raises(ValueError, match="collector_inference_device"):
        resolve_async_trainer_topology(4, 2, 1, None, "meta", torch.device("cpu"))


def test_ring_slice_partitions_collectors_by_ownership() -> None:
    topo = AsyncTrainerTopology(
        learners=tuple(
            LearnerInfo(
                rank=rank,
                device=torch.device("cpu"),
                numa_node=None,
                cpus=(),
            )
            for rank in range(3)
        ),
        collectors=tuple(CollectorInfo(i, 1, torch.device("cpu"), None, ()) for i in range(6)),
    )
    assert topo.num_learners == 3
    assert topo.num_collectors == 6
    assert topo.env_shards == (1,) * 6


def test_resolve_collector_inference_devices_colocates_with_owner(monkeypatch) -> None:
    cuda = lambda n: torch.device("cuda", n)  # noqa: E731
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert resolve_collector_inference_devices("cuda", [], 3, 1, torch.device("cuda")) == [cuda(0)] * 3
    assert resolve_collector_inference_devices("cuda", [], 2, 1, cuda(1)) == [cuda(1)] * 2
    assert resolve_collector_inference_devices("cuda", [cuda(0), cuda(1)], 4, 2, cuda(0)) == [
        cuda(0),
        cuda(0),
        cuda(1),
        cuda(1),
    ]
    assert resolve_collector_inference_devices("cuda:1", [], 2, 1, cuda(0)) == [cuda(1)] * 2
    assert resolve_collector_inference_devices("cuda", [], 2, 1, torch.device("cpu")) == [cuda(0)] * 2
    assert resolve_collector_inference_devices("cpu", [cuda(0), cuda(1)], 4, 2, cuda(0)) == [torch.device("cpu")] * 4

    with pytest.raises(ValueError, match="collector_inference_device"):
        resolve_collector_inference_devices("meta", [], 1, 1, cuda(0))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="only 1 device"):
        resolve_collector_inference_devices("cuda:1", [], 1, 1, cuda(0))


def test_generic_cuda_falls_back_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cpu = torch.device("cpu")
    assert resolve_collector_inference_devices("cuda", [], 2, 1, cpu) == [cpu, cpu]
    cuda = lambda n: torch.device("cuda", n)  # noqa: E731
    assert resolve_collector_inference_devices("cuda", [cuda(0), cuda(1)], 4, 2, cpu) == [
        cuda(0),
        cuda(0),
        cuda(1),
        cuda(1),
    ]


def test_resolve_learner_devices_expands_generic_cuda(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    assert [str(device) for device in resolve_learner_devices(None, 2, torch.device("cuda"))] == ["cuda:0", "cuda:1"]
    assert [str(device) for device in resolve_learner_devices(["cuda:3", "cuda:1"], 2, torch.device("cuda"))] == [
        "cuda:3",
        "cuda:1",
    ]


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
