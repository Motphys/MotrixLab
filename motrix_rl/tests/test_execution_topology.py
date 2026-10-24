# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Common worker layout contracts, independent of providers and GPU hardware."""

import pytest
import torch

from motrix_rl import numa as system_numa
from motrix_rl.execution.topology import (
    _learner_node,
    resolve_collector_inference_devices,
    resolve_learner_devices,
    resolve_worker_topology,
    same_cuda_device,
    select_collector_cpus,
    split_num_envs,
)


def test_split_num_envs_requires_even_division() -> None:
    assert split_num_envs(2048, 2) == [1024, 1024]
    assert split_num_envs(2048, 1) == [2048]
    with pytest.raises(ValueError, match="divide evenly"):
        split_num_envs(2048, 3)
    with pytest.raises(ValueError, match="num_collectors must be >= 1"):
        split_num_envs(2048, 0)


def test_select_collector_cpus_chunks_and_shares() -> None:
    base = [0, 1, 2, 3, 4, 5]
    assert select_collector_cpus(base, 0, None) == base
    assert select_collector_cpus(base, 1, None) == base
    assert select_collector_cpus(base, 0, 3) == [0, 1, 2]
    assert select_collector_cpus(base, 1, 3) == [3, 4, 5]
    with pytest.raises(ValueError, match="no CPUs"):
        select_collector_cpus(base, 2, 4)
    with pytest.raises(ValueError, match="positive"):
        select_collector_cpus(base, 0, 0)
    with pytest.raises(ValueError, match="empty"):
        select_collector_cpus([], 0, None)


def test_topology_pairs_collectors_with_owner_and_chunks_local_cpus(monkeypatch) -> None:
    node_cpus = {0: list(range(8)), 1: list(range(100, 108))}
    monkeypatch.setattr(system_numa, "available_numa_nodes", lambda: [0, 1])
    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: index)
    monkeypatch.setattr(system_numa, "numa_node_cpus", lambda node: node_cpus[node])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    topology = resolve_worker_topology(
        8, 4, 2, ["cuda:0", "cuda:1"], "cuda", torch.device("cuda:0"), cpus_per_collector=4
    )
    assert [str(c.device) for c in topology.collectors] == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
    assert [c.numa_node for c in topology.collectors] == [0, 0, 1, 1]
    assert [c.cpus for c in topology.collectors] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [100, 101, 102, 103],
        [104, 105, 106, 107],
    ]
    assert topology.env_shards == [2] * 4
    assert sum(topology.env_shards) == 8
    assert topology.num_learners == 2
    assert topology.num_collectors == 4
    assert topology.per_learner == 2
    assert [topology.collector_owner(i) for i in range(4)] == [0, 0, 1, 1]
    assert topology.items_for_rank(list(range(4)), 0) == [0, 1]
    assert topology.ring_slice_for_rank(list(range(4)), 1) == [2, 3]


@pytest.mark.parametrize("nodes,device,locality", [([0], "cuda:0", 0), ([0, 1], "cpu", 0), ([0, 1], "cuda:0", None)])
def test_topology_without_gpu_locality_uses_inherited_affinity(monkeypatch, nodes, device, locality) -> None:
    monkeypatch.setattr(system_numa, "available_numa_nodes", lambda: nodes)
    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: locality)
    monkeypatch.setattr("motrix_rl.execution.topology.os.sched_getaffinity", lambda pid: {2, 3, 5}, raising=False)
    topology = resolve_worker_topology(2, 2, 1, None, "cpu", torch.device(device))
    assert topology.learners[0].numa_node is None
    assert [c.numa_node for c in topology.collectors] == [None, None]
    assert [c.cpus for c in topology.collectors] == [[2, 3, 5], [2, 3, 5]]
    assert topology.items_for_rank(["a", "b"], 0) == ["a", "b"]


@pytest.mark.parametrize(("hip_version", "runtime"), [(None, "cuda"), ("6.3", "hip")])
def test_topology_passes_explicit_gpu_runtime(monkeypatch, hip_version, runtime) -> None:
    seen = []
    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(system_numa, "gpu_numa_node", lambda index, runtime: seen.append((index, runtime)) or 2)
    assert _learner_node(torch.device("cuda:1"), multi_node=True) == 2
    assert seen == [(1, runtime)]
    assert _learner_node(torch.device("cpu"), multi_node=True) is None
    assert _learner_node(torch.device("cuda:1"), multi_node=False) is None
    assert seen == [(1, runtime)]


def test_topology_rejects_invalid_counts_and_devices() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        resolve_worker_topology(3, 3, 2, None, "cpu", torch.device("cpu"))
    with pytest.raises(ValueError, match="invalid worker counts"):
        resolve_worker_topology(4, 0, 1, None, "cpu", torch.device("cpu"))
    with pytest.raises(ValueError, match="collector_inference_device"):
        resolve_worker_topology(4, 2, 1, None, "meta", torch.device("cpu"))


def test_collector_inference_devices_follow_owner(monkeypatch) -> None:
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


def test_generic_cuda_falls_back_without_explicit_layout(monkeypatch) -> None:
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
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert [str(d) for d in resolve_learner_devices(None, 2, torch.device("cuda"))] == ["cuda:0", "cuda:1"]
    assert [str(d) for d in resolve_learner_devices(["cuda:3", "cuda:1"], 2, torch.device("cuda"))] == [
        "cuda:3",
        "cuda:1",
    ]
    assert resolve_learner_devices(None, 1, torch.device("cpu")) == []


def test_resolve_learner_devices_rejects_bad_configs(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="exactly num_learners"):
        resolve_learner_devices(["cuda:0"], 2, torch.device("cuda"))
    with pytest.raises(ValueError, match="more than once"):
        resolve_learner_devices(["cuda:0", "cuda:0"], 2, torch.device("cuda"))
    with pytest.raises(ValueError, match="does not exist"):
        resolve_learner_devices(["cuda:0", "cuda:5"], 2, torch.device("cuda"))
    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_learner_devices(None, 2, torch.device("cpu"))


@pytest.mark.parametrize(
    "left,right,expected", [("cuda", "cuda:0", True), ("cuda", "cuda:1", False), ("cpu", "cpu", False)]
)
def test_same_cuda_device_compares_ordinals(left, right, expected) -> None:
    assert same_cuda_device(torch.device(left), torch.device(right)) is expected
