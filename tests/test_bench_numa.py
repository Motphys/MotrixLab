# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest

from bench.env_benchmark.runner import split_num_envs
from motrix_rl import numa


def test_parse_sparse_id_list():
    assert numa.parse_cpulist("0-2,5,8-9\n") == [0, 1, 2, 5, 8, 9]


def test_discovery_intersects_cpu_and_memory_constraints(tmp_path):
    (tmp_path / "online").write_text("0,2,5")
    for node, cpus in [(0, "0-3"), (2, "4-7"), (5, "8-11")]:
        directory = tmp_path / f"node{node}"
        directory.mkdir()
        (directory / "cpulist").write_text(cpus)
    status = tmp_path / "status"
    status.write_text("Mems_allowed_list:\t2,5\n")
    assert numa.discover_nodes(tmp_path, status, {1, 4, 6, 9}) == [
        numa.NodePlacement(2, (4, 6)),
        numa.NodePlacement(5, (9,)),
    ]


def test_missing_topology_degrades_to_unbound_worker(tmp_path):
    assert numa.discover_nodes(tmp_path, tmp_path / "status", {3}) == [numa.NodePlacement(None, ())]


def test_known_unusable_topology_is_not_bypassed(tmp_path):
    (tmp_path / "online").write_text("0")
    node = tmp_path / "node0"
    node.mkdir()
    (node / "cpulist").write_text("0-3")
    status = tmp_path / "status"
    status.write_text("Mems_allowed_list:\t0\n")
    with pytest.raises(ValueError, match="No online NUMA"):
        numa.discover_nodes(tmp_path, status, {8})


def test_batch_split_preserves_total_and_is_balanced():
    nodes = [numa.NodePlacement(index, (index,)) for index in (0, 2, 5)]
    assert split_num_envs(8, nodes) == [3, 3, 2]
    with pytest.raises(ValueError, match="must be at least"):
        split_num_envs(2, nodes)


def test_missing_libnuma_reports_cpu_only(monkeypatch):
    affinity = []
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda pid, cpus: affinity.append(tuple(cpus)))
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda pid: {4, 6})
    monkeypatch.setattr(numa.ctypes.util, "find_library", lambda name: None)
    result = numa.apply_placement(numa.NodePlacement(2, (4, 6)))
    assert affinity == [(4, 6)]
    assert result["cpu_bound"] is True
    assert result["actual_cpus"] == [4, 6]
    assert result["memory_bound"] is False
    assert result["memory_policy"] == "cpu-only"
    assert "libnuma" in result["memory_error"]


def test_successful_memory_policy_uses_valid_node_zero_mask(monkeypatch):
    requests = []

    class SetPolicy:
        def __call__(self, policy, mask, maxnode):
            requests.append((policy, mask[0], maxnode))
            return 0

    class Library:
        set_mempolicy = SetPolicy()

    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(numa.ctypes.util, "find_library", lambda name: "libnuma.so.1")
    monkeypatch.setattr(numa.ctypes, "CDLL", lambda *args, **kwargs: Library())
    result = numa.apply_placement(numa.NodePlacement(0, (0,)))
    assert result["memory_bound"] is True
    policy, mask, maxnode = requests[0]
    assert policy == 2
    assert mask == 1
    # Linux rejects maxnode=1 for node 0; include sufficient mask capacity.
    assert maxnode > 1


def test_failed_memory_syscall_is_not_reported_successful(monkeypatch):
    class SetPolicy:
        def __call__(self, policy, mask, maxnode):
            numa.ctypes.set_errno(1)
            return -1

    class Library:
        set_mempolicy = SetPolicy()

    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(numa.ctypes.util, "find_library", lambda name: "libnuma.so.1")
    monkeypatch.setattr(numa.ctypes, "CDLL", lambda *args, **kwargs: Library())
    result = numa.apply_placement(numa.NodePlacement(0, (0,)))
    assert result["memory_bound"] is False
    assert result["memory_errno"] == 1
    assert result["memory_policy"] == "cpu-only"
