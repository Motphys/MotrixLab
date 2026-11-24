# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest

from motrix_rl import numa
from motrix_rl.numa import spawn_placement


def test_shared_module_import_does_not_initialize_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import motrix_rl.numa; assert not {'numpy', 'torch', 'motrixsim'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_cpu_binding_never_expands_inherited_affinity(monkeypatch):
    requests = []
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda pid: {2, 3})
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda pid, cpus: requests.append(set(cpus)))
    numa.apply_cpu_affinity([1, 2, 3, 4], "test")
    assert requests == [{2, 3}]


def test_spawn_placement_restores_exact_policy_on_failure(monkeypatch):
    original = numa.MemoryPolicy(3, (5,))  # An inherited interleave policy, not default.
    affinity = []
    applied = []
    restored = []
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda pid, cpus: affinity.append(set(cpus)))
    monkeypatch.setattr(numa, "numa_node_cpus", lambda node: [0, 1])
    monkeypatch.setattr(numa, "get_memory_policy", lambda: original)
    monkeypatch.setattr(numa, "apply_memory_policy", lambda node, role: applied.append(node) or True)
    monkeypatch.setattr(numa, "set_memory_policy", restored.append)
    with pytest.raises(RuntimeError, match="worker startup"):
        with spawn_placement(0, "test"):
            raise RuntimeError("worker startup")
    assert applied == [0]
    assert restored == [original]
    assert affinity == [{0, 1}, {0, 1, 2, 3}]


def test_spawn_placement_keeps_memory_policy_when_snapshot_denied(monkeypatch):
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(numa, "numa_node_cpus", lambda node: [0])

    def denied():
        raise PermissionError("get_mempolicy denied")

    monkeypatch.setattr(numa, "get_memory_policy", denied)
    monkeypatch.setattr(numa, "apply_memory_policy", lambda *args: pytest.fail("cannot restore policy"))
    with spawn_placement(0, "test"):
        pass


def test_default_memory_policy_restore_uses_null_mask(monkeypatch):
    calls = []

    class SetPolicy:
        def __call__(self, mode, pointer, maxnode):
            calls.append((mode, pointer, maxnode))
            return 0

    class Library:
        set_mempolicy = SetPolicy()

    monkeypatch.setattr(numa, "_load_libnuma", lambda: Library())
    numa.set_memory_policy(numa.MemoryPolicy(0, (0,)))
    assert calls == [(0, None, 0)]


def test_worker_cpu_spec_intersects_allocation_and_preserves_empty_default(monkeypatch):
    monkeypatch.setattr(numa, "available_cpu_ids", lambda: {2, 4, 6})
    assert numa.resolve_worker_cpus(None, "worker") == set()
    assert numa.resolve_worker_cpus("0:4,6,6,99", "worker") == {2, 4, 6}
    assert numa.resolve_worker_cpus("0:1000000000", "worker") == {2, 4, 6}
    assert numa.resolve_worker_cpus("7:9", "worker") == set()
    with pytest.raises(ValueError, match="start > end"):
        numa.resolve_worker_cpus("6:2", "worker")
    with pytest.raises(ValueError, match="invalid core spec"):
        numa.resolve_worker_cpus("two", "worker")


def test_worker_budget_does_not_override_default_and_survives_denied_affinity(monkeypatch):
    import torch

    threads = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)

    def denied(cpus):
        raise PermissionError("restricted")

    monkeypatch.setattr(numa, "_set_worker_cpu_affinity", denied)
    numa.pin_worker_cpus(set())
    assert threads == []
    numa.pin_worker_cpus({2, 4})
    assert threads == [2]
