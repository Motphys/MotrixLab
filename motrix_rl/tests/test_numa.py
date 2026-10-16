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
            "import sys; import motrix_rl.numa; "
            "assert not {'numpy', 'torch', 'numba', 'motrixsim'} & sys.modules.keys()",
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
