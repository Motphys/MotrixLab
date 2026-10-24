# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from motrix_rl.execution.topology import resolve_worker_topology, wire_endpoints


def test_endpoint_wiring_uses_common_owner_and_host_fallback(monkeypatch):
    monkeypatch.setattr("motrix_rl.execution.topology.system_numa.available_numa_nodes", lambda: [0])
    topology = resolve_worker_topology(8, 2, 1, None, "cpu", torch.device("cpu"))
    endpoints = wire_endpoints(topology)
    assert [(item.collector_id, item.owner_rank, item.transport) for item in endpoints] == [
        (0, 0, "host"),
        (1, 0, "host"),
    ]


def test_endpoint_wiring_selects_colocated_cuda_or_rejects_explicit_request(monkeypatch):
    topology = resolve_worker_topology(8, 1, 1, ["cuda:0"], "cuda", torch.device("cuda:0"))
    assert all(item.transport == "cuda_ipc" for item in wire_endpoints(topology))
    cpu_topology = resolve_worker_topology(8, 1, 1, None, "cpu", torch.device("cpu"))
    with pytest.raises(ValueError, match="colocated CUDA"):
        wire_endpoints(cpu_topology, "cuda_ipc")
