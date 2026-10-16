# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC collector CPU slicing policy for topology-resolved CPU sets."""

from __future__ import annotations


def select_collector_cpus(
    base: list[int],
    collector_id: int,
    cpus_per_collector: int | None,
) -> list[int]:
    """Pick this collector's CPU slice from ``base`` (its node's or the process's CPUs).

    ``collector_id`` indexes WITHIN ``base`` — callers holding a per-node base
    pass the node-local ordinal, not the global collector id. Without
    ``cpus_per_collector`` all collectors share the full ``base`` set and the
    OS load-balances them; with it, the set is split into contiguous chunks
    so collectors never compete for the same cores.
    """
    if not base:
        raise ValueError("empty CPU set for collector binding")
    if cpus_per_collector is None:
        return list(base)
    if cpus_per_collector <= 0:
        raise ValueError(f"cpus_per_collector must be positive, got {cpus_per_collector}")
    start = collector_id * cpus_per_collector
    end = min(start + cpus_per_collector, len(base))
    if start >= end:
        raise ValueError(
            f"cpus_per_collector={cpus_per_collector} leaves no CPUs for collector {collector_id} "
            f"(base set has {len(base)} CPUs)"
        )
    return base[start:end]
