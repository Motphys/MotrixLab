# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pure rollout/timing statistics aggregation for the async trainer.

Parent-process-side helpers: collectors and learners emit per-process
snapshots; these functions merge them into the panel/TensorBoard-facing
views. No process, transport, or simulator knowledge — everything here is
a pure function over dicts and floats.
"""

from __future__ import annotations

import math
from typing import Any


def timing_mean(values: list[float]) -> float:
    """Mean of a non-empty timing sample list (ms)."""
    return sum(values) / len(values)


def nest_timing_path(tree: dict[str, Any], parts: tuple[str, ...], value: float) -> None:
    """Insert one dotted timing path into a nested mapping.

    A stage's scalar total and its sub-stage paths may arrive in either order
    (the collector emits parents before children); when both exist the scalar
    becomes the node's ``total`` alongside its children.
    """
    head, rest = parts[0], parts[1:]
    node = tree.get(head)
    if not rest:
        if isinstance(node, dict):
            node["total"] = value
        else:
            tree[head] = value
    else:
        if not isinstance(node, dict):
            node = {"total": node} if node is not None else {}
            tree[head] = node
        nest_timing_path(node, rest, value)


def aggregate_collector_stats(per_collector: dict[int, dict | None]) -> dict:
    """Merge per-collector rollout snapshots into one panel/TB-facing stats dict.

    Return/episode-length/reward/metric values are averaged over the collectors
    that reported since the last log window; episode counts are summed;
    ``policy_lag`` is the worst (max) staleness; timing values are averaged.
    """
    snapshots = [stats for stats in per_collector.values() if stats]

    def _nanmean(key: str) -> float:
        values = [stats[key] for stats in snapshots if not math.isnan(stats.get(key, float("nan")))]
        return (sum(values) / len(values)) if values else float("nan")

    def _mean_dicts(key: str) -> dict[str, float]:
        keys = {k for stats in snapshots for k in stats.get(key, {})}
        return {
            k: sum(stats[key][k] for stats in snapshots if k in stats.get(key, {}))
            / sum(1 for stats in snapshots if k in stats.get(key, {}))
            for k in keys
        }

    return {
        "return": _nanmean("return"),
        "ep_len": _nanmean("ep_len"),
        "episodes": sum(stats.get("episodes", 0) for stats in snapshots),
        "reward_terms": _mean_dicts("reward_terms"),
        "env_metrics": _mean_dicts("env_metrics"),
        "policy_lag": max((stats.get("policy_lag", 0) for stats in snapshots), default=0),
        "timing_ms": _mean_dicts("timing_ms"),
    }
