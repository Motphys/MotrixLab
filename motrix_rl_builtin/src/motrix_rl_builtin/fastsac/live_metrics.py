# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Host scalars FastSAC adds to its live metrics (see ``wiki/design/live-metrics.md``)."""

from __future__ import annotations

from motrix_rl.console import TrainingPanelStats

_GIB = 1024**3

# Rollout means that are a NaN placeholder until the first episode ends.
ROLLOUT_MEANS = ("rollout/mean_return", "rollout/mean_ep_len")


def system_scalars(stats: TrainingPanelStats) -> dict[str, float]:
    """The fixed ``sys/*`` keys from the samples a panel frame already holds.

    Memory is reported in GiB, like the panel. Per-core CPU load is left out on
    purpose: it would spend the service's per-run key budget on large hosts.
    """
    scalars: dict[str, float] = {}
    if stats.cpu_load is not None:
        scalars["sys/cpu_percent"] = stats.cpu_load.utilization_percent
    if stats.memory_usage is not None:
        scalars["sys/mem_used_gb"] = stats.memory_usage.used_bytes / _GIB
    for device in stats.gpu_devices or ():
        if device.utilization_percent is not None:
            scalars[f"sys/gpu{device.index}/util_percent"] = device.utilization_percent
        if device.memory is not None:
            scalars[f"sys/gpu{device.index}/mem_used_gb"] = device.memory.used_bytes / _GIB
    return scalars
