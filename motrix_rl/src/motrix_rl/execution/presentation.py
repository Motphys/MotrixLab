# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public ownership of training display, host sampling and metric sinks."""

from __future__ import annotations

from dataclasses import replace

from motrix_rl.console import emit_training_panel, open_training_live
from motrix_rl.execution.training import TrainingMetrics
from motrix_rl.system_metrics import (
    CpuLoadSampler,
    GpuMemoryUsageSampler,
    GpuUtilizationSampler,
    MemoryUsageSampler,
    sample_gpu_devices,
)
from motrix_rl.training_logger import TrainingLogger


class TrainingPresentation:
    """Consume algorithm frames without knowing their payload or budget policy.

    The caller controls cadence and supplies already-aggregated metrics. Invalid
    frames can still carry events, but cannot open the training panel or publish
    scalar windows. OS samples are attached here, never in an algorithm hook.
    """

    def __init__(self, context, *, title: str):
        self._title = title
        self._logger = TrainingLogger(context.run_dir, context.metrics)
        self._cpu = CpuLoadSampler()
        self._gpu = GpuUtilizationSampler()
        self._memory = MemoryUsageSampler()
        self._gpu_memory = GpuMemoryUsageSampler()
        self._opened = False
        self._live = None
        self._closed = False

    def publish(self, metrics: TrainingMetrics) -> None:
        for event in metrics.events:
            if self._logger.live is not None:
                self._logger.event(event.name, event.step, **event.data)
        for artifact in metrics.artifacts:
            if self._logger.live is not None:
                self._logger.event("artifact_created", kind=artifact.kind, path=artifact.path, **artifact.metadata)
        if not metrics.valid:
            return
        panel = metrics.panel
        if panel is not None:
            panel = replace(
                panel,
                cpu_load=self._cpu.sample(),
                gpu_utilization_percent=self._gpu.sample(),
                memory_usage=self._memory.sample(),
                gpu_memory_usage=self._gpu_memory.sample(),
                gpu_devices=sample_gpu_devices(self._gpu, self._gpu_memory),
            )
            if not self._opened:
                self._opened = True
                console, self._live = open_training_live()
                if console is not None:
                    console.clear()
            emit_training_panel(self._live, panel, title=self._title)
        if metrics.scalars:
            if metrics.step is None or metrics.env_steps is None:
                raise ValueError("scalar frames require explicit step and env_steps")
            self._logger.scalars(
                metrics.scalars,
                metrics.step,
                env_steps=metrics.env_steps,
                live_extra=lambda: _system_scalars(panel),
                placeholders=metrics.placeholders,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._live is not None:
                self._live.stop()
        finally:
            self._logger.close()


def _system_scalars(panel) -> dict[str, float]:
    if panel is None:
        return {}
    scalars = {}
    if panel.cpu_load is not None:
        scalars["sys/cpu_percent"] = panel.cpu_load.utilization_percent
    if panel.memory_usage is not None:
        scalars["sys/mem_used_gb"] = panel.memory_usage.used_bytes / 1024**3
    for device in panel.gpu_devices or ():
        if device.utilization_percent is not None:
            scalars[f"sys/gpu{device.index}/util_percent"] = device.utilization_percent
        if device.memory is not None:
            scalars[f"sys/gpu{device.index}/mem_used_gb"] = device.memory.used_bytes / 1024**3
    return scalars
