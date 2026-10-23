# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Algorithm-semantic template for supervised asynchronous training."""

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import replace
from typing import Any

from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.execution.building import BuildRuntime
from motrix_rl.execution.plan_execution import PlanExecution
from motrix_rl.execution.presentation import TrainingPresentation
from motrix_rl.execution.training import (
    TrainingDecision,
    TrainingMetrics,
    TrainingPlan,
    TrainingProgress,
    WorkerReports,
)


class AsyncTrainerBase(TrainerBase):
    """Own the runtime; algorithms describe computation, progress and metrics.

    The only required asynchronous extension points are ``build_training`` and
    ``evaluate_progress``. Metrics aggregation is optional. No hook receives a
    process, supervisor, control queue or console object. Worker entry points
    are module-level functions and run under the public worker protocol.
    """

    def __init__(self, *, context: TrainerContext[Any]) -> None:
        self._context = context

    @abstractmethod
    def build_training(self, runtime: BuildRuntime) -> TrainingPlan:
        """Declare algorithm resources, state and spawnable worker entries."""

    @abstractmethod
    def evaluate_progress(self, state: Any, progress: TrainingProgress) -> TrainingDecision:
        """Interpret algorithm budgets and legitimate worker completion."""

    def collect_metrics(self, state: Any, reports: WorkerReports) -> TrainingMetrics:
        """Interpret already-consumed reports; never poll queues or render UI."""
        return TrainingMetrics()

    def train(self) -> None:
        resources = ExitStack()
        primary: BaseException | None = None
        try:
            build = BuildRuntime(resources)
            presentation = resources.enter_context(_PresentationScope(self._context))
            plan = self.build_training(build)
            execution = PlanExecution(
                build,
                plan,
                run_dir=self._context.run_dir,
                startup_timeout=float(os.environ.get("MOTRIX_BOOT_TIMEOUT_S", "600")),
            )

            def observe(reports):
                metrics = self.collect_metrics(plan.state, reports)
                if reports.events and not metrics.valid:
                    metrics = replace(metrics, valid=True)
                events = reports.events + metrics.events
                if reports.events and metrics.events:
                    events = tuple(dict.fromkeys(events))
                presentation.publish(replace(metrics, events=events))

            execution.run(
                observe,
                lambda progress: self.evaluate_progress(plan.state, progress),
            )
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                resources.close()
            except BaseException:
                if primary is None:
                    raise
                logging.getLogger(__name__).exception("resource cleanup failed while handling training error")


class _PresentationScope:
    """Construct presentation before algorithm build, always close its sinks."""

    def __init__(self, context):
        self._context = context

    def __enter__(self):
        self._presentation = TrainingPresentation(
            self._context, title=getattr(self._context, "env_name", "training")
        )
        return self._presentation

    def __exit__(self, exc_type, exc, traceback):
        self._presentation.close()
