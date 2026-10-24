# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Declarative plans execute without exposing processes to the algorithm."""

import time
from contextlib import ExitStack

import pytest

from motrix_rl.execution.building import BuildRuntime
from motrix_rl.execution.plan_execution import PlanExecution
from motrix_rl.execution.training import TrainingDecision, TrainingPlan, WorkerSpec


def toy_worker(runtime):
    runtime.ready()
    runtime.report({"updates": 1})
    runtime.finish({"updates": 2})


@pytest.mark.integration
def test_plan_materializes_spawn_workers_and_drains_final_reports(tmp_path):
    reports = []
    deadline = time.monotonic() + 15

    def observe(batch):
        reports.extend(batch.reports.get("toy", ()))
        if time.monotonic() > deadline:
            raise TimeoutError("toy did not finish")

    def decide(progress):
        status = progress.workers["toy"]
        return TrainingDecision(status.finished, frozenset({"toy"}) if status.finished else frozenset())

    with ExitStack() as resources:
        build = BuildRuntime(resources)
        plan = TrainingPlan((WorkerSpec("toy", "learner", toy_worker),), state=None)
        PlanExecution(build, plan, run_dir=tmp_path, startup_timeout=10, poll_interval=0.01).run(observe, decide)
    assert reports == [{"updates": 1}, {"updates": 2}]


def returns_without_ready(runtime):
    pass


@pytest.mark.integration
def test_zero_exit_without_ready_is_a_failure(tmp_path):
    with ExitStack() as resources:
        build = BuildRuntime(resources)
        plan = TrainingPlan((WorkerSpec("early", "actor", returns_without_ready),), None)
        with pytest.raises(RuntimeError, match="early"):
            PlanExecution(build, plan, run_dir=tmp_path, startup_timeout=10, poll_interval=0.01).run(
                lambda batch: None, lambda progress: TrainingDecision(True)
            )
