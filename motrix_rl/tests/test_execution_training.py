# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pickle
from dataclasses import FrozenInstanceError

import pytest

from motrix_rl.console import TrainingPanelStats
from motrix_rl.execution.training import (
    TrainingArtifact,
    TrainingDecision,
    TrainingEvent,
    TrainingMetrics,
    TrainingPlan,
    TrainingProgress,
    WorkerReports,
    WorkerSpec,
    WorkerStatus,
)
from motrix_rl.execution.workers import WorkerSettings


def _worker(_runtime, value=0):
    return value


def test_worker_spec_is_spawn_pickle_friendly_and_has_no_process_state():
    spec = WorkerSpec(
        "collector-0",
        "collector",
        _worker,
        args=(3,),
        kwargs={"value": 4},
        settings=WorkerSettings(seed=11, cpus=(2, 3)),
    )

    restored = pickle.loads(pickle.dumps(spec))
    assert restored.id == "collector-0"
    assert restored.target is _worker
    assert restored.args == (3,)
    assert restored.kwargs == {"value": 4}
    assert restored.settings == spec.settings
    assert not hasattr(spec, "process")

    with pytest.raises(FrozenInstanceError):
        spec.id = "other"


def test_plan_and_read_only_observations_preserve_public_boundaries():
    status = WorkerStatus("running", progress={"updates": 2})
    progress = TrainingProgress(workers={"learner-0": status}, counters={"steps": 8})
    reports = WorkerReports({"learner-0": ({"loss": 0.25},)}, now=1.5, force=True)
    plan = TrainingPlan([WorkerSpec("learner-0", "learner", _worker)], state={"seed": 1})

    assert plan.workers[0].role == "learner"
    assert progress.workers["learner-0"] is status
    with pytest.raises(TypeError):
        progress.counters["steps"] = 9
    assert reports.now == 1.5
    assert reports.force is True
    with pytest.raises(TypeError):
        reports.reports["new"] = ()
    with pytest.raises(FrozenInstanceError):
        status.state = "ready"


def test_decision_and_metrics_carry_algorithm_completion_and_outputs():
    panel = TrainingPanelStats(
        iteration=1,
        total_iterations=2,
        steps_per_second=None,
        elapsed_seconds=0.5,
        mean_return=1.0,
        mean_episode_length=2.0,
        episodes=1,
        buffer_size=None,
        buffer_capacity=None,
        collect_ms=1.0,
        learn_ms=2.0,
        learn_percent=66.7,
    )
    metrics = TrainingMetrics(
        panel=panel,
        scalars={"loss": 0.25},
        step=1,
        env_steps=8,
        placeholders=frozenset({"mean_return"}),
        valid=True,
        events=(TrainingEvent("checkpoint_saved", step=1, data={"kind": "latest"}),),
        artifacts=(TrainingArtifact("checkpoint", "/tmp/model.pt"),),
    )
    decision = TrainingDecision(complete=True, completed_workers=frozenset({"learner-0"}))

    assert decision.complete is True
    assert decision.completed_workers == {"learner-0"}
    assert metrics.panel is panel
    assert metrics.scalars["loss"] == 0.25
    assert metrics.events[0].name == "checkpoint_saved"
    assert metrics.artifacts[0].path == "/tmp/model.pt"
    with pytest.raises(TypeError):
        metrics.scalars["loss"] = 1.0
