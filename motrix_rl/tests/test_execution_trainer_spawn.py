# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""CPU spawn integration coverage for the declarative asynchronous trainer."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from motrix_rl.execution.trainer import AsyncTrainerBase
from motrix_rl.execution.training import TrainingDecision, TrainingPlan, WorkerSpec


def _toy_worker(runtime) -> None:
    runtime.ready()
    if runtime.should_stop():
        return
    runtime.finish({"status": "finished", "value": 42})


class _SpawnTrainer(AsyncTrainerBase):
    def __init__(self, run_dir):
        super().__init__(context=SimpleNamespace(run_dir=Path(run_dir), env_name="toy", metrics=None))
        self.reports = []

    def build_training(self, runtime):
        return TrainingPlan((WorkerSpec("toy", "worker", _toy_worker),), self)

    def evaluate_progress(self, state, progress):
        if all(status.finished for status in progress.workers.values()):
            return TrainingDecision(True, frozenset(progress.workers))
        return TrainingDecision(False)

    def collect_metrics(self, state, reports):
        self.reports.append(reports)
        return super().collect_metrics(state, reports)

    def play(self, policy):
        raise NotImplementedError


@pytest.mark.integration
def test_async_trainer_base_spawn_barrier_queue_and_reaping(tmp_path, monkeypatch):
    monkeypatch.setenv("MOTRIX_BOOT_TIMEOUT_S", "10")
    trainer = _SpawnTrainer(tmp_path)
    deadline = time.monotonic() + 15
    trainer.train()
    assert time.monotonic() < deadline
    assert trainer.reports
    assert trainer.reports[-1].force is True
    assert any(frame.reports.get("toy") == ({"status": "finished", "value": 42},) for frame in trainer.reports)
