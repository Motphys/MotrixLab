# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from dataclasses import dataclass

import pytest

from motrix_rl.contracts import AgentProvider, RlFramework, TrainerBase, TrainerContext, TrainerHandle
from motrix_rl.execution.contracts import ExecutionCancelled
from motrix_rl.metrics import JSONL_FILENAME, JsonlSink, MetricsLogger


@dataclass
class Config:
    value: int = 1


class Provider(AgentProvider[Config]):
    config_type = Config

    @property
    def train_backend(self):
        return "test"

    @property
    def agent_name(self):
        return "unit"

    @property
    def checkpoint_format(self):
        return "bin"

    def create_trainer(self, context: TrainerContext[Config]):
        raise NotImplementedError


class Framework(RlFramework):
    @property
    def name(self):
        return "unit"


def test_provider_and_framework_contracts():
    framework = Framework((Provider(),))

    assert framework.supported_agents() == ("unit",)
    assert framework.supported_train_backends("unit") == ("test",)
    assert framework.get_agent_provider("unit", "test").checkpoint_format == "bin"
    assert Provider().validate_config(Config(3)).value == 3


class _Run:
    def __init__(self, run_dir):
        self.run_dir = run_dir


class _Trainer(TrainerBase):
    def __init__(self, error=None):
        self.error = error

    def train(self):
        if self.error is not None:
            raise self.error

    def play(self, policy):
        raise NotImplementedError


def _train(tmp_path, error=None):
    metrics = MetricsLogger("run-1", jsonl=JsonlSink(tmp_path / JSONL_FILENAME))
    handle = TrainerHandle(run=_Run(tmp_path), trainer=_Trainer(error), metrics=metrics, run_info={"total_steps": 8})
    raised = None
    try:
        handle.train()
    except BaseException as exc:  # noqa: BLE001 - the test inspects what propagated
        raised = exc
    metrics.event("after_close")  # dropped: the handle closed the logger
    metrics.join(5)  # the handle closed the logger without waiting for the sink; the test waits
    records = [json.loads(line) for line in (tmp_path / JSONL_FILENAME).read_text().splitlines()]
    return raised, [record["data"] for record in records]


def test_trainer_handle_reports_finished_after_running(tmp_path):
    raised, states = _train(tmp_path)

    assert raised is None
    assert states == [{"state": "running", "meta": {}, "total_steps": 8}, {"state": "finished"}]


def test_trainer_handle_does_not_wait_for_a_stuck_sink(tmp_path):
    unstick = threading.Event()

    class StuckDisk(JsonlSink):
        def _write_batch(self, batch):
            unstick.wait(5)
            super()._write_batch(batch)

    sink = StuckDisk(tmp_path / JSONL_FILENAME)
    handle = TrainerHandle(run=_Run(tmp_path), trainer=_Trainer(), metrics=MetricsLogger("run-1", jsonl=sink))

    handle.train()

    still_writing = sink._thread.is_alive()  # train() returned while the disk still held the records
    unstick.set()
    handle.metrics.join(5)
    assert still_writing
    states = [json.loads(line)["data"]["state"] for line in (tmp_path / JSONL_FILENAME).read_text().splitlines()]
    assert states == ["running", "finished"]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), ExecutionCancelled("cancel requested")])
def test_trainer_handle_reports_killed_on_cancellation(tmp_path, error):
    raised, states = _train(tmp_path, error)

    assert raised is error
    assert [state["state"] for state in states] == ["running", "killed"]


def test_trainer_handle_reports_failed_with_reason_and_reraises(tmp_path):
    (tmp_path / "async_errors").mkdir()
    (tmp_path / "async_errors" / "learner_error.log").write_text("Traceback")
    error = RuntimeError("x" * 5000)
    raised, states = _train(tmp_path, error)

    assert raised is error
    assert [state["state"] for state in states] == ["running", "failed"]
    reason = states[-1]["reason"]
    assert reason.startswith("RuntimeError: xxx")
    assert reason.endswith("(see async_errors/learner_error.log)")
    assert len(reason) == 2048


def test_trainer_handle_without_metrics_writes_nothing(tmp_path):
    handle = TrainerHandle(run=_Run(tmp_path), trainer=_Trainer())

    assert handle.train() is handle.run
    assert not handle.metrics.enabled
    assert list(tmp_path.iterdir()) == []
