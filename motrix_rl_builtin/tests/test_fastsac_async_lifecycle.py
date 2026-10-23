# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC parent resources must close even before workers can be started."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastsac_async_mocks import ThreadProcess
from motrix_rl_builtin.fastsac.async_impl import train
from test_fastsac_live_metrics import _RecordingSink, _Stage, _train_async, _Workers


@pytest.mark.parametrize("failure", ["backend", "probe_build", "probe_dims"])
def test_setup_failure_closes_logger_and_probe(tmp_path, monkeypatch, failure):
    logger = Mock()
    monkeypatch.setattr(train, "TrainingLogger", lambda *args: logger)
    trainer = train.Trainer.__new__(train.Trainer)
    trainer._rlcfg = SimpleNamespace(trainer=SimpleNamespace(num_learning_iterations=1, async_options=None))
    trainer._context = SimpleNamespace(
        run_dir=tmp_path,
        metrics=None,
        logging=SimpleNamespace(interval=1, backend="invalid" if failure == "backend" else "tensorboard"),
        checkpoint=SimpleNamespace(interval=1),
    )
    probe = Mock()
    trainer._make_env = Mock(return_value=probe)
    trainer._dims = Mock(side_effect=RuntimeError("probe dimensions failed"))
    if failure == "probe_build":
        trainer._make_env.side_effect = RuntimeError("probe creation failed")
    with pytest.raises((ValueError, RuntimeError)):
        trainer.train()
    logger.close.assert_called_once()
    if failure == "probe_dims":
        probe.close.assert_called_once()
    else:
        probe.close.assert_not_called()


@pytest.mark.parametrize("failure", ["partial_start", "zero_boot_exit", "boot_error"])
def test_worker_startup_failure_reaps_children_and_closes_resources(tmp_path, monkeypatch, failure):
    class BootWorkers(_Workers):
        def learner(self, process):
            if failure == "zero_boot_exit":
                return 0
            if failure == "boot_error":
                process.arguments["error_queue"].put(("learner", "boot traceback"))
                return 1
            return super().learner(process)

    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    workers = BootWorkers(sink, [_Stage((20, 20))])
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=20)
    logger = Mock()
    logger.live.enabled = False
    monkeypatch.setattr(train, "TrainingLogger", lambda *args: logger)
    queues = []
    original_queue = workers.ctx.Queue

    def queue(*args, **kwargs):
        result = original_queue(*args, **kwargs)
        result.close = Mock()
        queues.append(result)
        return result

    monkeypatch.setattr(workers.ctx, "Queue", queue)
    if failure == "partial_start":
        original_start = ThreadProcess.start

        def start(process):
            if "collector-0" in process.name:
                raise OSError("spawn failed")
            original_start(process)

        monkeypatch.setattr(ThreadProcess, "start", start)
    with pytest.raises((OSError, RuntimeError)):
        trainer.train()
    logger.close.assert_called_once()
    assert all(not process.is_alive() for process in workers.ctx.processes)
    assert all(queue.close.call_count == 1 for queue in queues)
    if failure == "boot_error":
        assert (tmp_path / "async_errors" / "learner_error.log").read_text() == "boot traceback"
