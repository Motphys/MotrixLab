# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import Mock

import pytest
import torch.utils.tensorboard

from motrix_rl.metrics import MetricsLogger
from motrix_rl.training_logger import TrainingLogger


class _Live(MetricsLogger):
    """An enabled live logger that keeps what it is sent."""

    def __init__(self):
        super().__init__("run-1")
        self.sent = []

    @property
    def enabled(self):
        return True

    def scalars(self, values, step, *, env_steps=None):
        self.sent.append(("scalars", dict(values), step, env_steps))

    def event(self, name, step=None, **data):
        self.sent.append(("event", name, step, data))


@pytest.fixture
def tensorboard(monkeypatch):
    writer = Mock()
    factory = Mock(return_value=writer)
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", factory)
    return factory, writer


def test_scalars_reach_tensorboard_and_live_from_one_call(tmp_path, tensorboard):
    factory, writer = tensorboard
    live = _Live()
    log = TrainingLogger(tmp_path, live)

    log.scalars(
        {"rollout/mean_return": math.nan, "train/qf_loss": 0.5},
        7,
        env_steps=56,
        live_extra=lambda: {"sys/cpu_percent": 12.0},
        placeholders=("rollout/mean_return",),
    )

    factory.assert_called_once_with(log_dir=str(tmp_path))
    tags = [(call.args[0], call.args[2]) for call in writer.add_scalar.call_args_list]
    assert tags == [("rollout/mean_return", 7), ("train/qf_loss", 7)]  # placeholders still reach TensorBoard
    assert live.sent == [("scalars", {"train/qf_loss": 0.5, "sys/cpu_percent": 12.0}, 7, 56)]


def test_tensorboard_opens_on_the_first_scalars_only(tmp_path, tensorboard):
    factory, writer = tensorboard
    live = _Live()
    log = TrainingLogger(tmp_path, live)

    log.event("run_started", step=0)
    log.close()

    factory.assert_not_called()  # a run that never logged leaves no event file
    assert live.sent == [("event", "run_started", 0, {})]


def test_close_closes_tensorboard_once_and_never_raises(tmp_path, tensorboard):
    _, writer = tensorboard
    writer.close.side_effect = OSError("disk gone")
    log = TrainingLogger(tmp_path, _Live())
    log.scalars({"a": 1.0}, 0, env_steps=1)

    log.close()
    log.close()

    writer.close.assert_called_once_with()


def test_live_still_gets_scalars_without_tensorboard(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", Mock(side_effect=ImportError("no tensorboard")))
    live = _Live()
    log = TrainingLogger(tmp_path, live)

    log.scalars({"a": 1.0}, 0, env_steps=1)
    log.scalars({"a": 2.0}, 1, env_steps=2)

    assert [sent[1] for sent in live.sent] == [{"a": 1.0}, {"a": 2.0}]


def test_disabled_live_writes_tensorboard_only(tmp_path, tensorboard):
    _, writer = tensorboard
    log = TrainingLogger(tmp_path, MetricsLogger.disabled())

    log.scalars({"a": 1.0}, 0, env_steps=1, live_extra=Mock(side_effect=AssertionError("sampled for live")))
    log.event("run_started", step=0)

    writer.add_scalar.assert_called_once_with("a", 1.0, 0)
    assert list(tmp_path.iterdir()) == []
