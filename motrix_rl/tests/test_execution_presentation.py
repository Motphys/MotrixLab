# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Presentation owns output lifetimes, while algorithms choose valid windows."""

from types import SimpleNamespace
from unittest.mock import Mock

from motrix_rl.execution import presentation
from motrix_rl.execution.training import TrainingEvent, TrainingMetrics


def test_invalid_metrics_still_deliver_events_without_opening_training_display(monkeypatch):
    logger = Mock()
    open_live = Mock()
    monkeypatch.setattr(presentation, "TrainingLogger", lambda *_: logger)
    monkeypatch.setattr(presentation, "open_training_live", open_live)
    context = SimpleNamespace(run_dir="unused", metrics=None)
    sink = presentation.TrainingPresentation(context, title="toy")
    sink.publish(TrainingMetrics(valid=False, events=(TrainingEvent("resumed", 2, {"path": "resume.pt"}),)))
    logger.event.assert_called_once_with("resumed", 2, path="resume.pt")
    logger.scalars.assert_not_called()
    open_live.assert_not_called()
    sink.close()
    sink.close()
    logger.close.assert_called_once_with()


def test_scalar_coordinates_and_placeholders_are_not_reinterpreted(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(presentation, "TrainingLogger", lambda *_: logger)
    sink = presentation.TrainingPresentation(SimpleNamespace(run_dir="unused", metrics=None), title="toy")
    sink.publish(TrainingMetrics(scalars={"loss": 3}, step=7, env_steps=120, placeholders=frozenset({"unknown"})))
    args, kwargs = logger.scalars.call_args
    assert args == ({"loss": 3}, 7)
    assert kwargs["env_steps"] == 120
    assert kwargs["placeholders"] == frozenset({"unknown"})
    assert kwargs["live_extra"]() == {}
    sink.close()
