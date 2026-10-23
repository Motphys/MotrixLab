# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from importlib.resources import files
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from motrix_rl_rslrl.torch.train import execution
from tensordict import TensorDict

from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.execution.contracts import Cancellation, ExecutionCancelled
from motrix_rl.metrics import MetricsLogger


class Env:
    num_envs = 2
    num_actions = 1
    device = "cpu"
    cfg = {}

    def __init__(self):
        self.steps = 0

    def get_observations(self):
        return TensorDict({"policy": torch.zeros(2, 3)}, [2])

    def step(self, actions):
        self.steps += 1
        return (
            self.get_observations(),
            torch.ones(2),
            torch.ones(2),
            {
                "Reward": {"alive": torch.ones(2)},
                "metrics": {"height": 2.0},
            },
        )


def context(tmp_path, **kwargs):
    return SimpleNamespace(
        env_name="toy",
        run_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_format="pt",
        logging=LoggingConfig(interval=2),
        checkpoint=CheckpointConfig(interval=1),
        metrics=kwargs.get("metrics", MetricsLogger.disabled()),
    )


def runner_stub():
    events = []
    env = Env()
    alg = SimpleNamespace(
        train_mode=lambda: events.append("train"),
        broadcast_parameters=lambda: events.append("broadcast"),
        act=lambda obs: torch.zeros(2, 1),
        process_env_step=lambda *args: events.append("step"),
        compute_returns=lambda obs: events.append("returns"),
        update=lambda: events.append("update") or {"value": 0.5},
        learning_rate=0.01,
        get_policy=lambda: SimpleNamespace(output_std=torch.ones(1)),
    )
    log = SimpleNamespace(
        init_logging_writer=Mock(),
        disable_logs=False,
        process_env_step=Mock(),
        log=Mock(),
        rewbuffer=[2.0],
        lenbuffer=[2.0],
    )
    runner = SimpleNamespace(
        env=env,
        alg=alg,
        logger=log,
        cfg={"max_iterations": 3, "num_steps_per_env": 2, "algorithm": {"rnd_cfg": None}},
        device="cpu",
        is_distributed=False,
        gpu_world_size=1,
        current_learning_iteration=0,
    )
    runner.save = lambda path: torch.save({"iter": runner.current_learning_iteration}, path)
    return runner, events


def test_sync_rollout_update_panel_and_checkpoint_boundaries(tmp_path, monkeypatch):
    runner, events = runner_stub()
    panels = []
    monkeypatch.setattr(execution, "open_training_live", lambda: (None, None))
    monkeypatch.setattr(execution, "emit_training_panel", lambda live, stats, **kw: panels.append(stats))
    metrics = Mock()
    execution.run_sync(runner, context(tmp_path, metrics=metrics))
    assert [call.args[1] for call in metrics.scalars.call_args_list] == [2, 3]
    assert [call.kwargs["env_steps"] for call in metrics.scalars.call_args_list] == [8, 12]
    assert metrics.scalars.call_args.args[0]["rollout/mean_return"] == 2.0
    assert events == ["train"] + ["step", "step", "returns", "update"] * 3
    assert runner.env.steps == 6
    assert [p.iteration for p in panels] == [2, 3]
    assert panels[-1].buffer_size is None and panels[-1].buffer_capacity is None
    assert panels[-1].training_metrics["value"] == 0.5
    assert panels[-1].reward_terms == {"alive": 1.0}
    assert panels[-1].env_metrics == {"height": 2.0}
    assert runner.logger.log.call_count == 2
    assert [call.kwargs["it"] for call in runner.logger.log.call_args_list] == [2, 3]
    assert [call.kwargs["num_iterations"] for call in runner.logger.log.call_args_list] == [2, 1]
    for iteration in (1, 2):
        assert torch.load(tmp_path / "checkpoints" / f"model_{iteration}.pt", weights_only=False)["iter"] == iteration
    assert torch.load(tmp_path / "checkpoints" / "latest.pt", weights_only=False)["iter"] == 3


def test_native_and_shared_metrics_publish_the_same_averaged_window(tmp_path, monkeypatch):
    from motrix_rl_rslrl.torch.train.logging import RslrlTrainingLogger

    runner, _ = runner_stub()
    logger = RslrlTrainingLogger(None, runner.cfg, {}, 2, False, 1, 0, "cpu")
    logger.log_dir = str(tmp_path)
    logger.logger_type = "tensorboard"
    writer = Mock()
    logger.writer = writer
    logger.init_logging_writer = Mock()
    logger.log = Mock(wraps=logger.log)
    runner.logger = logger
    update_losses = iter([1.0, 3.0, 8.0])
    runner.alg.update = lambda: {"value": next(update_losses)}
    original_step = runner.env.step

    def step(actions):
        obs, rewards, dones, extras = original_step(actions)
        extras["episode"] = {"score": float(runner.env.steps)}
        return obs, rewards, dones, extras

    runner.env.step = step
    panels = []
    monkeypatch.setattr(execution, "open_training_live", lambda: (None, None))
    monkeypatch.setattr(execution, "emit_training_panel", lambda live, stats, **kw: panels.append(stats))
    metrics = Mock()

    execution.run_sync(runner, context(tmp_path, metrics=metrics))

    calls = logger.log.call_args_list
    assert [call.kwargs["it"] for call in calls] == [2, 3]
    assert [call.kwargs["num_iterations"] for call in calls] == [2, 1]
    assert [call.kwargs["loss_dict"] for call in calls] == [{"value": 2.0}, {"value": 8.0}]
    assert [p.training_metrics["value"] for p in panels] == [2.0, 8.0]
    assert [call.args[0]["train/value"] for call in metrics.scalars.call_args_list] == [2.0, 8.0]
    native = writer.add_scalar.call_args_list
    assert [(float(c.args[1]), c.args[2]) for c in native if c.args[0] == "Loss/value"] == [(2.0, 2), (8.0, 3)]
    assert [(float(c.args[1]), c.args[2]) for c in native if c.args[0] == "Episode/score"] == [(2.5, 2), (5.5, 3)]
    fps = [c.args[1] for c in native if c.args[0] == "Perf/total_fps"]
    expected_fps = [
        int(call.kwargs["num_iterations"] * 4 / (call.kwargs["collect_time"] + call.kwargs["learn_time"]))
        for call in calls
    ]
    assert fps == expected_fps
    assert logger.tot_timesteps == 12
    assert logger.tot_time == pytest.approx(sum(c.kwargs["collect_time"] + c.kwargs["learn_time"] for c in calls))
    assert logger.ep_extras == []


def test_cancel_mid_rollout_skips_update_and_final_save(tmp_path, monkeypatch):
    runner, events = runner_stub()
    token = Cancellation()
    original = runner.alg.process_env_step

    def process(*args):
        original(*args)
        token.cancel()

    runner.alg.process_env_step = process
    live = Mock()
    monkeypatch.setattr(execution, "open_training_live", lambda: (None, live))
    with pytest.raises(ExecutionCancelled):
        execution.run_sync(runner, context(tmp_path), cancellation=token)
    assert events == ["train", "step"]
    assert not (tmp_path / "checkpoints").exists()
    live.stop.assert_called_once()


def test_distributed_secondary_updates_without_panel_or_checkpoints(tmp_path, monkeypatch):
    runner, events = runner_stub()
    runner.is_distributed = True
    runner.logger.disable_logs = True
    panel = Mock()
    monkeypatch.setattr(execution, "emit_training_panel", panel)
    execution.run_sync(runner, context(tmp_path))
    assert events[:2] == ["train", "broadcast"]
    assert events.count("update") == 3
    panel.assert_not_called()
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.integration
def test_real_rslrl_cpu_updates_and_runner_checkpoint_load(tmp_path, monkeypatch):
    from motrix_rl_rslrl.torch.train.logging import RslrlTrainingLogger
    from omegaconf import OmegaConf
    from rsl_rl.runners import OnPolicyRunner

    cfg = OmegaConf.to_container(OmegaConf.load(files("motrix_rl.configs").joinpath("algo_base/rslrl.ppo.yaml")))
    cfg.pop("defaults")
    cfg.update(max_iterations=2, num_steps_per_env=2, logger="tensorboard", save_interval=1)
    cfg["actor"]["hidden_dims"] = [8]
    cfg["critic"]["hidden_dims"] = [8]
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1, schedule="fixed")
    restore_cfg = deepcopy(cfg)  # Native construction consumes class_name fields.
    runner = OnPolicyRunner(Env(), cfg, log_dir=None, device="cpu")
    runner.logger = RslrlTrainingLogger(
        str(tmp_path),
        cfg,
        {},
        2,
        False,
        1,
        0,
        "cpu",
    )
    before = [p.detach().clone() for p in runner.alg.actor.parameters()]
    monkeypatch.setattr(execution, "open_training_live", lambda: (None, None))
    panel = Mock()
    monkeypatch.setattr(execution, "emit_training_panel", panel)
    try:
        execution.run_sync(runner, context(tmp_path))
    finally:
        runner.logger.close()
    assert any(not torch.equal(a, b) for a, b in zip(before, runner.alg.actor.parameters()))
    restored = OnPolicyRunner(Env(), restore_cfg, log_dir=None, device="cpu")
    restored.load(str(tmp_path / "checkpoints" / "latest.pt"))
    assert restored.current_learning_iteration == 2
    for a, b in zip(runner.alg.actor.parameters(), restored.alg.actor.parameters()):
        torch.testing.assert_close(a, b)
    assert panel.call_count == 1
    assert runner.logger.tot_timesteps == 8
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    recorded = EventAccumulator(str(tmp_path)).Reload()
    assert [event.step for event in recorded.Scalars("Loss/value")] == [2]
    assert [event.step for event in recorded.Scalars("Perf/total_fps")] == [2]
