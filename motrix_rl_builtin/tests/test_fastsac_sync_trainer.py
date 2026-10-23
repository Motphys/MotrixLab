# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.utils.tensorboard

from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.contracts import TrainerContext


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError("training failed"), KeyboardInterrupt()])
def test_sync_train_closes_tensorboard_writer_when_training_fails(tmp_path, monkeypatch, cleanup_fails, failure):
    from motrix_rl_builtin.fastsac.sync.train import Trainer

    writer = Mock()
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", Mock(return_value=writer))
    env = Mock()
    if cleanup_fails:
        env.close.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(Trainer, "_make_env", Mock(return_value=env))
    monkeypatch.setattr(Trainer, "_make_agent", Mock(return_value=SimpleNamespace(device="cpu", global_step=0)))

    def fail_after_logging(self, agent, env, training_log, **kwargs):
        training_log.scalars({"train/qf_loss": 1.0}, 0, env_steps=1)  # opens TensorBoard
        raise failure

    monkeypatch.setattr(Trainer, "_run_loop", fail_after_logging)
    context = TrainerContext(
        run=None,
        env_name="cartpole",
        run_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=1,
        play_num_envs=1,
        seed=None,
        rl_cfg=SimpleNamespace(trainer=SimpleNamespace(num_learning_iterations=1)),
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=0),
    )

    with pytest.raises(type(failure)) as caught:
        Trainer(context=context).train()
    assert caught.value is failure

    writer.close.assert_called_once_with()
    Trainer._make_env.return_value.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("learning_starts", "num_updates", "logging_interval", "expected_steps", "expected_collection"),
    [
        (1, 3, 100, [0, 3], [1, 4]),  # Final remainder is always published.
        (0, 3, 2, [2, 4], [2, 4]),  # Final threshold is emitted only once.
        (4, 3, 2, [0, 0], [1, 4]),  # Warmup never invents learning iterations.
        (0, 0, 2, [0], [4]),  # update(0) publishes final progress, no learner step.
    ],
)
def test_sync_phases_preserve_warmup_updates_and_progress(
    tmp_path, monkeypatch, learning_starts, num_updates, logging_interval, expected_steps, expected_collection
):
    import motrix_rl_builtin.fastsac.sync.train as module
    import torch

    events = []
    obs = torch.zeros(1, 2)
    env = SimpleNamespace(
        reset=lambda: (obs, obs),
        step=lambda actions: SimpleNamespace(
            obs=obs,
            critic_obs=obs,
            reward=torch.ones(1),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            info={},
        ),
    )
    agent = SimpleNamespace(
        cfg=SimpleNamespace(learning_starts=learning_starts, num_updates=num_updates),
        device="cpu",
        num_envs=1,
        act_dim=1,
        global_step=0,
        learning_iteration=0,
        update_idx=0,
        actor=SimpleNamespace(action_scale=1, action_bias=0),
        rb=SimpleNamespace(extend=lambda *args: events.append(("collect", agent.global_step))),
        set_train_mode=lambda: None,
        act=lambda *args, **kwargs: torch.zeros(1, 1),
        window_metrics=SimpleNamespace(stats_and_reset=lambda: {"actor_loss": 0.1}),
    )

    def update(count):
        events.append(("update", agent.global_step, count))
        if count > 0:
            agent.learning_iteration += 1
            agent.update_idx += count

    agent.update = update
    monkeypatch.setattr(module, "open_training_live", lambda: (None, None))
    panels = []
    monkeypatch.setattr(module, "emit_training_panel", lambda live, stats, **kwargs: panels.append(stats))
    agent.rb.num_stored = 1
    agent.rb.buffer_size = 10
    trainer = object.__new__(module.Trainer)
    trainer._env_name = "minimal"
    trainer._context = SimpleNamespace(checkpoint_dir=tmp_path)
    saved_steps = []
    agent.state_dict = lambda: {
        "global_step": agent.global_step,
        "learning_iteration": agent.learning_iteration,
        "update_idx": agent.update_idx,
    }
    monkeypatch.setattr(module.torch, "save", lambda state, path: saved_steps.append(state))
    log = Mock()
    trainer._run_loop(
        agent,
        env,
        log,
        num_iterations=4,
        logging_interval=logging_interval,
        save_interval=2,
        record_checkpoint=Mock(),
    )
    learned_at_save = max(2 - learning_starts, 0) if num_updates > 0 else 0
    assert saved_steps == [
        {"global_step": 2, "learning_iteration": learned_at_save, "update_idx": learned_at_save * num_updates}
    ]  # Saved collection count and independent learner counters describe the same boundary.
    assert agent.global_step == 4
    expected_events = []
    for collection in range(4):
        expected_events.append(("collect", collection))
        if collection >= learning_starts:
            expected_events.append(("update", collection, num_updates))
    assert events == expected_events
    assert agent.learning_iteration == (4 - learning_starts if num_updates > 0 else 0)
    assert agent.update_idx == agent.learning_iteration * num_updates
    assert [call.args[1] for call in log.scalars.call_args_list] == expected_steps
    assert [call.kwargs["env_steps"] for call in log.scalars.call_args_list] == expected_collection
    assert [call.args[0]["progress/collection_step"] for call in log.scalars.call_args_list] == expected_collection
    assert [panel.iteration for panel in panels] == expected_collection
    assert [panel.learning_iteration for panel in panels] == expected_steps
    assert all(panel.progress_unit == "collection steps" for panel in panels)
    assert panels[-1].iteration == 4
