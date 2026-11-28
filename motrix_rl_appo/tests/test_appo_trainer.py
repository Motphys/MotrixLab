# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from motrix_rl_appo.config import AppoCfg, NetworkCfg
from motrix_rl_appo.learner import APPOLearner
from motrix_rl_appo.trainer import AppoTrainer, check_worker, restore_rng, rng_state, stop_worker, wait_ready

from motrix_rl import checkpoints
from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.contracts import TrainerContext


class FakeProcess:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeWorker:
    def __init__(self, alive: bool = True) -> None:
        self.is_alive = alive
        self.exitcode = None if alive else 0
        self.process = FakeProcess()
        self.terminated = False
        self.killed = False

    def join(self, timeout: float) -> None:
        pass

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.is_alive = False


def test_startup_timeout_and_clean_exit_are_reported() -> None:
    ctx = mp.get_context("spawn")
    errors, error_tx = ctx.Pipe(duplex=False)
    startup, startup_tx = ctx.Pipe()
    try:
        with pytest.raises(TimeoutError, match="startup metadata"):
            wait_ready(FakeWorker(), errors, startup, timeout=0.001)
        with pytest.raises(RuntimeError, match="unexpectedly"):
            check_worker(FakeWorker(alive=False), errors)
        error_tx.send("original traceback")
        with pytest.raises(RuntimeError, match="original traceback"):
            check_worker(FakeWorker(), errors)
    finally:
        for pipe in (errors, error_tx, startup, startup_tx):
            pipe.close()


def test_cleanup_escalates_to_kill_and_closes_process() -> None:
    worker = FakeWorker()
    stop_worker(worker)
    assert worker.terminated and worker.killed and worker.process.closed


def test_periodic_and_final_checkpoint_manifest_and_warm_state(tmp_path) -> None:
    cfg = AppoCfg(actor=NetworkCfg(hidden_dims=[4]), critic=NetworkCfg(hidden_dims=[4]))
    context = TrainerContext(
        None,
        "fake",
        tmp_path,
        tmp_path / "checkpoints",
        None,
        "pt",
        2,
        1,
        4,
        cfg,
        LoggingConfig("tensorboard", 1),
        CheckpointConfig(1),
    )
    trainer = AppoTrainer(context=context)
    learner = APPOLearner(cfg, 2, 3, 1, "cpu")
    learner.update_count = 7
    learner.global_step = 28
    periodic = context.checkpoint_dir / "checkpoint_7.pt"
    trainer._save(learner, 4, periodic)
    assert checkpoints.resume_checkpoint(tmp_path) == periodic
    saved = torch.load(periodic, weights_only=False)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restored = APPOLearner(cfg, 2, 3, 1, "cpu")
    restored.load_state_dict(saved["learner"])
    restore_rng(saved["rng"])
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])
    assert restored.update_count == 7 and restored.global_step == 28
    final = checkpoints.final_checkpoint_path("pt", tmp_path)
    trainer._save(restored, 4, final)
    assert checkpoints.resume_checkpoint(tmp_path) == final
    policy_artifact = checkpoints.read_manifest(tmp_path).artifacts[checkpoints.BEST_POLICY]
    assert context.checkpoint_dir / policy_artifact.path == final


@pytest.mark.integration
def test_real_cartpole_spawn_resume_and_deterministic_play(tmp_path, monkeypatch) -> None:
    """Protect real spawn/transport/environment/checkpoint integration on CPU."""
    from motrix_rl_appo.networks import build_actor

    from motrix_env_core.renderer import RenderConfig
    from motrix_rl_appo import trainer as trainer_module

    cfg = AppoCfg(
        actor=NetworkCfg(hidden_dims=[8]),
        critic=NetworkCfg(hidden_dims=[8]),
        rollout_steps=2,
        max_iterations=2,
        device="cpu",
        collector_device="cpu",
        worker_timeout=30,
    )
    cfg.algorithm.num_learning_epochs = 1
    cfg.algorithm.num_mini_batches = 1
    context = TrainerContext(
        None,
        "cartpole",
        tmp_path,
        tmp_path / "checkpoints",
        None,
        "pt",
        2,
        1,
        4,
        cfg,
        LoggingConfig("tensorboard", 1),
        CheckpointConfig(1),
    )
    AppoTrainer(context=context).train()
    final = checkpoints.resume_checkpoint(tmp_path)
    saved = torch.load(final, weights_only=False)
    assert saved["iteration"] == 2 and saved["learner"]["global_step"] == 8
    assert (context.checkpoint_dir / "checkpoint_1.pt").is_file()
    cfg.max_iterations = 3
    AppoTrainer(context=replace(context, resume_from=str(final))).train()
    saved = torch.load(final, weights_only=False)
    assert saved["iteration"] == 3 and saved["learner"]["global_step"] == 12

    class OneStepRenderer:
        def __init__(self) -> None:
            self.closed = False
            self.calls = 0

        def render(self) -> bool:
            self.calls += 1
            return False

        def close(self) -> None:
            self.closed = True

    renderer = OneStepRenderer()
    monkeypatch.setattr(trainer_module, "create_renderer", lambda env, config: renderer)
    original_build = build_actor
    observed: list[torch.Tensor] = []

    def playback_actor(network_cfg, obs_dim: int, action_dim: int):
        actor = original_build(network_cfg, obs_dim, action_dim)
        mean = actor.mean

        def tracked_mean(obs: torch.Tensor) -> torch.Tensor:
            result = mean(obs)
            observed.append(result.detach().clone())
            torch.testing.assert_close(result, mean(obs))
            return result

        def reject_sample(obs: torch.Tensor):
            raise AssertionError("play must use deterministic mean, not sample")

        actor.mean = tracked_mean
        actor.sample = reject_sample
        return actor

    monkeypatch.setattr(trainer_module, "build_actor", playback_actor)
    AppoTrainer(context=replace(context, render=RenderConfig(headless=True))).play(str(final))
    assert len(observed) == 1 and renderer.calls == 1 and renderer.closed


@pytest.mark.integration
def test_real_go2_timeout_hook_and_collector_bootstrap() -> None:
    """Exercise the real ArrayEnv capture hook, not just its helper in isolation."""
    from motrix_rl_appo.collector import ChunkCollector
    from motrix_rl_appo.networks import build_actor, build_critic

    from motrix_env_core import registry

    build = registry.resolve("go2-walk-flat")
    build.env_cfg.max_episode_seconds = build.env_cfg.ctrl_dt
    env = build.make(num_envs=2, headless=True)
    cfg = AppoCfg(
        actor=NetworkCfg(hidden_dims=[8]),
        critic=NetworkCfg(hidden_dims=[8]),
        rollout_steps=1,
        device="cpu",
        collector_device="cpu",
    )
    actor = build_actor(cfg.actor, env.observation_spec.policy.space.shape[0], env.action_space.shape[0])
    critic = build_critic(cfg.critic, env.observation_spec.value_or_policy.space.shape[0])
    # A constant nonzero bootstrap makes reward folding observable independently
    # of random initialization or noisy observation values.
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.mlp[-1].bias.fill_(2.0)
    collector = ChunkCollector(env, actor, critic, cfg, torch.device("cpu"))
    chunk, returns, lengths, _ = collector.collect()
    state = env.state
    final = state.final_obs
    assert final is not None
    assert state.truncated.all() and not state.terminated.any()
    assert final.env_ids.tolist() == [0, 1]
    assert final.obs.policy.shape == (2, env.observation_spec.policy.space.shape[0])
    assert final.obs.value.shape == (2, env.observation_spec.value_or_policy.space.shape[0])
    torch.testing.assert_close(chunk["rewards"][:, 0], state.reward + cfg.algorithm.gamma * 2.0)
    torch.testing.assert_close(torch.tensor(returns), state.reward)
    assert lengths == [1, 1]
    assert chunk["dones"].all()
    torch.testing.assert_close(chunk["last_critic_obs"], state.obs.value_or_policy)
    owned = final.obs.value.clone()
    env.step(torch.zeros((2, env.action_space.shape[0])))
    torch.testing.assert_close(final.obs.value, owned)


def test_rng_roundtrip() -> None:
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(2), expected[2])
