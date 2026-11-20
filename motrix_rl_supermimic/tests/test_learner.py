# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Learner update contracts using the async transition ingestion path."""

import copy
import time

import pytest
import torch
from _helpers import ReferenceContextCfg, ingest_toy_steps, make_toy, make_toy_task
from motrix_rl_supermimic.config import AgentCfg, SuperMimicCfg, TrainerCfg
from motrix_rl_supermimic.learner import Learner
from motrix_rl_supermimic.mbrl_env import assemble_task
from motrix_rl_supermimic.toy import ToyEnvCfg
from motrix_rl_supermimic.transitions import Transitions


def test_critic_history_is_rejected_at_assembly_boundary(tmp_path):
    from dataclasses import replace

    import pytest
    from _helpers import make_toy_task

    task, task_binding = make_toy_task()
    task.observation_spec = replace(
        task.observation_spec,
        groups={**task.observation_spec.groups, "value": replace(task.observation_spec.value, history_length=2)},
    )
    with pytest.raises(ValueError, match="value history_length=1"):
        Learner(SuperMimicCfg(), task, task_binding, 1, 0.02, "cpu", tmp_path)


def test_observe_consumes_selected_steps_and_advances_normalizers(tmp_path):
    """Only the supplied span enters replay and normalizer statistics."""
    torch.manual_seed(0)
    env = make_toy(num_envs=2, cfg=ToyEnvCfg(goal_frames=2))
    env.init_state()
    cfg = SuperMimicCfg(agent=AgentCfg(hidden=8), trainer=TrainerCfg())
    learner = Learner(
        cfg, *assemble_task(env.cfg.for_imagined(), "cpu"), env.num_envs, env.cfg.ctrl_dt, "cpu", tmp_path
    )
    steps, n = 5, env.num_envs
    views = {
        "s": torch.randn(steps, n, learner.mdp.world_state.size),
        "s1": torch.randn(steps, n, learner.mdp.world_state.size),
        "a": torch.randn(steps, n, 1),
        "a_prev": torch.randn(steps, n, 1),
        "xi": torch.zeros(steps, n, 0),
        "r": torch.randn(steps, n),
        "f": torch.zeros(steps, n, dtype=torch.long),
        "ep_step": torch.zeros(steps, n, dtype=torch.int32),
        "done": torch.zeros(steps, n, dtype=torch.bool),
        "obs": torch.randn(steps, n, 4),
        "goal": torch.randn(steps, n, 2),
    }
    span_reward = learner.observe(Transitions.from_dict({name: values[:3] for name, values in views.items()}))

    assert learner.sim_steps == 3 * n
    for norm in (
        learner.pi.obs_norm,
        learner.pi.goal_norm,
        learner.wm.in_norm,
        learner.wm.out_norm,
        learner.wm.inc_norm,
    ):
        assert int(norm.count) == 3 * n
    assert learner.buffer.ptr == 3
    torch.testing.assert_close(learner.buffer.data["s"][2], views["s"][2])
    torch.testing.assert_close(learner.buffer.data["r"][2], views["r"][2])
    expected_span = views["r"][:3].mean() / learner.return_normalizer
    torch.testing.assert_close(span_reward, expected_span)


def test_toy_training_smoke(tmp_path):
    torch.manual_seed(0)
    env = make_toy(num_envs=8, cfg=ToyEnvCfg(goal_frames=2))
    env.init_state()
    cfg = SuperMimicCfg(
        agent=AgentCfg(hidden=32),
        trainer=TrainerCfg(
            iterations=30,
            buffer_steps=128,
            utd=0.5,
            horizon=4,
            wm_window=2,
            wm_batches=2,
            wm_batch=64,
            pi_batch=64,
            warmup_steps=8,
            wm_only_iters=2,
            critic_epochs=1,
            critic_minibatches=2,
            lr=3e-3,
            log_every=10,
            ckpt_every=15,
        ),
    )
    env.init_state()
    learner = Learner(
        cfg, *assemble_task(env.cfg.for_imagined(), "cpu"), env.num_envs, env.cfg.ctrl_dt, "cpu", tmp_path
    )
    # Async ingestion is the only test feeder; WM loss must decrease.
    ingest_toy_steps(learner, env, 64)
    wm_losses = [learner.wm_update() for _ in range(60)]
    assert wm_losses[-1] < wm_losses[0]

    # Direct updates cover policy/critic finiteness without a sync train loop.
    learner._apply_lr()
    policy = learner.policy_update()
    assert all(torch.isfinite(torch.as_tensor(value)) for value in policy)
    learner.iteration = cfg.trainer.iterations
    learner.save(time.monotonic())
    assert (tmp_path / "last.pt").exists()

    # Resume path round-trips module weights.
    fresh = Learner(
        cfg, *assemble_task(env.cfg.for_imagined(), "cpu"), env.num_envs, env.cfg.ctrl_dt, "cpu", tmp_path / "fresh"
    )
    fresh.load(str(tmp_path / "last.pt"))
    assert fresh.iteration == learner.iteration
    for key in ("wm", "pi", "critic"):
        for a, b in zip(getattr(learner, key).parameters(), getattr(fresh, key).parameters()):
            torch.testing.assert_close(a, b)


def test_cached_critic_matches_uncached_updates(tmp_path):
    """Caching normalized features preserves losses and optimizer updates."""
    torch.manual_seed(17)
    env = make_toy(num_envs=4, cfg=ToyEnvCfg(goal_frames=2))
    env.init_state()
    cfg = SuperMimicCfg(
        agent=AgentCfg(hidden=16),
        trainer=TrainerCfg(
            buffer_steps=16,
            horizon=2,
            wm_window=2,
            wm_batch=8,
            pi_batch=8,
            critic_epochs=2,
            critic_minibatches=2,
        ),
    )
    learner = Learner(
        cfg, *assemble_task(env.cfg.for_imagined(), "cpu"), env.num_envs, env.cfg.ctrl_dt, "cpu", tmp_path
    )
    ingest_toy_steps(learner, env, 8)
    learner.wm_update()
    cached = copy.deepcopy(learner)
    cached.cfg.trainer.cache_critic_features = True
    rng = torch.get_rng_state()
    expected = learner.policy_update()
    torch.set_rng_state(rng)
    actual = cached.policy_update()
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
    for model in ("pi", "critic"):
        for a, e in zip(getattr(cached, model).parameters(), getattr(learner, model).parameters()):
            torch.testing.assert_close(a, e)


class _SagittalMirror:
    """Reflect physical vectors and a signed joint across the sagittal plane."""

    def state(self, state: torch.Tensor) -> torch.Tensor:
        return -state

    def action(self, action: torch.Tensor) -> torch.Tensor:
        return -action

    def context(self, context: torch.Tensor) -> torch.Tensor:
        return -context


def test_mirror_augmentation_reflects_only_selected_windows(tmp_path) -> None:
    task, task_binding = make_toy_task(cfg=ToyEnvCfg(world_model_context={"reference": ReferenceContextCfg()}))
    task.imagination.mirror = _SagittalMirror()
    learner = Learner(
        SuperMimicCfg(agent=AgentCfg(hidden=8, wm_mirror=0.5)), task, task_binding, 1, 0.02, "cpu", tmp_path
    )
    state = torch.randn(4, 2, task.world_state.size)
    batch = {
        "s": state,
        "s1": state.clone(),
        "a": torch.randn(4, 2, 1),
        "xi": torch.randn(4, 2, 1),
        "mask": torch.ones(4, 2),
    }
    before = {key: value.clone() for key, value in batch.items()}
    result = learner._mirror_windows(batch, 2)
    for key in ("s", "s1"):
        torch.testing.assert_close(result[key][:2], -batch[key][:2])
    for key in ("a", "xi"):
        torch.testing.assert_close(result[key][:2], -batch[key][:2])
    for key in batch:
        torch.testing.assert_close(result[key][2:], batch[key][2:])
        torch.testing.assert_close(batch[key], before[key])
    restored = learner._mirror_windows(result, 2)
    for key in batch:
        torch.testing.assert_close(restored[key], batch[key])


def test_world_model_mirror_requires_task_capability(tmp_path) -> None:
    with pytest.raises(ValueError, match="wm_mirror requires.*mirror"):
        Learner(SuperMimicCfg(agent=AgentCfg(hidden=8, wm_mirror=0.5)), *make_toy_task(), 1, 0.02, "cpu", tmp_path)


@pytest.mark.parametrize("normalizer", [None, 0.5])
@pytest.mark.parametrize("metadata", [{}, {"task": "tracking", "clip": 7, "physics_dt": 0.01}])
def test_extension_data_controls_return_scale_and_checkpoint_metadata(tmp_path, normalizer, metadata) -> None:
    task, task_binding = make_toy_task()
    task.imagination.return_normalizer = normalizer
    task.imagination.evaluation_metadata.update(metadata)
    # Every independently assembled task owns its mutable metadata.
    other_task, _ = make_toy_task()
    assert other_task.imagination.evaluation_metadata == {}
    learner = Learner(SuperMimicCfg(agent=AgentCfg(hidden=8)), task, task_binding, 1, 0.02, "cpu", tmp_path)
    assert learner.return_normalizer == (0.02 if normalizer is None else normalizer)
    learner.save(0.0)
    for name in ("last.pt", "policy_0000000.pt"):
        checkpoint = torch.load(tmp_path / name, map_location="cpu", weights_only=False)
        if metadata:
            assert checkpoint["args"].items() >= metadata.items()
            assert checkpoint["args"]["hidden"] == learner.cfg.agent.hidden
            assert checkpoint["args"]["hist"] == task.observation_spec.policy.history_length
            assert checkpoint["args"]["goal_frames"] == task.goal_frames
        else:
            assert "args" not in checkpoint
