# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import Path

import torch
from motrix_rl_appo.config import AlgorithmCfg, AppoCfg, NetworkCfg
from motrix_rl_appo.learner import APPOLearner


def make_learner() -> APPOLearner:
    cfg = AppoCfg(
        actor=NetworkCfg(hidden_dims=[8]),
        critic=NetworkCfg(hidden_dims=[8]),
        algorithm=AlgorithmCfg(num_learning_epochs=2, num_mini_batches=2, target_update_interval=2),
    )
    return APPOLearner(cfg, 3, 4, 2, "cpu")


def make_batch(learner: APPOLearner) -> dict[str, torch.Tensor]:
    obs = torch.randn(3, 2, 3) + 2
    with torch.no_grad():
        actions, logp = learner.actor.sample(obs)
    return {
        "obs": obs,
        "critic_obs": torch.randn(3, 2, 4),
        "actions": actions,
        "behaviour_logp": logp,
        "rewards": torch.randn(3, 2),
        "dones": torch.zeros(3, 2, dtype=torch.bool),
        "last_critic_obs": torch.randn(2, 4),
    }


def test_multi_epoch_training_and_complete_target_snapshot() -> None:
    torch.manual_seed(5)
    learner = make_learner()
    batch = make_batch(learner)
    old_target = deepcopy(learner.target_actor.state_dict())
    old_actor = deepcopy(learner.actor.state_dict())
    metrics = learner.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert learner.actor.obs_normalizer.count.item() == 6
    assert any(not torch.equal(value, old_actor[key]) for key, value in learner.actor.state_dict().items())
    for key, value in learner.target_actor.state_dict().items():
        torch.testing.assert_close(value, old_target[key])
    learner.update(batch)
    assert learner.actor.obs_normalizer.count.item() == 12
    for key, value in learner.target_actor.state_dict().items():
        torch.testing.assert_close(value, learner.actor.state_dict()[key])
    assert learner.update_count == 2
    assert learner.global_step == 12


def test_checkpoint_roundtrip_preserves_target_optimizer_and_next_update(tmp_path: Path) -> None:
    torch.manual_seed(7)
    learner = make_learner()
    batch = make_batch(learner)
    learner.update(batch)
    path = tmp_path / "state.pt"
    torch.save(learner.state_dict(), path)
    restored = make_learner()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.update_count == learner.update_count
    assert restored.global_step == learner.global_step
    for module_name in ("actor", "critic", "target_actor"):
        expected = learner.state_dict()[module_name]
        actual = restored.state_dict()[module_name]
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
    torch.manual_seed(17)
    expected_metrics = learner.update(batch)
    torch.manual_seed(17)
    actual_metrics = restored.update(batch)
    assert actual_metrics == expected_metrics
    for key, value in learner.actor.state_dict().items():
        torch.testing.assert_close(value, restored.actor.state_dict()[key])


def test_prepared_targets_are_detached_and_keep_time_axis() -> None:
    learner = make_learner()
    batch = make_batch(learner)
    batch["rewards"].requires_grad_(True)
    prepared = learner.process(batch)
    assert learner.actor.obs_normalizer.count.item() == 0
    torch.testing.assert_close(prepared.target_logp, batch["behaviour_logp"].flatten())
    assert prepared.returns.shape == (6,)
    assert not prepared.returns.requires_grad
    assert not prepared.returns.is_inference()
    torch.testing.assert_close(prepared.obs, batch["obs"].reshape(6, 3))
    assert not any(parameter.grad is not None for parameter in learner.target_actor.parameters())
