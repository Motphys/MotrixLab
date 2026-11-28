# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Execution modes preserve objective, state transitions and checkpoint semantics."""

import os
from copy import deepcopy

import pytest
import torch
from motrix_rl_appo.config import AlgorithmCfg, AppoCfg, NetworkCfg
from motrix_rl_appo.learner import APPOLearner
from torch.distributions import Normal, kl_divergence


def learner(mode: str = "eager", fused: bool = False, device: str = "cpu") -> APPOLearner:
    return APPOLearner(
        AppoCfg(
            actor=NetworkCfg(hidden_dims=[8]),
            critic=NetworkCfg(hidden_dims=[8]),
            algorithm=AlgorithmCfg(
                execution=mode,
                fused_optimizer=fused,
                num_learning_epochs=2,
                num_mini_batches=2,
                target_update_interval=2,
                desired_kl=1e-5,
            ),
        ),
        3,
        4,
        2,
        device,
    )


def batch(model: APPOLearner) -> dict[str, torch.Tensor]:
    device = model.device
    obs = torch.randn(3, 4, 3, device=device)
    with torch.no_grad():
        actions, logp = model.actor.sample(obs)
    return {
        "obs": obs,
        "critic_obs": torch.randn(3, 4, 4, device=device),
        "actions": actions,
        "behaviour_logp": logp,
        "rewards": torch.randn(3, 4, device=device),
        "dones": torch.zeros(3, 4, device=device, dtype=torch.bool),
        "last_critic_obs": torch.randn(4, 4, device=device),
    }


def test_tensor_objective_matches_distribution_values_and_gradients() -> None:
    torch.manual_seed(6)
    model = learner()
    prepared = model.process(batch(model))
    args = (
        prepared.obs,
        prepared.critic_obs,
        prepared.actions,
        prepared.behaviour_logp,
        prepared.target_logp,
        prepared.values,
        prepared.returns,
        prepared.advantages,
        prepared.old_mean + 0.1,
        prepared.old_std * 1.2,
    )
    loss, metrics = model.execution.objective(*args)
    dist = model.actor.distribution(prepared.obs)
    ratio = (dist.log_prob(prepared.actions).sum(-1) - prepared.behaviour_logp).exp()
    cfg = model.cfg.algorithm
    policy = -torch.minimum(ratio * prepared.advantages, ratio.clamp(0.8, 1.2) * prepared.advantages).mean()
    value = model.critic(prepared.critic_obs)
    clipped = prepared.values + (value - prepared.values).clamp(-0.2, 0.2)
    value_loss = torch.maximum((value - prepared.returns).square(), (clipped - prepared.returns).square()).mean()
    reference = policy + cfg.value_loss_coef * value_loss - cfg.entropy_coef * dist.entropy().sum(-1).mean()
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(metrics[3], kl_divergence(Normal(args[8], args[9]), dist).sum(-1).mean())
    parameters = [*model.actor.parameters(), *model.critic.parameters()]
    expected_grads = torch.autograd.grad(reference, parameters, retain_graph=True)
    actual_grads = torch.autograd.grad(loss, parameters)
    for expected, actual in zip(expected_grads, actual_grads, strict=True):
        torch.testing.assert_close(actual, expected)


def test_cuda_graph_rejects_cpu_and_requires_explicit_fused() -> None:
    with pytest.raises(ValueError, match="fused_optimizer"):
        learner("cuda_graph")
    with pytest.raises(ValueError, match="CUDA"):
        learner("cuda_graph", True)


def test_cpu_fused_tensor_lr_and_checkpoint() -> None:
    torch.manual_seed(3)
    plain, fused = learner(), learner(fused=True)
    fused.load_state_dict(deepcopy(plain.state_dict()))
    data = batch(plain)
    for _ in range(2):
        torch.manual_seed(15)
        expected = plain.update(data)
        torch.manual_seed(15)
        actual = fused.update(data)
        assert actual == pytest.approx(expected, abs=2e-6, rel=2e-5)
    for key, value in plain.actor.state_dict().items():
        torch.testing.assert_close(value, fused.actor.state_dict()[key])


@pytest.mark.parametrize("kl, factor", [(0.1, 1 / 1.5), (1e-6, 1.5), (0.0, 1.0)])
def test_adaptive_learning_rate_stays_on_device(kl: float, factor: float) -> None:
    model = learner()
    prepared = model.process(batch(model))
    # Equal variances make KL = mean displacement squared / (2 * variance).
    old_mean = prepared.old_mean + prepared.old_std * (kl**0.5)
    args = (
        prepared.obs,
        prepared.critic_obs,
        prepared.actions,
        prepared.behaviour_logp,
        prepared.target_logp,
        prepared.values,
        prepared.returns,
        prepared.advantages,
        old_mean,
        prepared.old_std,
    )
    original_rate = model.learning_rate.clone()
    model.execution(args)
    assert model.optimizer.param_groups[0]["lr"] is model.learning_rate
    torch.testing.assert_close(model.learning_rate, original_rate * factor)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("APPO_TEST_COMPILE") != "1", reason="opt-in compiler runtime")
def test_compile_fullgraph_cpu_parity() -> None:
    _parity("compile", "cpu")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("APPO_TEST_CUDA") != "1" or not torch.cuda.is_available(), reason="opt-in CUDA runtime"
)
@pytest.mark.parametrize("mode", ["compile", "cuda_graph"])
def test_cuda_execution_parity_and_checkpoint(mode: str) -> None:
    _parity(mode, "cuda")


def _parity(mode: str, device: str) -> None:
    torch.manual_seed(8)
    eager = learner(fused=True, device=device)
    accelerated = learner(mode, True, device)
    if mode == "compile":
        # A strict KL > 0 adaptive branch is discontinuous at the initial
        # identical-policy snapshot. Fused compiler arithmetic can make its
        # roundoff positive; use fixed LR for numerical execution parity.
        eager.cfg.algorithm.schedule = "fixed"
        accelerated.cfg.algorithm.schedule = "fixed"
    accelerated.load_state_dict(deepcopy(eager.state_dict()))
    data = batch(eager)
    rng = torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state()
    # Three updates also check normalizer publication and frozen target cadence.
    for index in range(3):
        torch.manual_seed(20 + index)
        expected = eager.update(data)
        expected_rng = torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state()
        torch.manual_seed(20 + index)
        actual = accelerated.update(data)
        actual_rng = torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state()
        assert torch.equal(expected_rng, actual_rng)
        assert actual == pytest.approx(expected, abs=3e-5, rel=3e-4)
        for name in ("actor", "critic", "target_actor"):
            for key, value in eager.state_dict()[name].items():
                torch.testing.assert_close(value, accelerated.state_dict()[name][key], atol=3e-6, rtol=3e-5)
        for state in accelerated.optimizer.state.values():
            assert state["step"].item() == (index + 1) * 4
        # Loading an already-captured learner must invalidate old optimizer pointers.
        if index == 1:
            accelerated.load_state_dict(deepcopy(eager.state_dict()))
    restored = learner(device=device)
    restored.cfg.algorithm.schedule = eager.cfg.algorithm.schedule
    restored.load_state_dict(deepcopy(accelerated.state_dict()))
    torch.manual_seed(40)
    expected = accelerated.update(data)
    torch.manual_seed(40)
    actual = restored.update(data)
    assert actual == pytest.approx(expected, abs=3e-5, rel=3e-4)
    if device == "cpu":
        torch.set_rng_state(rng)
    else:
        torch.cuda.set_rng_state(rng)
