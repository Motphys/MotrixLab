# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
import torch
from motrix_rl_appo.config import AlgorithmCfg, AppoCfg, NetworkCfg
from motrix_rl_appo.networks import EmpiricalNormalization, build_actor, build_critic


def test_pooled_moments_include_between_batch_variance() -> None:
    normalizer = EmpiricalNormalization(2)
    samples = torch.tensor([[1.0, 5.0], [3.0, 7.0], [10.0, 1.0], [14.0, 3.0], [20.0, 9.0]])
    normalizer.update(samples[:2])
    normalizer.update(samples[2:])
    torch.testing.assert_close(normalizer._mean[0], samples.mean(0))
    torch.testing.assert_close(normalizer._var[0], samples.var(0, correction=0))
    torch.testing.assert_close(
        normalizer(samples), (samples - samples.mean(0)) / (samples.std(0, correction=0) + normalizer.epsilon)
    )
    normalizer(samples)
    assert normalizer.count.item() == len(samples)


def test_policy_logp_and_time_major_shapes() -> None:
    cfg = NetworkCfg(hidden_dims=[8])
    actor = build_actor(cfg, 3, 2)
    obs = torch.randn(4, 5, 3)
    actions, logp = actor.sample(obs)
    assert actions.shape == (4, 5, 2)
    assert logp.shape == (4, 5)
    torch.testing.assert_close(actor.distribution(obs).log_prob(actions).sum(-1), logp)
    assert build_critic(cfg, 3)(obs).shape == (4, 5)
    with torch.no_grad():
        actor.log_std.fill_(-100)
    assert (actor.distribution(obs).scale > 0).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: NetworkCfg(init_noise_std=0),
        lambda: NetworkCfg(hidden_dims=[0]),
        lambda: NetworkCfg(activation="invalid"),
        lambda: AlgorithmCfg(gamma=1.1),
        lambda: AlgorithmCfg(target_update_interval=0),
        lambda: AppoCfg(queue_chunks=0),
    ],
)
def test_invalid_config_rejected(factory: Callable[[], NetworkCfg | AlgorithmCfg | AppoCfg]) -> None:
    with pytest.raises(ValueError):
        factory()
