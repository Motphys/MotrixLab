# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Gaussian policy, scalar value network and explicit pooled observation stats."""

import math

import torch
from torch import nn
from torch.distributions import Normal

from motrix_rl_appo.config import NetworkCfg


class EmpiricalNormalization(nn.Module):
    """Population moments; forward never changes the inference snapshot."""

    def __init__(self, dim: int, epsilon: float = 1e-2) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_var", torch.ones(1, dim))
        self.register_buffer("_std", torch.ones(1, dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    @torch.no_grad()
    def update(self, obs: torch.Tensor) -> None:
        samples = obs.reshape(-1, self._mean.shape[-1])
        batch_var, batch_mean = torch.var_mean(samples, dim=0, correction=0, keepdim=True)
        batch_count = samples.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self._mean
        # Parallel population-variance merge includes between-group variance.
        variance = (
            self._var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        ) / total
        self._mean.add_(delta * (batch_count / total))
        self._var.copy_(variance)
        self._std.copy_(variance.sqrt())
        self.count.copy_(total)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self._mean) / (self._std + self.epsilon)


def _mlp(cfg: NetworkCfg, input_dim: int, output_dim: int) -> nn.Sequential:
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    layers: list[nn.Module] = []
    previous = input_dim
    for width in cfg.hidden_dims:
        layers.extend((nn.Linear(previous, width), activations[cfg.activation]()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, cfg: NetworkCfg, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(obs_dim) if cfg.normalize_observations else nn.Identity()
        self.mlp = _mlp(cfg, obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(cfg.init_noise_std)))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))

    def distribution(self, obs: torch.Tensor) -> Normal:
        # Bounds prevent exp under/overflow without permitting negative scales.
        return Normal(self.mean(obs), self.log_std.clamp(-20, 2).exp())

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(obs)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions).sum(dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(obs)


class Critic(nn.Module):
    def __init__(self, cfg: NetworkCfg, obs_dim: int) -> None:
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(obs_dim) if cfg.normalize_observations else nn.Identity()
        self.mlp = _mlp(cfg, obs_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs)).squeeze(-1)


def build_actor(cfg: NetworkCfg, obs_dim: int, action_dim: int) -> Actor:
    return Actor(cfg, obs_dim, action_dim)


def build_critic(cfg: NetworkCfg, obs_dim: int) -> Critic:
    return Critic(cfg, obs_dim)
