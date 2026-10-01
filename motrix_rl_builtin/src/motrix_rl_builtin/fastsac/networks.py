# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Network modules for FastSAC.

Ported from holosoma's ``fast_sac`` (a FastTD3/FastSAC-style distributional SAC):
a tanh-squashed Gaussian actor and an ensemble of distributional (C51-style)
Q networks. The CNN/asymmetric-observation paths from the original are dropped;
observations here are a single flat vector shared by actor and critic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        log_std_max: float,
        log_std_min: float,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.use_tanh = use_tanh

        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 4, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim // 4, n_act, device=device)
        nn.init.constant_(self.fc_mu.weight, 0.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

        if action_scale is None:
            action_scale = torch.ones(n_act)
        if action_bias is None:
            action_bias = torch.zeros(n_act)
        self.register_buffer("action_scale", action_scale.to(device))
        self.register_buffer("action_bias", action_bias.to(device))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.fc_mu(x)
        log_std = torch.tanh(self.fc_logstd(x))
        # From SpinUp / Denis Yarats: map tanh output to [log_std_min, log_std_max].
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        if self.use_tanh:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = mean
        return action, mean, log_std

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
            log_prob = dist.log_prob(raw_action)
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            action = raw_action
            log_prob = dist.log_prob(raw_action)

        return action, log_prob.sum(1)

    @torch.no_grad()
    def explore(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        _, mean, log_std = self(obs)
        if deterministic:
            if self.use_tanh:
                return torch.tanh(mean) * self.action_scale + self.action_bias
            return mean
        std = log_std.exp()
        raw_action = torch.distributions.Normal(mean, std).rsample()
        if self.use_tanh:
            return torch.tanh(raw_action) * self.action_scale + self.action_bias
        return raw_action


class DistributionalQNetwork(nn.Module):
    """A single C51-style distributional Q network."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, actions], 1))

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.arange(batch_size, device=device, dtype=torch.long)
            .mul_(self.num_atoms)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
        )
        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist


class Critic(nn.Module):
    """Ensemble of distributional Q networks."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: torch.device | None = None,
    ):
        super().__init__()
        if num_q_networks < 1:
            raise ValueError("num_q_networks must be at least 1")
        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    n_obs=n_obs,
                    n_act=n_act,
                    num_atoms=num_atoms,
                    v_min=v_min,
                    v_max=v_max,
                    hidden_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                    device=device,
                )
                for _ in range(num_q_networks)
            ]
        )
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([qnet(obs, actions) for qnet in self.qnets], dim=0)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                qnet.projection(obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device)
                for qnet in self.qnets
            ],
            dim=0,
        )

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(probs * self.q_support, dim=-1)
