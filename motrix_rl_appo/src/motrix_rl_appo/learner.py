# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Fixed-chunk multi-epoch APPO optimization with a frozen target snapshot."""

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TypedDict

import torch

from motrix_rl_appo.config import AppoCfg
from motrix_rl_appo.execution import METRIC_NAMES, MinibatchExecution
from motrix_rl_appo.networks import EmpiricalNormalization, build_actor, build_critic
from motrix_rl_appo.vtrace import compute_vtrace_targets


@dataclass
class PreparedBatch:
    obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    behaviour_logp: torch.Tensor
    target_logp: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor


class LearnerState(TypedDict):
    actor: dict[str, torch.Tensor]
    critic: dict[str, torch.Tensor]
    target_actor: dict[str, torch.Tensor]
    optimizer: dict
    update_count: int
    global_step: int


class APPOLearner:
    def __init__(
        self, cfg: AppoCfg, obs_dim: int, critic_dim: int, action_dim: int, device: str | torch.device
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = build_actor(cfg.actor, obs_dim, action_dim).to(self.device)
        self.critic = build_critic(cfg.critic, critic_dim).to(self.device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False).eval()
        parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        algorithm = cfg.algorithm
        if algorithm.execution == "cuda_graph" and self.device.type != "cuda":
            raise ValueError("cuda_graph execution requires a CUDA learner device")
        self.learning_rate = torch.tensor(algorithm.learning_rate, device=self.device)
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            fused=algorithm.fused_optimizer,
            foreach=not algorithm.fused_optimizer and self.device.type == "cuda",
            capturable=self.device.type == "cuda",
        )
        self.execution = MinibatchExecution(self.actor, self.critic, self.optimizer, algorithm)
        self.update_count = 0
        self.global_step = 0

    @torch.no_grad()
    def process(self, batch: Mapping[str, torch.Tensor]) -> PreparedBatch:
        """Prepare detached targets without changing either inference snapshot."""
        data = {name: tensor.to(self.device) for name, tensor in batch.items()}
        obs, critic_obs = data["obs"], data["critic_obs"]
        values = self.critic(critic_obs)
        bootstrap = self.critic(data["last_critic_obs"])
        target_mean = self.target_actor.mean(obs)
        target_std = self.target_actor.log_std.clamp(-20, 2).exp()
        target_logp = (
            -(data["actions"] - target_mean).square() / (2 * target_std.square())
            - target_std.log()
            - math.log(math.sqrt(2 * math.pi))
        ).sum(-1)
        algorithm = self.cfg.algorithm
        trace = compute_vtrace_targets(
            data["rewards"],
            algorithm.gamma * (~data["dones"].bool()).float(),
            values,
            bootstrap,
            target_logp,
            data["behaviour_logp"],
            algorithm.rho_clip,
            algorithm.c_clip,
            algorithm.pg_rho_clip,
        )
        advantages = trace.advantages.flatten()
        if algorithm.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)
        old_mean = self.actor.mean(obs)
        old_std = self.actor.log_std.clamp(-20, 2).exp().expand_as(old_mean)
        return PreparedBatch(
            obs=obs.flatten(0, 1),
            critic_obs=critic_obs.flatten(0, 1),
            actions=data["actions"].flatten(0, 1),
            behaviour_logp=data["behaviour_logp"].flatten(),
            target_logp=target_logp.flatten(),
            values=values.flatten(),
            returns=trace.values.flatten(),
            advantages=advantages,
            old_mean=old_mean.flatten(0, 1),
            old_std=old_std.flatten(0, 1),
        )

    def update(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        sample_count = batch["rewards"].numel()
        algorithm = self.cfg.algorithm
        if algorithm.num_mini_batches > sample_count:
            raise ValueError("num_mini_batches cannot exceed the number of rollout transitions")
        prepared = self.process(batch)
        if algorithm.execution == "cuda_graph" and sample_count % algorithm.num_mini_batches:
            raise ValueError("cuda_graph requires equally sized minibatches")
        totals = torch.zeros(len(METRIC_NAMES), device=self.device)
        updates = 0
        tensors = (
            prepared.obs,
            prepared.critic_obs,
            prepared.actions,
            prepared.behaviour_logp,
            prepared.target_logp,
            prepared.values,
            prepared.returns,
            prepared.advantages,
            prepared.old_mean,
            prepared.old_std,
        )
        for _ in range(algorithm.num_learning_epochs):
            indices = torch.randperm(sample_count, device=self.device)
            for ids in indices.tensor_split(algorithm.num_mini_batches):
                totals.add_(self.execution(tuple(tensor[ids] for tensor in tensors)))
                updates += 1
        # Keep policy coordinates fixed throughout optimization. New sample
        # moments become part of the next published actor/critic snapshot.
        if isinstance(self.actor.obs_normalizer, EmpiricalNormalization):
            self.actor.obs_normalizer.update(prepared.obs)
        if isinstance(self.critic.obs_normalizer, EmpiricalNormalization):
            self.critic.obs_normalizer.update(prepared.critic_obs)
        self.update_count += 1
        self.global_step += sample_count
        # Parameters AND normalization buffers form one immutable target version.
        if self.update_count % algorithm.target_update_interval == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
        diagnostics = torch.stack(
            (
                self.learning_rate,
                (prepared.target_logp - prepared.behaviour_logp)
                .clamp(max=math.log(algorithm.pg_rho_clip))
                .exp()
                .mean(),
                ((prepared.behaviour_logp - prepared.target_logp) > math.log(algorithm.worker_target_clip))
                .float()
                .mean(),
            )
        )
        # One host transfer per update, never one synchronization per minibatch.
        values = torch.cat((totals / updates, diagnostics)).detach().cpu().tolist()
        return dict(zip((*METRIC_NAMES, "learning_rate", "pg_rho", "worker_cap_fraction"), values, strict=True))

    def state_dict(self) -> LearnerState:
        return LearnerState(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            target_actor=self.target_actor.state_dict(),
            optimizer=self.optimizer.state_dict(),
            update_count=self.update_count,
            global_step=self.global_step,
        )

    def load_state_dict(self, state: LearnerState) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_actor.load_state_dict(state["target_actor"])
        # Loading Adam replaces storage. Discard captured pointers before load,
        # then retain this learner's execution settings, not the saved mode's.
        self.execution = MinibatchExecution(self.actor, self.critic, self.optimizer, self.cfg.algorithm)
        self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        for group in self.optimizer.param_groups:
            self.learning_rate.copy_(group["lr"])
            group["lr"] = self.learning_rate
            group["fused"] = self.cfg.algorithm.fused_optimizer
            group["capturable"] = self.device.type == "cuda"
            group["foreach"] = not self.cfg.algorithm.fused_optimizer and self.device.type == "cuda"
        for optimizer_state in self.optimizer.state.values():
            optimizer_state["step"] = optimizer_state["step"].to(self.device if self.device.type == "cuda" else "cpu")
        self.update_count = state["update_count"]
        self.global_step = state["global_step"]
