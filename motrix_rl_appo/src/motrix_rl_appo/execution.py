# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Tensor-only APPO objective and optional fixed-minibatch CUDA execution."""

import math
from collections.abc import Callable

import torch
from torch import nn

from motrix_rl_appo.config import AlgorithmCfg
from motrix_rl_appo.networks import Actor, Critic
from motrix_rl_appo.vtrace import impact_ratio

METRIC_NAMES = (
    "policy_loss",
    "value_loss",
    "entropy",
    "kl",
    "importance_ratio",
    "clip_fraction",
    "policy_std",
    "grad_norm",
)


class MinibatchExecution:
    """Own execution caches, not model or optimizer state.

    Compilation is intentionally fullgraph and errors propagate to the caller.
    CUDA capture is lazy, so its first-call cost can be measured separately.
    """

    def __init__(self, actor: Actor, critic: Critic, optimizer: torch.optim.Adam, cfg: AlgorithmCfg) -> None:
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.cfg = cfg
        self.parameters = list(actor.parameters()) + list(critic.parameters())
        self.loss: Callable[..., tuple[torch.Tensor, torch.Tensor]] = self.objective
        if cfg.execution == "compile":
            self.loss = torch.compile(self.objective, fullgraph=True)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.inputs: tuple[torch.Tensor, ...] = ()
        self.metrics: torch.Tensor

    def objective(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        behaviour_logp: torch.Tensor,
        target_logp: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        mean = self.actor.mean(obs)
        std = self.actor.log_std.clamp(-20, 2).exp()
        # Match torch.distributions.Normal, without distribution validation's
        # data-dependent Python branches (and consequent CUDA synchronization).
        logp = (-((actions - mean).square()) / (2 * std.square()) - std.log() - math.log(math.sqrt(2 * math.pi))).sum(
            -1
        )
        ratio = impact_ratio(logp, behaviour_logp, target_logp, cfg.worker_target_clip)
        policy_loss = -torch.minimum(
            ratio * advantages, ratio.clamp(1 - cfg.clip_param, 1 + cfg.clip_param) * advantages
        ).mean()
        values = self.critic(critic_obs)
        value_error = (values - returns).square()
        if cfg.use_clipped_value_loss:
            clipped = old_values + (values - old_values).clamp(-cfg.clip_param, cfg.clip_param)
            value_error = torch.maximum(value_error, (clipped - returns).square())
        value_loss = value_error.mean()
        entropy = (std.log() + 0.5 + 0.5 * math.log(2 * math.pi)).sum()
        with torch.no_grad():
            var_ratio = (old_std / std).square()
            mean_term = ((old_mean - mean) / std).square()
            kl = (0.5 * (var_ratio - 1 - var_ratio.log() + mean_term)).sum(-1).mean()
            metrics = torch.stack(
                (
                    policy_loss.detach(),
                    value_loss.detach(),
                    entropy.detach(),
                    kl,
                    ratio.detach().mean(),
                    ((ratio.detach() - 1).abs() > cfg.clip_param).float().mean(),
                    std.detach().mean(),
                )
            )
        loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy
        return loss, metrics

    def step(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(*inputs)
        with torch.no_grad():
            if self.cfg.schedule == "adaptive":
                rate = self.optimizer.param_groups[0]["lr"]
                kl = metrics[3]
                rate.copy_(
                    torch.where(
                        kl > 2 * self.cfg.desired_kl,
                        (rate / 1.5).clamp(min=1e-5),
                        torch.where((kl > 0) & (kl < self.cfg.desired_kl / 2), (rate * 1.5).clamp(max=1e-2), rate),
                    )
                )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.cfg.max_grad_norm)
        self.optimizer.step()
        return torch.cat((metrics, grad_norm.detach().reshape(1)))

    def __call__(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.cfg.execution != "cuda_graph":
            return self.step(inputs)
        if self.graph is None:
            self.capture(inputs)
        if any(buffer.shape != value.shape for buffer, value in zip(self.inputs, inputs, strict=True)):
            raise ValueError("cuda_graph requires the same minibatch shapes until checkpoint load")
        for buffer, value in zip(self.inputs, inputs, strict=True):
            buffer.copy_(value)
        self.graph.replay()
        return self.metrics

    def capture(self, inputs: tuple[torch.Tensor, ...]) -> None:
        self.inputs = tuple(value.clone() for value in inputs)
        device = inputs[0].device
        # Adam lazily allocates state. Warmup/capture must not constitute extra
        # training: restore every mutable value in place, retaining graph pointers.
        parameters = [parameter.detach().clone() for parameter in self.parameters]
        rate = self.optimizer.param_groups[0]["lr"].clone()
        saved_state = {
            parameter: {key: value.clone() for key, value in state.items()}
            for parameter, state in self.optimizer.state.items()
        }
        rng = torch.cuda.get_rng_state(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.step(self.inputs)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.metrics = self.step(self.inputs)
        with torch.no_grad():
            for parameter, saved in zip(self.parameters, parameters, strict=True):
                parameter.copy_(saved)
                for key, value in self.optimizer.state[parameter].items():
                    if parameter in saved_state:
                        value.copy_(saved_state[parameter][key])
                    else:
                        value.zero_()
            self.optimizer.param_groups[0]["lr"].copy_(rate)
        torch.cuda.set_rng_state(rng, device)
