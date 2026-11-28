# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Time-major IMPALA value and policy-gradient corrections."""

import math
from dataclasses import dataclass

import torch


@dataclass
class VTraceReturns:
    values: torch.Tensor
    advantages: torch.Tensor


@torch.no_grad()
def compute_vtrace_targets(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    target_logp: torch.Tensor,
    behaviour_logp: torch.Tensor,
    rho_clip: float = 1.0,
    c_clip: float = 1.0,
    pg_rho_clip: float = 1.0,
) -> VTraceReturns:
    """All sequence tensors are [T,N], bootstrap_value is [N].

    ``discounts`` is gamma times not-done. Timeout bootstrap is already in
    rewards, so terminal and timeout boundaries both stop the trace.
    """
    log_ratio = target_logp - behaviour_logp
    rho = log_ratio.clamp(max=math.log(rho_clip)).exp()
    c = log_ratio.clamp(max=math.log(c_clip)).exp()
    pg_rho = log_ratio.clamp(max=math.log(pg_rho_clip)).exp()
    next_values = torch.cat((values[1:], bootstrap_value.unsqueeze(0)), dim=0)
    deltas = rho * (rewards + discounts * next_values - values)
    correction = torch.zeros_like(bootstrap_value)
    corrected = torch.empty_like(values)
    for step in range(rewards.shape[0] - 1, -1, -1):
        correction = deltas[step] + discounts[step] * c[step] * correction
        corrected[step] = values[step] + correction
    next_corrected = torch.cat((corrected[1:], bootstrap_value.unsqueeze(0)), dim=0)
    advantages = pg_rho * (rewards + discounts * next_corrected - values)
    return VTraceReturns(values=corrected, advantages=advantages)


def impact_ratio(
    current_logp: torch.Tensor,
    behaviour_logp: torch.Tensor,
    target_logp: torch.Tensor,
    worker_target_clip: float = 2.0,
) -> torch.Tensor:
    """min(mu/target, kappa) * current/mu, kept separate from rhoPG."""
    worker_correction = (behaviour_logp - target_logp).clamp(max=math.log(worker_target_clip))
    return (worker_correction + current_logp - behaviour_logp).exp()
