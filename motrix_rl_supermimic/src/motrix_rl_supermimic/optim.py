# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Optimizer pairing and the policy-LR schedule."""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["MuonAdam", "policy_multiplier", "trigger_decay"]


def _linears(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


class MuonAdam:
    """Muon on hidden MLP matrices and fused Adam on remaining parameters.

    Only the inner Linear weights of each listed MLP go to Muon; first/last
    layer weights, all biases and LayerNorm parameters stay on Adam.
    """

    def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float, betas=(0.9, 0.95), fused: bool = True):
        hidden = {id(layer.weight) for mlp in mlps for layer in _linears(mlp)[1:-1]}
        muon_params = [p for p in model.parameters() if id(p) in hidden]
        adam_params = [p for p in model.parameters() if id(p) not in hidden and p.requires_grad]
        if not muon_params:
            raise ValueError("MuonAdam requires at least one hidden Linear weight")
        self.muon = torch.optim.Muon(muon_params, lr=lr, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
        self.adam = torch.optim.Adam(adam_params, lr=lr, betas=betas, fused=fused)

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])


def policy_multiplier(cfg, iteration: int, decay_at: int | None) -> float:
    """Policy-lr multiplier: pi_lr_mult until the decay trigger, a linear ramp
    back to x1 over lr_decay_len, then a cosine anneal to pi_anneal_min."""
    if decay_at is None:
        return cfg.pi_lr_mult
    multiplier = cfg.pi_lr_mult + (1.0 - cfg.pi_lr_mult) * min(1.0, (iteration - decay_at) / max(cfg.lr_decay_len, 1))
    if cfg.pi_anneal_len > 0 and iteration >= decay_at + cfg.lr_decay_len:
        u = min(1.0, (iteration - decay_at - cfg.lr_decay_len) / cfg.pi_anneal_len)
        multiplier = cfg.pi_anneal_min + (1.0 - cfg.pi_anneal_min) * 0.5 * (1.0 + math.cos(math.pi * u))
    return multiplier


def trigger_decay(cfg, iteration: int, normalized_return: float, decay_at: int | None) -> int | None:
    """One-shot sticky decay trigger on the aggregated normalized training
    return or the iteration deadline; returns the new trigger or None."""
    if decay_at is not None or cfg.lr_decay_ret <= 0:
        return decay_at
    if normalized_return >= cfg.lr_decay_ret or (cfg.lr_decay_by > 0 and iteration >= cfg.lr_decay_by):
        return iteration
    return None
