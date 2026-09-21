# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared policy-variant dispatch for sync and async FastSAC execution."""

from __future__ import annotations

import torch
from torch import nn

# Importing the SONIC module registers its bundled PolicyVariant. This side
# effect lives in the dispatch module rather than the package initializer so
# that ``import motrix_rl`` stays free of heavy imports; every consumer below
# already imports torch anyway.
from motrix_rl.fastsac import sonic as _sonic  # noqa: E402,F401
from motrix_rl.fastsac.api import PolicyVariant, policy_variant_registry
from motrix_rl.fastsac.config import FastSacCfg


def resolve_policy_variant(name: str) -> PolicyVariant:
    """Resolve a registered policy variant by name."""
    return policy_variant_registry.resolve(name)


def make_actor(
    cfg: FastSacCfg,
    *,
    dims: tuple[int, int],
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    device: torch.device | str,
) -> nn.Module:
    """Build the configured actor and validate the environment contract."""
    return resolve_policy_variant(cfg.policy_variant).build_actor(
        cfg,
        dims,
        action_scale,
        action_bias,
        device,
    )
