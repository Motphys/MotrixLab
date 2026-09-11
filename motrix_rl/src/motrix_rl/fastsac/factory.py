# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared FastSAC actor construction for sync and async execution."""

from __future__ import annotations

import torch
from torch import nn

from motrix_rl.fastsac.config import FastSacAgentCfg, SonicSacCfg
from motrix_rl.fastsac.networks import Actor
from motrix_rl.fastsac.sonic import SonicActor, SonicModelConfig


def make_actor(
    cfg: FastSacAgentCfg,
    sonic_cfg: SonicSacCfg | None = None,
    *,
    obs_dim: int,
    act_dim: int,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    device: torch.device | str,
) -> nn.Module:
    """Build the configured actor and validate the environment contract."""
    if sonic_cfg is None or not sonic_cfg.enabled:
        return Actor(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=cfg.actor_hidden_dim,
            log_std_max=cfg.log_std_max,
            log_std_min=cfg.log_std_min,
            use_tanh=cfg.use_tanh,
            use_layer_norm=cfg.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            device=device,
        )

    model_values = dict(sonic_cfg.model) if getattr(sonic_cfg, "model", None) else {}
    model = (
        SonicModelConfig.from_mapping(model_values)
        if model_values
        else SonicModelConfig.from_profile(sonic_cfg.profile)
    )
    if obs_dim != model.packed_obs_dim:
        raise ValueError(
            f"SONIC profile {sonic_cfg.profile!r} expects packed actor dim {model.packed_obs_dim}, got {obs_dim}"
        )
    if act_dim != model.action_dim:
        raise ValueError(f"SONIC profile {sonic_cfg.profile!r} expects action dim {model.action_dim}, got {act_dim}")
    if sonic_cfg.actor_num_blocks <= 0:
        raise ValueError("SONIC actor_num_blocks must be positive")
    actor = SonicActor(
        model,
        sonic_cfg.auxiliary,
        log_std_min=cfg.log_std_min,
        log_std_max=cfg.log_std_max,
        actor_num_blocks=sonic_cfg.actor_num_blocks,
        device=device,
    )
    # The SONIC action term owns the per-joint scale (scale * effort_limit / kp).
    # Do not apply the generic environment Box affine transform a second time.
    actor.action_scale.fill_(1.0)
    actor.action_bias.zero_()
    return actor
