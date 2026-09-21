# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Neutral policy-variant extension contract for FastSAC."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from motrix_rl.fastsac.config import FastSacCfg

if TYPE_CHECKING:
    import torch
    from torch import nn


class PolicyVariant(Protocol):
    """Build and describe an actor that follows the frozen FastSAC actor protocol."""

    name: str

    def build_actor(
        self,
        cfg: FastSacCfg,
        dims: tuple[int, int],
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        device: torch.device | str,
    ) -> nn.Module:
        """Return an actor for ``(obs_dim, act_dim)``."""

    def passthrough_dims(self, cfg: FastSacCfg) -> int:
        """Return trailing observation dimensions bypassing normalization."""

    def aux_loss_weights(self, cfg: FastSacCfg) -> dict[str, float]:
        """Return named auxiliary-loss weights; an empty mapping means none."""

    def checkpoint_metadata(self, cfg: FastSacCfg) -> dict[str, Any]:
        """Return metadata sufficient to reconstruct the configured actor."""

    @classmethod
    def build_from_checkpoint(
        cls,
        metadata: Mapping[str, Any],
        cfg: FastSacCfg,
        *,
        device: torch.device | str,
    ) -> nn.Module:
        """Reconstruct an actor from checkpoint metadata."""


class PolicyVariantRegistry:
    def __init__(self) -> None:
        self._variants: dict[str, PolicyVariant] = {}

    def register(self, name: str, variant: PolicyVariant) -> None:
        self._variants[name] = variant

    def resolve(self, name: str) -> PolicyVariant:
        try:
            return self._variants[name]
        except KeyError as exc:
            expected = ", ".join(sorted(self._variants)) or "none"
            raise ValueError(f"unknown FastSAC policy variant {name!r}; expected one of: {expected}") from exc


policy_variant_registry = PolicyVariantRegistry()


class DefaultPolicyVariant:
    name = "default"

    def build_actor(
        self,
        cfg: FastSacCfg,
        dims: tuple[int, int],
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        device: torch.device | str,
    ) -> nn.Module:
        from motrix_rl.fastsac.networks import Actor

        obs_dim, act_dim = dims
        return Actor(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=cfg.agent.actor_hidden_dim,
            log_std_max=cfg.agent.log_std_max,
            log_std_min=cfg.agent.log_std_min,
            use_tanh=cfg.agent.use_tanh,
            use_layer_norm=cfg.agent.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            device=device,
        )

    def passthrough_dims(self, cfg: FastSacCfg) -> int:
        return 0

    def aux_loss_weights(self, cfg: FastSacCfg) -> dict[str, float]:
        return {}

    def checkpoint_metadata(self, cfg: FastSacCfg) -> dict[str, Any]:
        return {}

    @classmethod
    def build_from_checkpoint(
        cls,
        metadata: Mapping[str, Any],
        cfg: FastSacCfg,
        *,
        device: torch.device | str,
    ) -> nn.Module:
        import torch

        return cls().build_actor(
            cfg,
            (int(metadata["obs_dim"]), int(metadata["act_dim"])),
            torch.ones(int(metadata["act_dim"])),
            torch.zeros(int(metadata["act_dim"])),
            device,
        )


policy_variant_registry.register(DefaultPolicyVariant.name, DefaultPolicyVariant())
