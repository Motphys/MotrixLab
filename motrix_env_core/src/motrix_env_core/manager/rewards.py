# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.terms import BaseTerm, canonicalize_term_args


@dataclass(frozen=True, slots=True, init=False)
class RewardTerm(BaseTerm):
    """Host-side reward dispatch and its static canonicalized arguments."""

    def __init__(self, dispatch: Callable[..., float], *args: Any) -> None:
        BaseTerm.__init__(self, dispatch, *args)


@configclass(kw_only=True)
class RewardTermCfg(abc.ABC):
    weight: float

    @abc.abstractmethod
    def __call__(self, ctx: BuildContext) -> RewardTerm:
        """Assemble the runtime term dispatched by the torch executor."""


@configclass
class ManagerRewardsCfg:
    """Typed declaration group for one environment's reward terms."""

    def to_dict(self) -> dict[str, RewardTermCfg]:
        """Return reward configs keyed by their declaration names."""
        term_cfgs: dict[str, RewardTermCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if not isinstance(term_cfg, RewardTermCfg):
                raise TypeError(
                    f"Manager reward {term_field.name!r} must be a RewardTermCfg, got {type(term_cfg).__name__}."
                )
            term_cfgs[term_field.name] = term_cfg
        return term_cfgs


def create_reward_terms(cfg: dict[str, RewardTermCfg], ctx: BuildContext) -> dict[str, RewardTerm]:
    terms = {}
    for name, term_cfg in cfg.items():
        created = term_cfg(ctx)
        if not isinstance(created, RewardTerm):
            raise TypeError(f"Manager reward {name} __call__() must return RewardTerm, got {type(created).__name__}.")
        terms[name] = RewardTerm(
            created.dispatch,
            *canonicalize_term_args(created.args, context=f"Manager term reward.{name}"),
        )
    return terms


class RewardManager:
    """Build reward terms and own their evaluate-kernel output buffers.

    Per-term columns are allocated here in declaration order; the manager
    program resolver assigns ``term.index`` by the same order, so the
    evaluate kernel's column writes and the ``reward_terms`` views stay
    aligned by contract rather than by coincidence.
    """

    def __init__(self, cfgs: dict[str, RewardTermCfg], context: BuildContext, num_envs: int) -> None:
        self.terms = create_reward_terms(cfgs, context)
        self.total_reward = torch.zeros(num_envs, dtype=torch.float32)
        self.term_rewards = torch.empty((num_envs, len(self.terms)), dtype=torch.float32)
        self.weighted_term_rewards = torch.empty_like(self.term_rewards)
        self.reward_terms = {name: self.weighted_term_rewards[:, index] for index, name in enumerate(self.terms)}


__all__ = [
    "ManagerRewardsCfg",
    "RewardTerm",
    "RewardTermCfg",
    "RewardManager",
    "create_reward_terms",
]
