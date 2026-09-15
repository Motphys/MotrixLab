# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from motrix_env_core.config import configclass
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.terms import BaseTerm, canonicalize_term_args

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.env import ManagerEnv


@dataclass(frozen=True, slots=True, init=False)
class RewardTerm(BaseTerm):
    """Host-side reward dispatch and its static Numba-compatible arguments."""

    def __init__(self, dispatch: Callable[..., float], *args: Any) -> None:
        BaseTerm.__init__(self, dispatch, *args)


@configclass(kw_only=True)
class RewardTermCfg(abc.ABC):
    weight: float

    @abc.abstractmethod
    def __call__(self, ctx) -> RewardTerm:
        """Assemble the runtime term (dispatch plus static arguments)."""


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


def create_reward_terms(cfg: dict[str, RewardTermCfg], env: ManagerEnv) -> dict[str, RewardTerm]:
    terms = {}
    for name, term_cfg in cfg.items():
        created = term_cfg(BuildContext(env, f"rewards.{name}"))
        if not isinstance(created, RewardTerm):
            raise TypeError(f"Manager reward {name} __call__() must return RewardTerm, got {type(created).__name__}.")
        terms[name] = RewardTerm(
            created.dispatch,
            *canonicalize_term_args(created.args, context=f"Manager term reward.{name}"),
        )
    return terms


__all__ = ["ManagerRewardsCfg", "RewardTerm", "RewardTermCfg", "create_reward_terms"]
