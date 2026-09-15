# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import fields
from typing import TYPE_CHECKING

from motrix_env_core.config.decorate import configclass

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.sim_reset import ResetTerm


@configclass(kw_only=True)
class ResetTermCfg(abc.ABC):
    """Configuration that creates one reset dispatch descriptor."""

    @abc.abstractmethod
    def __call__(self, ctx) -> ResetTerm:
        """Assemble the concrete reset dispatch descriptor."""


@configclass
class ManagerResetCfg:
    """Typed declaration group for one environment's reset terms."""

    def to_dict(self) -> dict[str, ResetTermCfg]:
        """Return reset term configs keyed by their declaration names."""
        term_cfgs: dict[str, ResetTermCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if not isinstance(term_cfg, ResetTermCfg):
                raise TypeError(
                    f"Manager reset term {term_field.name!r} must be a ResetTermCfg, got {type(term_cfg).__name__}."
                )
            term_cfgs[term_field.name] = term_cfg
        return term_cfgs


__all__ = ["ManagerResetCfg", "ResetTermCfg"]
