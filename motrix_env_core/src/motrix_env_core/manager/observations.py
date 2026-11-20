# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import numbers
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.terms import BaseTerm, canonicalize_term_args

if TYPE_CHECKING:
    from motrix_env_core.manager.env import ManagerBasedEnvCfg


@configclass(kw_only=True)
class ObservationTermCfg(abc.ABC):
    """Configuration that creates one environment-local observation term."""

    @abc.abstractmethod
    def __call__(self, ctx: BuildContext) -> ObsTerm:
        """Assemble the runtime term (dispatch plus static arguments)."""


@dataclass(frozen=True, slots=True, init=False)
class ObsTerm(BaseTerm):
    """Pure observation dispatch with a fixed output width.

    Read context and write only the supplied output, except observation noise
    may consume the default Torch stream through ctx.rand. Do not advance
    history or commands: truncated steps may observe again before reset.
    """

    size: int

    def __init__(self, size: int, dispatch: Callable[..., Any], *args: Any) -> None:
        object.__setattr__(self, "size", size)
        BaseTerm.__init__(self, dispatch, *args)

    def __post_init__(self) -> None:
        BaseTerm.__post_init__(self)
        if not isinstance(self.size, numbers.Integral) or isinstance(self.size, (bool, np.bool_)):
            raise TypeError("Observation term size must be an integer.")
        if self.size <= 0:
            raise ValueError("Observation term size must be positive.")


@configclass
class ManagerObservationGroupCfg:
    """Typed declaration group for one named observation group's terms."""

    history_length: int = 1

    def to_dict(self) -> dict[str, ObservationTermCfg]:
        """Return observation term configs keyed by their declaration names."""
        term_cfgs: dict[str, ObservationTermCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if isinstance(term_cfg, ObservationTermCfg):
                term_cfgs[term_field.name] = term_cfg
        return term_cfgs


@configclass
class ManagerObservationsCfg:
    """Typed declaration groups for one environment's observation terms."""

    def to_dict(self) -> dict[str, ManagerObservationGroupCfg]:
        """Return observation groups keyed by their declaration names."""
        group_cfgs: dict[str, ManagerObservationGroupCfg] = {}
        for group_field in fields(self):
            group_name = group_field.name
            group_cfg = getattr(self, group_name)
            if not isinstance(group_cfg, ManagerObservationGroupCfg):
                raise TypeError(
                    f"Observation group {group_name!r} must be a ManagerObservationGroupCfg, "
                    f"got {type(group_cfg).__name__}."
                )
            group_cfgs[group_name] = group_cfg
        return group_cfgs


@dataclass(frozen=True)
class ObservationTermEntry:
    name: str
    term: ObsTerm
    size: int


@dataclass(frozen=True)
class ObservationGroupEntry:
    name: str
    terms: tuple[ObservationTermEntry, ...]
    size: int
    history_length: int = 1


def create_observation_groups(
    cfg: ManagerBasedEnvCfg,
    ctx: BuildContext,
) -> dict[str, ObservationGroupEntry]:
    """Create and validate runtime observation terms grouped by output buffer."""
    groups: dict[str, ObservationGroupEntry] = {}
    for group_name, term_cfgs in cfg.observation_cfgs().items():
        entries = []
        group_size = 0
        for term_name, term_cfg in term_cfgs.items():
            created = term_cfg(ctx)
            if not isinstance(created, ObsTerm):
                raise TypeError(
                    f"Observation term {group_name}.{term_name} __call__() must return ObsTerm, "
                    f"got {type(created).__name__}."
                )
            term = ObsTerm(
                int(created.size),
                created.dispatch,
                *canonicalize_term_args(created.args, context=f"Observation term {group_name}.{term_name}"),
            )
            resolved_size = int(term.size)
            entries.append(ObservationTermEntry(term_name, term, resolved_size))
            group_size += resolved_size
        if entries:
            group_cfg = cfg.observation_group_cfgs()[group_name]
            history_length = group_cfg.history_length if isinstance(group_cfg, ManagerObservationGroupCfg) else 1
            if not isinstance(history_length, numbers.Integral) or isinstance(history_length, (bool, np.bool_)):
                raise TypeError("Observation history_length must be an integer.")
            if history_length < 1:
                raise ValueError("Observation history_length must be at least 1.")
            groups[group_name] = ObservationGroupEntry(group_name, tuple(entries), group_size, history_length)
    if "policy" not in groups:
        raise ValueError("Manager environment config requires a 'policy' observation group.")
    return groups


__all__ = [
    "ManagerObservationGroupCfg",
    "ManagerObservationsCfg",
    "ObsTerm",
    "ObservationGroupEntry",
    "ObservationTermCfg",
    "ObservationTermEntry",
    "create_observation_groups",
]
