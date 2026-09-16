# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.terms import BaseTerm, canonicalize_term_args

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.env import ManagerEnv


@dataclass(frozen=True, slots=True, init=False)
class TerminationTerm(BaseTerm):
    """Termination dispatch invocation with manager-allocated metric names."""

    metric_names: tuple[str, ...]

    def __init__(self, dispatch: Callable[..., bool], *args: Any, metric_names: tuple[str, ...] = ()) -> None:
        object.__setattr__(self, "metric_names", tuple(metric_names))
        BaseTerm.__init__(self, dispatch, *args)

    def __post_init__(self) -> None:
        BaseTerm.__post_init__(self)
        if any(not isinstance(name, str) or not name for name in self.metric_names):
            raise TypeError("Termination metric names must be non-empty strings.")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("Termination metric names must be unique.")


@configclass(kw_only=True)
class TerminationTermCfg(abc.ABC):
    """Configuration that creates one immutable termination term.

    Terms that read simulator state may declare their queries by overriding
    :meth:`required_sim_queries`; declarations are merged into the compiled
    read set like task-declared queries.
    """

    @abc.abstractmethod
    def __call__(self, ctx) -> TerminationTerm:
        """Assemble the runtime term (dispatch plus static arguments)."""


@configclass
class ManagerTerminationsCfg:
    """Typed declaration group for one environment's termination terms."""

    def to_dict(self) -> dict[str, TerminationTermCfg]:
        """Return termination configs keyed by their declaration names."""
        term_cfgs: dict[str, TerminationTermCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if not isinstance(term_cfg, TerminationTermCfg):
                raise TypeError(
                    f"Manager termination {term_field.name!r} must be a TerminationTermCfg, "
                    f"got {type(term_cfg).__name__}."
                )
            term_cfgs[term_field.name] = term_cfg
        return term_cfgs


class TerminationManager:
    """Create and own configured termination terms."""

    def __init__(self, cfg: dict[str, TerminationTermCfg], env: ManagerEnv) -> None:
        self._env = env
        self._terms: dict[str, TerminationTerm] = {}
        self._initialize(cfg)

    def _initialize(self, cfg: dict[str, TerminationTermCfg]) -> None:
        termination_types: dict[type[TerminationTermCfg], str] = {}
        for name, term_cfg in cfg.items():
            existing_name = termination_types.get(type(term_cfg))
            if existing_name is not None:
                raise ValueError(
                    f"Termination config type {type(term_cfg).__name__} is used by both "
                    f"{existing_name!r} and {name!r}; each configured termination type must be unique."
                )
            termination_types[type(term_cfg)] = name
            created = term_cfg(BuildContext(self._env, f"terminations.{name}"))
            if not isinstance(created, TerminationTerm):
                raise TypeError(
                    f"Manager termination {name} __call__() must return TerminationTerm, got {type(created).__name__}."
                )
            canonical_args = canonicalize_term_args(created.args, context=f"Manager term termination.{name}")
            for metric_name in created.metric_names:
                if metric_name in self._env.metrics:
                    raise ValueError(f"Duplicate per-environment metric name: {metric_name!r}")
                self._env.metrics[metric_name] = np.empty((self._env.num_envs, 1), dtype=np.float32)
            self._terms[name] = TerminationTerm(
                created.dispatch,
                *canonical_args,
                metric_names=created.metric_names,
            )

    @property
    def terms(self) -> dict[str, TerminationTerm]:
        return self._terms
