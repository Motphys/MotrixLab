# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Interval-only simulator events with independent per-environment timers."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.terms import BaseTerm, canonicalize_term_args
from motrix_env_core.sim.write import SimWrite, WriteProgram

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.env import ManagerEnv


@dataclass(frozen=True, slots=True, init=False)
class IntervalEvent(BaseTerm):
    """Dispatch descriptor: ``dispatch(ctx, writes, *args)`` fills selected rows."""

    writes: dict[str, SimWrite]

    def __init__(self, dispatch: Callable[..., None], *args: Any, writes: dict[str, SimWrite]) -> None:
        object.__setattr__(self, "writes", dict(writes))
        BaseTerm.__init__(self, dispatch, *args)


@configclass(kw_only=True)
class IntervalEventCfg(abc.ABC):
    """Construct an event with independently sampled positive intervals in seconds."""

    interval_range_s: tuple[float, float]

    @abc.abstractmethod
    def __call__(self, ctx: BuildContext) -> IntervalEvent:
        """Declare the numeric entry point and simulator writes."""


@dataclass
class _IntervalEventRuntime:
    term: IntervalEvent
    program: WriteProgram
    buffers: tuple[np.ndarray, ...]
    remaining_s: np.ndarray
    interval_range_s: tuple[np.float32, np.float32]
    kernel: Any = None
    inputs: tuple[Any, ...] = ()

    def execute(self, env_ids: np.ndarray, *, reset: bool) -> None:
        low, high = self.interval_range_s
        self.kernel(self.inputs, env_ids, self.buffers, self.remaining_s, low, high, reset)
        if not reset:
            self.program.execute(env_ids)


@dataclass
class IntervalEventManager:
    """Schedule events before physics; only episode resets resample their timers."""

    events: dict[str, _IntervalEventRuntime] = field(default_factory=dict)

    @classmethod
    def create(cls, env: ManagerEnv, configs: dict[str, IntervalEventCfg]) -> IntervalEventManager:
        events = {}
        for name, cfg in configs.items():
            if not isinstance(cfg, IntervalEventCfg):
                raise TypeError(f"Interval event {name!r} must be an IntervalEventCfg.")
            low, high = (np.float32(value) for value in cfg.interval_range_s)
            if not np.isfinite(low) or not np.isfinite(high) or not 0 < low <= high:
                raise ValueError(f"Interval event {name!r} requires finite 0 < low <= high intervals.")
            created = cfg(BuildContext(env, f"interval_events.{name}"))
            if not isinstance(created, IntervalEvent):
                raise TypeError(f"Interval event {name!r} must create an IntervalEvent.")
            term = IntervalEvent(
                created.dispatch,
                *canonicalize_term_args(created.args, context=f"Interval event {name!r}"),
                writes=created.writes,
            )
            if not term.writes or any(
                not isinstance(key, str) or not key or not isinstance(write, SimWrite)
                for key, write in term.writes.items()
            ):
                raise ValueError(f"Interval event {name!r} must declare named SimWrite outputs.")
            program = env.sim.compile_writes(term.writes, reset=False, forward_kinematics=False)
            events[name] = _IntervalEventRuntime(
                term,
                program,
                tuple(program.buffer(key) for key in term.writes),
                np.zeros(env.num_envs, dtype=np.float32),
                (np.float32(low), np.float32(high)),
            )
        return cls(events)

    def compile(self, env: ManagerEnv) -> None:
        from motrix_env_core.numba.manager.compiler.compiler import NumbaKernelCompiler

        for name, event in self.events.items():
            compiler = NumbaKernelCompiler(env)
            event.kernel, event.inputs = compiler.build_interval_event(name, event.term)
            low, high = event.interval_range_s
            args = (event.inputs, np.empty(0, dtype=np.int64), event.buffers, event.remaining_s, low, high, False)
            compiler._compile_specialization("interval_event", event.kernel, args)

    def reset(self, env_ids: np.ndarray) -> None:
        if env_ids.size:
            for event in self.events.values():
                event.execute(env_ids, reset=True)

    def apply(self, dt: float) -> None:
        for event in self.events.values():
            event.remaining_s -= np.float32(dt)
            due = np.flatnonzero(event.remaining_s <= 0)
            if due.size:
                event.execute(due, reset=False)
