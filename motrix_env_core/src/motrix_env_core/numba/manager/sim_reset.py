# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager-owned simulator reset writes, terms, and output buffers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from motrix_env_core.config.sim_reset import ManagerResetCfg
from motrix_env_core.numba.kernel_data import canonicalize_kernel_data, is_kernel_data
from motrix_env_core.perf import active_perf_scope
from motrix_env_core.sim.write import SimWrite, WriteProgram

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.env import ManagerEnv
    from motrix_env_core.numba.program import NumbaTaskProgram
    from motrix_env_core.sim.backend import SimBackend


@dataclass(frozen=True, slots=True, init=False)
class ResetTerm:
    """Immutable reset dispatch descriptor and its static Numba-compatible arguments."""

    dispatch: Callable[..., None]
    args: tuple[Any, ...]
    writes: dict[str, SimWrite]

    def __init__(self, dispatch: Callable[..., None], *args: Any, writes: dict[str, SimWrite]) -> None:
        object.__setattr__(self, "dispatch", dispatch)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "writes", dict(writes))
        self.__post_init__()

    def __post_init__(self) -> None:
        if not inspect.isfunction(self.dispatch):
            raise TypeError(f"Reset dispatch must be a Python function, got {type(self.dispatch).__name__}.")
        if not getattr(self.dispatch, "__motrix_manager_dispatch__", False):
            raise TypeError(f"Reset dispatch {self.dispatch.__qualname__!r} must be decorated with @dispatch.")
        if not isinstance(self.args, tuple):
            raise TypeError("Reset term args must be a tuple.")
        if not isinstance(self.writes, dict):
            raise TypeError("Reset term writes must be a dict.")


def _canonicalize_reset_args(args: tuple[Any, ...], *, context: str) -> tuple[Any, ...]:
    """Validate and canonicalize positional Numba-compatible reset arguments."""
    values: list[Any] = []
    for index, value in enumerate(args):
        if is_kernel_data(value):
            values.append(canonicalize_kernel_data(value, context=f"{context} args[{index}]"))
        elif isinstance(value, tuple) and all(isinstance(item, (bool, int, float, np.generic)) for item in value):
            values.append(value)
        elif isinstance(value, (bool, int, float, np.generic)):
            values.append(value)
        else:
            raise TypeError(
                f"{context} args[{index}] must be a scalar, a scalar tuple, or a @kernel_data value; "
                f"got {type(value).__name__}. Raw np.ndarray values are not supported: wrap array data in a "
                f"@kernel_data type (e.g. with SharedArray fields) so it lowers into kernel inputs."
            )
    return tuple(values)


@dataclass
class SimResetRuntime:
    """Compiled reset-mode write program, descriptors, and output buffers."""

    program: WriteProgram
    terms: dict[str, ResetTerm]
    writes: dict[str, dict[str, SimWrite]]
    buffers: tuple[np.ndarray, ...]

    @classmethod
    def create(
        cls,
        env: ManagerEnv,
        cfg: ManagerResetCfg,
        sim: SimBackend,
    ) -> SimResetRuntime:
        """Create terms, compile their declared writes, and bind output buffers."""
        terms = {}
        for name, term_cfg in cfg.to_dict().items():
            created = term_cfg(env)
            if not isinstance(created, ResetTerm):
                raise TypeError(
                    f"Manager simulator reset term {name!r} __call__() must return ResetTerm, "
                    f"got {type(created).__name__}."
                )
            terms[name] = ResetTerm(
                created.dispatch,
                *_canonicalize_reset_args(created.args, context=f"Manager simulator reset term {name!r}"),
                writes=created.writes,
            )
        writes = {name: term.writes for name, term in terms.items()}
        for term_name, term_writes in writes.items():
            if not term_writes:
                raise ValueError(f"Manager simulator reset term {term_name!r} must declare at least one write output.")
            for output_name, write in term_writes.items():
                if not isinstance(output_name, str) or not output_name:
                    raise ValueError("Manager reset output names must be non-empty strings.")
                if not isinstance(write, SimWrite):
                    raise TypeError(
                        f"Manager reset output {output_name!r} must be a SimWrite, got {type(write).__name__}."
                    )
        flat_writes = {
            f"{term_name}.{output_name}": write
            for term_name, term_writes in writes.items()
            for output_name, write in term_writes.items()
        }
        program = sim.compile_writes(flat_writes, reset=True)
        buffers = tuple(
            program.buffer(f"{term_name}.{output_name}")
            for term_name, term_writes in writes.items()
            for output_name in term_writes
        )
        return cls(program=program, terms=terms, writes=writes, buffers=buffers)

    def apply(
        self,
        task_program: NumbaTaskProgram,
        inputs: tuple[Any, ...],
        env_ids: np.ndarray,
    ) -> None:
        """Materialize compact term outputs, scatter them, and execute reset."""
        with active_perf_scope("reset_kernel"):
            task_program.reset_kernel(inputs, env_ids, self.buffers)
        for buffer in self.buffers:
            compact = buffer[: env_ids.size].copy()
            buffer[env_ids] = compact
        with active_perf_scope("backend_apply"):
            self.program.execute(env_ids)


__all__ = ["ResetTerm", "SimResetRuntime"]
