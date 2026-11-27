# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager-owned simulator reset writes, terms, and output buffers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch

from motrix_env_core.config.sim_reset import ManagerResetCfg
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.terms import BaseTerm, canonicalize_term_args
from motrix_env_core.perf import active_perf_scope
from motrix_env_core.sim.write import SimWrite, WriteProgram

if TYPE_CHECKING:
    from motrix_env_core.sim.backend import SimBackend
    from motrix_env_core.torch_manager.runtime import TorchTaskProgram


@dataclass(frozen=True, slots=True, init=False)
class ResetTerm(BaseTerm):
    """Describe a simulator reset dispatch and its write outputs.

    With ``stateless=True``, dispatch may receive all world IDs and full-size
    buffers even if few worlds reset. It may read the context and draw from
    its RNG, and must write only to ``sim_writes``. Only selected rows are
    committed. Other terms always receive compact IDs and outputs.
    """

    writes: dict[str, SimWrite]
    stateless: bool

    def __init__(
        self, dispatch: Callable[..., None], *args: Any, writes: dict[str, SimWrite], stateless: bool = False
    ) -> None:
        object.__setattr__(self, "writes", dict(writes))
        object.__setattr__(self, "stateless", stateless)
        BaseTerm.__init__(self, dispatch, *args)

    def __post_init__(self) -> None:
        BaseTerm.__post_init__(self)
        if not isinstance(self.writes, dict):
            raise TypeError("Reset term writes must be a dict.")


@dataclass
class SimResetRuntime:
    """Compiled reset-mode write program, descriptors, and output buffers."""

    program: WriteProgram
    terms: dict[str, ResetTerm]
    writes: dict[str, dict[str, SimWrite]]
    buffers: tuple[torch.Tensor, ...]

    @cached_property
    def supports_stateless_masked_reset(self) -> bool:
        return all(term.stateless for term in self.terms.values())

    @classmethod
    def create(
        cls,
        ctx: BuildContext,
        cfg: ManagerResetCfg,
        sim: SimBackend,
    ) -> SimResetRuntime:
        """Create terms, compile their declared writes, and bind output buffers."""
        terms = {}
        for name, term_cfg in cfg.to_dict().items():
            created = term_cfg(ctx)
            if not isinstance(created, ResetTerm):
                raise TypeError(
                    f"Manager simulator reset term {name!r} __call__() must return ResetTerm, "
                    f"got {type(created).__name__}."
                )
            terms[name] = ResetTerm(
                created.dispatch,
                *canonicalize_term_args(created.args, context=f"Manager simulator reset term {name!r}"),
                writes=created.writes,
                stateless=created.stateless,
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
        task_program: TorchTaskProgram,
        env_ids: torch.Tensor,
        episode_reset: torch.Tensor,
    ) -> None:
        """Apply compact term outputs to the selected simulator rows."""
        with active_perf_scope("reset_kernel"):
            task_program.reset_kernel(env_ids, episode_reset, self.buffers)
        with active_perf_scope("backend_apply"):
            self.program.execute_compact(env_ids, self.buffers)

    def apply_masked(
        self,
        task_program: TorchTaskProgram,
        world_mask: torch.Tensor,
        episode_reset: torch.Tensor,
        env_ids: torch.Tensor | None,
    ) -> None:
        with active_perf_scope("reset_kernel"):
            task_program.reset_masked_kernel(world_mask, episode_reset, env_ids, self.buffers)
        with active_perf_scope("backend_apply"):
            if env_ids is None:
                self.program.execute_masked(world_mask)
            else:
                self.program.execute_compact(env_ids, self.buffers)


__all__ = ["ResetTerm", "SimResetRuntime"]
