# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numbers
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from numba.core.errors import ForceLiteralArg
from numba.core.types import StringLiteral

from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.numba.kernel import clone_kernel_value
from motrix_env_core.numba.kernel_data import KernelDataScope, rebuild_lowered
from motrix_env_core.numba.manager.env import CompiledManagerProgram, KernelInputSource, ManagerEnv


@dataclass(frozen=True)
class PreparedInvocation:
    """Runtime metadata required to warm up and generate one compiled manager term."""

    context: str
    kind: str
    dispatcher: Any
    receiver_key: str | None
    args_expressions: tuple[str, ...] = ()
    args_values: tuple[Any, ...] = ()
    args_prepared_indices: tuple[int | None, ...] = ()
    output_size: int = 0


@dataclass(frozen=True)
class ResolvedManagerContext:
    prepared_index: int
    expression: str


@dataclass(frozen=True)
class ResolvedSimReset:
    invocation: PreparedInvocation
    sim_writes_type: str
    output_offset: int
    output_count: int


@dataclass(frozen=True)
class _CompiledManagerProgram(CompiledManagerProgram):
    invocations: tuple[PreparedInvocation, ...]
    prepared_terms: tuple[KernelInputSource, ...]
    input_offsets: tuple[int, ...]
    context: ResolvedManagerContext

    def _compile_dispatcher(self, dispatcher, args: tuple) -> None:
        """Compile one dispatcher for the warmup argument types.

        ``numba.literally`` in a dispatch body requests literal string
        arguments via a ``ForceLiteralArg`` pseudo-exception; fold the
        compile-time string values into the signature and retry.
        """
        try:
            dispatcher.compile(tuple(numba.typeof(arg) for arg in args))
        except ForceLiteralArg:
            literal_types = tuple(
                StringLiteral(str(arg)) if isinstance(arg, str) else numba.typeof(arg) for arg in args
            )
            dispatcher.compile(literal_types)

    def warmup_terms(self, env: ManagerEnv, state: ArrayEnvState, buffers: tuple[Any, ...]) -> None:
        del state, buffers
        inputs = clone_kernel_value(self.read_plan.read(env))
        context = self._prepared_context(inputs, 0)
        for invocation in self.invocations:
            if invocation.kind == "reset":
                continue
            receiver = self._context_receiver(context, invocation)
            if invocation.kind == "observation":
                args = [context, np.empty((invocation.output_size,), dtype=np.float32)]
                args.extend(
                    self._prepared_value(prepared_index, inputs, 0)
                    if prepared_index is not None
                    else invocation.args_values[index]
                    for index, prepared_index in enumerate(invocation.args_prepared_indices)
                )
                args = tuple(args)
            elif invocation.kind == "reward":
                args = [context]
                args.extend(
                    self._prepared_value(prepared_index, inputs, 0)
                    if prepared_index is not None
                    else invocation.args_values[index]
                    for index, prepared_index in enumerate(invocation.args_prepared_indices)
                )
                args = tuple(args)
            elif invocation.kind == "termination":
                args = [context]
                args.extend(
                    self._prepared_value(prepared_index, inputs, 0)
                    if prepared_index is not None
                    else invocation.args_values[index]
                    for index, prepared_index in enumerate(invocation.args_prepared_indices)
                )
                args = tuple(args)
            elif invocation.kind.startswith("command"):
                args = (receiver, context)
            else:
                args = (context,)
            self._compile_dispatcher(invocation.dispatcher, args)
            result = invocation.dispatcher(*args)
            if invocation.kind.startswith("command") or invocation.kind == "observation":
                if result is not None:
                    raise TypeError(f"Numba {invocation.kind} term {invocation.context} must return None.")
            if invocation.kind == "reward" and (
                isinstance(result, (bool, np.bool_)) or not isinstance(result, numbers.Number)
            ):
                raise TypeError(f"Numba reward term {invocation.context} must return a numeric scalar.")
            if invocation.kind == "termination" and not isinstance(result, (bool, np.bool_)):
                raise TypeError(f"Numba termination term {invocation.context} must return bool.")
            if not invocation.dispatcher.nopython_signatures:
                raise RuntimeError(f"Numba manager term {invocation.context} did not produce a nopython signature.")

    def _prepared_value(self, prepared_index: int, inputs: tuple[Any, ...], env_id: int) -> Any:
        prepared = self.prepared_terms[prepared_index]
        offset = self.input_offsets[prepared_index]
        values = tuple(
            inputs[offset + index][env_id] if field.scope is KernelDataScope.PER_ENV else inputs[offset + index]
            for index, field in enumerate(prepared.fields)
        )
        return rebuild_lowered(prepared.layout, values)

    def _prepared_context(self, inputs: tuple[Any, ...], env_id: int) -> Any:
        return self._prepared_value(self.context.prepared_index, inputs, env_id)

    @staticmethod
    def _context_receiver(context: Any, invocation: PreparedInvocation) -> Any:
        if invocation.receiver_key is None:
            return None
        keys = context.commands.__class__.__motrix_kernel_data_map_keys__
        return context.commands[keys.index(invocation.receiver_key)]


__all__ = [
    "PreparedInvocation",
    "ResolvedManagerContext",
    "ResolvedSimReset",
    "_CompiledManagerProgram",
]
