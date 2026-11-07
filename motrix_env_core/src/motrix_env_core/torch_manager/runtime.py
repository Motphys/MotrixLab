# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic Torch executor for resolved Manager programs.

The runtime consumes the metadata produced by the manager resolver (read
plan, layout, the live :class:`ManagerContext` host instance, and per-term
typed call records) and executes the manager term schedule — command
``update`` hooks, rewards, terminations, command ``advance`` hooks for
evaluate; per-group observation terms for observe; command ``reset_env``
hooks plus simulator reset terms for reset — with one batched tensor call
per term. Terms receive the real host objects: Dynamo traces dataclass
attribute access, frozen ``Map`` lookups, and term-state methods directly,
guarding on object identity.

By default each kernel schedule is additionally fused with ``torch.compile``
into a compiled closure over the captured persistent tensors (one fused
graph per kernel instead of one eager dispatch per tensor op; the reset
graph computes over the full batch with a lane mask so every shape stays
static). The eager schedule remains available as a debugging fallback
(``TorchManagerRuntime(env, program, compile=False)``).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.kernel import KernelOutputs
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.resolver import (
    CommandAdvanceHook,
    CommandResetHook,
    CommandUpdateHook,
    ObservationTermCall,
    RewardTermCall,
    SimArg,
    SimResetTermCall,
    TerminationTermCall,
)
from motrix_env_core.torch_manager import merged_omp

if TYPE_CHECKING:
    from motrix_env_core.manager.env import ManagerEnv
    from motrix_env_core.manager.resolver import ResolvedManagerProgram


def _compile_options() -> dict[str, Any]:
    """Inductor options for compiled term schedules.

    Small fused graphs oversubscribe when every generated loop grabs all
    OpenMP threads; capping intra-graph threads near 8 is faster for
    manager-term-sized ops.
    """
    threads = int(os.environ.get("MOTRIX_TORCH_COMPILE_THREADS", "0")) or min(8, torch.get_num_threads())
    # Inductor serializes any loop whose per-thread work would fall below
    # ``cpp.min_chunk_size`` (default 4096 elements — with capped threads that
    # means whole-batch loops under ~16k elements run single-threaded inside
    # ``#pragma omp single``). Manager term loops are exactly that size, so
    # lower the threshold to let them parallelize.
    chunk = int(os.environ.get("MOTRIX_TORCH_MIN_CHUNK", "256"))
    # Experiment: barrier-free OpenMP loops in the generated C++ (no-op
    # unless MOTRIX_TORCH_MERGED_OMP is set).
    merged_omp.maybe_install()
    return {"cpp.threads": threads, "cpp.min_chunk_size": chunk}


@dataclass(frozen=True)
class TorchTaskProgram:
    """Drop-in replacement task program backed by Torch kernels."""

    evaluate_kernel: Callable[[torch.Tensor, tuple[torch.Tensor, ...], Any], None]
    observe_kernel: Callable[[Any], None]
    reset_kernel: Callable[[np.ndarray, tuple[np.ndarray, ...]], None]
    reward_weights: torch.Tensor


def _lane_clone(
    value: Any,
    num_envs: int,
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    clones_by_storage: dict[tuple[int, tuple[int, ...], tuple[int, ...], torch.dtype], torch.Tensor],
) -> Any:
    """Clone one kernel-data record's lane-shaped tensor fields.

    Reset terms write only the receiver command terms, the RNG state, and
    metrics; every other field (shared clip arrays, sim views, scalars) is
    read-only during reset and passes through uncloned.  The storage map
    preserves aliases between a term metric field and the exported
    ``ctx.metrics`` view, so both surfaces observe one masked reset result.
    """
    replacements = {}
    for data_field in fields(value):
        field_value = getattr(value, data_field.name)
        if isinstance(field_value, torch.Tensor) and field_value.dim() >= 1 and field_value.shape[0] == num_envs:
            key = (field_value.data_ptr(), tuple(field_value.shape), tuple(field_value.stride()), field_value.dtype)
            clone = clones_by_storage.get(key)
            if clone is None:
                clone = field_value.clone()
                clones_by_storage[key] = clone
                pairs.append((field_value, clone))
            replacements[data_field.name] = clone
    return replace(value, **replacements) if replacements else value


class TorchManagerRuntime:
    """Execute one resolved manager program with batched Torch kernels.

    ``compile`` fuses each kernel's whole term schedule into one Inductor
    graph over the captured persistent tensors; ``compile=False`` keeps the
    per-term eager schedule for debugging.
    """

    def __init__(self, env: ManagerEnv, program: ResolvedManagerProgram, compile: bool = True) -> None:
        self.env = env
        self.program = program
        self.layout = program.layout
        self.compile_enabled = compile
        self.ctx = program.context

        # Typed call sequences encode the execution contract structurally:
        # evaluate follows the fixed phase order command update hooks ->
        # rewards -> terminations -> command advance hooks (rewards and
        # terminations read pre-advance command state); each phase keeps its
        # declaration order and reward/termination indices match
        # ``layout.rewards``/``layout.terminations``.
        self.command_updates: tuple[CommandUpdateHook, ...] = program.command_updates
        self.rewards: tuple[RewardTermCall, ...] = program.rewards
        self.terminations: tuple[TerminationTermCall, ...] = program.terminations
        self.command_advances: tuple[CommandAdvanceHook, ...] = program.command_advances
        self.command_reset_hooks: tuple[CommandResetHook, ...] = program.command_reset_hooks
        self.observations: dict[str, tuple[ObservationTermCall, ...]] = program.observations
        reward_cfgs = env.cfg.reward_cfgs()
        self.reward_weights = torch.tensor(
            [float(reward_cfgs[term.name].weight) for term in self.layout.rewards], dtype=torch.float32
        )
        # Reset-term compact-write layout mirrors SimResetRuntime.buffers:
        # one entry per declared write, ordered by term then output name.
        # The write-field names come from the host (same insertion order the
        # resolver used when assigning offsets).
        self._reset_terms: tuple[tuple[SimResetTermCall, int, tuple[str, ...]], ...] = tuple(
            (reset.call, reset.output_offset, tuple(env.sim_reset_writes[name]))
            for name, reset in zip(env.sim_reset_terms, program.sim_reset_terms, strict=True)
        )

        self._compiled_kernels: dict[Any, Any] = {}

    # -- kernel API ---------------------------------------------------------

    def task_program(self) -> TorchTaskProgram:
        return TorchTaskProgram(
            evaluate_kernel=self.evaluate,
            observe_kernel=self.observe,
            reset_kernel=self.reset,
            reward_weights=self.reward_weights,
        )

    def evaluate(
        self,
        reward_weights: torch.Tensor,
        buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        outputs: KernelOutputs,
    ) -> None:
        kernel = self._compiled_kernels.get(("evaluate", id(buffers)))
        if kernel is None:
            kernel = self._build_compiled_evaluate(reward_weights, buffers)
            self._compiled_kernels[("evaluate", id(buffers))] = kernel
        kernel(outputs.reward, outputs.terminated)

    def observe(self, outputs: KernelOutputs) -> None:
        kernel = self._compiled_kernels.get("observe")
        if kernel is None:
            kernel = self._build_compiled_observe()
            self._compiled_kernels["observe"] = kernel
        value_obs = outputs.value_obs
        kernel(
            outputs.policy_obs,
            value_obs if value_obs is not None and value_obs.numel() else None,
        )

    def reset(self, env_ids: torch.Tensor, reset_buffers: tuple[torch.Tensor, ...]) -> None:
        if not self.command_reset_hooks and not self._reset_terms:
            return
        ids = env_ids.detach().cpu().contiguous().to(torch.int64)
        kernel = self._compiled_kernels.get(("reset", id(reset_buffers)))
        if kernel is None:
            kernel = self._build_compiled_reset(reset_buffers)
            self._compiled_kernels[("reset", id(reset_buffers))] = kernel
        kernel(ids)

    def _build_compiled_evaluate(
        self,
        reward_weights: torch.Tensor,
        buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Callable[[torch.Tensor, torch.Tensor], None]:
        ctx = self.ctx
        weights = reward_weights
        raw_terms = buffers[0]
        weighted_terms = buffers[1]
        termination_masks = buffers[2]
        dt = ctx.dt
        update_hooks = self.command_updates
        advance_hooks = self.command_advances
        reward_plan = tuple((term, self._invocation_args(term)) for term in self.rewards)
        termination_plan = tuple((term, self._invocation_args(term)) for term in self.terminations)

        def kernel(reward_view: torch.Tensor, terminated_view: torch.Tensor) -> None:
            for hook in update_hooks:
                receiver = ctx.commands[hook.receiver_key]
                hook.function(receiver, ctx)
            total = None
            for term, args in reward_plan:
                value = term.function(ctx, *args).to(torch.float32)
                weighted = value * weights[term.index] * dt
                raw_terms[:, term.index] = value
                weighted_terms[:, term.index] = weighted
                total = weighted if total is None else total + weighted
            terminated = None
            for term, args in termination_plan:
                mask = term.function(ctx, *args).to(torch.bool)
                termination_masks[:, term.index] = mask
                terminated = mask if terminated is None else terminated | mask
            for hook in advance_hooks:
                receiver = ctx.commands[hook.receiver_key]
                hook.function(receiver, ctx)
            reward_view.copy_(total if total is not None else torch.zeros_like(reward_view))
            terminated_view.copy_(terminated if terminated is not None else torch.zeros_like(terminated_view))

        if self.compile_enabled:
            return torch.compile(kernel, dynamic=False, options=_compile_options())
        return kernel

    def _build_compiled_observe(self) -> Callable[[torch.Tensor, torch.Tensor | None], None]:
        ctx = self.ctx
        # Each term writes a private contiguous buffer; a single fused
        # ``cat + copy`` assembles the group output. Writing through strided
        # slice views of the group buffer forces per-term unfusable store
        # regions in the generated code (aliased memory), which multiplies
        # the OpenMP fork/join cost of the fused kernel.
        num_envs = ctx.env_id.shape[0]
        term_bufs = {
            group_name: tuple(
                torch.empty(num_envs, term.output_slice.stop - term.output_slice.start, dtype=torch.float32)
                for term in group.terms
            )
            for group_name, group in self.layout.observations.items()
        }
        per_term_plan = tuple(
            (
                group_name,
                tuple(
                    (call, self._invocation_args(call), buf)
                    for call, buf in zip(self.observations[group_name], bufs, strict=True)
                ),
            )
            for group_name, bufs in term_bufs.items()
        )

        # One compiled graph per observation group: smaller generated kernels
        # keep their code hot through the interleaved physics/evaluate passes,
        # which measurably beats a single fused graph on many-core hosts.

        def make_group_kernel(
            entries: tuple[tuple[ObservationTermCall, tuple[Any, ...], torch.Tensor], ...],
        ) -> Callable[[torch.Tensor], None]:
            def group_kernel(target: torch.Tensor) -> None:
                for call, args, buf in entries:
                    call.function(ctx, buf, *args)
                target.copy_(torch.cat([buf for _call, _args, buf in entries], dim=1))

            if self.compile_enabled:
                return torch.compile(group_kernel, dynamic=False, options=_compile_options())
            return group_kernel

        group_kernels = {group_name: make_group_kernel(entries) for group_name, entries in per_term_plan}
        policy_kernel = group_kernels.get("policy")
        value_kernel = group_kernels.get("value")

        def kernel(policy_obs: torch.Tensor, value_obs: torch.Tensor | None) -> None:
            if policy_kernel is not None:
                policy_kernel(policy_obs)
            if value_obs is not None and value_kernel is not None:
                value_kernel(value_obs)

        return kernel

    def _build_compiled_reset(self, reset_buffers: tuple[np.ndarray, ...]) -> Callable[[torch.Tensor], None]:
        """Compile the reset schedule as a static full-batch masked graph.

        The reset math runs for ALL lanes with the selected lanes chosen by a
        boolean mask. Each receiver command term, the RNG state, and every
        metric view is rebuilt once as a lane clone (``dataclasses.replace``
        with lane-shaped tensor fields cloned); per-env state written by
        terms is merged back with ``where(mask, clone, original)`` inside the
        graph, so non-selected lanes (including their RNG state) keep their
        values while selected lanes advance exactly as the gathered schedule
        computed them (each lane's SplitMix64 sequence is independent).
        Compaction into the reset buffers happens outside the graph via one
        ``index_select`` per write field. Static shapes let Inductor
        parallelize every loop.
        """
        ctx = self.ctx
        command_reset_hooks = self.command_reset_hooks
        num_envs = ctx.env_id.shape[0]
        mask = torch.zeros(num_envs, dtype=torch.bool)
        # Lane write surfaces, cloned once at compile time and reused every
        # reset: receiver command terms, the RNG state, and metrics. Sim
        # writes go through the sim_writes map; everything else (sim views,
        # action state, clip arrays, shared params) is read-only during reset
        # and passes through uncloned.
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        clones_by_storage: dict[tuple[int, tuple[int, ...], tuple[int, ...], torch.dtype], torch.Tensor] = {}
        lane_commands = {
            hook.receiver_key: _lane_clone(ctx.commands[hook.receiver_key], num_envs, pairs, clones_by_storage)
            for hook in command_reset_hooks
        }
        lane_rand_state = ctx.rand.state.clone()
        pairs.append((ctx.rand.state, lane_rand_state))
        lane_metrics = {}
        for name, view in ctx.metrics.items():
            key = (view.data_ptr(), tuple(view.shape), tuple(view.stride()), view.dtype)
            clone = clones_by_storage.get(key)
            if clone is None:
                clone = view.clone()
                clones_by_storage[key] = clone
                pairs.append((view, clone))
            lane_metrics[name] = clone
        lane_ctx = ManagerContext(
            env_id=ctx.env_id,
            actions=ctx.actions,
            commands=Map(lane_commands),
            metrics=Map(lane_metrics),
            rand=replace(ctx.rand, state=lane_rand_state),
            sim=ctx.sim,
            dt=ctx.dt,
            sim_reset_requested=ctx.sim_reset_requested,
        )
        write_plan = tuple(
            (
                call,
                tuple(reset_buffers[offset + index] for index in range(len(fields_))),
                tuple(torch.empty_like(reset_buffers[offset + index]) for index in range(len(fields_))),
                fields_,
                self._invocation_args(call),
            )
            for call, offset, fields_ in self._reset_terms
        )

        def kernel() -> None:
            # Refresh the lane clones from the live originals first: the
            # clones persist across reset invocations, so each run starts
            # from the current batch state exactly like a fresh clone would.
            for original, clone in pairs:
                clone.copy_(original)
            for hook in command_reset_hooks:
                receiver = lane_ctx.commands[hook.receiver_key]
                hook.function(receiver, lane_ctx)
            for call, _buffers, staging, fields_, args in write_plan:
                sim_writes = {name: buffer for name, buffer in zip(fields_, staging, strict=True)}
                call.function(lane_ctx, _WritesMap(sim_writes), *args)
            for original, clone in pairs:
                lane_mask = mask.view((num_envs,) + (1,) * (original.dim() - 1))
                original.copy_(torch.where(lane_mask, clone, original))

        if self.compile_enabled:
            kernel = torch.compile(kernel, dynamic=False, options=_compile_options())
        compact_plan = tuple(
            (buffers, staging)
            for _invocation, buffers, staging, _fields, _args in write_plan
        )

        def wrapper(ids: torch.Tensor) -> None:
            mask.fill_(False)
            mask[ids] = True
            kernel()
            count = ids.shape[0]
            for buffers, staging in compact_plan:
                for buffer, stage in zip(buffers, staging, strict=True):
                    buffer[:count] = stage[ids]

        return wrapper

    # -- helpers ------------------------------------------------------------

    def _invocation_args(
        self,
        call: RewardTermCall | TerminationTermCall | ObservationTermCall | SimResetTermCall,
    ) -> tuple[Any, ...]:
        args: list[Any] = []
        for arg in call.args:
            if isinstance(arg, SimArg):
                args.append(self.ctx.sim[arg.key])
            else:
                args.append(arg.value)
        return tuple(args)


class _WritesMap:
    def __init__(self, writes: dict[str, torch.Tensor]) -> None:
        self._writes = writes

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._writes[name]


__all__ = ["TorchManagerRuntime", "TorchTaskProgram"]
