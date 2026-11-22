# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic Torch executor for resolved Manager programs.

The runtime consumes the metadata produced by the manager resolver (read
plan, layout, the live :class:`ManagerContext` host instance, and per-term
typed call records) and executes the manager term schedule — command
``update`` hooks, rewards, terminations, command ``advance`` hooks for
evaluate; per-group observation terms for observe; command ``reset``
hooks plus simulator reset terms for reset — with one batched tensor call
per term. Terms receive the real host objects: Dynamo traces dataclass
attribute access, fixed-key ``dict`` lookups, and term-state methods directly,
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from motrix_env_core.array.env import EnvObs
from motrix_env_core.manager.kernel import EvaluationOutputs
from motrix_env_core.manager.resolver import (
    CommandAdvanceHook,
    CommandResetHook,
    CommandUpdateHook,
    ObservationTermCall,
    RewardTermCall,
    SimResetTermCall,
    TerminationTermCall,
)

if TYPE_CHECKING:
    from motrix_env_core.manager.env import ManagerEnv
    from motrix_env_core.manager.resolver import ResolvedManagerProgram


def _compile_options() -> dict[str, Any]:
    """Inductor options for compiled term schedules.

    The process intra-op pool is capped by batch size by default (see
    ``_apply_default_thread_cap``): manager-sized batch loops keep a small
    pool fully busy, and wider pools add fork/join overhead faster than they
    add throughput. Compiled graphs default to the same pool;
    ``MOTRIX_TORCH_COMPILE_THREADS`` overrides just the graphs.
    """
    threads = int(os.environ.get("MOTRIX_TORCH_COMPILE_THREADS", "0")) or torch.get_num_threads()
    # Inductor serializes any loop whose per-thread work would fall below
    # ``cpp.min_chunk_size`` (default 512 — with a capped pool that means
    # whole-batch loops under a few thousand elements run single-threaded
    # inside ``#pragma omp single``). Manager term loops are exactly that
    # size, so lower the threshold to let them parallelize; 64 measured
    # fastest on 192-core EPYC (2048–8192 envs) and 16-core desktop hosts.
    chunk = int(os.environ.get("MOTRIX_TORCH_MIN_CHUNK", "64"))
    return {"cpp.threads": threads, "cpp.min_chunk_size": chunk}


def _apply_default_thread_cap(num_envs: int) -> None:
    """Cap the process intra-op thread pool unless the caller chose a value.

    ``OMP_NUM_THREADS`` is respected outright. A pool the caller already
    narrowed below the machine's CPU count is also respected — e.g. async
    collectors pin their worker CPUs and call ``torch.set_num_threads``
    before ``build_env()``. Otherwise the pool is capped at
    ``min(16, max(8, num_envs // 256))`` threads. Larger batches amortize a
    wider pool over longer loops, so the cap grows with the batch — measured
    optima on the 192-core EPYC host: 8 threads at 2048 envs (11.3ms vs 12.9
    at 32 and 33.8 uncapped), 16 at 4096 and 8192 (16.9/27.3ms); the curve is
    shallow, so the heuristic stays within ~4% of the best fixed value across
    the 2048–8192 range.
    """
    if os.environ.get("OMP_NUM_THREADS"):
        return
    if torch.get_num_threads() < (os.cpu_count() or 1):
        return
    capped = min(min(16, max(8, num_envs // 256)), torch.get_num_threads())
    if capped < torch.get_num_threads():
        torch.set_num_threads(capped)


@dataclass(frozen=True)
class TorchTaskProgram:
    """Drop-in replacement task program backed by Torch kernels."""

    evaluate_kernel: Callable[[torch.Tensor, tuple[torch.Tensor, ...], EvaluationOutputs], None]
    observe_kernel: Callable[[EnvObs], None]
    reset_kernel: Callable[[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]], None]
    reward_weights: torch.Tensor


_K = TypeVar("_K")


class CompiledKernels:
    """Memo of compiled kernels keyed by kind and captured-tensor identity.

    A compiled kernel is bound to the specific tensor objects it captured at
    build time, so the cache key includes the identity (``id``) of the
    captured buffer tuple alongside the kernel kind. Entries are created
    lazily on first call and stay valid as long as the caller keeps passing
    the same buffer objects; a rebuilt buffer tuple misses the cache and
    compiles a fresh graph bound to the new tensors.
    """

    def __init__(self) -> None:
        self._kernels: dict[Any, _K] = {}

    def get_or_build(self, key: Any, build: Callable[[], _K]) -> _K:
        """Return the cached kernel for ``key``, compiling it on first use."""
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = build()
            self._kernels[key] = kernel
        return kernel


class TorchManagerRuntime:
    """Execute one resolved manager program with batched Torch kernels.

    ``compile`` traces numerical work and default Torch random draws over
    persistent tensors. ``compile=False`` keeps the per-term eager schedule
    for debugging.
    """

    def __init__(self, env: ManagerEnv, program: ResolvedManagerProgram, compile: bool = True) -> None:
        self.env = env
        self.program = program
        self.layout = program.layout
        self.compile_enabled = compile
        self.ctx = program.context
        _apply_default_thread_cap(env.num_envs)

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

        self._compiled_kernels = CompiledKernels()

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
        evaluation_outputs: EvaluationOutputs,
    ) -> None:
        kernel = self._compiled_kernels.get_or_build(
            ("evaluate", id(buffers), id(reward_weights)),
            lambda: self._build_compiled_evaluate(reward_weights, buffers),
        )
        kernel(evaluation_outputs.reward, evaluation_outputs.terminated)

    def observe(self, outputs: EnvObs) -> None:
        kernel = self._compiled_kernels.get_or_build("observe", self._build_compiled_observe)
        kernel(outputs)

    def reset(
        self, env_ids: torch.Tensor, episode_reset: torch.Tensor, reset_buffers: tuple[torch.Tensor, ...]
    ) -> None:
        if not self.command_reset_hooks and not self._reset_terms:
            return
        # Normalize ids onto the kernel tensors' device (via the context
        # anchor) so the cached static graph sees one form regardless of the
        # caller's device, dtype, layout, or autograd history.
        ids = env_ids.detach().to(device=self.ctx.env_id.device, dtype=torch.int64).contiguous()
        kernel = self._compiled_kernels.get_or_build(
            ("reset", id(reset_buffers)), lambda: self._build_compiled_reset(reset_buffers)
        )
        kernel(ids, episode_reset)

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
        reward_plan = self.rewards
        termination_plan = self.terminations

        def kernel(reward_view: torch.Tensor, terminated_view: torch.Tensor) -> None:
            for hook in update_hooks:
                receiver = ctx.commands[hook.receiver_key]
                hook.function(receiver, ctx)
            total = None
            for term in reward_plan:
                value = term.invoke(ctx).to(torch.float32)
                weighted = value * weights[term.index] * dt
                raw_terms[:, term.index] = value
                weighted_terms[:, term.index] = weighted
                total = weighted if total is None else total + weighted
            terminated = None
            for term in termination_plan:
                mask = term.invoke(ctx).to(torch.bool)
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

    def _build_compiled_observe(self) -> Callable[[EnvObs], None]:
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
                tuple((call, buf) for call, buf in zip(self.observations[group_name], bufs, strict=True)),
            )
            for group_name, bufs in term_bufs.items()
        )

        # One compiled graph per observation group: smaller generated kernels
        # keep their code hot through the interleaved physics/evaluate passes,
        # which measurably beats a single fused graph on many-core hosts.

        def make_group_kernel(
            entries: tuple[tuple[ObservationTermCall, torch.Tensor], ...],
        ) -> Callable[[torch.Tensor], None]:
            def group_kernel(target: torch.Tensor) -> None:
                for call, buf in entries:
                    call.invoke(ctx, buf)
                target.copy_(torch.cat([buf for _call, buf in entries], dim=1))

            if self.compile_enabled:
                return torch.compile(group_kernel, dynamic=False, options=_compile_options())
            return group_kernel

        group_kernels = {group_name: make_group_kernel(entries) for group_name, entries in per_term_plan}

        def kernel(outputs: EnvObs) -> None:
            for name, group_kernel in group_kernels.items():
                group_kernel(outputs[name])

        return kernel

    def _build_compiled_reset(
        self, reset_buffers: tuple[torch.Tensor, ...]
    ) -> Callable[[torch.Tensor, torch.Tensor], None]:
        """Compile the reset schedule as a dynamic gather-compact graph.

        The kernel receives the reset lanes' ids (``env_ids``, dynamic
        length K). Command ``reset`` hooks run against the live receiver
        instances and scatter their per-lane writes back through those ids;
        sim reset terms fill compact ``(K, ...)`` staging buffers, and the
        wrapper copies each staging block to the head of the full-batch reset
        buffers (``SimResetRuntime.apply`` scatters them to the backend rows).
        Only the selected lanes are computed — no full-batch clone/merge —
        with a symbolic compact axis where supported. Terms use default Torch
        random draws through ``ctx.rand`` inside the numerical graph.
        """
        ctx = self.ctx
        command_reset_hooks = self.command_reset_hooks
        write_plan = tuple(
            (
                call,
                tuple(reset_buffers[offset + index] for index in range(len(fields_))),
                fields_,
            )
            for call, offset, fields_ in self._reset_terms
        )
        buffer_plan = tuple(
            reset_buffers[offset + index]
            for _call, offset, fields_ in self._reset_terms
            for index in range(len(fields_))
        )

        def kernel(
            env_ids: torch.Tensor,
            episode_reset: torch.Tensor,
            writes_by_term: tuple[dict[str, torch.Tensor], ...],
        ) -> None:
            for hook in command_reset_hooks:
                receiver = ctx.commands[hook.receiver_key]
                hook.function(receiver, ctx, env_ids, episode_reset)
            for (call, _buffers, _fields), sim_writes in zip(write_plan, writes_by_term, strict=True):
                call.invoke(ctx, env_ids, sim_writes)

        if self.compile_enabled:
            kernel = torch.compile(kernel, dynamic=True, options=_compile_options())

        def wrapper(ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
            # Request a symbolic compact axis; Torch's zero/one handling
            # can still specialize graph segments.
            torch._dynamo.mark_dynamic(ids, 0)
            torch._dynamo.mark_dynamic(episode_reset, 0)
            count = ids.shape[0]
            writes_by_term = tuple(
                {
                    name: torch.empty((count, *buffer.shape[1:]), dtype=buffer.dtype)
                    for name, buffer in zip(fields_, buffers, strict=True)
                }
                for _call, buffers, fields_ in write_plan
            )
            # Allocate eagerly so the compact axis can be marked dynamic before
            # tracing, including when a term resumes after a graph break.
            for sim_writes in writes_by_term:
                for output in sim_writes.values():
                    torch._dynamo.mark_dynamic(output, 0)
                    for dim in range(1, output.ndim):
                        torch._dynamo.mark_static(output, dim)
            kernel(ids, episode_reset, writes_by_term)
            staging = tuple(
                sim_writes[name]
                for (_call, _buffers, fields_), sim_writes in zip(write_plan, writes_by_term, strict=True)
                for name in fields_
            )
            for buffer, stage in zip(buffer_plan, staging, strict=True):
                buffer[:count] = stage

        return wrapper


__all__ = ["TorchManagerRuntime", "TorchTaskProgram"]
