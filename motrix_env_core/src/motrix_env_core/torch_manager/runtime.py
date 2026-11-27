# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic Torch executor for resolved Manager programs.

The runtime consumes the metadata produced by the manager resolver (read
plan, layout, the live :class:`ManagerContext` host instance, and per-term
typed call records) and executes the manager term schedule — command
``update`` hooks, terminations, rewards, command ``advance`` hooks for
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
from motrix_env_core.manager.kernel_data.data import iter_leaves
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
    """Cap the process intra-op thread pool unless OMP_NUM_THREADS is set.

    ``OMP_NUM_THREADS`` is respected outright. Otherwise an existing pool
    below the cap is left unchanged, while a larger pool is capped at
    ``min(16, max(8, num_envs // 256))`` threads. Larger batches amortize a
    wider pool over longer loops, so the cap grows with the batch — measured
    optima on the 192-core EPYC host: 8 threads at 2048 envs (11.3ms vs 12.9
    at 32 and 33.8 uncapped), 16 at 4096 and 8192 (16.9/27.3ms); the curve is
    shallow, so the heuristic stays within ~4% of the best fixed value across
    the 2048–8192 range.
    """
    if os.environ.get("OMP_NUM_THREADS"):
        return
    # Torch may default to physical cores while os.cpu_count() reports
    # logical CPUs. A smaller pool is not evidence of a caller override.
    current = torch.get_num_threads()
    capped = min(16, max(8, num_envs // 256), current)
    if capped < current:
        torch.set_num_threads(capped)


@dataclass(frozen=True)
class TorchTaskProgram:
    """Drop-in replacement task program backed by Torch kernels."""

    evaluate_kernel: Callable[[torch.Tensor, tuple[torch.Tensor, ...], EvaluationOutputs], None]
    observe_kernel: Callable[[EnvObs], None]
    reset_kernel: Callable[[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]], None]
    reward_weights: torch.Tensor
    reset_masked_kernel: (
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...]], None] | None
    ) = None


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
        if self.ctx.env_id.is_cuda:
            self._static_options = {"triton.cudagraphs": True}
            if compile:
                # Terms mutate the persistent command, action and metric buffers in
                # place. Dynamo guards their addresses when recording a CUDA graph.
                for _, _, tensor in iter_leaves(self.ctx):
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        torch._dynamo.mark_static_address(tensor)
        else:
            _apply_default_thread_cap(env.num_envs)
            self._static_options = _compile_options()

        # Typed call sequences encode the execution contract structurally:
        # evaluate follows the fixed phase order command update hooks ->
        # terminations -> rewards -> command advance hooks. Terminations write
        # the union back before rewards run, so reward terms read the current
        # transition's ctx.terminated; rewards and terminations both read
        # pre-advance command state; each phase keeps its declaration order
        # and reward/termination indices match
        # ``layout.rewards``/``layout.terminations``.
        self.command_updates: tuple[CommandUpdateHook, ...] = program.command_updates
        self.rewards: tuple[RewardTermCall, ...] = program.rewards
        self.terminations: tuple[TerminationTermCall, ...] = program.terminations
        self.command_advances: tuple[CommandAdvanceHook, ...] = program.command_advances
        self.command_reset_hooks: tuple[CommandResetHook, ...] = program.command_reset_hooks
        self.observations: dict[str, tuple[ObservationTermCall, ...]] = program.observations
        reward_cfgs = env.cfg.reward_cfgs()
        self.reward_weights = torch.tensor(
            [float(reward_cfgs[term.name].weight) for term in self.layout.rewards],
            dtype=torch.float32,
            device=self.ctx.env_id.device,
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
            reset_masked_kernel=self.reset_masked,
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
        self,
        env_ids: torch.Tensor,
        episode_reset: torch.Tensor,
        reset_buffers: tuple[torch.Tensor, ...],
        *,
        reset_commands: bool = True,
    ) -> None:
        if not self._reset_terms and not (reset_commands and self.command_reset_hooks):
            return
        # Use a consistent ID tensor for the cached compact reset graph.
        ids = env_ids.detach().to(device=self.ctx.env_id.device, dtype=torch.int64).contiguous()
        kernel = self._compiled_kernels.get_or_build(
            ("reset", id(reset_buffers), reset_commands),
            lambda: self._build_compiled_reset(reset_buffers, reset_commands=reset_commands),
        )
        kernel(ids, episode_reset)

    def reset_masked(
        self,
        world_mask: torch.Tensor,
        episode_reset: torch.Tensor,
        env_ids: torch.Tensor | None,
        reset_buffers: tuple[torch.Tensor, ...],
    ) -> None:
        # Compact terms keep their row contract when a program contains any
        # stateful term. Mask-aware commands still run before those terms.
        buffers = reset_buffers if env_ids is None else ()
        if self.command_reset_hooks or buffers:
            kernel = self._compiled_kernels.get_or_build(
                ("reset_masked", id(buffers)), lambda: self._build_masked_reset(buffers)
            )
            kernel(world_mask, episode_reset)
        if env_ids is not None and self._reset_terms:
            self.reset(env_ids, episode_reset[env_ids], reset_buffers, reset_commands=False)

    def _build_masked_reset(self, reset_buffers: tuple[torch.Tensor, ...]) -> Callable:
        ctx = self.ctx
        receivers = tuple(ctx.commands[hook.receiver_key] for hook in self.command_reset_hooks)
        writes = (
            tuple(
                (call, {name: reset_buffers[offset + index] for index, name in enumerate(names)})
                for call, offset, names in self._reset_terms
            )
            if reset_buffers
            else ()
        )
        if self.compile_enabled and ctx.env_id.is_cuda:
            for buffer in reset_buffers:
                torch._dynamo.mark_static_address(buffer)

        def kernel(world_mask: torch.Tensor, episode_reset: torch.Tensor) -> None:
            for receiver in receivers:
                receiver.reset_masked(ctx, world_mask, episode_reset)
            for call, sim_writes in writes:
                call.invoke(ctx, ctx.env_id, sim_writes)

        return torch.compile(kernel, dynamic=False, options=self._static_options) if self.compile_enabled else kernel

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

        if self.compile_enabled and ctx.env_id.is_cuda:
            for buffer in buffers:
                torch._dynamo.mark_static_address(buffer)

        def kernel(reward_view: torch.Tensor, terminated_view: torch.Tensor) -> None:
            for hook in update_hooks:
                receiver = ctx.commands[hook.receiver_key]
                hook.function(receiver, ctx)
            terminated = None
            for term in termination_plan:
                mask = term.invoke(ctx).to(torch.bool)
                termination_masks[:, term.index] = mask
                terminated = mask if terminated is None else terminated | mask
            # Write the union back before rewards run: reward terms read the
            # current transition's ctx.terminated through the shared buffer.
            terminated_view.copy_(terminated if terminated is not None else torch.zeros_like(terminated_view))
            total = None
            for term in reward_plan:
                value = term.invoke(ctx).to(torch.float32)
                weighted = value * weights[term.index] * dt
                raw_terms[:, term.index] = value
                weighted_terms[:, term.index] = weighted
                total = weighted if total is None else total + weighted
            reward_view.copy_(total if total is not None else torch.zeros_like(reward_view))
            for hook in advance_hooks:
                receiver = ctx.commands[hook.receiver_key]
                hook.function(receiver, ctx)

        if self.compile_enabled:
            return torch.compile(kernel, dynamic=False, options=self._static_options)
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
                torch.empty(
                    num_envs,
                    term.output_slice.stop - term.output_slice.start,
                    dtype=torch.float32,
                    device=ctx.env_id.device,
                )
                for term in group.terms
            )
            for group_name, group in self.layout.observations.items()
        }
        if self.compile_enabled and ctx.env_id.is_cuda:
            for group_buffers in term_bufs.values():
                for buffer in group_buffers:
                    torch._dynamo.mark_static_address(buffer)

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
                return torch.compile(group_kernel, dynamic=False, options=self._static_options)
            return group_kernel

        group_kernels = {group_name: make_group_kernel(entries) for group_name, entries in per_term_plan}

        def kernel(outputs: EnvObs) -> None:
            for name, group_kernel in group_kernels.items():
                target = outputs[name]
                if self.compile_enabled and target.is_cuda:
                    torch._dynamo.mark_static_address(target)
                group_kernel(target)

        return kernel

    def _build_compiled_reset(
        self, reset_buffers: tuple[torch.Tensor, ...], *, reset_commands: bool = True
    ) -> Callable[[torch.Tensor, torch.Tensor], None]:
        """Compile the reset schedule as a dynamic gather-compact graph.

        The kernel receives the reset lanes' ids (``env_ids``, dynamic
        length K). Command ``reset`` hooks run against the live receiver
        instances and scatter their per-lane writes back through those ids;
        sim reset terms fill compact views over the first K rows of the persistent
        backend write buffers. SimResetRuntime clones that prefix before scattering
        to the selected original rows, preserving unsorted selection semantics.
        Terms must fill their declared outputs completely; failed dispatches do not
        execute the backend and may leave write-buffer scratch partially written.
        Only selected lanes are computed, with a symbolic compact axis where supported.
        Terms use default Torch random draws through ``ctx.rand`` inside the numerical graph.
        """
        ctx = self.ctx
        command_reset_hooks = self.command_reset_hooks if reset_commands else ()
        write_plan = tuple(
            (
                call,
                tuple(reset_buffers[offset + index] for index in range(len(fields_))),
                fields_,
            )
            for call, offset, fields_ in self._reset_terms
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
            kernel = torch.compile(kernel, dynamic=True, options={} if ctx.env_id.is_cuda else _compile_options())

        last_count: int | None = None
        writes_by_term: tuple[dict[str, torch.Tensor], ...] = ()

        def wrapper(ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
            nonlocal last_count, writes_by_term
            torch._dynamo.mark_dynamic(ids, 0)
            torch._dynamo.mark_dynamic(episode_reset, 0)
            count = ids.shape[0]
            if count != last_count:
                writes_by_term = tuple(
                    {name: buffer[:count] for name, buffer in zip(fields_, buffers, strict=True)}
                    for _call, buffers, fields_ in write_plan
                )
                # Mark compact views before tracing so graph breaks cannot
                # conflate their row count with fixed full-batch/field widths.
                for sim_writes in writes_by_term:
                    for output in sim_writes.values():
                        torch._dynamo.mark_dynamic(output, 0)
                        for dim in range(1, output.ndim):
                            torch._dynamo.mark_static(output, dim)
                last_count = count
            kernel(ids, episode_reset, writes_by_term)

        return wrapper


__all__ = ["TorchManagerRuntime", "TorchTaskProgram"]
