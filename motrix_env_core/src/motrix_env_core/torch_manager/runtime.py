# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic Torch executor for resolved Manager programs.

The runtime consumes the metadata produced by the manager resolver (read
plan, layout, resolved context/term-argument trees, and per-term-kind typed
call records) and executes the manager term schedule — command ``update``
hooks, rewards, terminations, command ``advance`` hooks for evaluate;
per-group observation terms for observe; command ``reset_env`` hooks plus
simulator reset terms for reset — with one batched tensor call per term
instead of a per-lane loop. All writes go back into the persistent numpy
buffers the host lifecycle already owns, so ``ManagerEnv`` keeps its current
API and state surfaces unchanged.

By default each kernel schedule is additionally fused with ``torch.compile``
into a compiled closure over the captured persistent tensors (one fused
graph per kernel instead of one eager dispatch per tensor op; the reset
graph computes over the full batch with a lane mask so every shape stays
static). The eager schedule
remains available as a debugging fallback
(``TorchManagerRuntime(env, program, compile=False)``).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from motrix_env_core.manager.kernel import KernelOutputs
from motrix_env_core.manager.resolver import (
    CommandAdvanceHook,
    CommandResetHook,
    CommandUpdateHook,
    LeafArg,
    ObservationTermCall,
    RewardTermCall,
    SimArg,
    SimResetTermCall,
    TerminationTermCall,
    TreeArg,
)
from motrix_env_core.torch_manager.rand import draw_uniform

if TYPE_CHECKING:
    from motrix_env_core.manager.env import ManagerEnv
    from motrix_env_core.manager.resolver import ResolvedManagerProgram

# Simulator-backed read-only arrays are wrapped as tensors but never written
# by the executor; suppress PyTorch's non-writable-array warning once instead
# of copying every read-only leaf on every view build.
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable",
    category=UserWarning,
)


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
    return {"cpp.threads": threads, "cpp.min_chunk_size": chunk}


@dataclass(frozen=True)
class TorchTaskProgram:
    """Drop-in replacement task program backed by Torch kernels."""

    evaluate_kernel: Callable[[tuple[Any, ...], np.ndarray, tuple[np.ndarray, ...], Any], None]
    observe_kernel: Callable[[tuple[Any, ...], Any], None]
    reset_kernel: Callable[[tuple[Any, ...], np.ndarray, tuple[np.ndarray, ...]], None]
    reward_weights: np.ndarray


class _TensorCache:
    """Tensor views over one resolved-input tree anchor.

    Leaves are held by reference in the resolved trees and simulator-backed
    arrays are rewritten in place by the read program, so views stay valid for
    the lifetime of the anchor. The cache rebuilds only when the anchor
    identity changes; tensors are keyed by leaf-value identity.
    """

    def __init__(self) -> None:
        self._anchor: Any = None
        self._tensors: dict[int, Any] = {}

    def tensors(self, anchor: Any) -> dict[int, Any]:
        if self._anchor is not anchor:
            self._anchor = anchor
            self._tensors = {}
        return self._tensors

    def tensor(self, anchor: Any, value: Any) -> Any:
        tensors = self.tensors(anchor)
        key = id(value)
        tensor = tensors.get(key)
        if tensor is None:
            if isinstance(value, torch.Tensor):
                # 0-dim tensor leaves are configuration constants (body
                # indices, sizes, noise amplitudes, mode flags); as tensors
                # they force tensor-dependent control flow that Dynamo cannot
                # trace (``if amplitude != 0`` / ``if sequential_clips``) and
                # a scalar extraction Inductor cannot compile. Python scalars
                # carry no dtype-precision concern and specialize branches at
                # trace time. Non-0-dim tensors are already zero-copy views
                # over their owning arenas and pass through unchanged.
                if value.dim() == 0:
                    if value.dtype in (torch.int64, torch.int32):
                        tensors[key] = int(value)
                        return tensors[key]
                    if value.dtype in (torch.float32, torch.float64):
                        tensors[key] = float(value)
                        return tensors[key]
                    if value.dtype is torch.bool:
                        tensors[key] = bool(value)
                        return tensors[key]
                tensors[key] = value
                return value
            if isinstance(value, np.ndarray):
                # 0-dim numpy scalar leaves follow the same rule.
                if value.shape == ():
                    if np.issubdtype(value.dtype, np.integer):
                        tensors[key] = int(value)
                        return tensors[key]
                    if np.issubdtype(value.dtype, np.floating):
                        tensors[key] = float(value)
                        return tensors[key]
                    if np.issubdtype(value.dtype, np.bool_):
                        tensors[key] = bool(value)
                        return tensors[key]
                if value.dtype == np.uint64:
                    value = value.view(np.int64)
                tensor = torch.from_numpy(value)
            elif isinstance(value, np.integer):
                tensors[key] = int(value)
                return tensors[key]
            elif isinstance(value, (np.floating, float)):
                tensors[key] = float(value)
                return tensors[key]
            elif isinstance(value, (np.bool_, bool)):
                tensors[key] = bool(value)
                return tensors[key]
            else:
                tensor = torch.as_tensor(value)
            tensors[key] = tensor
        return tensor


def _tensorize_leaves(cache: _TensorCache, anchor: Any, tree: Any) -> dict[tuple[str, ...], Any]:
    """Eagerly materialize every leaf of one resolved tree.

    All tensor views are built up front so namespace resolution is a plain
    dict lookup — no per-leaf branches for Dynamo to specialize when a graph
    break forces it to compile as its own frame.
    """
    return {path: cache.tensor(anchor, leaf.value) for path, leaf in tree.leaves.items()}


def _leaf_lookup(leaves: dict[tuple[str, ...], Any]) -> Callable[[tuple[str, ...]], Any]:
    """Traceable leaf getter over one materialized leaf-view mapping.

    ``dict.__getitem__`` bound methods are builtins Dynamo refuses to trace;
    a plain closure keeps namespace resolution inside the compiled frame.
    """

    def get(path: tuple[str, ...]) -> Any:
        return leaves[path]

    return get


class _Namespace:
    """Attribute access over one subtree of the resolved manager context.

    Leaf paths resolve to tensors through ``get`` (the batch view materializer
    or the reset-lane gather); intermediate paths build child namespaces;
    names absent from the tree fall back to the owning host object (plain
    Python attributes such as scalar parameters). Resolved children are
    memoized per instance: the tree is immutable, and tensor views stay valid
    for the lifetime of the enclosing input anchor.
    """

    def __init__(
        self,
        leaves: dict[tuple[str, ...], Any],
        prefix: tuple[str, ...],
        host: Any,
        get: Callable[[tuple[str, ...]], Any],
    ) -> None:
        self._leaves = leaves
        self._prefix = prefix
        self._host = host
        self._get = get
        self._resolved: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        resolved = self._resolved.get(name, _MISSING)
        if resolved is not _MISSING:
            return resolved
        resolved = self._resolve(name)
        self._resolved[name] = resolved
        return resolved

    def _resolve(self, name: str) -> Any:
        path = (*self._prefix, name)
        if path in self._leaves:
            return self._get(path)
        children = {rest: path + rest[:1] for p in self._leaves if p[: len(path)] == path and (rest := p[len(path) :])}
        if children:
            # Single-leaf wrappers resolve directly to their only leaf;
            # structured records with nested kernel_data build a child
            # namespace so chained paths like ``motion.clip.joint_pos`` stay
            # reachable.
            if len(children) == 1:
                first = next(iter(children))
                if len(first) == 1:
                    return self._get(children[first])
            return _Namespace(self._leaves, path, self._host, self._get)
        if self._host is None:
            raise AttributeError(name)
        value = getattr(self._host, name)
        if isinstance(value, torch.Tensor):
            # Host records keep tensor fields; expose the same 0-dim
            # integer-constant rule as _TensorCache.tensor.
            if value.dim() == 0 and value.dtype in (torch.int64, torch.int32):
                return int(value)
            return value
        if isinstance(value, np.ndarray):
            # Same 0-dim integer-constant rule as _TensorCache.tensor.
            if value.shape == () and np.issubdtype(value.dtype, np.integer):
                return int(value)
            if value.dtype == np.uint64:
                value = value.view(np.int64)
            return torch.from_numpy(value)
        return value


_MISSING = object()


class _BatchRand:
    """Batched SplitMix64 draws over the shared per-env state buffer."""

    def __init__(self, state: torch.Tensor) -> None:
        self.state = state

    def next_uniform(self, count: int = 1) -> torch.Tensor:
        """Draw ``count`` values per lane in ``[-1, 1)``; shape ``(N, count)``."""
        samples, state = draw_uniform(self.state, count)
        self.state.copy_(state)
        return samples

    def uniform_range(self, count: int, low: float, high: float) -> torch.Tensor:
        """Draw ``count`` values per lane in ``[low, high)``; shape ``(N, count)``."""
        return low + (high - low) * np.float32(0.5) * (self.next_uniform(count) + np.float32(1.0))


class _BatchActions:
    """One action term's ring-buffer state with batched current/previous."""

    def __init__(self, namespace: _Namespace) -> None:
        self._namespace = namespace

    @property
    def action_queue(self) -> torch.Tensor:
        return self._namespace.action_queue

    @property
    def action_ptr(self) -> torch.Tensor:
        return self._namespace.action_ptr

    def current(self) -> torch.Tensor:
        # ``action_ptr`` is the shared (1,) cursor; gather every lane's row
        # from the same slot.
        queue = self.action_queue
        index = self.action_ptr.view(1, 1, 1).expand(queue.shape[0], 1, queue.shape[-1])
        return queue.gather(1, index).squeeze(1)

    def previous(self) -> torch.Tensor:
        queue = self.action_queue
        prev = (self.action_ptr.view(1, 1) - 1) % queue.shape[1]
        index = prev.expand(queue.shape[0], 1, queue.shape[-1])
        return queue.gather(1, index).squeeze(1)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._namespace, name)


class _SimMap:
    def __init__(self, get: Callable[[tuple[str, ...]], Any], paths: dict[str, tuple[str, ...]]) -> None:
        self._get = get
        self._paths = paths

    def __getitem__(self, key: str) -> Any:
        return self._get(self._paths[key])


class _CommandsMap:
    def __init__(
        self,
        leaves: dict[tuple[str, ...], Any],
        command_terms: dict[str, Any],
        get: Callable[[tuple[str, ...]], Any],
    ) -> None:
        self._leaves = leaves
        self._command_terms = command_terms
        self._get = get
        self._cache: dict[str, _Namespace] = {}

    def __getitem__(self, name: str) -> _Namespace:
        namespace = self._cache.get(name)
        if namespace is None:
            prefix = ("commands", name)
            if not any(path[:2] == prefix for path in self._leaves):
                raise KeyError(name)
            namespace = _Namespace(self._leaves, prefix, self._command_terms[name], self._get)
            self._cache[name] = namespace
        return namespace


class _ActionsMap:
    def __init__(
        self,
        leaves: dict[tuple[str, ...], Any],
        action_terms: dict[str, Any],
        get: Callable[[tuple[str, ...]], Any],
    ) -> None:
        self._leaves = leaves
        self._action_terms = action_terms
        self._get = get
        self._cache: dict[str, _BatchActions] = {}

    def __getitem__(self, name: str) -> _BatchActions:
        actions = self._cache.get(name)
        if actions is None:
            prefix = ("actions", name)
            if not any(path[:2] == prefix for path in self._leaves):
                raise KeyError(name)
            namespace = _Namespace(self._leaves, prefix, self._action_terms[name].state, self._get)
            actions = _BatchActions(namespace)
            self._cache[name] = actions
        return actions


class _MetricsMap:
    def __init__(self, get: Callable[[tuple[str, ...]], Any]) -> None:
        self._get = get

    def __getitem__(self, name: str) -> Any:
        return self._get(("metrics", name))


class _BatchContext:
    """Batched stand-in for the per-lane ``ManagerContext``.

    Per-env fields are full ``(N, ...)`` tensors; terms written for this
    context operate on whole batches.
    """

    def __init__(
        self,
        tree: Any,
        tensors: dict[tuple[str, ...], Any],
        env: ManagerEnv,
    ):
        self.env_id = tensors[("env_id",)]
        self.dt = float(tree.leaves[("dt",)].value)
        self.sim_reset_requested = tensors[("sim_reset_requested",)]
        self.rand = _BatchRand(tensors[("rand", "state")])
        get = _leaf_lookup(tensors)
        self.sim = _SimMap(
            get,
            {path[1]: path for path in tensors if len(path) == 2 and path[0] == "sim"},
        )
        self.commands = _CommandsMap(tensors, env.command_terms, get)
        self.actions = _ActionsMap(tensors, env.action_terms, get)
        self.metrics = _MetricsMap(get)


class _WritesMap:
    def __init__(self, writes: dict[str, torch.Tensor]) -> None:
        self._writes = writes

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._writes[name]


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
        self._cache = _TensorCache()

        self._context_tree = program.prepared_trees[program.context.tree_index]
        self._context_leaves = self._context_tree.leaves
        # Reset terms write through three surfaces only: each reset hook's
        # receiver command tree, the RNG state, and metrics. Everything
        # else (sim views, action state, clip arrays, shared params) is
        # read-only during reset and passes through uncloned.
        self._reset_clone_prefixes: tuple[tuple[str, ...], ...] = (
            *(("commands", hook.receiver_key) for hook in program.command_reset_hooks),
            ("rand", "state"),
            ("metrics",),
        )
        self._sim_paths = {path[1]: path for path in self._context_leaves if len(path) == 2 and path[0] == "sim"}
        self._rand_path = ("rand", "state")

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
        self.reward_weights = np.asarray(
            [float(reward_cfgs[term.name].weight) for term in self.layout.rewards], dtype=np.float32
        )
        # Reset-term compact-write layout mirrors SimResetRuntime.buffers:
        # one entry per declared write, ordered by term then output name.
        # The write-field names come from the host (same insertion order the
        # resolver used when assigning offsets).
        self._reset_terms: tuple[tuple[SimResetTermCall, int, tuple[str, ...]], ...] = tuple(
            (reset.call, reset.output_offset, tuple(env.sim_reset_writes[name]))
            for name, reset in zip(env.sim_reset_terms, program.sim_reset_terms, strict=True)
        )

        # Per-anchor caches: the host reuses one persistent resolved-input
        # anchor (and at most two kernel-output buffer sets), so the batched
        # context and per-invocation argument tuples are built once.
        self._batch_ctx: _BatchContext | None = None
        self._context_tensors: dict[tuple[str, ...], Any] | None = None
        self._tree_tensors: tuple[dict[tuple[str, ...], Any], ...] | None = None
        self._batch_ctx_anchor: Any = None
        self._invocation_args_cache: dict[int, dict[int, tuple[Any, ...]]] = {}
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
        inputs: tuple[Any, ...],
        reward_weights: np.ndarray,
        buffers: tuple[np.ndarray, np.ndarray, np.ndarray],
        outputs: KernelOutputs,
    ) -> None:
        kernel = self._compiled_kernels.get(("evaluate", id(inputs), id(buffers)))
        if kernel is None:
            kernel = self._build_compiled_evaluate(inputs, reward_weights, buffers)
            self._compiled_kernels[("evaluate", id(inputs), id(buffers))] = kernel
        kernel(torch.from_numpy(outputs.reward), torch.from_numpy(outputs.terminated))

    def observe(self, inputs: tuple[Any, ...], outputs: KernelOutputs) -> None:
        kernel = self._compiled_kernels.get(("observe", id(inputs)))
        if kernel is None:
            kernel = self._build_compiled_observe(inputs)
            self._compiled_kernels[("observe", id(inputs))] = kernel
        value_obs = outputs.value_obs
        kernel(
            torch.from_numpy(outputs.policy_obs),
            torch.from_numpy(value_obs) if value_obs is not None and value_obs.size else None,
        )

    def reset(self, inputs: tuple[Any, ...], env_ids: np.ndarray, reset_buffers: tuple[np.ndarray, ...]) -> None:
        if not self.command_reset_hooks and not self._reset_terms:
            return
        ids = torch.from_numpy(np.ascontiguousarray(env_ids, dtype=np.int64))
        kernel = self._compiled_kernels.get(("reset", id(inputs), id(reset_buffers)))
        if kernel is None:
            kernel = self._build_compiled_reset(inputs, reset_buffers)
            self._compiled_kernels[("reset", id(inputs), id(reset_buffers))] = kernel
        kernel(ids)

    def _build_compiled_evaluate(
        self,
        inputs: tuple[Any, ...],
        reward_weights: np.ndarray,
        buffers: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Callable[[torch.Tensor, torch.Tensor], None]:
        ctx = self._context_for(inputs)
        weights = torch.from_numpy(np.asarray(reward_weights, dtype=np.float32))
        raw_terms = torch.from_numpy(buffers[0])
        weighted_terms = torch.from_numpy(buffers[1])
        termination_masks = torch.from_numpy(buffers[2])
        dt = ctx.dt
        update_hooks = self.command_updates
        advance_hooks = self.command_advances
        reward_plan = tuple((term, self._invocation_args_for(term, inputs)) for term in self.rewards)
        termination_plan = tuple((term, self._invocation_args_for(term, inputs)) for term in self.terminations)

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

    def _build_compiled_observe(self, inputs: tuple[Any, ...]) -> Callable[[torch.Tensor, torch.Tensor | None], None]:
        ctx = self._context_for(inputs)
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
                    (call, self._invocation_args_for(call, inputs), buf)
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

    def _build_compiled_reset(
        self, inputs: tuple[Any, ...], reset_buffers: tuple[np.ndarray, ...]
    ) -> Callable[[torch.Tensor], None]:
        """Compile the reset schedule as a static full-batch masked graph.

        The reset math runs for ALL lanes with the selected lanes chosen by a
        boolean mask; per-env state written by terms is clone-modified and
        merged back with ``where(mask, clone, original)``, so non-selected
        lanes (including their RNG state) keep their values while selected
        lanes advance exactly as the gathered schedule computed them (each
        lane's SplitMix64 sequence is independent). Compaction into the
        reset buffers happens outside the graph via one ``index_select`` per
        write field. Static shapes let Inductor parallelize every loop.
        """
        ctx = self._context_for(inputs)
        assert self._context_tensors is not None
        clone_prefixes = self._reset_clone_prefixes
        get = _leaf_lookup(self._context_tensors)
        command_reset_hooks = self.command_reset_hooks
        num_envs = ctx.rand.state.shape[0]
        mask = torch.zeros(num_envs, dtype=torch.bool)
        all_ids = torch.arange(num_envs)
        write_plan = tuple(
            (
                call,
                tuple(torch.from_numpy(reset_buffers[offset + index]) for index in range(len(fields))),
                tuple(
                    torch.empty_like(torch.from_numpy(reset_buffers[offset + index])) for index in range(len(fields))
                ),
                fields,
                self._invocation_args_for(call, inputs),
            )
            for call, offset, fields in self._reset_terms
        )

        def kernel() -> None:
            clones: dict[tuple[str, ...], torch.Tensor] = {}

            def lane_get(path: tuple[str, ...]) -> torch.Tensor:
                cached = clones.get(path)
                if cached is None:
                    full = get(path)
                    # Reset terms only write the receiver command trees,
                    # RNG state and metrics (sim writes go through the
                    # sim_writes map); every other leaf is read-only and
                    # passes through uncloned.
                    cached = (
                        full.clone()
                        if isinstance(full, torch.Tensor)
                        and full.shape[:1] == (num_envs,)
                        and any(path[: len(prefix)] == prefix for prefix in clone_prefixes)
                        else full
                    )
                    clones[path] = cached
                return cached

            lane_rand = _LaneRand(self._rand_path, lane_get)
            lane_ctx = _LaneContext(
                ctx,
                _CommandsMap(self._context_tensors, self.env.command_terms, lane_get),
                lane_rand,
                _SimMap(lane_get, self._sim_paths),
                _MetricsMap(lane_get),
                all_ids,
                actions=_ActionsMap(self._context_tensors, self.env.action_terms, lane_get),
            )
            for hook in command_reset_hooks:
                receiver = lane_ctx.commands[hook.receiver_key]
                hook.function(receiver, lane_ctx)
            for call, _buffers, staging, fields, args in write_plan:
                sim_writes = {name: buffer for name, buffer in zip(fields, staging, strict=True)}
                call.function(lane_ctx, _WritesMap(sim_writes), *args)
            for path, tensor in clones.items():
                if tensor is not get(path):
                    full = get(path)
                    lane_mask = mask.view((num_envs,) + (1,) * (tensor.dim() - 1))
                    full.copy_(torch.where(lane_mask, tensor, full))

        if self.compile_enabled:
            kernel = torch.compile(kernel, dynamic=False, options=_compile_options())
        mask_np = mask.numpy()
        compact_plan = tuple(
            (buffer.numpy(), stage.numpy())
            for _invocation, buffers, staging, _fields, _args in write_plan
            for buffer, stage in zip(buffers, staging, strict=True)
        )

        def wrapper(ids: torch.Tensor) -> None:
            ids_np = ids.numpy()
            mask_np.fill(False)
            mask_np[ids_np] = True
            kernel()
            count = ids_np.shape[0]
            for buffer, stage in compact_plan:
                buffer[:count] = stage[ids_np]

        return wrapper

    # -- helpers ------------------------------------------------------------

    def _context_for(self, inputs: tuple[Any, ...]) -> _BatchContext:
        if self._batch_ctx_anchor is not inputs:
            tensors = tuple(_tensorize_leaves(self._cache, inputs, tree) for tree in inputs)
            self._tree_tensors = tensors
            self._context_tensors = tensors[self.program.context.tree_index]
            self._batch_ctx = _BatchContext(self._context_tree, self._context_tensors, self.env)
            self._batch_ctx_anchor = inputs
            self._invocation_args_cache = {}
        return self._batch_ctx

    def _plan_cache(self, inputs: tuple[Any, ...]) -> dict[int, tuple[Any, ...]]:
        cache = self._invocation_args_cache.get(id(inputs))
        if cache is None:
            cache = {}
            self._invocation_args_cache[id(inputs)] = cache
        return cache

    def _invocation_args_for(self, call: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        cache = self._plan_cache(inputs)
        key = id(call)
        args = cache.get(key)
        if args is None:
            args = self._invocation_args(call, inputs)
            cache[key] = args
        return args

    def _invocation_args(
        self,
        call: RewardTermCall | TerminationTermCall | ObservationTermCall | SimResetTermCall,
        inputs: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert self._tree_tensors is not None
        args: list[Any] = []
        for arg in call.args:
            if isinstance(arg, TreeArg):
                leaves = self._tree_tensors[arg.tree_index]
                host = self.program.prepared_trees[arg.tree_index].value
                args.append(_Namespace(leaves, (), host, _leaf_lookup(leaves)))
            elif isinstance(arg, LeafArg):
                args.append(self._cache.tensor(inputs, arg.leaf.value))
            elif isinstance(arg, SimArg):
                assert self._context_tensors is not None
                args.append(self._context_tensors[self._sim_paths[arg.key]])
            else:
                args.append(arg.value)
        return tuple(args)


class _LaneRand:
    """SplitMix64 draws on gathered reset rows, updated in place."""

    def __init__(self, rand_path: tuple[str, ...], gather: Callable[[tuple[str, ...]], torch.Tensor]) -> None:
        self._rand_path = rand_path
        self._gather = gather
        self._state: torch.Tensor | None = None

    @property
    def state(self) -> torch.Tensor:
        if self._state is None:
            self._state = self._gather(self._rand_path)
        return self._state

    def next_uniform(self, count: int = 1) -> torch.Tensor:
        samples, state = draw_uniform(self.state, count)
        self.state.copy_(state)
        return samples

    def uniform_range(self, count: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * np.float32(0.5) * (self.next_uniform(count) + np.float32(1.0))


class _LaneContext:
    """Reset-lane context: per-env leaves cloned to the selected rows."""

    def __init__(
        self,
        base: _BatchContext,
        commands: _CommandsMap,
        rand: _LaneRand,
        sim: _SimMap,
        metrics: _MetricsMap,
        ids: torch.Tensor,
        actions: _ActionsMap | None = None,
    ) -> None:
        self._base = base
        self.commands = commands
        self.rand = rand
        self.sim = sim
        self.metrics = metrics
        self.env_id = ids
        self.dt = base.dt
        self.sim_reset_requested = base.sim_reset_requested
        self.actions = actions if actions is not None else base.actions


__all__ = ["TorchManagerRuntime", "TorchTaskProgram"]
