# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License: Apache-2.0

"""Generic Torch executor for resolved Manager programs.

The runtime consumes the metadata produced by the manager resolver (read
plan, layout, and prepared invocations) and executes the same term schedule
as the previous fused Numba kernels — command ``update`` hooks, rewards,
terminations, command ``advance`` hooks for evaluate; per-group observation
terms for observe; command ``reset_env`` hooks plus simulator reset terms
for reset — with one batched tensor call per term instead of a per-lane
loop. All writes go back into the persistent numpy buffers the host
lifecycle already owns, so ``ManagerEnv`` keeps its current API and state
surfaces unchanged.

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
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from motrix_env_core.manager.kernel import KernelOutputs
from motrix_env_core.manager.kernel_data import KernelDataScope
from motrix_env_core.torch_manager.rand import draw_uniform

if TYPE_CHECKING:
    from motrix_env_core.manager.env import ManagerEnv
    from motrix_env_core.manager.resolver import PreparedInvocation, ResolvedManagerProgram

_INPUT_EXPRESSION = re.compile(r"^input_(\d+)(\[env_id\])?$")

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
    """Tensor views over one persistent flat-input tuple.

    Simulator-backed arrays are rewritten in place by the read program, so
    views stay valid for the lifetime of the input tuple. The cache rebuilds
    only when the tuple identity changes.
    """

    def __init__(self) -> None:
        self._inputs: tuple[Any, ...] | None = None
        self._tensors: dict[int, torch.Tensor] = {}

    def tensors(self, inputs: tuple[Any, ...]) -> dict[int, torch.Tensor]:
        if self._inputs is not inputs:
            self._inputs = inputs
            self._tensors = {}
        return self._tensors

    def tensor(self, inputs: tuple[Any, ...], slot: int) -> torch.Tensor:
        tensors = self.tensors(inputs)
        tensor = tensors.get(slot)
        if tensor is None:
            value = inputs[slot]
            if isinstance(value, np.ndarray):
                # 0-dim scalar leaves are configuration constants (body
                # indices, sizes, noise amplitudes, mode flags); as tensors
                # they force tensor-dependent control flow that Dynamo cannot
                # trace (``if amplitude != 0`` / ``if sequential_clips``) and
                # a scalar extraction Inductor cannot compile. Python scalars
                # carry no dtype-precision concern and specialize branches at
                # trace time.
                if value.shape == ():
                    if np.issubdtype(value.dtype, np.integer):
                        tensors[slot] = int(value)
                        return tensors[slot]
                    if np.issubdtype(value.dtype, np.floating):
                        tensors[slot] = float(value)
                        return tensors[slot]
                    if np.issubdtype(value.dtype, np.bool_):
                        tensors[slot] = bool(value)
                        return tensors[slot]
                if value.dtype == np.uint64:
                    value = value.view(np.int64)
                tensor = torch.from_numpy(value)
            elif isinstance(value, np.integer):
                tensors[slot] = int(value)
                return tensors[slot]
            elif isinstance(value, (np.floating, float)):
                tensors[slot] = float(value)
                return tensors[slot]
            elif isinstance(value, (np.bool_, bool)):
                tensors[slot] = bool(value)
                return tensors[slot]
            else:
                tensor = torch.as_tensor(value)
            tensors[slot] = tensor
        return tensor


def _leaf_getter(cache: _TensorCache, inputs: tuple[Any, ...]) -> Callable[[int], torch.Tensor]:
    """Leaf resolver over one input tuple.

    All tensor views are materialized eagerly so the getter body is a plain
    list index — no per-slot branches for Dynamo to specialize when a graph
    break forces it to compile as its own frame.
    """

    tensors = [cache.tensor(inputs, slot) for slot in range(len(inputs))]

    def get(slot: int) -> torch.Tensor:
        return tensors[slot]

    return get


class _Namespace:
    """Attribute access over one subtree of the flattened manager context.

    Leaf paths resolve to tensors through ``get``; intermediate paths build
    child namespaces; names absent from the layout fall back to the owning
    host object (plain Python attributes such as scalar parameters).
    Resolved children are memoized per instance: the layout is immutable, and
    tensor views stay valid for the lifetime of the enclosing input tuple.
    """

    def __init__(
        self,
        paths: dict[tuple[str, ...], int],
        prefix: tuple[str, ...],
        host: Any,
        get: Callable[[int], torch.Tensor],
    ) -> None:
        self._paths = paths
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
        if path in self._paths:
            return self._get(self._paths[path])
        children = {
            rest: slot for p, slot in self._paths.items() if p[: len(path)] == path and (rest := p[len(path) :])
        }
        if children:
            # Single-leaf wrappers (for example scalar/array term buffers)
            # resolve directly to their only leaf; structured records with
            # nested kernel_data build a child namespace so chained paths
            # like ``motion.clip.joint_pos`` stay reachable.
            if len(children) == 1:
                first = next(iter(children))
                if len(first) == 1:
                    return self._get(children[first])
            return _Namespace(self._paths, path, self._host, self._get)
        if self._host is None:
            raise AttributeError(name)
        value = getattr(self._host, name)
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
    def __init__(self, get: Callable[[int], torch.Tensor], slots: dict[str, int]) -> None:
        self._get = get
        self._slots = slots

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._get(self._slots[key])


class _CommandsMap:
    def __init__(
        self,
        paths: dict[tuple[str, ...], int],
        command_terms: dict[str, Any],
        get: Callable[[int], torch.Tensor],
    ) -> None:
        self._paths = paths
        self._command_terms = command_terms
        self._get = get
        self._cache: dict[str, _Namespace] = {}

    def __getitem__(self, name: str) -> _Namespace:
        namespace = self._cache.get(name)
        if namespace is None:
            prefix = ("commands", name)
            if not any(path[:2] == prefix for path in self._paths):
                raise KeyError(name)
            namespace = _Namespace(self._paths, prefix, self._command_terms[name], self._get)
            self._cache[name] = namespace
        return namespace


class _ActionsMap:
    def __init__(
        self,
        paths: dict[tuple[str, ...], int],
        action_terms: dict[str, Any],
        get: Callable[[int], torch.Tensor],
    ) -> None:
        self._paths = paths
        self._action_terms = action_terms
        self._get = get
        self._cache: dict[str, _BatchActions] = {}

    def __getitem__(self, name: str) -> _BatchActions:
        actions = self._cache.get(name)
        if actions is None:
            prefix = ("actions", name)
            if not any(path[:2] == prefix for path in self._paths):
                raise KeyError(name)
            namespace = _Namespace(self._paths, prefix, self._action_terms[name].state, self._get)
            actions = _BatchActions(namespace)
            self._cache[name] = actions
        return actions


class _MetricsMap:
    def __init__(self, paths: dict[tuple[str, ...], int], get: Callable[[int], torch.Tensor]) -> None:
        self._paths = paths
        self._get = get

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._get(self._paths[("metrics", name)])


class _BatchContext:
    """Batched stand-in for the per-lane ``ManagerContext``.

    Per-env fields are full ``(N, ...)`` tensors; terms written for this
    context operate on whole batches.
    """

    def __init__(
        self,
        inputs: tuple[Any, ...],
        cache: _TensorCache,
        paths: dict[tuple[str, ...], int],
        env: ManagerEnv,
    ):
        get = _leaf_getter(cache, inputs)
        self._paths = paths
        self.env_id = get(paths[("env_id",)])
        self.dt = float(inputs[paths[("dt",)]])
        self.sim_reset_requested = get(paths[("sim_reset_requested",)])
        self.rand = _BatchRand(get(paths[("rand", "state")]))
        self.sim = _SimMap(get, {path[1]: slot for path, slot in paths.items() if len(path) == 2 and path[0] == "sim"})
        self.commands = _CommandsMap(paths, env.command_terms, get)
        self.actions = _ActionsMap(paths, env.action_terms, get)
        self.metrics = _MetricsMap(paths, get)


class _WritesMap:
    def __init__(self, writes: dict[str, torch.Tensor]) -> None:
        self._writes = writes

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._writes[name]


def _parse_input_expression(expression: str) -> tuple[int, bool] | None:
    """Return ``(slot, is_lane)`` for ``input_N`` / ``input_N[env_id]``."""
    match = _INPUT_EXPRESSION.match(expression)
    if match is None:
        return None
    return int(match.group(1)), match.group(2) is not None


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

        context_source = program.prepared_terms[program.context.prepared_index]
        context_base = program.input_offsets[program.context.prepared_index]
        self._context_paths: dict[tuple[str, ...], int] = {
            tuple(field.path): context_base + field.slot_index for field in context_source.fields
        }
        self._per_env_slots = frozenset(
            context_base + field.slot_index for field in context_source.fields if field.scope is KernelDataScope.PER_ENV
        )

        # Command hooks are static functions on the CommandTerm class taking
        # the command's kernel-data namespace as their receiver.
        self.invocations = tuple(program.invocations)
        # The resolver groups invocations by term kind; the evaluate schedule
        # must instead follow the fused-kernel statement order: command
        # update hooks -> rewards -> terminations -> command advance hooks
        # (rewards and terminations read pre-advance command state).
        self.evaluate_invocations = tuple(
            sorted(
                (i for i in self.invocations if i.kind in _EVALUATE_KINDS),
                key=lambda invocation: _EVALUATE_ORDER[invocation.kind],
            )
        )
        self.observe_invocations = tuple(i for i in self.invocations if i.kind == "observation")
        self._reward_indices = {
            i.context: index for index, i in enumerate(i for i in self.evaluate_invocations if i.kind == "reward")
        }
        self._termination_indices = {
            i.context: index for index, i in enumerate(i for i in self.evaluate_invocations if i.kind == "termination")
        }
        self._observation_by_name = {i.context: i for i in self.observe_invocations}
        reward_cfgs = env.cfg.reward_cfgs()
        self.reward_weights = np.asarray(
            [float(reward_cfgs[term.name].weight) for term in self.layout.rewards], dtype=np.float32
        )
        # Reset-term compact-write layout mirrors SimResetRuntime.buffers:
        # one entry per declared write, ordered by term then output name.
        self._reset_terms: tuple[tuple[Any, int, tuple[str, ...]], ...] = ()
        offsets: dict[str, int] = {}
        offset = 0
        for name, writes in env.sim_reset_writes.items():
            fields = tuple(writes)
            offsets[name] = offset
            offset += len(fields)
        self._reset_terms = tuple(
            (
                invocation,
                offsets[name],
                tuple(env.sim_reset_writes[name]),
            )
            for invocation in program.invocations
            if invocation.kind == "reset"
            for name in (invocation.context[len("sim_reset.") :],)
        )
        self._reset_env_invocations = tuple(i for i in self.invocations if i.kind == "command_reset_env")

        # Per-inputs-identity caches: the host reuses one persistent input
        # tuple (and at most two kernel-output buffer sets), so the batched
        # context and per-invocation argument tuples are built once.
        self._batch_ctx: _BatchContext | None = None
        self._batch_ctx_inputs: tuple[Any, ...] | None = None
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
        if not self._reset_env_invocations and not self._reset_terms:
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
        reward_indices = self._reward_indices
        termination_indices = self._termination_indices
        plan = tuple((invocation.kind, invocation, args) for invocation, args in self._evaluate_plan(inputs))

        def kernel(reward_view: torch.Tensor, terminated_view: torch.Tensor) -> None:
            total = None
            terminated = None
            for kind, invocation, args in plan:
                if kind in ("command_update", "command_advance"):
                    receiver = ctx.commands[invocation.receiver_key]
                    invocation.dispatcher(receiver, ctx)
                elif kind == "reward":
                    value = invocation.dispatcher(ctx, *args).to(torch.float32)
                    index = reward_indices[invocation.context]
                    weighted = value * weights[index] * dt
                    raw_terms[:, index] = value
                    weighted_terms[:, index] = weighted
                    total = weighted if total is None else total + weighted
                else:
                    mask = invocation.dispatcher(ctx, *args).to(torch.bool)
                    index = termination_indices[invocation.context]
                    termination_masks[:, index] = mask
                    terminated = mask if terminated is None else terminated | mask
            reward_view.copy_(total if total is not None else torch.zeros_like(reward_view))
            terminated_view.copy_(terminated if terminated is not None else torch.zeros_like(terminated_view))

        if self.compile_enabled:
            return torch.compile(kernel, dynamic=False, options=_compile_options())
        return kernel

    def _build_compiled_observe(self, inputs: tuple[Any, ...]) -> Callable[[torch.Tensor, torch.Tensor | None], None]:
        ctx = self._context_for(inputs)
        observation_by_name = self._observation_by_name
        plan = tuple(
            (group_name, term, observation_by_name[f"{group_name}.{term.name}"])
            for group_name, group in self.layout.observations.items()
            for term in group.terms
        )
        invocation_args = {id(inv): self._invocation_args_for(inv, inputs) for _, _, inv in plan}
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
                    (inv, invocation_args[id(inv)], buf)
                    for (_group_name, _term, inv), buf in zip(
                        tuple(entry for entry in plan if entry[0] == group_name),
                        bufs,
                        strict=True,
                    )
                ),
            )
            for group_name, bufs in term_bufs.items()
        )

        # One compiled graph per observation group: smaller generated kernels
        # keep their code hot through the interleaved physics/evaluate passes,
        # which measurably beats a single fused graph on many-core hosts.

        def make_group_kernel(
            entries: tuple[tuple[PreparedInvocation, tuple[Any, ...], torch.Tensor], ...],
        ) -> Callable[[torch.Tensor], None]:
            def group_kernel(target: torch.Tensor) -> None:
                for invocation, args, buf in entries:
                    invocation.dispatcher(ctx, buf, *args)
                target.copy_(torch.cat([buf for _inv, _args, buf in entries], dim=1))

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
        rand_slot = self._context_paths[("rand", "state")]
        per_env = self._per_env_slots
        get = _leaf_getter(self._cache, inputs)
        sim_slots = {path[1]: slot for path, slot in self._context_paths.items() if len(path) == 2 and path[0] == "sim"}
        sim_slot_ids = frozenset(sim_slots.values())
        command_reset_env = self._reset_env_invocations
        num_envs = ctx.rand.state.shape[0]
        mask = torch.zeros(num_envs, dtype=torch.bool)
        all_ids = torch.arange(num_envs)
        write_plan = tuple(
            (
                invocation,
                tuple(torch.from_numpy(reset_buffers[offset + index]) for index in range(len(fields))),
                tuple(
                    torch.empty_like(torch.from_numpy(reset_buffers[offset + index])) for index in range(len(fields))
                ),
                fields,
                self._invocation_args_for(invocation, inputs),
            )
            for invocation, offset, fields in self._reset_terms
        )

        def kernel() -> None:
            clones: dict[int, torch.Tensor] = {}

            def lane_get(slot: int) -> torch.Tensor:
                cached = clones.get(slot)
                if cached is None:
                    full = get(slot)
                    # Simulator reads are never written by reset terms (writes
                    # go through the sim_writes map), so they pass through
                    # uncloned; every other per-env leaf may be mutated.
                    cached = full.clone() if slot in per_env and slot not in sim_slot_ids else full
                    clones[slot] = cached
                return cached

            lane_rand = _LaneRand(rand_slot, ctx, lambda slot, _tensor: lane_get(slot))
            lane_ctx = _LaneContext(
                ctx,
                _CommandsMap(self._context_paths, self.env.command_terms, lane_get),
                lane_rand,
                _SimMap(lane_get, sim_slots),
                _MetricsMap(self._context_paths, lane_get),
                all_ids,
                actions=_ActionsMap(self._context_paths, self.env.action_terms, lane_get),
            )
            for invocation in command_reset_env:
                receiver = lane_ctx.commands[invocation.receiver_key]
                invocation.dispatcher(receiver, lane_ctx)
            for invocation, _buffers, staging, fields, args in write_plan:
                sim_writes = {name: buffer for name, buffer in zip(fields, staging, strict=True)}
                invocation.dispatcher(lane_ctx, _WritesMap(sim_writes), *args)
            for slot, tensor in clones.items():
                if tensor is not get(slot):
                    full = get(slot)
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
        if self._batch_ctx_inputs is not inputs:
            self._batch_ctx = _BatchContext(inputs, self._cache, self._context_paths, self.env)
            self._batch_ctx_inputs = inputs
            self._invocation_args_cache = {}
        return self._batch_ctx

    def _evaluate_plan(self, inputs: tuple[Any, ...]) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
        plan = self._plan_cache(inputs).get("evaluate")
        if plan is None:
            plan = tuple(
                (invocation, self._invocation_args_for(invocation, inputs)) for invocation in self.evaluate_invocations
            )
            self._plan_cache(inputs)["evaluate"] = plan
        return plan

    def _plan_cache(self, inputs: tuple[Any, ...]) -> dict[str, Any]:
        cache = self._invocation_args_cache.get(id(inputs))
        if cache is None:
            cache = {}
            self._invocation_args_cache[id(inputs)] = cache
        return cache

    def _invocation_args_for(self, invocation: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        cache = self._plan_cache(inputs)
        key = id(invocation)
        args = cache.get(key)
        if args is None:
            args = self._invocation_args(invocation, inputs)
            cache[key] = args
        return args

    def _invocation_args(self, invocation: PreparedInvocation, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        args: list[Any] = []
        for value, prepared_index, expression in zip(
            invocation.args_values, invocation.args_prepared_indices, invocation.args_expressions, strict=True
        ):
            if prepared_index is not None:
                source = self.program.prepared_terms[prepared_index]
                base = self.program.input_offsets[prepared_index]
                paths = {tuple(field.path): base + field.slot_index for field in source.fields}
                args.append(_Namespace(paths, (), value, _leaf_getter(self._cache, inputs)))
                continue
            parsed = _parse_input_expression(expression)
            if parsed is not None:
                slot, _is_lane = parsed
                leaf_value = inputs[slot]
                if isinstance(leaf_value, np.ndarray):
                    args.append(self._cache.tensor(inputs, slot))
                else:
                    args.append(float(leaf_value))
                continue
            args.append(value)
        return tuple(args)


class _LaneRand:
    """SplitMix64 draws on gathered reset rows, updated in place."""

    def __init__(
        self, rand_slot: int, base: _BatchContext, gather: Callable[[int, torch.Tensor], torch.Tensor]
    ) -> None:
        self._rand_slot = rand_slot
        self._base = base
        self._gather = gather
        self._state: torch.Tensor | None = None

    @property
    def state(self) -> torch.Tensor:
        if self._state is None:
            self._state = self._gather(self._rand_slot, self._base.rand.state)
        return self._state

    def next_uniform(self, count: int = 1) -> torch.Tensor:
        samples, state = draw_uniform(self.state, count)
        self.state.copy_(state)
        return samples

    def uniform_range(self, count: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * np.float32(0.5) * (self.next_uniform(count) + np.float32(1.0))


class _LaneContext:
    """Reset-lane context: per-env leaves gathered to the selected rows."""

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


_EVALUATE_KINDS = ("command_update", "reward", "termination", "command_advance")
_EVALUATE_ORDER = {"command_update": 0, "reward": 1, "termination": 2, "command_advance": 3}


__all__ = ["TorchManagerRuntime", "TorchTaskProgram"]
