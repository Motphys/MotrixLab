# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import inspect
import logging
import pickle
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, get_args, get_origin, get_type_hints

import numba
import numpy as np
from numba.extending import register_jitable

from motrix_env_core.numba.fingerprint import function_fingerprint
from motrix_env_core.numba.kernel import clone_kernel_value
from motrix_env_core.numba.kernel_data import (
    KernelDataLayout,
    KernelDataLowering,
    Map,
    SharedArray,
    flatten_kernel_data,
    is_kernel_data,
    iter_layout_leaves,
    kernel_data,
    lane_expression,
    map_proxy,
    proxy_symbol,
    proxy_types,
)
from motrix_env_core.numba.manager.compiler.cache import (
    _KERNEL_CACHE,
    _TERM_CACHE,
    generated_source_path,
    has_generated_kernel_cache,
    invalidate_generated_cache,
    invalidate_term_cache,
    materialize_source,
)
from motrix_env_core.numba.manager.compiler.codegen import KernelSourceGenerator
from motrix_env_core.numba.manager.compiler.fingerprint import (
    plan_key,
    type_name,
)
from motrix_env_core.numba.manager.compiler.plan import (
    ActionTermLayout,
    InputSlotLayout,
    ManagerLayout,
    MetricFieldLayout,
    ObservationGroupLayout,
    ObservationTermLayout,
    ScalarTermLayout,
    SimInputLayout,
)
from motrix_env_core.numba.manager.compiler.program import (
    PreparedInvocation,
    ResolvedManagerContext,
    ResolvedSimReset,
    _CompiledManagerProgram,
)
from motrix_env_core.numba.manager.context import ManagerContext
from motrix_env_core.numba.manager.dispatch import is_dispatched
from motrix_env_core.numba.manager.env import (
    CompiledManagerProgram,
    KernelInputSource,
    ManagerBasedEnvCfg,
    ManagerEnv,
    ReadPlan,
)
from motrix_env_core.numba.manager.observations import ObservationGroupEntry, ObsTerm
from motrix_env_core.numba.manager.rewards import RewardTerm
from motrix_env_core.numba.manager.terminations import TerminationTerm
from motrix_env_core.numba.program import NumbaTaskProgram
from motrix_env_core.sim import PhysicsReadProgram, SimDataQuery

_SCHEMA_VERSION = 34
logger = logging.getLogger(__name__)

_KERNEL_KINDS = ("evaluate", "observe", "reset")


@kernel_data
class TermArrayBuffer:
    """Shared array slot for one ndarray term argument.

    Arrays passed directly as manager term arguments are lowered with shared
    scope (every kernel lane sees the full array): they hold env-invariant
    reference data such as default poses, weights, or lookup tables. Unlike
    :class:`TermScalarBuffer` the full value participates in the plan
    fingerprint, mirroring kernel-data array leaves.
    """

    value: SharedArray


@kernel_data
class TermScalarBuffer:
    """Runtime buffer slot for one numeric term argument.

    Numeric tuning arguments (thresholds, scales, gains, ...) are lowered into
    a fixed-dtype shared input slot: only the type/layout fingerprint enters
    the plan key, and the value itself is read at runtime. This keeps
    tuning-variant manager configs on one compiled plan instead of triggering
    a full recompile per value.
    """

    value: np.float32


@dataclass(frozen=True)
class ManagerSimInput:
    """One compiled simulator input exposed through ``ManagerContext.sim``."""

    value: np.ndarray
    query: SimDataQuery


def _manager_sim_inputs(sim_data: PhysicsReadProgram) -> tuple[Mapping[str, ManagerSimInput], tuple[Any, ...]]:
    inputs = {key: ManagerSimInput(sim_data.view(key), sim_data.query(key)) for key in sim_data.keys}
    fingerprint = tuple(
        (key, repr(binding.query), binding.value.shape[1:], binding.value.strides) for key, binding in inputs.items()
    )
    return MappingProxyType(inputs), fingerprint


class NumbaKernelCompiler:
    """Lower one fixed manager config to a fused Numba task program."""

    def __init__(self, env: ManagerEnv):
        self._env = env
        self._sim_inputs, sim_input_fingerprint = _manager_sim_inputs(env.sim_data)
        self._sim_slots_by_key: dict[str, int] = {}
        self._prepared_terms: list[KernelInputSource] = []
        self._input_offsets: list[int] = []
        self._flat_input_count = 0
        # Plan parts are partitioned per fused kernel so every kernel gets an
        # independent plan key and disk cache: editing reward terms must not
        # recompile the observation or reset kernels. Parts shared by all
        # kernels (schema, context layout, sim inputs) live in _shared_parts.
        self._shared_parts: list[Any] = [_SCHEMA_VERSION, numba.__version__]
        self._evaluate_parts: list[Any] = []
        self._observe_parts: list[Any] = []
        self._reset_parts: list[Any] = []
        self._term_functions: dict[str, Any] = {}
        self._prepared_types: dict[str, type[tuple]] = {}
        self._kernel_data_lowering = KernelDataLowering()
        self._shared_parts.append(("sim_inputs", sim_input_fingerprint))

    def build(self) -> CompiledManagerProgram:
        build_started = perf_counter()
        self._resolve_runtime_terms()
        observation_groups = self._env.observation_groups
        reward_terms = self._env._reward_terms
        termination_terms = self._env.termination_manager.terms
        context = self._resolve_manager_context(observation_groups, reward_terms, termination_terms)
        command_updates = self._resolve_command_hooks("update")
        command_advances = self._resolve_command_hooks("advance")
        command_resets = self._resolve_command_hooks("reset_env")
        observation_terms, observation_layout = self._resolve_observations(observation_groups)
        reward_invocations, reward_layout, reward_weights = self._resolve_rewards(self._env.cfg, reward_terms)
        termination_invocations, termination_layout = self._resolve_terminations(termination_terms)
        reset_invocations = self._resolve_resets()
        per_env_metric_layout = tuple(
            MetricFieldLayout(name, index, np.dtype(value.dtype))
            for index, (name, value) in enumerate(self._env.metrics.items())
            if value.shape in {(self._env.num_envs,), (self._env.num_envs, 1)}
        )
        self._validate_observation_spaces(observation_groups)

        # Reward weights stay unscaled in the runtime buffer; the dt scaling
        # happens inside the kernel via ctx.dt, so ctrl_dt does not need to
        # participate in the plan key.
        kernel_reward_weights = np.asarray(reward_weights, dtype=np.float32)
        # The kernel kind participates in the key: two kernels with identical
        # term sets (e.g. both empty) still have different generated sources.
        part_lists = {
            "evaluate": self._evaluate_parts,
            "observe": self._observe_parts,
            "reset": self._reset_parts,
        }
        plan_keys = {
            kind: plan_key([("kernel_kind", kind)] + self._shared_parts + part_lists[kind]) for kind in _KERNEL_KINDS
        }
        logger.info(
            "Manager startup %s: build plan finished in %.3fs (evaluate=%.12s observe=%.12s reset=%.12s)",
            self._env_name(),
            perf_counter() - build_started,
            plan_keys["evaluate"],
            plan_keys["observe"],
            plan_keys["reset"],
        )
        sources = KernelSourceGenerator(self._flat_input_count).generate(
            observation_terms,
            observation_layout,
            reward_invocations,
            termination_invocations,
            context,
            command_updates,
            command_advances,
            command_resets,
            reset_invocations,
        )
        kernels: dict[str, Any] = {}
        generated_filenames: dict[str, str] = {}
        for kind, source in zip(_KERNEL_KINDS, sources):
            kernels[kind], generated_filenames[kind] = self._load_kernel(kind, source, plan_keys[kind])
        evaluate_kernel = kernels["evaluate"]
        observe_kernel = kernels["observe"]
        reset_kernel = kernels["reset"]

        layout = ManagerLayout(
            inputs=tuple(
                InputSlotLayout(
                    offset + field_index,
                    f"{prepared.source_name}.{field_name}",
                    prepared.fields[field_index].scope,
                )
                for prepared, offset in zip(self._prepared_terms, self._input_offsets, strict=True)
                for field_index, field_name in enumerate(prepared.field_names)
            ),
            sim_inputs=self._sim_input_layout(),
            actions=tuple(
                ActionTermLayout(
                    name,
                    action_slice,
                    None
                    if self._env.action_actuators[name] is None
                    else tuple(spec.name for spec in self._env.action_actuators[name]),
                )
                for name, action_slice in self._env.action_slices.items()
            ),
            observations=observation_layout,
            rewards=reward_layout,
            terminations=termination_layout,
            metrics=per_env_metric_layout,
            plan_keys=(plan_keys["evaluate"], plan_keys["observe"], plan_keys["reset"]),
            generated_filenames=(
                generated_filenames["evaluate"],
                generated_filenames["observe"],
                generated_filenames["reset"],
            ),
        )
        invocations = (
            command_updates
            + command_advances
            + command_resets
            + tuple(term for terms in observation_terms.values() for term in terms)
            + reward_invocations
            + termination_invocations
        )
        return _CompiledManagerProgram(
            task=NumbaTaskProgram(
                evaluate_kernel=evaluate_kernel,
                observe_kernel=observe_kernel,
                reset_kernel=reset_kernel,
                reward_weights=kernel_reward_weights,
            ),
            read_plan=ReadPlan(
                self._env.sim_data,
                tuple(self._prepared_terms),
                tuple(value for prepared in self._prepared_terms for value in prepared.values),
            ),
            layout=layout,
            sources=sources,
            invocations=invocations,
            prepared_terms=tuple(self._prepared_terms),
            input_offsets=tuple(self._input_offsets),
            context=context,
        )

    def _resolve_resets(self) -> tuple[ResolvedSimReset, ...]:
        resolved = []
        output_offset = 0
        for index, (name, term) in enumerate(self._env.sim_reset_terms.items()):
            writes = self._env.sim_reset_writes[name]
            outputs = tuple(writes)
            resolved.append(self._resolve_reset(name, term, index, output_offset, outputs, writes))
            output_offset += len(outputs)
        return tuple(resolved)

    def _resolve_reset(
        self,
        name: str,
        term: Any,
        term_index: int,
        output_offset: int,
        fields: tuple[str, ...],
        descriptors: object,
    ) -> ResolvedSimReset:
        function = term.dispatch
        parameters = tuple(inspect.signature(function).parameters.values())
        if any(
            parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for parameter in parameters
        ):
            raise TypeError(f"Manager simulator reset term {name!r} dispatch args must be positional parameters.")
        if len(parameters) != 2 + len(term.args):
            raise TypeError(
                f"Manager simulator reset term {name!r} dispatch must take a ManagerContext parameter, a "
                f"Map[np.ndarray] sim-writes parameter, and {len(term.args)} positional args; got "
                f"{tuple(parameter.name for parameter in parameters)}."
            )
        annotations = get_type_hints(function, include_extras=True)
        if annotations.get(parameters[0].name) is not ManagerContext:
            raise TypeError(
                f"Manager simulator reset term {name!r} dispatch must annotate its first parameter as "
                f"ManagerContext (conventionally named ctx)."
            )
        sim_writes_annotation = annotations.get(parameters[1].name)
        if get_origin(sim_writes_annotation) is not Map or get_args(sim_writes_annotation) != (np.ndarray,):
            raise TypeError(
                f"Manager simulator reset term {name!r} dispatch must annotate its second parameter as "
                f"Map[np.ndarray] (conventionally named sim_writes)."
            )
        sim_writes_proxy = map_proxy(
            f"sim_reset_{name}_writes",
            fields,
            module_name=__name__,
            schema_fingerprint=hashlib.sha256(repr(descriptors).encode()).hexdigest(),
        )
        self._prepared_types[proxy_symbol(sim_writes_proxy)] = sim_writes_proxy
        dispatcher = self._compile_term(function)
        symbol = f"term_reset_{term_index}"
        self._term_functions[symbol] = self._generated_term(function, dispatcher)
        expressions = []
        plan_expressions = []
        prepared_indices = []
        for index, value in enumerate(term.args):
            if is_kernel_data(value):
                _, tree_def = flatten_kernel_data(value)
                layout = self._kernel_data_lowering.lower(
                    tree_def,
                    context=f"Simulator reset {name} args[{index}]",
                    force_shared=True,
                )
                prepared_index, expression = self._register_prepared(
                    f"sim_reset.{name}.args[{index}]", value, type(value), layout
                )
                prepared_indices.append(prepared_index)
                expressions.append(expression)
                plan_expressions.append(expression)
            else:
                expression, _, plan_part, prepared_index = self._resolve_plain_arg(
                    f"sim_reset.{name}.args[{index}]", value
                )
                prepared_indices.append(prepared_index)
                expressions.append(expression)
                plan_expressions.append(plan_part)
        self._reset_parts.append(
            (
                "sim_reset",
                name,
                self._function_fingerprint(function),
                plan_expressions,
                repr(descriptors),
                fields,
            )
        )
        invocation = PreparedInvocation(
            "sim_reset." + name,
            "reset",
            dispatcher,
            None,
            args_expressions=tuple(expressions),
            args_values=term.args,
            args_prepared_indices=tuple(prepared_indices),
        )
        return ResolvedSimReset(
            invocation,
            proxy_symbol(sim_writes_proxy),
            output_offset,
            len(fields),
        )

    def _sim_input_layout(self) -> tuple[SimInputLayout, ...]:
        slots_by_key = self._sim_slots_by_key
        return tuple(
            SimInputLayout(
                slot=slots_by_key[key],
                key=key,
                shape=binding.value.shape[1:],
                consumers=("ManagerContext.sim",),
            )
            for key, binding in self._sim_inputs.items()
        )

    def _resolve_runtime_terms(self) -> None:
        for name, action_term in self._env.action_terms.items():
            _, tree_def = flatten_kernel_data(action_term.state)
            self._shared_parts.append(
                (
                    "action",
                    name,
                    self._type_name(type(action_term)),
                    tree_def.fingerprint,
                    self._env.action_slices[name],
                    None
                    if self._env.action_actuators[name] is None
                    else tuple(spec.name for spec in self._env.action_actuators[name]),
                )
            )
        for name, command_term in self._env.command_terms.items():
            _, tree_def = flatten_kernel_data(command_term)
            self._shared_parts.append(
                (
                    "command",
                    name,
                    self._type_name(type(command_term)),
                    tree_def.fingerprint,
                )
            )

    def _resolve_manager_context(
        self,
        observation_groups: dict[str, ObservationGroupEntry],
        reward_terms: dict[str, RewardTerm],
        termination_terms: dict[str, TerminationTerm],
    ) -> ResolvedManagerContext:
        actions = {name: term.state for name, term in self._env.action_terms.items()}
        sim = {key: binding.value for key, binding in self._sim_inputs.items()}
        commands = dict(self._env.command_terms)
        value = ManagerContext(
            env_id=np.arange(self._env.num_envs, dtype=np.int64),
            actions=Map(actions),
            commands=Map(commands),
            metrics=Map(self._env.metrics),
            rand=self._env._rand,
            sim=Map(sim),
            dt=np.float32(self._env.cfg.ctrl_dt),
            sim_reset_requested=self._env._sim_reset_requested,
        )
        _, tree_def = flatten_kernel_data(value)
        layout = self._kernel_data_lowering.lower(
            tree_def,
            context="ManagerContext",
        )
        prepared_index, expression = self._register_prepared("manager_context", value, ManagerContext, layout)
        self._shared_parts.append(("manager_context", layout.fingerprint))
        self._shared_parts.append(
            (
                "metrics",
                tuple((name, value.shape, value.dtype.str) for name, value in self._env.metrics.items()),
            )
        )
        # Map each declared sim key to its global input slot inside the
        # flattened context, so term args naming a sim key can resolve to the
        # key's lane view instead of crossing the kernel as a string.
        context_source = self._prepared_terms[prepared_index]
        context_offset = self._input_offsets[prepared_index]
        self._sim_slots_by_key = {
            field.path[1]: context_offset + field_index
            for field_index, field in enumerate(context_source.fields)
            if len(field.path) == 2 and field.path[0] == "sim"
        }
        return ResolvedManagerContext(prepared_index, expression)

    def _resolve_command_hooks(self, function_name: str) -> tuple[PreparedInvocation, ...]:
        hooks = []
        for index, (name, command_term) in enumerate(self._env.command_terms.items()):
            function = inspect.getattr_static(type(command_term), function_name)
            hooks.append(
                self._resolve_receiver_term(
                    "command",
                    name,
                    command_term,
                    function_name,
                    index,
                    0,
                    function=function,
                    receiver_key=name,
                )
            )
        return tuple(hooks)

    def _resolve_observations(
        self,
        groups: dict[str, ObservationGroupEntry],
    ) -> tuple[dict[str, tuple[PreparedInvocation, ...]], dict[str, ObservationGroupLayout]]:
        resolved_groups = {}
        layout_groups = {}
        term_index = 0
        for group_name, group in groups.items():
            terms = []
            layouts = []
            offset = 0
            for entry in group.terms:
                terms.append(self._resolve_observation(group_name, entry.name, entry.term, term_index, entry.size))
                output_slice = slice(offset, offset + entry.size)
                layouts.append(ObservationTermLayout(entry.name, output_slice))
                offset = output_slice.stop
                term_index += 1
            if offset != group.size:
                raise RuntimeError(f"Observation group {group_name!r} resolved width changed during compilation.")
            resolved_groups[group_name] = tuple(terms)
            layout_groups[group_name] = ObservationGroupLayout(group_name, tuple(layouts), group.size)
        return resolved_groups, layout_groups

    def _resolve_observation_dispatch(
        self, group: str, name: str, term: ObsTerm, term_index: int, output_size: int
    ) -> PreparedInvocation:
        function = term.dispatch
        parameters = tuple(inspect.signature(function).parameters.values())
        if any(
            parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for parameter in parameters
        ):
            raise TypeError(f"Manager observation {group}.{name} dispatch args must be positional parameters.")
        if len(parameters) != 2 + len(term.args):
            raise TypeError(
                f"Manager observation {group}.{name} dispatch must take a ManagerContext parameter, an "
                f"np.ndarray output parameter, and {len(term.args)} positional args; got "
                f"{tuple(parameter.name for parameter in parameters)}."
            )
        annotations = get_type_hints(function, include_extras=True)
        if (
            annotations.get(parameters[0].name) is not ManagerContext
            or annotations.get(parameters[1].name) is not np.ndarray
        ):
            raise TypeError(
                f"Manager observation {group}.{name} dispatch must annotate its first two parameters as "
                f"ManagerContext and np.ndarray (conventionally named ctx and out)."
            )
        expressions = []
        plan_expressions = []
        prepared_indices = []
        warmup_values = []
        for index, value in enumerate(term.args):
            if is_kernel_data(value):
                _, tree_def = flatten_kernel_data(value)
                layout = self._kernel_data_lowering.lower(
                    tree_def, context=f"Observation {group}.{name} args[{index}]", force_shared=True
                )
                prepared_index, expression = self._register_prepared(
                    f"observation.{group}.{name}.args[{index}]", value, type(value), layout
                )
                prepared_indices.append(prepared_index)
                expressions.append(expression)
                plan_expressions.append(expression)
                warmup_values.append(value)
            else:
                expression, warmup, plan_part, prepared_index = self._resolve_plain_arg(
                    f"observation.{group}.{name}.args[{index}]", value
                )
                prepared_indices.append(prepared_index)
                expressions.append(expression)
                plan_expressions.append(plan_part)
                warmup_values.append(warmup)
        expressions = tuple(expressions)
        dispatcher = self._compile_term(function)
        symbol = f"term_observation_{term_index}"
        self._term_functions[symbol] = self._generated_term(function, dispatcher)
        self._observe_parts.append(
            (group, name, "observation_dispatch", self._function_fingerprint(function), output_size, plan_expressions)
        )
        return PreparedInvocation(
            f"{group}.{name}",
            "observation",
            dispatcher,
            None,
            args_expressions=expressions,
            args_values=tuple(warmup_values),
            args_prepared_indices=tuple(prepared_indices),
            output_size=output_size,
        )

    def _resolve_observation(
        self,
        group: str,
        name: str,
        term: Any,
        term_index: int,
        output_size: int,
    ) -> PreparedInvocation:
        return self._resolve_observation_dispatch(group, name, term, term_index, output_size)

    def _resolve_rewards(
        self,
        cfg: ManagerBasedEnvCfg,
        reward_terms: dict[str, RewardTerm],
    ) -> tuple[tuple[PreparedInvocation, ...], tuple[ScalarTermLayout, ...], tuple[float, ...]]:
        term_cfgs = cfg.reward_cfgs()
        terms = []
        layout = []
        weights = []
        for index, (name, term) in enumerate(reward_terms.items()):
            terms.append(self._resolve_reward(name, term, index))
            layout.append(ScalarTermLayout(name, index))
            try:
                weights.append(float(term_cfgs[name].weight))
            except (TypeError, ValueError) as error:
                raise TypeError(f"Numba reward term reward.{name} weight must be float-compatible.") from error
        return tuple(terms), tuple(layout), tuple(weights)

    def _resolve_reward(self, name: str, term: RewardTerm, term_index: int) -> PreparedInvocation:
        return self._resolve_dispatch_term("reward", name, term, term_index)

    def _resolve_terminations(
        self,
        termination_terms: dict[str, TerminationTerm],
    ) -> tuple[tuple[PreparedInvocation, ...], tuple[ScalarTermLayout, ...]]:
        terms = []
        layout = []
        for index, (name, term) in enumerate(termination_terms.items()):
            terms.append(self._resolve_termination(name, term, index))
            layout.append(ScalarTermLayout(name, index))
        return tuple(terms), tuple(layout)

    def _resolve_dispatch_term(
        self,
        group: str,
        name: str,
        term: Any,
        term_index: int,
    ) -> PreparedInvocation:
        function = term.dispatch
        parameters = tuple(inspect.signature(function).parameters.values())
        if any(
            parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for parameter in parameters
        ):
            raise TypeError(f"Manager {group} {name} dispatch args must be positional parameters.")
        if len(parameters) != 1 + len(term.args):
            raise TypeError(
                f"Manager {group} {name} dispatch must take a ManagerContext parameter followed by "
                f"{len(term.args)} positional args; got {tuple(parameter.name for parameter in parameters)}."
            )
        annotations = get_type_hints(function, include_extras=True)
        if annotations.get(parameters[0].name) is not ManagerContext:
            raise TypeError(
                f"Manager {group} {name} dispatch must annotate its first parameter as ManagerContext "
                f"(conventionally named ctx)."
            )
        expressions = []
        plan_expressions = []
        prepared_indices = []
        warmup_values = []
        for index, value in enumerate(term.args):
            if is_kernel_data(value):
                _, tree_def = flatten_kernel_data(value)
                layout = self._kernel_data_lowering.lower(
                    tree_def, context=f"{group}.{name} args[{index}]", force_shared=True
                )
                prepared_index, expression = self._register_prepared(
                    f"{group}.{name}.args[{index}]", value, type(value), layout
                )
                prepared_indices.append(prepared_index)
                expressions.append(expression)
                plan_expressions.append(expression)
                warmup_values.append(value)
            else:
                expression, warmup, plan_part, prepared_index = self._resolve_plain_arg(
                    f"{group}.{name}.args[{index}]", value
                )
                prepared_indices.append(prepared_index)
                expressions.append(expression)
                plan_expressions.append(plan_part)
                warmup_values.append(warmup)
        dispatcher = self._compile_term(function)
        symbol = f"term_{group}_{term_index}"
        self._term_functions[symbol] = self._generated_term(function, dispatcher)
        self._evaluate_parts.append((group, name, "dispatch", self._function_fingerprint(function), plan_expressions))
        return PreparedInvocation(
            f"{group}.{name}",
            group,
            dispatcher,
            None,
            args_expressions=tuple(expressions),
            args_values=tuple(warmup_values),
            args_prepared_indices=tuple(prepared_indices),
        )

    def _resolve_termination(
        self,
        name: str,
        term: TerminationTerm,
        term_index: int,
    ) -> PreparedInvocation:
        return self._resolve_dispatch_term("termination", name, term, term_index)

    def _resolve_receiver_term(
        self,
        group: str,
        name: str,
        term: Any,
        function_name: str,
        term_index: int,
        output_size: int,
        *,
        function: Callable[..., Any] | None = None,
        receiver_key: str,
    ) -> PreparedInvocation:
        function = getattr(term, function_name) if function is None else function
        has_receiver = receiver_key is not None
        if not inspect.isfunction(function) or (not has_receiver and inspect.ismethod(function)):
            raise TypeError(f"Manager term {group}.{name} {function_name} must be a static function.")
        parameters = tuple(inspect.signature(function).parameters.values())
        if has_receiver:
            if not parameters:
                raise TypeError(f"Manager term {group}.{name} {function_name} must declare a receiver parameter.")
            parameters = parameters[1:]
        if not parameters:
            raise TypeError(f"Manager term {group}.{name} {function_name} must declare a ManagerContext parameter.")
        argument_parameters = parameters[1:]
        annotations = get_type_hints(function, include_extras=True)
        if annotations.get(parameters[0].name) is not ManagerContext:
            raise TypeError(
                f"Manager term {group}.{name} {function_name} must annotate its first parameter after the "
                f"receiver as ManagerContext (conventionally named ctx)."
            )
        if argument_parameters:
            raise TypeError(
                f"Manager term {group}.{name} {function_name} has unsupported dependency parameters "
                f"{[parameter.name for parameter in argument_parameters]}; read manager and simulator data "
                f"from the ManagerContext parameter."
            )
        kind = f"command_{function_name}" if group == "command" else group
        context = f"{group}.{name} ({function.__module__}.{function.__qualname__})"
        symbol = f"term_{kind}_{term_index}"
        dispatcher = self._compile_term(function)
        self._term_functions[symbol] = self._generated_term(function, dispatcher)
        parts = self._reset_parts if function_name == "reset_env" else self._evaluate_parts
        parts.append(
            (
                group,
                name,
                kind,
                function_name,
                self._function_fingerprint(function),
                output_size,
                has_receiver,
            )
        )
        return PreparedInvocation(context, kind, dispatcher, receiver_key, output_size=output_size)

    def _register_prepared(
        self,
        source_name: str,
        value: Any,
        value_type: type[Any],
        layout: KernelDataLayout,
    ) -> tuple[int, str]:
        prepared_index = len(self._prepared_terms)
        input_offset = self._flat_input_count
        self._prepared_terms.append(KernelInputSource.prepare(source_name, value_type, layout, value))
        self._input_offsets.append(input_offset)
        self._flat_input_count += len(iter_layout_leaves(layout))
        for proxy in proxy_types(layout):
            self._prepared_types[proxy_symbol(proxy)] = proxy
        return prepared_index, lane_expression(layout, input_offset)

    def _scalar_buffer_arg(self, source_name: str, value: float) -> tuple[str, np.float32]:
        """Lower one numeric term argument into a runtime scalar buffer slot.

        Returns the kernel-lane argument expression and the normalized warmup
        value. Only the fixed float32 layout enters the plan; the value itself
        is read from the input slot at runtime.
        """
        scalar = np.float32(value)
        wrapped = TermScalarBuffer(scalar)
        _, tree_def = flatten_kernel_data(wrapped)
        layout = self._kernel_data_lowering.lower(
            tree_def,
            context=source_name,
            force_shared=True,
        )
        prepared_index, _ = self._register_prepared(source_name, wrapped, TermScalarBuffer, layout)
        leaf = iter_layout_leaves(layout)[0]
        return f"input_{self._input_offsets[prepared_index] + leaf.slot_index}", scalar

    def _array_buffer_arg(self, source_name: str, value: np.ndarray) -> tuple[str, np.ndarray]:
        """Lower one ndarray term argument into a per-environment input slot.

        Returns the kernel-lane argument expression (the lane row) and the
        warmup value. The array content participates in the plan fingerprint
        via its shape and dtype, and the value is read from the input slot at
        runtime.
        """
        wrapped = TermArrayBuffer(value=value)
        _, tree_def = flatten_kernel_data(wrapped)
        layout = self._kernel_data_lowering.lower(
            tree_def,
            context=source_name,
        )
        prepared_index, _ = self._register_prepared(source_name, wrapped, TermArrayBuffer, layout)
        leaf = iter_layout_leaves(layout)[0]
        return f"input_{self._input_offsets[prepared_index] + leaf.slot_index}", value

    def _resolve_plain_arg(self, source_name: str, value: Any) -> tuple[str | None, Any, str, int | None]:
        """Resolve one non-kernel-data term argument.

        Float-like scalars become runtime scalar buffers; every
        other scalar stays a compile-time constant. Returns the kernel-lane
        argument expression, the warmup value, the plan fingerprint part, and
        the prepared index (``None`` for both plain scalars and buffers).
        """
        if isinstance(value, (float, np.floating)):
            expression, warmup = self._scalar_buffer_arg(source_name, value)
            return expression, warmup, "scalar_buffer(np.float32)", None
        if isinstance(value, np.ndarray):
            expression, warmup = self._array_buffer_arg(source_name, value)
            return expression, warmup, f"array_buffer({value.dtype.str}{value.shape})", None
        if isinstance(value, str):
            # String arguments are inlined as source-level literals. Dispatch
            # bodies keep them literal via ``numba.literally`` so Map lookups
            # like ``ctx.actions[name]`` resolve at compile time.
            return repr(value), value, f"str({value!r})", None
        if isinstance(value, SimDataQuery):
            # A query passed as a term argument resolves to its registered
            # read-program lane view; the dispatch receives the array in the
            # argument's position.
            key = self._env._term_query_keys[value]
            slot = self._sim_slots_by_key[key]
            lane_view = self._sim_inputs[key].value[0]
            return f"input_{slot}[env_id]", lane_view, f"sim_input({key!r})", None
        return repr(value), value, repr(value), None

    def _validate_observation_spaces(self, groups: dict[str, ObservationGroupEntry]) -> None:
        expected_policy = self._env.policy_observation_space.shape
        if expected_policy != (groups["policy"].size,):
            raise ValueError(
                f"Policy observation space has shape {expected_policy}, "
                f"manager config produces {(groups['policy'].size,)}."
            )
        if self._env.policy_observation_space.dtype != np.dtype(np.float32):
            raise TypeError("Manager policy observation space must use float32 dtype.")
        if "value" in groups:
            expected_value = self._env.value_observation_space.shape
            if expected_value != (groups["value"].size,):
                raise ValueError(
                    f"Value observation space has shape {expected_value}, "
                    f"manager config produces {(groups['value'].size,)}."
                )
            if self._env.value_observation_space.dtype != np.dtype(np.float32):
                raise TypeError("Manager value observation space must use float32 dtype.")
        elif self._env.has_value_observation:
            raise ValueError("Environment declares a value observation space but manager config has no 'value' group.")

    # Cache and fingerprint helpers live in sibling modules; they stay bound as
    # staticmethods so callers (and tests) can intercept them on the class.
    _generated_source_path = staticmethod(generated_source_path)
    _materialize_source = staticmethod(materialize_source)
    _has_generated_kernel_cache = staticmethod(has_generated_kernel_cache)
    _invalidate_generated_cache = staticmethod(invalidate_generated_cache)
    _type_name = staticmethod(type_name)
    _function_fingerprint = staticmethod(function_fingerprint)

    def compile_specializations(self, inputs: tuple[Any, ...]) -> None:
        """Compile all term and fused-kernel specializations for the environment."""
        started = perf_counter()
        try:
            self._compile_specializations_once(inputs)
        except (
            AttributeError,
            EOFError,
            ImportError,
            IndexError,
            KeyError,
            OSError,
            pickle.UnpicklingError,
            ValueError,
        ) as error:
            plan_keys = self._env.manager_layout.plan_keys
            logger.warning("Manager specialization cache fallback: plan_keys=%s error=%s", plan_keys, error)
            for plan_key in plan_keys:
                self._invalidate_generated_cache(plan_key)
                _KERNEL_CACHE.pop(plan_key, None)
            invalidate_term_cache()
            self._env._task_program = self.build().task
            self._compile_specializations_once(inputs)
        logger.info(
            "Manager startup %s: term specializations finished in %.3fs",
            self._env_name(),
            perf_counter() - started,
        )

    def precompile_reset_kernel(self, inputs: tuple[Any, ...]) -> None:
        """Compile the reset kernel before the first reset lifecycle call.

        The base ``init_state`` lifecycle resets rows through the reset kernel
        before term specializations run; without this precompile the reset
        kernel compiles lazily inside that first reset. env_ids always come
        from ``np.flatnonzero`` (int64). Cache errors are deferred to the
        specialization phase, whose fallback invalidates and rebuilds.
        """
        task = self._env._task_program
        if task is None:
            return
        reset_args = (clone_kernel_value(inputs), np.arange(2, dtype=np.int64), self._env._sim_reset_runtime.buffers)
        try:
            task.reset_kernel.compile(tuple(numba.typeof(arg) for arg in reset_args))
        except (
            AttributeError,
            EOFError,
            ImportError,
            IndexError,
            KeyError,
            OSError,
            pickle.UnpicklingError,
            ValueError,
        ):
            logger.warning("Manager reset kernel precompile deferred to specializations", exc_info=True)

    def _env_name(self) -> str:
        return type(self._env).__name__

    def _compile_specializations_once(self, inputs: tuple[Any, ...]) -> None:
        task = self._env._task_program
        assert task is not None
        # Terms are not compiled standalone: their dispatch bodies are inlined
        # into the fused kernels below, which type-checks the whole plan in one
        # compilation instead of paying a separate dispatcher compile per term.
        warmup_args = tuple(
            clone_kernel_value(value)
            for value in (inputs, task.reward_weights, self._env._kernel_buffers, self._env._kernel_outputs)
        )
        task.evaluate_kernel.compile(tuple(numba.typeof(arg) for arg in warmup_args))
        observe_args = (clone_kernel_value(inputs), clone_kernel_value(self._env._kernel_outputs))
        task.observe_kernel.compile(tuple(numba.typeof(arg) for arg in observe_args))
        # Precompile the reset kernel here instead of paying its compilation on
        # the first reset; env_ids always come from np.flatnonzero (int64).
        reset_args = (clone_kernel_value(inputs), np.arange(2, dtype=np.int64), self._env._sim_reset_runtime.buffers)
        task.reset_kernel.compile(tuple(numba.typeof(arg) for arg in reset_args))

    def _load_kernel(self, kind: str, source: str, plan_key: str) -> tuple[Any, str]:
        """Load one fused kernel from the in-process or disk cache, else compile it."""
        cached = _KERNEL_CACHE.get(plan_key)
        if cached is not None:
            logger.info(
                "Manager startup %s: %s kernel reused (cache=in-process, plan_key=%.12s)",
                self._env_name(),
                kind,
                plan_key,
            )
            return cached, NumbaKernelCompiler._generated_source_path(plan_key)
        disk_cache = self._has_generated_kernel_cache(plan_key)
        if disk_cache:
            logger.info(
                "Manager startup %s: %s kernel cache found, loading (cache=disk, plan_key=%.12s)",
                self._env_name(),
                kind,
                plan_key,
            )
        else:
            logger.info(
                "Manager startup %s: no cache for this %s kernel, first compile may take a while "
                "(cache=miss, plan_key=%.12s)",
                self._env_name(),
                kind,
                plan_key,
            )
        compile_started = perf_counter()
        filename = self._materialize_source(source, plan_key)
        try:
            kernel = self._compile_kernel(kind, source, filename)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("Manager kernel cache fallback: plan_key=%s error=%s", plan_key, error)
            self._invalidate_generated_cache(plan_key)
            filename = self._materialize_source(source, plan_key)
            kernel = self._compile_kernel(kind, source, filename)
        _KERNEL_CACHE[plan_key] = kernel
        logger.info(
            "Manager startup %s: %s kernel compile finished in %.3fs (cache=%s)",
            self._env_name(),
            kind,
            perf_counter() - compile_started,
            "disk" if disk_cache else "miss",
        )
        return kernel, filename

    def _compile_kernel(self, kind: str, source: str, filename: str) -> Any:
        module_name = f"motrixlab_generated_{kind}_{Path(filename).stem}"
        spec = importlib.util.spec_from_file_location(module_name, filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load generated Manager module from {filename!r}.")
        module = importlib.util.module_from_spec(spec)
        module.__dict__.update({"numba": numba, "np": np, **self._term_functions, **self._prepared_types})
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        options = {"cache": True, "nogil": True, "parallel": True}
        return numba.njit(**options)(getattr(module, f"generated_{kind}_kernel"))

    @staticmethod
    def _generated_term(function: Callable[..., Any], dispatcher: Any) -> Any:
        module_name = function.__module__ or ""
        if "." in module_name and not module_name.startswith("test") and "<locals>" not in function.__qualname__:
            return register_jitable(function)
        return dispatcher

    def _compile_term(self, function: Callable[..., Any]) -> Any:
        if not is_dispatched(function):
            raise TypeError(f"Manager kernel entry {function.__qualname__!r} must be decorated with @dispatch.")
        dispatcher = _TERM_CACHE.get(function)
        if dispatcher is None:
            # Functions declared inside a caller (notably test fixtures) do not
            # have a stable importable locator, so their cache entries cannot be
            # restored safely in another interpreter.
            module_name = function.__module__ or ""
            cache = (
                "." in module_name and not module_name.startswith("test") and "<locals>" not in function.__qualname__
            )
            dispatcher = numba.njit(cache=cache, nogil=True, inline="always")(function)
            _TERM_CACHE[function] = dispatcher
        return dispatcher


__all__ = ["NumbaKernelCompiler"]
