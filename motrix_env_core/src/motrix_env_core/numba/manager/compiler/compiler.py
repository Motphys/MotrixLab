# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import inspect
import logging
import os
import pickle
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, get_args, get_origin, get_type_hints

import numba
import numpy as np
from numba.extending import register_jitable

from motrix_env_core.numba.kernel import clone_kernel_value
from motrix_env_core.numba.kernel_data import (
    KernelDataLayout,
    KernelDataLowering,
    Map,
    flatten_kernel_data,
    is_kernel_data,
    iter_layout_leaves,
    kernel_data,
    lane_expression,
    map_proxy,
    proxy_symbol,
    proxy_types,
)
from motrix_env_core.numba.manager.compiler.codegen import KernelSourceGenerator
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

_SCHEMA_VERSION = 33
_KERNEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}
logger = logging.getLogger(__name__)
_TERM_CACHE: dict[Callable[..., Any], Any] = {}


def _invalidate_term_cache() -> None:
    for dispatcher in _TERM_CACHE.values():
        try:
            dispatcher._cache.flush()
        except (AttributeError, OSError):
            logger.debug("Unable to remove invalid manager term cache", exc_info=True)
    _TERM_CACHE.clear()


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
        self._prepared_terms: list[KernelInputSource] = []
        self._input_offsets: list[int] = []
        self._flat_input_count = 0
        self._plan_parts: list[Any] = [_SCHEMA_VERSION, numba.__version__]
        self._term_functions: dict[str, Any] = {}
        self._prepared_types: dict[str, type[tuple]] = {}
        self._kernel_data_lowering = KernelDataLowering()
        self._plan_parts.append(("sim_inputs", sim_input_fingerprint))

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
        plan_key = hashlib.sha256(repr(tuple(self._plan_parts)).encode()).hexdigest()
        logger.info(
            "Manager startup %s: build plan finished in %.3fs (plan_key=%.12s)",
            self._env_name(),
            perf_counter() - build_started,
            plan_key,
        )
        source = KernelSourceGenerator(self._flat_input_count).generate(
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
        generated_filename = self._materialize_source(source, plan_key)
        kernels = _KERNEL_CACHE.get(plan_key)
        if kernels is not None:
            logger.info(
                "Manager startup %s: kernels reused (cache=in-process, plan_key=%.12s)",
                self._env_name(),
                plan_key,
            )
        else:
            disk_cache = self._has_generated_kernel_cache(plan_key)
            if disk_cache:
                logger.info(
                    "Manager startup %s: compiled kernel cache found, loading (cache=disk, plan_key=%.12s)",
                    self._env_name(),
                    plan_key,
                )
            else:
                logger.info(
                    "Manager startup %s: no cache for this plan, first compile may take a while "
                    "(cache=miss, plan_key=%.12s)",
                    self._env_name(),
                    plan_key,
                )
            compile_started = perf_counter()
            try:
                kernels = self._compile_kernels(source, generated_filename)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                logger.warning("Manager kernel cache fallback: plan_key=%s error=%s", plan_key, error)
                self._invalidate_generated_cache(plan_key)
                generated_filename = self._materialize_source(source, plan_key)
                kernels = self._compile_kernels(source, generated_filename)
            _KERNEL_CACHE[plan_key] = kernels
            logger.info(
                "Manager startup %s: compile kernels finished in %.3fs (cache=%s)",
                self._env_name(),
                perf_counter() - compile_started,
                "disk" if disk_cache else "miss",
            )
        evaluate_kernel, observe_kernel, reset_kernel = kernels

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
            plan_key=plan_key,
            generated_filename=generated_filename,
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
            source=source,
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
        self._plan_parts.append(
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
        context_index = next(
            index for index, source in enumerate(self._prepared_terms) if source.value_type is ManagerContext
        )
        context_source = self._prepared_terms[context_index]
        context_offset = self._input_offsets[context_index]
        slots_by_key = {
            field.path[1]: context_offset + field_index
            for field_index, field in enumerate(context_source.fields)
            if len(field.path) == 2 and field.path[0] == "sim"
        }
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
            _, tree_def = flatten_kernel_data(action_term)
            self._plan_parts.append(
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
            self._plan_parts.append(
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
        actions = dict(self._env.action_terms)
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
        self._plan_parts.append(("manager_context", layout.fingerprint))
        self._plan_parts.append(
            (
                "metrics",
                tuple((name, value.shape, value.dtype.str) for name, value in self._env.metrics.items()),
            )
        )
        return ResolvedManagerContext(prepared_index, expression)

    def _resolve_command_hooks(self, function_name: str) -> tuple[PreparedInvocation, ...]:
        hooks = []
        for index, (name, command_term) in enumerate(self._env.command_terms.items()):
            function = inspect.getattr_static(type(command_term), function_name)
            hooks.append(
                self._resolve_receiver_term(
                    "command", name, command_term, function_name, index, 0, function=function, receiver_key=name
                )
            )
        return tuple(hooks)

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
        self._plan_parts.append(
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
        self._plan_parts.append((group, name, "dispatch", self._function_fingerprint(function), plan_expressions))
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
        self._plan_parts.append(
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

    @staticmethod
    def _materialize_source(source: str, plan_key: str) -> str:
        cache_dir = NumbaKernelCompiler._generated_cache_dir()
        path = Path(cache_dir) / "generated" / f"{plan_key}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding="utf-8") != source:
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
                ) as temporary:
                    temporary.write(source)
                    temporary_name = temporary.name
                os.replace(temporary_name, path)
            finally:
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)
        return str(path)

    @staticmethod
    def _generated_cache_dir() -> str:
        cache_dir = os.environ.get("NUMBA_CACHE_DIR")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "motrixlab", "numba-manager")
        return cache_dir

    @staticmethod
    def _generated_cache_paths(plan_key: str) -> tuple[Path, ...]:
        """Return compiled-kernel cache files for one plan key.

        Numba stores the cached specializations of the generated module next to
        its cache locator: under ``generated_<hash>/`` when ``NUMBA_CACHE_DIR``
        is set, and under ``generated/__pycache__/`` (beside the generated
        source) when it is not. Both layouts are covered so probe and
        invalidation agree with where the files actually live.
        """
        cache_dir = Path(NumbaKernelCompiler._generated_cache_dir())
        paths: list[Path] = []
        for pattern in (f"generated_*/*{plan_key}*", f"generated/__pycache__/*{plan_key}*"):
            try:
                paths.extend(cache_dir.glob(pattern))
            except OSError:
                logger.debug("Unable to scan generated kernel cache: %s", cache_dir, exc_info=True)
        return tuple(paths)

    @staticmethod
    def _has_generated_kernel_cache(plan_key: str) -> bool:
        """Return whether compiled-kernel cache files exist for one plan key."""
        return any(path.suffix in {".nbi", ".nbc"} for path in NumbaKernelCompiler._generated_cache_paths(plan_key))

    @staticmethod
    def _invalidate_generated_cache(plan_key: str) -> None:
        cache_dir = NumbaKernelCompiler._generated_cache_dir()
        source_path = Path(cache_dir) / "generated" / f"{plan_key}.py"
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Unable to remove invalid generated source: %s", source_path, exc_info=True)
        for cache_path in NumbaKernelCompiler._generated_cache_paths(plan_key):
            try:
                cache_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Unable to remove invalid generated cache: %s", cache_path, exc_info=True)

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
            plan_key = self._env.manager_layout.plan_key
            logger.warning("Manager specialization cache fallback: plan_key=%s error=%s", plan_key, error)
            self._invalidate_generated_cache(plan_key)
            _invalidate_term_cache()
            _KERNEL_CACHE.pop(plan_key, None)
            self._env._task_program = self.build().task
            self._compile_specializations_once(inputs)
        logger.info(
            "Manager startup %s: term specializations finished in %.3fs",
            self._env_name(),
            perf_counter() - started,
        )

    def _env_name(self) -> str:
        return type(self._env).__name__

    def _compile_specializations_once(self, inputs: tuple[Any, ...]) -> None:
        program = self._env._compiled_manager_program
        task = self._env._task_program
        assert program is not None
        assert task is not None
        assert self._env._kernel_buffers is not None
        assert self._env._state is not None
        program.warmup_terms(self._env, self._env._state, self._env._kernel_buffers)
        warmup_args = tuple(
            clone_kernel_value(value)
            for value in (inputs, task.reward_weights, self._env._kernel_buffers, self._env._kernel_outputs)
        )
        task.evaluate_kernel.compile(tuple(numba.typeof(arg) for arg in warmup_args))
        observe_args = (clone_kernel_value(inputs), clone_kernel_value(self._env._kernel_outputs))
        task.observe_kernel.compile(tuple(numba.typeof(arg) for arg in observe_args))

    def _compile_kernels(self, source: str, filename: str) -> tuple[Any, Any, Any]:

        module_name = f"motrixlab_generated_{Path(filename).stem}"
        spec = importlib.util.spec_from_file_location(module_name, filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load generated Manager module from {filename!r}.")
        module = importlib.util.module_from_spec(spec)
        module.__dict__.update({"numba": numba, "np": np, **self._term_functions, **self._prepared_types})
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        options = {"cache": True, "nogil": True, "parallel": True}
        return (
            numba.njit(**options)(module.generated_evaluate_kernel),
            numba.njit(**options)(module.generated_observe_kernel),
            numba.njit(**options)(module.generated_reset_kernel),
        )

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

    @staticmethod
    def _type_name(value_type: type[Any]) -> str:
        return f"{value_type.__module__}.{value_type.__qualname__}"

    @staticmethod
    def _function_fingerprint(function: Callable[..., Any]) -> str:
        try:
            source = inspect.getsource(function).encode()
        except (OSError, TypeError):
            source = function.__code__.co_code
        identity = f"{function.__module__}.{function.__qualname__}".encode()
        return hashlib.sha256(identity + b"\0" + source).hexdigest()


__all__ = ["NumbaKernelCompiler"]
