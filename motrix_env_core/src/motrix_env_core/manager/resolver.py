# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral resolver for manager task programs.

Lowers one constructed manager environment into its execution metadata: the
simulator read plan, the flat kernel-input layout, prepared term arguments,
and one invocation per manager term (command hooks, observations, rewards,
terminations, simulator resets). Numerical executors — currently the Torch
executor in :mod:`motrix_env_core.torch_manager` — consume this metadata and
schedule the invocations; no code generation or dispatcher compilation
happens here.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np
import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.dispatch import is_dispatched
from motrix_env_core.manager.env import (
    KernelInputSource,
    ManagerEnv,
    ReadPlan,
)
from motrix_env_core.manager.kernel_data import (
    KernelDataLayout,
    KernelDataLowering,
    KernelDataScope,
    Map,
    SharedArray,
    flatten_kernel_data,
    is_kernel_data,
    iter_layout_leaves,
    kernel_data,
    lane_expression,
)
from motrix_env_core.manager.observations import ObservationGroupEntry, ObsTerm
from motrix_env_core.manager.rewards import RewardTerm
from motrix_env_core.manager.terminations import TerminationTerm
from motrix_env_core.sim import PhysicsReadProgram, SimDataQuery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolved layouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputSlotLayout:
    index: int
    source: str
    scope: KernelDataScope


@dataclass(frozen=True)
class SimInputLayout:
    slot: int
    key: str
    shape: tuple[int, ...]
    consumers: tuple[str, ...]


@dataclass(frozen=True)
class ActionTermLayout:
    name: str
    input_slice: slice
    actuator_names: tuple[str, ...] | None


@dataclass(frozen=True)
class ObservationTermLayout:
    name: str
    output_slice: slice


@dataclass(frozen=True)
class ObservationGroupLayout:
    name: str
    terms: tuple[ObservationTermLayout, ...]
    size: int


@dataclass(frozen=True)
class ScalarTermLayout:
    name: str
    index: int


@dataclass(frozen=True)
class MetricFieldLayout:
    name: str
    index: int
    dtype: np.dtype


@dataclass(frozen=True)
class ManagerLayout:
    inputs: tuple[InputSlotLayout, ...]
    sim_inputs: tuple[SimInputLayout, ...]
    actions: tuple[ActionTermLayout, ...]
    observations: dict[str, ObservationGroupLayout]
    rewards: tuple[ScalarTermLayout, ...]
    terminations: tuple[ScalarTermLayout, ...]
    metrics: tuple[MetricFieldLayout, ...]

    def dump(self) -> str:
        lines = [
            "inputs:",
        ]
        lines.extend(f"  [{slot.index}] scope={slot.scope.value} source={slot.source}" for slot in self.inputs)
        lines.append("sim_inputs:")
        lines.extend(
            f"  [{sim_input.slot}] {sim_input.key}: shape={sim_input.shape} consumers={sim_input.consumers}"
            for sim_input in self.sim_inputs
        )
        lines.append("actions:")
        lines.extend(
            f"  {term.name}: input=[{term.input_slice.start}:{term.input_slice.stop}] actuators={term.actuator_names}"
            for term in self.actions
        )
        lines.append("observations:")
        lines.extend(
            f"  {group.name}.{term.name}: [{term.output_slice.start}:{term.output_slice.stop}]"
            for group in self.observations.values()
            for term in group.terms
        )
        lines.append("rewards:")
        lines.extend(f"  [{term.index}] {term.name}" for term in self.rewards)
        lines.append("terminations:")
        lines.extend(f"  [{term.index}] {term.name}" for term in self.terminations)
        lines.append("metrics:")
        lines.extend(f"  [{metric.index}] dtype={metric.dtype.name} {metric.name}" for metric in self.metrics)
        return "\n".join(lines)


@dataclass(frozen=True)
class PreparedInvocation:
    """Runtime metadata required to execute one manager term."""

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
class ResolvedManagerProgram:
    """Resolved execution metadata for one manager environment."""

    read_plan: ReadPlan
    layout: ManagerLayout
    invocations: tuple[PreparedInvocation, ...]
    prepared_terms: tuple[KernelInputSource, ...]
    input_offsets: tuple[int, ...]
    context: ResolvedManagerContext


__all__ = [
    "ActionTermLayout",
    "InputSlotLayout",
    "ManagerLayout",
    "ManagerProgramResolver",
    "MetricFieldLayout",
    "ObservationGroupLayout",
    "ObservationTermLayout",
    "PreparedInvocation",
    "ResolvedManagerContext",
    "ResolvedManagerProgram",
    "ResolvedSimReset",
    "ScalarTermLayout",
    "SimInputLayout",
]


# ---------------------------------------------------------------------------
# Term-argument buffer wrappers
# ---------------------------------------------------------------------------


@kernel_data
class TermArrayBuffer:
    """Shared array slot for one ndarray term argument.

    Arrays passed directly as manager term arguments are lowered with shared
    scope (every execution lane sees the full array): they hold env-invariant
    reference data such as default poses, weights, or lookup tables.
    """

    value: SharedArray


@kernel_data
class TermScalarBuffer:
    """Runtime buffer slot for one numeric term argument.

    Numeric tuning arguments (thresholds, scales, gains, ...) are lowered
    into a fixed-dtype shared input slot so executors read the value at
    runtime instead of baking it into a compiled artifact.
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


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class ManagerProgramResolver:
    """Resolve one fixed manager config into executor-neutral metadata."""

    def __init__(self, env: ManagerEnv):
        self._env = env
        self._sim_inputs: Mapping[str, ManagerSimInput]
        self._sim_inputs, sim_input_fingerprint = _manager_sim_inputs(env.sim_data)
        self._sim_slots_by_key: dict[str, int] = {}
        self._prepared_terms: list[KernelInputSource] = []
        self._input_offsets: list[int] = []
        self._flat_input_count = 0
        self._kernel_data_lowering = KernelDataLowering()
        self._layout_identity: list[Any] = [sim_input_fingerprint]

    def build(self) -> ResolvedManagerProgram:
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
        reward_invocations, reward_layout, _ = self._resolve_rewards(self._env.cfg, reward_terms)
        termination_invocations, termination_layout = self._resolve_terminations(termination_terms)
        reset_invocations = self._resolve_resets()
        per_env_metric_layout = tuple(
            MetricFieldLayout(name, index, np.dtype(value.dtype))
            for index, (name, value) in enumerate(self._env.metrics.items())
            if value.shape in {(self._env.num_envs,), (self._env.num_envs, 1)}
        )
        self._validate_observation_spaces(observation_groups)

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
        )
        invocations = (
            command_updates
            + command_advances
            + command_resets
            + tuple(term for terms in observation_terms.values() for term in terms)
            + reward_invocations
            + termination_invocations
            + tuple(reset.invocation for reset in reset_invocations)
        )
        logger.info(
            "Manager startup %s: resolve finished in %.3fs (%d invocations, %d inputs)",
            type(self._env).__name__,
            perf_counter() - build_started,
            len(invocations),
            self._flat_input_count,
        )
        return ResolvedManagerProgram(
            read_plan=ReadPlan(
                self._env.sim_data,
                tuple(self._prepared_terms),
                tuple(value for prepared in self._prepared_terms for value in prepared.values),
            ),
            layout=layout,
            invocations=invocations,
            prepared_terms=tuple(self._prepared_terms),
            input_offsets=tuple(self._input_offsets),
            context=context,
        )

    # -- resets -------------------------------------------------------------

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
                f"Map[torch.Tensor] sim-writes parameter, and {len(term.args)} positional args; got "
                f"{tuple(parameter.name for parameter in parameters)}."
            )
        annotations = get_type_hints(function, include_extras=True)
        if annotations.get(parameters[0].name) is not ManagerContext:
            raise TypeError(
                f"Manager simulator reset term {name!r} dispatch must annotate its first parameter as "
                f"ManagerContext (conventionally named ctx)."
            )
        sim_writes_annotation = annotations.get(parameters[1].name)
        if get_origin(sim_writes_annotation) is not Map or get_args(sim_writes_annotation) != (torch.Tensor,):
            raise TypeError(
                f"Manager simulator reset term {name!r} dispatch must annotate its second parameter as "
                f"Map[torch.Tensor] (conventionally named sim_writes)."
            )
        dispatcher = self._term_entry(function)
        expressions = []
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
            else:
                expression, _, prepared_index = self._resolve_plain_arg(f"sim_reset.{name}.args[{index}]", value)
                prepared_indices.append(prepared_index)
                expressions.append(expression)
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
            f"sim_reset_{name}_writes",
            output_offset,
            len(fields),
        )

    # -- layouts ------------------------------------------------------------

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
            self._layout_identity.append(
                (
                    "action",
                    name,
                    type(action_term).__qualname__,
                    tree_def.fingerprint,
                    self._env.action_slices[name],
                    None
                    if self._env.action_actuators[name] is None
                    else tuple(spec.name for spec in self._env.action_actuators[name]),
                )
            )
        for name, command_term in self._env.command_terms.items():
            _, tree_def = flatten_kernel_data(command_term)
            self._layout_identity.append(
                (
                    "command",
                    name,
                    type(command_term).__qualname__,
                    tree_def.fingerprint,
                )
            )

    def _resolve_manager_context(
        self,
        observation_groups: dict[str, ObservationGroupEntry],
        reward_terms: dict[str, RewardTerm],
        termination_terms: dict[str, TerminationTerm],
    ) -> ResolvedManagerContext:
        del observation_groups, reward_terms, termination_terms
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
        self._layout_identity.append(("manager_context", layout.fingerprint))
        self._layout_identity.append(
            (
                "metrics",
                tuple((name, value.shape, value.dtype.str) for name, value in self._env.metrics.items()),
            )
        )
        # Map each declared sim key to its global input slot inside the
        # flattened context, so term args naming a sim key resolve to the
        # key's batched view instead of crossing the executor as a string.
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
            if isinstance(function, staticmethod):
                function = function.__func__
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

    # -- observations --------------------------------------------------------

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
                raise RuntimeError(f"Observation group {group_name!r} resolved width changed during resolution.")
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
                f"torch.Tensor output parameter, and {len(term.args)} positional args; got "
                f"{tuple(parameter.name for parameter in parameters)}."
            )
        annotations = get_type_hints(function, include_extras=True)
        if (
            annotations.get(parameters[0].name) is not ManagerContext
            or annotations.get(parameters[1].name) is not torch.Tensor
        ):
            raise TypeError(
                f"Manager observation {group}.{name} dispatch must annotate its first two parameters as "
                f"ManagerContext and torch.Tensor (conventionally named ctx and out)."
            )
        expressions = []
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
                warmup_values.append(value)
            else:
                expression, warmup, prepared_index = self._resolve_plain_arg(
                    f"observation.{group}.{name}.args[{index}]", value
                )
                prepared_indices.append(prepared_index)
                expressions.append(expression)
                warmup_values.append(warmup)
        expressions = tuple(expressions)
        dispatcher = self._term_entry(function)
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

    # -- rewards and terminations --------------------------------------------

    def _resolve_rewards(
        self,
        cfg: Any,
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
                raise TypeError(f"Manager reward term reward.{name} weight must be float-compatible.") from error
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
                warmup_values.append(value)
            else:
                expression, warmup, prepared_index = self._resolve_plain_arg(f"{group}.{name}.args[{index}]", value)
                prepared_indices.append(prepared_index)
                expressions.append(expression)
                warmup_values.append(warmup)
        dispatcher = self._term_entry(function)
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

    # -- command hooks -------------------------------------------------------

    def _resolve_receiver_term(
        self,
        group: str,
        name: str,
        term: Any,
        function_name: str,
        term_index: int,
        output_size: int,
        *,
        function: Any | None = None,
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
                "from the ManagerContext parameter."
            )
        kind = f"command_{function_name}" if group == "command" else group
        context = f"{group}.{name} ({function.__module__}.{function.__qualname__})"
        dispatcher = self._term_entry(function)
        return PreparedInvocation(context, kind, dispatcher, receiver_key, output_size=output_size)

    # -- input preparation ----------------------------------------------------

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
        return prepared_index, lane_expression(layout, input_offset)

    def _scalar_buffer_arg(self, source_name: str, value: float) -> tuple[str, np.float32]:
        """Lower one numeric term argument into a runtime scalar buffer slot.

        Returns the lane argument expression and the normalized warmup value.
        Only the fixed float32 layout enters the resolved metadata; the value
        itself is read from the input slot at runtime.
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

        Returns the lane argument expression (the lane row) and the warmup
        value.
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

    def _resolve_plain_arg(self, source_name: str, value: Any) -> tuple[str | None, Any, int | None]:
        """Resolve one non-kernel-data term argument.

        Float-like scalars become runtime scalar buffers; every other scalar
        stays a plain constant. Returns the lane argument expression, the
        warmup value, and the prepared index (``None`` for both plain scalars
        and buffers).
        """
        if isinstance(value, (float, np.floating)):
            expression, warmup = self._scalar_buffer_arg(source_name, value)
            return expression, warmup, None
        if isinstance(value, np.ndarray):
            expression, warmup = self._array_buffer_arg(source_name, value)
            return expression, warmup, None
        if isinstance(value, torch.Tensor):
            return f"torch.Tensor({value.dtype}{tuple(value.shape)})", value, None
        if isinstance(value, str):
            return repr(value), value, None
        if isinstance(value, SimDataQuery):
            # A query passed as a term argument resolves to its registered
            # read-program batched view; the dispatch receives the array in
            # the argument's position.
            key = self._env._term_query_keys[value]
            slot = self._sim_slots_by_key[key]
            return f"input_{slot}[env_id]", self._sim_inputs[key].value[0], None
        return repr(value), value, None

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

    def _term_entry(self, function: Any) -> Any:
        """Validate one dispatch entry and keep it as the Python callable."""
        if not is_dispatched(function):
            raise TypeError(f"Manager kernel entry {function.__qualname__!r} must be decorated with @dispatch.")
        return function
