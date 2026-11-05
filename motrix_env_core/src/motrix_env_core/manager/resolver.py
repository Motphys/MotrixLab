# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral resolver for manager task programs.

Lowers one constructed manager environment into its execution metadata: the
simulator read plan, the resolved context/term-argument trees (path-keyed
leaf descriptors with live value references and data scopes), prepared term
arguments, and one invocation per manager term (command hooks, observations,
rewards, terminations, simulator resets). Numerical executors — currently the
Torch executor in :mod:`motrix_env_core.torch_manager` — consume this
metadata and schedule the invocations; no code generation or dispatcher
compilation happens here.
"""

from __future__ import annotations

import hashlib
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
from motrix_env_core.manager.env import ManagerEnv, ReadPlan
from motrix_env_core.manager.kernel_data import (
    KernelDataLayout,
    KernelDataLowering,
    KernelDataScope,
    Map,
    ResolvedLeaf,
    ResolvedTree,
    flatten_kernel_data,
    is_kernel_data,
    iter_layout_leaves,
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
class SimInputLayout:
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
    sim_inputs: tuple[SimInputLayout, ...]
    actions: tuple[ActionTermLayout, ...]
    observations: dict[str, ObservationGroupLayout]
    rewards: tuple[ScalarTermLayout, ...]
    terminations: tuple[ScalarTermLayout, ...]
    metrics: tuple[MetricFieldLayout, ...]

    def dump(self) -> str:
        lines = [
            "sim_inputs:",
        ]
        lines.extend(
            f"  {sim_input.key}: shape={sim_input.shape} consumers={sim_input.consumers}"
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


# ---------------------------------------------------------------------------
# Prepared arguments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantArg:
    """Literal term argument passed through unchanged (strings, tensors...)."""

    value: Any


@dataclass(frozen=True)
class TreeArg:
    """Term argument resolved as a structured kernel-data tree."""

    tree_index: int


@dataclass(frozen=True)
class LeafArg:
    """Term argument resolved as one leaf value (numeric/ndarray argument)."""

    leaf: ResolvedLeaf


@dataclass(frozen=True)
class SimArg:
    """Term argument resolving to one sim read-program batched array."""

    key: str


PreparedArg = ConstantArg | TreeArg | LeafArg | SimArg


@dataclass(frozen=True)
class PreparedInvocation:
    """Runtime metadata required to execute one manager term."""

    context: str
    kind: str
    dispatcher: Any
    receiver_key: str | None
    args: tuple[PreparedArg, ...] = ()
    output_size: int = 0


@dataclass(frozen=True)
class ResolvedManagerContext:
    tree_index: int


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
    prepared_trees: tuple[ResolvedTree, ...]
    context: ResolvedManagerContext
    fingerprint: str


__all__ = [
    "ActionTermLayout",
    "ConstantArg",
    "LeafArg",
    "ManagerLayout",
    "ManagerProgramResolver",
    "MetricFieldLayout",
    "ObservationGroupLayout",
    "ObservationTermLayout",
    "PreparedArg",
    "PreparedInvocation",
    "ResolvedManagerContext",
    "ResolvedManagerProgram",
    "ResolvedSimReset",
    "ScalarTermLayout",
    "SimArg",
    "SimInputLayout",
    "TreeArg",
]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


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


def _tree_leaf_fingerprint(tree: ResolvedTree) -> tuple[Any, ...]:
    """Per-leaf content fingerprint: scope plus array shape/dtype identity."""
    return tuple(
        (
            path,
            leaf.scope.value,
            None
            if not isinstance(leaf.value, np.ndarray)
            else (leaf.value.shape, leaf.value.dtype.str, leaf.value.strides),
        )
        for path, leaf in tree.leaves.items()
    )


class ManagerProgramResolver:
    """Resolve one fixed manager config into executor-neutral metadata."""

    def __init__(self, env: ManagerEnv):
        self._env = env
        self._sim_inputs: Mapping[str, ManagerSimInput]
        self._sim_inputs, sim_input_fingerprint = _manager_sim_inputs(env.sim_data)
        self._prepared_trees: list[ResolvedTree] = []
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
            sim_inputs=tuple(
                SimInputLayout(
                    key=key,
                    shape=binding.value.shape[1:],
                    consumers=("ManagerContext.sim",),
                )
                for key, binding in self._sim_inputs.items()
            ),
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
        leaf_count = sum(len(tree.leaves) for tree in self._prepared_trees)
        logger.info(
            "Manager startup %s: resolve finished in %.3fs (%d invocations, %d resolved leaves)",
            type(self._env).__name__,
            perf_counter() - build_started,
            len(invocations),
            leaf_count,
        )
        context_tree = self._prepared_trees[context.tree_index]
        return ResolvedManagerProgram(
            read_plan=ReadPlan(self._env.sim_data, context_tree),
            layout=layout,
            invocations=invocations,
            prepared_trees=tuple(self._prepared_trees),
            context=context,
            fingerprint=hashlib.sha256(repr(tuple(self._layout_identity)).encode()).hexdigest(),
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
                "ManagerContext (conventionally named ctx)."
            )
        sim_writes_annotation = annotations.get(parameters[1].name)
        if get_origin(sim_writes_annotation) is not Map or get_args(sim_writes_annotation) != (torch.Tensor,):
            raise TypeError(
                f"Manager simulator reset term {name!r} dispatch must annotate its second parameter as "
                "Map[torch.Tensor] (conventionally named sim_writes)."
            )
        dispatcher = function
        args = tuple(
            self._resolve_term_arg(f"sim_reset.{name}.args[{index}]", value) for index, value in enumerate(term.args)
        )
        invocation = PreparedInvocation(
            "sim_reset." + name,
            "reset",
            dispatcher,
            None,
            args=args,
        )
        return ResolvedSimReset(
            invocation,
            f"sim_reset_{name}_writes",
            output_offset,
            len(fields),
        )

    # -- layouts ------------------------------------------------------------

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
        tree_index = self._register_tree("manager_context", value, layout)
        tree = self._prepared_trees[tree_index]
        self._layout_identity.append(("manager_context", layout.fingerprint, _tree_leaf_fingerprint(tree)))
        self._layout_identity.append(
            (
                "metrics",
                tuple((name, value.shape, value.dtype.str) for name, value in self._env.metrics.items()),
            )
        )
        return ResolvedManagerContext(tree_index)

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
                "torch.Tensor output parameter, and "
                f"{len(term.args)} positional args; got "
                f"{tuple(parameter.name for parameter in parameters)}."
            )
        annotations = get_type_hints(function, include_extras=True)
        if (
            annotations.get(parameters[0].name) is not ManagerContext
            or annotations.get(parameters[1].name) is not torch.Tensor
        ):
            raise TypeError(
                f"Manager observation {group}.{name} dispatch must annotate its first two parameters as "
                "ManagerContext and torch.Tensor (conventionally named ctx and out)."
            )
        args = tuple(
            self._resolve_term_arg(f"observation.{group}.{name}.args[{index}]", value)
            for index, value in enumerate(term.args)
        )
        dispatcher = function
        return PreparedInvocation(
            f"{group}.{name}",
            "observation",
            dispatcher,
            None,
            args=args,
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
        args = tuple(
            self._resolve_term_arg(f"{group}.{name}.args[{index}]", value) for index, value in enumerate(term.args)
        )
        dispatcher = function
        return PreparedInvocation(
            f"{group}.{name}",
            group,
            dispatcher,
            None,
            args=args,
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
                "receiver as ManagerContext (conventionally named ctx)."
            )
        if argument_parameters:
            raise TypeError(
                f"Manager term {group}.{name} {function_name} has unsupported dependency parameters "
                f"{[parameter.name for parameter in argument_parameters]}; read manager and simulator data "
                "from the ManagerContext parameter."
            )
        kind = f"command_{function_name}" if group == "command" else group
        context = f"{group}.{name} ({function.__module__}.{function.__qualname__})"
        dispatcher = function
        return PreparedInvocation(context, kind, dispatcher, receiver_key, output_size=output_size)

    # -- argument resolution ---------------------------------------------------

    def _resolve_term_arg(self, source_name: str, value: Any) -> PreparedArg:
        """Resolve one manager term argument into a prepared descriptor.

        KernelData arguments become structured trees; float-like scalars and
        ndarrays become shared leaf slots read by reference; simulator queries
        resolve to their read-program batched array; every other value is a
        literal constant.
        """
        if is_kernel_data(value):
            _, tree_def = flatten_kernel_data(value)
            layout = self._kernel_data_lowering.lower(
                tree_def,
                context=source_name,
                force_shared=True,
            )
            return TreeArg(self._register_tree(source_name, value, layout))
        if isinstance(value, (float, np.floating)):
            return LeafArg(ResolvedLeaf(np.float32(value), KernelDataScope.SHARED))
        if isinstance(value, np.ndarray):
            return LeafArg(ResolvedLeaf(value, KernelDataScope.SHARED))
        if isinstance(value, SimDataQuery):
            # A query passed as a term argument resolves to its registered
            # read-program batched view; the dispatch receives the array in
            # the argument's position.
            return SimArg(self._env._term_query_keys[value])
        return ConstantArg(value)

    def _register_tree(
        self,
        source_name: str,
        value: Any,
        layout: KernelDataLayout,
    ) -> int:
        """Validate one kernel-data value against its layout and record its tree.

        Leaves are captured by reference: array leaves are refreshed in place
        by their owning runtime objects and scalar/shared leaves are reused
        unchanged.
        """
        values, tree_def = flatten_kernel_data(value)
        if tree_def.fingerprint != layout.tree_def.fingerprint:
            raise TypeError(f"{source_name} value has a different KernelData schema.")
        leaves: dict[tuple[str, ...], ResolvedLeaf] = {}
        for prepared_field, field_value in zip(iter_layout_leaves(layout), values, strict=True):
            if prepared_field.value_type is np.ndarray and not isinstance(field_value, np.ndarray):
                raise TypeError(f"{source_name} field {'.'.join(prepared_field.path)!r} must be np.ndarray.")
            if prepared_field.value_type is not np.ndarray and type(field_value) is not prepared_field.value_type:
                raise TypeError(
                    f"{source_name} field {'.'.join(prepared_field.path)!r} must return "
                    f"{prepared_field.value_type.__name__}, got {type(field_value).__name__}."
                )
            leaves[prepared_field.path] = ResolvedLeaf(field_value, prepared_field.scope)
        self._prepared_trees.append(ResolvedTree(value, leaves))
        return len(self._prepared_trees) - 1

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
