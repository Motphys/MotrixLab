# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral resolver for manager task programs.

Lowers one constructed manager environment into its execution metadata: the
simulator read plan, the live manager context handed to terms, prepared term
arguments, and one typed call record per manager term (command hooks,
observations, rewards, terminations, simulator resets). Numerical executors —
currently the Torch executor in :mod:`motrix_env_core.torch_manager` —
consume this metadata and schedule the typed calls; no code generation or
dispatcher compilation happens here.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np
import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.env import ManagerEnv, ReadPlan
from motrix_env_core.manager.kernel_data import (
    Map,
    is_kernel_data,
    iter_leaves,
    leaf_value_type,
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
    dtype: torch.dtype


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
        lines.extend(f"  [{metric.index}] dtype={metric.dtype} {metric.name}" for metric in self.metrics)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prepared arguments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantArg:
    """Literal term argument passed through unchanged (strings, tensors...)."""

    value: Any


@dataclass(frozen=True)
class SimArg:
    """Term argument resolving to one sim read-program batched array."""

    key: str


PreparedArg = ConstantArg | SimArg


@dataclass(frozen=True)
class CommandUpdateHook:
    """One command term's ``update`` hook (receiver + static function)."""

    receiver_key: str
    function: Callable


@dataclass(frozen=True)
class CommandAdvanceHook:
    """One command term's ``advance`` hook (receiver + static function)."""

    receiver_key: str
    function: Callable


@dataclass(frozen=True)
class CommandResetHook:
    """One command term's ``reset`` hook (receiver + static function)."""

    receiver_key: str
    function: Callable


@dataclass(frozen=True)
class RewardTermCall:
    """One reward term call with its resolver-assigned buffer column."""

    index: int
    function: Callable
    args: tuple[PreparedArg, ...]


@dataclass(frozen=True)
class TerminationTermCall:
    """One termination term call with its resolver-assigned buffer column."""

    index: int
    function: Callable
    args: tuple[PreparedArg, ...]


@dataclass(frozen=True)
class ObservationTermCall:
    """One observation term call writing its group slice."""

    group: str
    name: str
    function: Callable
    args: tuple[PreparedArg, ...]


@dataclass(frozen=True)
class SimResetTermCall:
    """One simulator reset term call (write offsets stay on ``ResolvedSimReset``)."""

    function: Callable
    args: tuple[PreparedArg, ...]


@dataclass(frozen=True)
class ResolvedSimReset:
    call: SimResetTermCall
    sim_writes_type: str
    output_offset: int
    output_count: int


@dataclass(frozen=True)
class ResolvedManagerProgram:
    """Resolved execution metadata for one manager environment.

    The phase sequences encode the execution contract structurally: evaluate
    runs command updates, then rewards, then terminations, then command
    advances (declaration order within each phase); observe iterates the
    per-group observation calls; reset runs the command reset hooks followed
    by the simulator reset terms. Reward and termination indices match
    ``ManagerLayout.rewards``/``terminations`` ordering.

    ``context`` is the live :class:`ManagerContext` host instance handed to
    every term; its tensors and Map entries alias the environment's
    persistent state, so it is built once and never rebuilt.
    """

    read_plan: ReadPlan
    layout: ManagerLayout
    context: ManagerContext
    command_updates: tuple[CommandUpdateHook, ...]
    rewards: tuple[RewardTermCall, ...]
    terminations: tuple[TerminationTermCall, ...]
    command_advances: tuple[CommandAdvanceHook, ...]
    command_reset_hooks: tuple[CommandResetHook, ...]
    sim_reset_terms: tuple[ResolvedSimReset, ...]
    observations: dict[str, tuple[ObservationTermCall, ...]]


__all__ = [
    "ActionTermLayout",
    "CommandAdvanceHook",
    "CommandResetHook",
    "CommandUpdateHook",
    "ConstantArg",
    "ManagerLayout",
    "ManagerProgramResolver",
    "MetricFieldLayout",
    "ObservationGroupLayout",
    "ObservationTermCall",
    "ObservationTermLayout",
    "PreparedArg",
    "ResolvedManagerProgram",
    "ResolvedSimReset",
    "RewardTermCall",
    "ScalarTermLayout",
    "SimArg",
    "SimInputLayout",
    "SimResetTermCall",
    "TerminationTermCall",
]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManagerSimInput:
    """One compiled simulator input exposed through ``ManagerContext.sim``."""

    value: torch.Tensor
    query: SimDataQuery


def _manager_sim_inputs(sim_data: PhysicsReadProgram) -> Mapping[str, ManagerSimInput]:
    inputs = {key: ManagerSimInput(sim_data.view(key), sim_data.query(key)) for key in sim_data.keys}
    return MappingProxyType(inputs)


class ManagerProgramResolver:
    """Resolve one fixed manager config into executor-neutral metadata."""

    def __init__(self, env: ManagerEnv):
        self._env = env
        self._sim_inputs = _manager_sim_inputs(env.sim_data)

    def build(self) -> ResolvedManagerProgram:
        build_started = perf_counter()
        observation_groups = self._env.observation_groups
        reward_terms = self._env._reward_terms
        termination_terms = self._env.termination_manager.terms
        context = self._resolve_manager_context()
        command_updates = tuple(
            CommandUpdateHook(key, function) for key, function in self._resolve_command_hooks("update")
        )
        command_advances = tuple(
            CommandAdvanceHook(key, function) for key, function in self._resolve_command_hooks("advance")
        )
        command_reset_hooks = tuple(
            CommandResetHook(key, function) for key, function in self._resolve_command_hooks("reset")
        )
        observation_terms, observation_layout = self._resolve_observations(observation_groups)
        reward_calls, reward_layout, _ = self._resolve_rewards(self._env.cfg, reward_terms)
        termination_calls, termination_layout = self._resolve_terminations(termination_terms)
        sim_reset_terms = self._resolve_resets()
        per_env_metric_layout = tuple(
            MetricFieldLayout(name, index, value.dtype)
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
        call_count = (
            len(command_updates)
            + len(command_advances)
            + len(command_reset_hooks)
            + sum(len(terms) for terms in observation_terms.values())
            + len(reward_calls)
            + len(termination_calls)
            + len(sim_reset_terms)
        )
        logger.info(
            "Manager startup %s: resolve finished in %.3fs (%d term calls)",
            type(self._env).__name__,
            perf_counter() - build_started,
            call_count,
        )
        return ResolvedManagerProgram(
            read_plan=ReadPlan(self._env.sim_data),
            layout=layout,
            context=context,
            command_updates=command_updates,
            rewards=reward_calls,
            terminations=termination_calls,
            command_advances=command_advances,
            command_reset_hooks=command_reset_hooks,
            sim_reset_terms=sim_reset_terms,
            observations=observation_terms,
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
        args = tuple(self._resolve_term_arg(value) for index, value in enumerate(term.args))
        return ResolvedSimReset(
            SimResetTermCall(dispatcher, args),
            f"sim_reset_{name}_writes",
            output_offset,
            len(fields),
        )

    # -- layouts ------------------------------------------------------------

    def _resolve_manager_context(self) -> ManagerContext:
        actions = {name: term.state for name, term in self._env.action_terms.items()}
        # Sim views enter the context as zero-copy tensor views over the
        # read-program arena: the read program rewrites the arena contents in
        # place, so the pre-built views stay valid for the program's lifetime.
        sim = {key: binding.value for key, binding in self._sim_inputs.items()}
        metrics = dict(self._env.metrics)
        commands = dict(self._env.command_terms)
        value = ManagerContext(
            env_id=torch.arange(self._env.num_envs, dtype=torch.int64),
            actions=Map(actions),
            commands=Map(commands),
            metrics=Map(metrics),
            rand=self._env._rand,
            sim=Map(sim),
            dt=float(self._env.cfg.ctrl_dt),
            sim_reset_requested=self._env._sim_reset_requested,
            # Reset-pass context owned here: ``reset_ids`` is the compact row
            # index of the current reset pass (rebound per call by the reset
            # kernel wrapper), the episode mask is full-batch scratch filled
            # by _reset_rows and gathered per pass, and ``terminated`` is the
            # termination buffer the manager state is assembled over (see
            # ManagerEnv._new_state), so kernels and ``state.terminated``
            # share one tensor.
            reset_ids=torch.zeros(0, dtype=torch.int64),
            reset_episode=torch.zeros((self._env.num_envs, 1), dtype=torch.bool),
            terminated=torch.zeros(self._env.num_envs, dtype=torch.bool),
        )
        self._validate_context_leaves(value)
        return value

    def _resolve_command_hooks(self, function_name: str) -> tuple[tuple[str, Callable], ...]:
        """Resolve one hook family over all command terms in declaration order.

        Returns ``(receiver_key, function)`` pairs; the static function
        signature is validated against the command hook contract.
        """
        hooks = []
        for name, command_term in self._env.command_terms.items():
            function = inspect.getattr_static(type(command_term), function_name)
            if isinstance(function, staticmethod):
                function = function.__func__
            self._validate_command_hook(name, function, function_name)
            hooks.append((name, function))
        return tuple(hooks)

    def _validate_command_hook(self, name: str, function: Any, function_name: str) -> None:
        if not inspect.isfunction(function):
            raise TypeError(f"Manager term command.{name} {function_name} must be a static function.")
        parameters = tuple(inspect.signature(function).parameters.values())
        if not parameters:
            raise TypeError(f"Manager term command.{name} {function_name} must declare a receiver parameter.")
        ctx_parameters = parameters[1:]
        if not ctx_parameters:
            raise TypeError(f"Manager term command.{name} {function_name} must declare a ManagerContext parameter.")
        argument_parameters = ctx_parameters[1:]
        annotations = get_type_hints(function, include_extras=True)
        if annotations.get(ctx_parameters[0].name) is not ManagerContext:
            raise TypeError(
                f"Manager term command.{name} {function_name} must annotate its first parameter after the "
                "receiver as ManagerContext (conventionally named ctx)."
            )
        if argument_parameters:
            raise TypeError(
                f"Manager term command.{name} {function_name} has unsupported dependency parameters "
                f"{[parameter.name for parameter in argument_parameters]}; read manager and simulator data "
                "from the ManagerContext parameter."
            )

    # -- observations --------------------------------------------------------

    def _resolve_observations(
        self,
        groups: dict[str, ObservationGroupEntry],
    ) -> tuple[dict[str, tuple[ObservationTermCall, ...]], dict[str, ObservationGroupLayout]]:
        resolved_groups = {}
        layout_groups = {}
        for group_name, group in groups.items():
            terms = []
            layouts = []
            offset = 0
            for entry in group.terms:
                terms.append(self._resolve_observation(group_name, entry.name, entry.term))
                output_slice = slice(offset, offset + entry.size)
                layouts.append(ObservationTermLayout(entry.name, output_slice))
                offset = output_slice.stop
            if offset != group.size:
                raise RuntimeError(f"Observation group {group_name!r} resolved width changed during resolution.")
            resolved_groups[group_name] = tuple(terms)
            layout_groups[group_name] = ObservationGroupLayout(group_name, tuple(layouts), group.size)
        return resolved_groups, layout_groups

    def _resolve_observation(self, group: str, name: str, term: ObsTerm) -> ObservationTermCall:
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
        args = tuple(self._resolve_term_arg(value) for index, value in enumerate(term.args))
        return ObservationTermCall(group, name, function, args)

    # -- rewards and terminations --------------------------------------------

    def _resolve_rewards(
        self,
        cfg: Any,
        reward_terms: dict[str, RewardTerm],
    ) -> tuple[tuple[RewardTermCall, ...], tuple[ScalarTermLayout, ...], tuple[float, ...]]:
        term_cfgs = cfg.reward_cfgs()
        terms = []
        layout = []
        weights = []
        for index, (name, term) in enumerate(reward_terms.items()):
            args = self._resolve_dispatch_args("reward", name, term)
            terms.append(RewardTermCall(index, term.dispatch, args))
            layout.append(ScalarTermLayout(name, index))
            try:
                weights.append(float(term_cfgs[name].weight))
            except (TypeError, ValueError) as error:
                raise TypeError(f"Manager reward term reward.{name} weight must be float-compatible.") from error
        return tuple(terms), tuple(layout), tuple(weights)

    def _resolve_terminations(
        self,
        termination_terms: dict[str, TerminationTerm],
    ) -> tuple[tuple[TerminationTermCall, ...], tuple[ScalarTermLayout, ...]]:
        terms = []
        layout = []
        for index, (name, term) in enumerate(termination_terms.items()):
            args = self._resolve_dispatch_args("termination", name, term)
            terms.append(TerminationTermCall(index, term.dispatch, args))
            layout.append(ScalarTermLayout(name, index))
        return tuple(terms), tuple(layout)

    def _resolve_dispatch_args(self, group: str, name: str, term: Any) -> tuple[PreparedArg, ...]:
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
        return tuple(self._resolve_term_arg(value) for index, value in enumerate(term.args))

    # -- argument resolution ---------------------------------------------------

    def _resolve_term_arg(self, value: Any) -> PreparedArg:
        """Resolve one manager term argument into a prepared descriptor.

        Carrier dataclass and tensor/ndarray arguments pass through as
        constants: Dynamo traces their attribute access and tensor ops
        directly. Float-like numpy scalars normalize to Python floats;
        simulator queries resolve to their read-program batched array; every
        other value is a literal constant.
        """
        if is_kernel_data(value):
            return ConstantArg(value)
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return ConstantArg(value)
        if isinstance(value, np.floating):
            return ConstantArg(float(value))
        if isinstance(value, SimDataQuery):
            # A query passed as a term argument resolves to its registered
            # read-program batched view; the dispatch receives the array in
            # the argument's position.
            return SimArg(self._env._term_query_keys[value])
        return ConstantArg(value)

    def _validate_context_leaves(self, value: ManagerContext) -> None:
        """Validate the context tree's leaves once at resolution time.

        Command terms, action states, and the rand value are already
        canonicalized at construction; this walk covers the assembled context
        itself (env id, sim views, metric views) with the same leaf-type
        contract. Nothing is stored: terms hold the live host objects.
        """
        for leaf_path, annotation, leaf_value in iter_leaves(value):
            value_type = leaf_value_type(annotation)
            if value_type is None:
                raise TypeError(
                    f"manager_context KernelData {'.'.join(leaf_path)} expected a tensor, an ndarray, or a numeric "
                    "scalar."
                )
            if value_type in (np.ndarray, torch.Tensor):
                if not isinstance(leaf_value, value_type):
                    raise TypeError(f"manager_context field {'.'.join(leaf_path)!r} must be {value_type.__name__}.")
            elif type(leaf_value) is not value_type:
                raise TypeError(
                    f"manager_context field {'.'.join(leaf_path)!r} must return "
                    f"{value_type.__name__}, got {type(leaf_value).__name__}."
                )

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
