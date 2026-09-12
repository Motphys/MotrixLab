# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from time import perf_counter
from typing import Any, TypeVar

import gymnasium as gym
import numpy as np

from motrix_env_core.array.env import ArrayEnv, ArrayEnvState, EnvCfgType, NpObs
from motrix_env_core.base import EnvCfg, ObsSpace
from motrix_env_core.config import configclass
from motrix_env_core.config.sim_reset import ManagerResetCfg, ResetTermCfg
from motrix_env_core.numba.kernel import (
    ManagerWarmupResult,
    NumbaKernelOutputs,
    clone_kernel_value,
    get_num_threads,
    get_threading_layer,
    validate_kernel_context,
)
from motrix_env_core.numba.kernel_data import (
    KernelDataLayout,
    KernelDataScope,
    KernelLeafLayout,
    canonicalize_kernel_data,
    flatten_kernel_data,
    iter_layout_leaves,
)
from motrix_env_core.numba.manager.actions import ActionCfg, ActionTerm, ManagerActionsCfg
from motrix_env_core.numba.manager.commands import CommandCfg, CommandTerm, ManagerCommandsCfg, ResetContext
from motrix_env_core.numba.manager.compiler.plan import ManagerLayout
from motrix_env_core.numba.manager.metrics import collect_metrics, materialize_metrics
from motrix_env_core.numba.manager.observations import (
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ObservationGroupEntry,
    ObservationTermCfg,
    create_observation_groups,
)
from motrix_env_core.numba.manager.rewards import ManagerRewardsCfg, RewardTermCfg, create_reward_terms
from motrix_env_core.numba.manager.sim_reset import SimResetRuntime
from motrix_env_core.numba.manager.terminations import (
    ManagerTerminationsCfg,
    TerminationManager,
    TerminationTermCfg,
)
from motrix_env_core.numba.program import NumbaTaskProgram
from motrix_env_core.sim import (
    ModelQuery,
    PhysicsReadProgram,
    SimDataQuery,
    SimQueriesCfg,
)
from motrix_env_core.sim.backend import ActuatorSpec, RenderConfig, SimBackend, SimRenderer
from motrix_env_core.sim.write import CtrlTargetsWrite, SimWrite

_CfgT = TypeVar("_CfgT")
_Q = TypeVar("_Q")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequiredQueryContribution:
    """One observation term's resolved ``required_sim_queries()`` declaration."""

    source: str
    data: dict[str, SimDataQuery]
    model: dict[str, ModelQuery]


def observation_required_sim_queries(cfg: ManagerBasedEnvCfg) -> tuple[RequiredQueryContribution, ...]:
    """Resolve every observation term's ``required_sim_queries()`` against ``cfg``."""
    contributions = []
    for group_name, term_cfgs in cfg.observation_cfgs().items():
        for term_name, term_cfg in term_cfgs.items():
            source = f"observations.{group_name}.{term_name}"
            required = term_cfg.required_sim_queries(cfg)
            if not isinstance(required, SimQueriesCfg):
                raise TypeError(
                    f"Observation term {source!r} required_sim_queries() must return SimQueriesCfg, "
                    f"got {type(required).__name__}."
                )
            required.validate()
            contributions.append(
                RequiredQueryContribution(
                    source=source,
                    data=dict(required.data),
                    model=dict(required.model),
                )
            )
    return tuple(contributions)


def _merge_query_declarations(
    declared: dict[str, _Q],
    required: Iterable[tuple[str, str, _Q]],
    *,
    label: str,
) -> dict[str, _Q]:
    """Merge query declarations from all sources.

    Declarations sharing a key are allowed only when their query definitions are
    equal. Equal declarations collapse to one query; any unequal declarations
    fail with the key and the contributing sources. This applies uniformly to
    task-owned and observation-term-owned declarations.
    """
    by_key: dict[str, list[tuple[str, _Q]]] = {key: [("task queries", query)] for key, query in declared.items()}
    for source, key, query in required:
        by_key.setdefault(key, []).append((source, query))

    merged = {}
    for key, contributors in by_key.items():
        first_source, first = contributors[0]
        for other_source, other in contributors[1:]:
            if other != first:
                sources = ", ".join(source for source, _ in contributors)
                raise ValueError(
                    f"{label} query key {key!r} has unequal declarations from [{sources}]: "
                    f"{first_source}={first!r}, {other_source}={other!r}."
                )
        merged[key] = first
    return merged


def _manager_group_to_dict(
    value: object,
    group_type: type[object],
    item_type: type[_CfgT],
    *,
    label: str,
) -> dict[str, _CfgT]:
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, group_type):
        items = ((item_field.name, getattr(value, item_field.name)) for item_field in fields(value))
    else:
        raise TypeError(f"Manager {label} config must be a dict or {group_type.__name__}, got {type(value).__name__}.")

    configs = {}
    for name, config in items:
        if not isinstance(config, item_type):
            raise TypeError(f"Manager {label} {name!r} must be a {item_type.__name__}, got {type(config).__name__}.")
        configs[name] = config
    return configs


@configclass
class ManagerBasedEnvCfg(EnvCfg):
    """Complete environment config with manager groups at the top level."""

    sim_reset: ManagerResetCfg = ManagerResetCfg()
    commands: dict[str, CommandCfg] = field(default_factory=dict)
    actions: dict[str, ActionCfg] = field(default_factory=dict)
    queries: SimQueriesCfg = SimQueriesCfg()
    observations: dict[str, dict[str, ObservationTermCfg]] = field(default_factory=dict)
    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)

    def command_cfgs(self) -> dict[str, CommandCfg]:
        return _manager_group_to_dict(self.commands, ManagerCommandsCfg, CommandCfg, label="command")

    def action_cfgs(self) -> dict[str, ActionCfg]:
        return _manager_group_to_dict(self.actions, ManagerActionsCfg, ActionCfg, label="action")

    def sim_reset_cfgs(self) -> dict[str, ResetTermCfg]:
        if not isinstance(self.sim_reset, ManagerResetCfg):
            raise TypeError(
                f"Manager simulator reset config must be a ManagerResetCfg, got {type(self.sim_reset).__name__}."
            )
        return self.sim_reset.to_dict()

    def sim_query_cfgs(self) -> dict[str, SimDataQuery]:
        """Return task-declared plus term-required simulator data queries.

        Observation terms contribute their ``required_sim_queries()`` declarations;
        declarations from any source may share a key only when their query
        definitions are equal, while unequal declarations fail.
        """
        self.queries.validate()
        declared = dict(self.queries.data)
        required = (
            (contribution.source, key, query)
            for contribution in observation_required_sim_queries(self)
            for key, query in contribution.data.items()
        )
        return _merge_query_declarations(declared, required, label="simulator data")

    def model_query_cfgs(self) -> dict[str, ModelQuery]:
        """Return task-declared plus term-required model queries.

        Merged like :meth:`sim_query_cfgs`; declarations from any source may
        share a key only when their query definitions are equal.
        """
        self.queries.validate()
        declared = dict(self.queries.model)
        required = (
            (contribution.source, key, query)
            for contribution in observation_required_sim_queries(self)
            for key, query in contribution.model.items()
        )
        return _merge_query_declarations(declared, required, label="model")

    def observation_cfgs(self) -> dict[str, dict[str, ObservationTermCfg]]:
        if isinstance(self.observations, dict):
            groups = self.observations
        elif isinstance(self.observations, ManagerObservationsCfg):
            groups = self.observations.to_dict()
        else:
            raise TypeError(
                "Manager observation config must be a dict or "
                f"ManagerObservationsCfg, got {type(self.observations).__name__}."
            )

        resolved = {}
        for group_name, group in groups.items():
            if group_name not in {"policy", "value"}:
                raise ValueError(f"Unsupported observation group {group_name!r}; expected 'policy' or 'value'.")
            resolved[group_name] = _manager_group_to_dict(
                group,
                ManagerObservationGroupCfg,
                ObservationTermCfg,
                label=f"observation group {group_name}",
            )
        return resolved

    def reward_cfgs(self) -> dict[str, RewardTermCfg]:
        return _manager_group_to_dict(self.rewards, ManagerRewardsCfg, RewardTermCfg, label="reward")

    def termination_cfgs(self) -> dict[str, TerminationTermCfg]:
        return _manager_group_to_dict(
            self.terminations,
            ManagerTerminationsCfg,
            TerminationTermCfg,
            label="termination",
        )


@dataclass(frozen=True)
class ReadPlan:
    sim_data: PhysicsReadProgram
    sources: tuple[KernelInputSource, ...]
    flat_inputs: tuple[Any, ...]

    def read(
        self,
        env: ManagerEnv,
        env_ids: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        with env.perf.scope("sim_read"):
            self.sim_data.execute(env_ids)
        return self.flat_inputs


@dataclass(frozen=True)
class KernelInputSource:
    """Compile-time description of one manager-kernel input source.

    Manager inputs are flattened and validated once while the Numba program is
    compiled.  ``values`` then keeps those flattened leaves by reference so
    runtime reads only refresh simulator-backed arrays and can return the same
    tuple without rebuilding the input tree.  ``source_name`` is a stable,
    human-readable label used in manager-layout diagnostics and validation
    errors; it is intentionally independent of a source implementation class.

    Attributes:
        source_name: Stable semantic label for layout entries and validation
            messages (for example, ``"manager_context"`` or ``"sim_reset"``).
        value_type: Original structured input type reconstructed by the kernel
            compiler when preparing a manager context or reset term.
        layout: Lowered tree layout describing the order, scope, and types of
            the flattened input leaves.
        values: Flattened leaves captured at compile time.  Array leaves are
            retained by reference and are refreshed in place by their owning
            runtime objects; scalar/shared leaves are reused unchanged.
    """

    source_name: str
    value_type: type[Any]
    layout: KernelDataLayout
    values: tuple[Any, ...]

    @property
    def fields(self) -> tuple[KernelLeafLayout, ...]:
        return iter_layout_leaves(self.layout)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(".".join(field.path) or "data" for field in self.fields)

    @classmethod
    def prepare(
        cls,
        source_name: str,
        value_type: type[Any],
        layout: KernelDataLayout,
        value: Any,
    ) -> KernelInputSource:
        """Flatten and validate one static manager-kernel input at compile time."""
        values, tree_def = flatten_kernel_data(value)
        if tree_def.fingerprint != layout.tree_def.fingerprint:
            raise TypeError(f"{source_name} value has a different KernelData schema.")
        fields = iter_layout_leaves(layout)
        for prepared_field, field_value in zip(fields, values, strict=True):
            if prepared_field.value_type is np.ndarray and not isinstance(field_value, np.ndarray):
                raise TypeError(f"{source_name} field {'.'.join(prepared_field.path)!r} must be np.ndarray.")
            if prepared_field.value_type is not np.ndarray and type(field_value) is not prepared_field.value_type:
                raise TypeError(
                    f"{source_name} field {'.'.join(prepared_field.path)!r} must return "
                    f"{prepared_field.value_type.__name__}, got {type(field_value).__name__}."
                )
        return cls(source_name, value_type, layout, values)


@dataclass(frozen=True)
class CompiledManagerProgram(abc.ABC):
    task: NumbaTaskProgram
    read_plan: ReadPlan
    layout: ManagerLayout
    source: str

    @abc.abstractmethod
    def warmup_terms(self, env: ManagerEnv, state: ArrayEnvState, buffers: tuple[Any, ...]) -> None:
        """Compile and validate each manager term independently."""


@dataclass(frozen=True)
class _ObsBufferSlot:
    observation: NpObs
    kernel_outputs: NumbaKernelOutputs


@dataclass
class SwapObsBuffer:
    """Alternate two preallocated observation slots without copying.

    Each slot binds an exposed ``NpObs`` to ``NumbaKernelOutputs`` that write
    into the same policy/value arrays. Swapping both together before the next
    kernel call keeps ``obs_t`` unchanged while a transition consumer retains
    it and the kernel produces ``obs_t+1`` in the other slot.

    Reward and termination arrays are shared because only observations need
    the extra step of lifetime. This preserves that lifetime without per-step
    observation allocations or copies.
    """

    _slots: tuple[_ObsBufferSlot, _ObsBufferSlot]
    _index: int = 0

    @classmethod
    def create(
        cls,
        first: NpObs,
        second: NpObs,
        reward: np.ndarray,
        terminated: np.ndarray,
    ) -> SwapObsBuffer:
        def make_slot(observation: NpObs) -> _ObsBufferSlot:
            value = observation.value
            if value is None:
                value = np.empty((observation.policy.shape[0], 0), dtype=observation.policy.dtype)
            return _ObsBufferSlot(
                observation=observation,
                kernel_outputs=NumbaKernelOutputs(
                    policy_obs=observation.policy,
                    value_obs=value,
                    reward=reward,
                    terminated=terminated,
                ),
            )

        return cls((make_slot(first), make_slot(second)))

    @property
    def current(self) -> _ObsBufferSlot:
        return self._slots[self._index]

    def swap(self) -> tuple[NpObs, NumbaKernelOutputs]:
        self._index = 1 - self._index
        slot = self.current
        return slot.observation, slot.kernel_outputs


class ManagerEnv(ArrayEnv[EnvCfgType]):
    """NumPy environment whose task program is generated from manager config."""

    def __init__(self, cfg: EnvCfgType, num_envs: int = 1, backend: str | None = None, seed: int | None = None):
        """Construct a manager environment using the selected simulator backend."""
        if not isinstance(cfg, ManagerBasedEnvCfg):
            raise TypeError(f"{type(cfg).__name__} must inherit ManagerBasedEnvCfg.")
        super().__init__(cfg, num_envs)
        self._rand_seed = 1 if seed is None else seed
        from motrix_env_core.sim.registry import create_sim_backend, default_sim_backend_name

        factory = create_sim_backend(backend or default_sim_backend_name())
        # Construction compiles the scene inside the backend: no compiled
        # artifact crosses the boundary.
        self.sim: SimBackend = factory(cfg.scene, cfg.sim, num_envs)
        self.model = self.sim.compile_model(cfg.model_query_cfgs())
        # Full-width ctrl buffer in canonical actuator order, owned by the
        # compiled ctrl write program; routing happens here, the backend only
        # receives the merged targets.
        self._sim_reset_runtime = SimResetRuntime.create(self, cfg.sim_reset, self.sim)
        self.sim_data: PhysicsReadProgram = self.sim.compile_reads(cfg.sim_query_cfgs())
        from motrix_env_core.mdp.state import _create_rand_value

        self._task_program: NumbaTaskProgram | None = None
        self._kernel_buffers: tuple[Any, ...] | None = None
        self.metrics: dict[str, np.ndarray] = {}
        self._kernel_outputs: NumbaKernelOutputs | None = None
        self._observation_buffer: SwapObsBuffer | None = None
        # Persistent kernel-bound sim-reset-request buffer
        # (``ctx.sim_reset_requested``), cleared before each physics step and
        # consumed by _reset_done_envs.
        self._sim_reset_requested = np.zeros((self.num_envs, 1), dtype=bool)
        self._compiled_manager_program: CompiledManagerProgram | None = None
        self._action_cfgs = cfg.action_cfgs()
        if not self._action_cfgs:
            raise ValueError("Manager environment config requires at least one action config.")
        self._action_actuators = self._resolve_action_actuators()
        self._action_writes = self.sim.write_compiler.compile(
            {
                name: CtrlTargetsWrite(None if action_cfg.actuator_names == () else action_cfg.actuator_names)
                for name, action_cfg in self._action_cfgs.items()
                if action_cfg.actuator_names is not None
            }
        )
        self._action_terms: dict[str, ActionTerm] = {
            name: canonicalize_kernel_data(
                action_cfg(self, self._action_actuators[name]),
                context=f"Manager action {name!r} __call__()",
            )
            for name, action_cfg in self._action_cfgs.items()
        }
        self._command_cfgs = cfg.command_cfgs()
        self._command_terms: dict[str, CommandTerm] = {
            name: canonicalize_kernel_data(
                materialize_metrics(
                    command_cfg(self),
                    self.num_envs,
                    context=f"Manager command {name!r} __call__()",
                ),
                context=f"Manager command {name!r} __call__()",
            )
            for name, command_cfg in self._command_cfgs.items()
        }
        self._rand = canonicalize_kernel_data(_create_rand_value(self), context="Manager random state")
        self._action_space, self._action_slices = self._build_action_space()
        self._reward_terms = create_reward_terms(cfg.reward_cfgs(), self)
        self.termination_manager = TerminationManager(cfg.termination_cfgs(), self)
        self._observation_groups = create_observation_groups(cfg, self)
        for name, term in self._command_terms.items():
            for binding in collect_metrics(
                term,
                self.num_envs,
                context=f"Manager command {name!r}",
                context_path=("commands", name),
            ):
                metric_name, value = binding
                if metric_name in self.metrics:
                    raise ValueError(f"Duplicate per-environment metric name: {metric_name!r}")
                self.metrics[metric_name] = value

        policy_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (self._observation_groups["policy"].size,),
            dtype=np.float32,
        )
        value_space = (
            gym.spaces.Box(-np.inf, np.inf, (self._observation_groups["value"].size,), dtype=np.float32)
            if "value" in self._observation_groups
            else None
        )
        self._observation_space = ObsSpace(policy=policy_space, value=value_space)

    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        return self.sim.create_renderer(
            config,
            num_envs=self.num_envs,
            render_spacing=self.render_spacing,
            system_camera=self.cfg.scene.system_camera,
        )

    @property
    def num_dof_pos(self) -> int:
        return self.sim.num_dof_pos

    @property
    def num_dof_vel(self) -> int:
        return self.sim.num_dof_vel

    @property
    def num_actuators(self) -> int:
        return self.sim.num_actuators

    def physics_step(self) -> None:
        self.sim.step(self._cfg.sim_substeps)

    def init_state(self) -> ArrayEnvState:
        startup_started = perf_counter()
        env_name = type(self).__name__
        if self._task_program is None:
            logger.info("Manager env %r startup begin: num_envs=%d", env_name, self.num_envs)
            self._task_program = self._build_task_program()
            # Initialize the simulator arena before the base lifecycle resets
            # rows through the reset kernel.
            self._refresh_sim_reads()
        state = super().init_state()
        self._kernel_buffers = self._make_kernel_buffers(state)
        state.metrics = self._make_metrics_view()
        # Alternate two preallocated outputs so consumers can retain observation t
        # until they have stored the transition produced by step t + 1.
        alternate_obs = self._zeros_obs(self.observation_space)
        self._observation_buffer = SwapObsBuffer.create(
            state.obs,
            alternate_obs,
            state.reward,
            state.terminated,
        )
        self._kernel_outputs = self._observation_buffer.current.kernel_outputs
        self._refresh_sim_reads()
        inputs = self._kernel_inputs
        self._validate_kernel_context(inputs)
        self._compile_manager_specializations(inputs)
        warmup_started = perf_counter()
        self._execute_observe_kernel(inputs)
        logger.info(
            "Manager startup %s: warmup execution finished in %.3fs",
            env_name,
            perf_counter() - warmup_started,
        )
        logger.info("Manager env %r startup complete in %.3fs", env_name, perf_counter() - startup_started)
        return state

    def _prev_physics_step(self) -> None:
        self._sim_reset_requested.fill(False)
        super()._prev_physics_step()
        assert self._observation_buffer is not None
        obs, self._kernel_outputs = self._observation_buffer.swap()
        self._state = self._state.replace(obs=obs)

    def _execute_evaluate_kernel(self, inputs: tuple[Any, ...]) -> None:
        """Evaluate rewards and terminations against the current command."""
        assert self._task_program is not None
        assert self._kernel_buffers is not None
        assert self._kernel_outputs is not None
        self._task_program.evaluate_kernel(
            inputs,
            self._task_program.reward_weights,
            self._kernel_buffers,
            self._kernel_outputs,
        )

    def _execute_observe_kernel(self, inputs: tuple[Any, ...]) -> None:
        """Evaluate observations against the command for the next action."""
        assert self._task_program is not None
        assert self._kernel_outputs is not None
        self._task_program.observe_kernel(inputs, self._kernel_outputs)

    def _execute_task_kernel(self, inputs: tuple[Any, ...]) -> None:
        """Evaluate and observe without advancing host-side commands."""
        self._execute_evaluate_kernel(inputs)
        self._execute_observe_kernel(inputs)

    def _resolve_action_actuators(self) -> dict[str, tuple[ActuatorSpec, ...] | None]:
        routes = {}
        owners = {}
        by_name = {spec.name: spec for spec in self.model.actuators}
        for term_name, cfg in self._action_cfgs.items():
            if cfg.actuator_names is None:
                routes[term_name] = None
                continue
            names = cfg.actuator_names or tuple(by_name)
            if not names:
                raise ValueError(f"Manager action {term_name!r} actuator route must not be empty.")
            if len(set(names)) != len(names):
                raise ValueError(f"Manager action {term_name!r} actuator names must be unique.")
            actuators = []
            for actuator_name in names:
                try:
                    actuator = by_name[actuator_name]
                except KeyError:
                    raise ValueError(
                        f"Manager action {term_name!r} references unknown actuator {actuator_name!r}."
                    ) from None
                existing = owners.get(actuator_name)
                if existing is not None:
                    raise ValueError(
                        f"Manager actions {existing!r} and {term_name!r} both control actuator {actuator_name!r}."
                    )
                owners[actuator_name] = term_name
                actuators.append(actuator)
            routes[term_name] = tuple(actuators)
        return routes

    def _build_action_space(self) -> tuple[gym.spaces.Box, dict[str, slice]]:
        lows = []
        highs = []
        action_slices = {}
        offset = 0
        for name, term in self._action_terms.items():
            space = term.action_space(self, self._action_actuators[name])
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(f"Manager action {name!r} must produce a gym.spaces.Box.")
            if len(space.shape) != 1 or space.shape[0] <= 0:
                raise ValueError(f"Manager action {name!r} space must have a non-empty one-dimensional shape.")
            if space.dtype != np.dtype(np.float32):
                raise TypeError(f"Manager action {name!r} space dtype must be float32, got {space.dtype}.")
            size = space.shape[0]
            action_slices[name] = slice(offset, offset + size)
            lows.append(space.low)
            highs.append(space.high)
            offset += size
        return (
            gym.spaces.Box(np.concatenate(lows), np.concatenate(highs), dtype=np.float32),
            action_slices,
        )

    @property
    def action_cfgs(self) -> dict[str, ActionCfg]:
        return self._action_cfgs

    @property
    def action_terms(self) -> dict[str, ActionTerm]:
        return self._action_terms

    @property
    def sim_reset_terms(self) -> dict[str, Any]:
        return self._sim_reset_runtime.terms

    @property
    def sim_reset_writes(self) -> dict[str, dict[str, SimWrite]]:
        return self._sim_reset_runtime.writes

    @property
    def action_slices(self) -> dict[str, slice]:
        return self._action_slices

    @property
    def action_actuators(self) -> dict[str, tuple[ActuatorSpec, ...] | None]:
        return self._action_actuators

    @property
    def observation_space(self) -> ObsSpace:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
        expected_shape = (self.num_envs, self.action_space.shape[0])
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {actions.shape}.")
        for name, term in self._action_terms.items():
            controls = term.process(actions[:, self._action_slices[name]])
            actuators = self._action_actuators[name]
            if actuators is None:
                assert controls is None
                continue
            assert controls is not None
            self._action_writes.buffer(name)[:] = controls
        self._action_writes.execute()
        return state

    def _reset_sim_rows(self, env_ids: np.ndarray, inputs: tuple[Any, ...]) -> None:
        """Run the simulator-only stages of the reset pipeline for selected rows.

        Clears frontend ctrl targets to mirror the backend's reset-row ctrl
        clearing, reruns the reset kernel (command ``reset_env`` hooks followed
        by the configured sim reset terms), and rereads kernel inputs.
        """
        # The backend clears its own ctrl channels on reset; mirror that for the
        # frontend buffer so unrouted actuators do not see stale targets.
        with self.perf.scope("clear_ctrl_targets"):
            for name, actuators in self._action_actuators.items():
                if actuators is not None:
                    self._action_writes.buffer(name)[env_ids] = 0.0
        with self.perf.scope("apply_sim_reset"):
            assert self._task_program is not None
            self._sim_reset_runtime.apply(self._task_program, inputs, env_ids)
        assert self._compiled_manager_program is not None
        with self.perf.scope("reset_read_inputs"):
            self._compiled_manager_program.read_plan.read(self, env_ids)

    def _reset_done_envs(self) -> None:
        """Reset done lanes and requested sim-reset lanes in one pass.

        Sim-reset requests (command terms setting ``ctx.sim_reset_requested``
        mid-transition) join the episode resets in a single reset-kernel run.
        They are not episode boundaries: lanes that also finished their episode
        take the full reset path only, and episode bookkeeping stays untouched
        for sim-reset-only lanes.
        """
        state = self._state
        with self.perf.scope("done_mask"):
            done = state.done
            sim_reset_ids = np.flatnonzero(self._sim_reset_requested[:, 0])
            if sim_reset_ids.size:
                sim_reset_ids = sim_reset_ids[~done[sim_reset_ids]]
            env_ids = np.flatnonzero(done)
        if not env_ids.size and not sim_reset_ids.size:
            return
        if env_ids.size:
            with self.perf.scope("select_done"):
                np.putmask(state.episode_steps, done, 0)
        with self.perf.scope("reset_envs"):
            info1 = self.reset(env_ids, sim_reset_ids)
        self._merge_reset_info(state, info1, done)

    def reset(self, env_ids: np.ndarray, sim_reset_ids: np.ndarray | None = None) -> dict[str, Any]:
        """Reset episode lanes fully and sim-reset the rest in one pass.

        Episode lanes (``env_ids``) run the host lifecycle resets (command
        ``reset`` with termination statistics, action resets); sim-reset lanes
        (``ctx.sim_reset_requested`` set mid-transition) skip them — with no
        lane terminating, they would not change any state. Both sets join one
        reset-kernel run: command ``reset_env`` hooks resample lane state, the
        configured sim reset terms rewrite simulator rows, and kernel inputs
        are reread once for the union.
        """
        assert self._state is not None
        if env_ids.size:
            with self.perf.scope("command_reset"):
                for command_term in self._command_terms.values():
                    command_term.reset(
                        ResetContext(
                            env_ids=env_ids,
                            terminated=self._state.terminated,
                            metrics=self._state.metrics,
                        )
                    )
            with self.perf.scope("action_reset"):
                for action_term in self._action_terms.values():
                    action_term.reset(env_ids)
        if sim_reset_ids is not None and sim_reset_ids.size:
            env_ids = np.concatenate([env_ids, sim_reset_ids])
        if env_ids.size:
            self._reset_sim_rows(env_ids, self._kernel_inputs)
        return {}

    def _make_metrics_view(self) -> dict[str, Any]:
        """Assemble the persistent live metrics view for the current state.

        Values are views into the kernel reward/termination buffers and the
        term-owned per-env arrays, so kernels write them in place every step
        and nothing needs rebuilding per transition. Host-side writers may add
        batch-level scalar entries (e.g. sampler statistics at reset time).
        Consumers retain values across steps through
        ``ArrayEnvState.process_metrics()``.
        """
        assert self._compiled_manager_program is not None
        assert self._kernel_buffers is not None
        termination_masks = self._kernel_buffers[2]
        metrics = {
            term.name: termination_masks[:, term.index] for term in self._compiled_manager_program.layout.terminations
        }
        metrics.update(
            {
                name: value[:, 0] if value.ndim == 2 and value.shape[1] == 1 else value
                for name, value in self.metrics.items()
            }
        )
        return metrics

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        with self.perf.scope("read_inputs"):
            self._refresh_sim_reads()
        with self.perf.scope("evaluate"):
            self._execute_evaluate_kernel(self._kernel_inputs)
        with self.perf.scope("command_on_transition"):
            for command_term in self._command_terms.values():
                command_term.on_transition()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        # During initialisation, ArrayEnv computes the first observation
        # before the manager's kernel outputs exist. The manager-specific
        # init_state() performs that first kernel pass itself.
        if self._kernel_outputs is None:
            return state
        with self.perf.scope("observe"):
            self._execute_observe_kernel(self._kernel_inputs)
        return state

    @property
    def command_cfgs(self) -> dict[str, CommandCfg]:
        return self._command_cfgs

    @property
    def command_terms(self) -> dict[str, CommandTerm]:
        return self._command_terms

    @property
    def manager_layout(self) -> ManagerLayout:
        if self._compiled_manager_program is None:
            raise RuntimeError("Manager layout is available after init_state() builds the task program.")
        return self._compiled_manager_program.layout

    def _build_task_program(self) -> NumbaTaskProgram:
        from motrix_env_core.numba.manager.compiler import NumbaKernelCompiler

        self._compiled_manager_program = NumbaKernelCompiler(self).build()
        return self._compiled_manager_program.task

    def _compile_manager_specializations(self, inputs: tuple[Any, ...]) -> None:
        from motrix_env_core.numba.manager.compiler import NumbaKernelCompiler

        NumbaKernelCompiler(self).compile_specializations(inputs)

    def compile(self) -> float:
        """Compile all manager term and kernel specializations for this environment."""
        if self._state is None:
            self.init_state()
        self._refresh_sim_reads()
        inputs = self._kernel_inputs
        self._validate_kernel_context(inputs)
        started = perf_counter()
        self._compile_manager_specializations(inputs)
        return perf_counter() - started

    @property
    def observation_groups(self) -> dict[str, ObservationGroupEntry]:
        """Return the environment-local runtime observation groups."""
        return self._observation_groups

    def _make_kernel_buffers(self, state: ArrayEnvState) -> tuple[np.ndarray, ...]:
        assert self._compiled_manager_program is not None
        layout = self._compiled_manager_program.layout
        reward_terms = np.empty((self.num_envs, len(layout.rewards)), dtype=np.float32)
        weighted_reward_terms = np.empty_like(reward_terms)
        termination_masks = np.empty((self.num_envs, len(layout.terminations)), dtype=bool)
        state.info["Reward"] = {term.name: weighted_reward_terms[:, term.index] for term in layout.rewards}
        buffers = (
            reward_terms,
            weighted_reward_terms,
            termination_masks,
        )
        self._kernel_buffers = buffers
        return buffers

    @property
    def _kernel_inputs(self) -> tuple[Any, ...]:
        """Persistent flat kernel-input bundle shared by all three kernels.

        The tuple is built once by the read plan. Simulator-backed arrays are
        rewritten in place by :meth:`_refresh_sim_reads`; every other entry
        aliases host state, so the bundle is never rebuilt.
        """
        assert self._compiled_manager_program is not None
        return self._compiled_manager_program.read_plan.flat_inputs

    def _refresh_sim_reads(self, env_ids: np.ndarray | None = None) -> None:
        """Refresh the simulator-backed kernel inputs in place.

        One native batch read rewrites the simulator-owned arena; ``env_ids``
        refreshes only the selected rows (post-reset rereads).
        """
        assert self._compiled_manager_program is not None
        self._compiled_manager_program.read_plan.read(self, env_ids)

    def _validate_kernel_context(self, inputs: tuple[Any, ...]) -> None:
        assert self._task_program is not None
        kernel = self._task_program.evaluate_kernel
        if not callable(kernel) or not hasattr(kernel, "nopython_signatures"):
            raise TypeError(f"{type(self).__name__} manager compiler must produce a Numba dispatcher.")
        assert self._kernel_buffers is not None
        assert self._kernel_outputs is not None
        try:
            validate_kernel_context(
                inputs,
                self._task_program.reward_weights,
                self._kernel_buffers,
                self._kernel_outputs,
                self.num_envs,
                self._kernel_input_batch_axes(),
            )
        except (TypeError, ValueError) as error:
            message = f"Invalid kernel context for {type(self).__name__} task kernel: {error}"
            raise type(error)(message) from error

    def _kernel_input_batch_axes(self) -> tuple[bool, ...]:
        assert self._compiled_manager_program is not None
        return tuple(
            field.scope is KernelDataScope.PER_ENV
            for source in self._compiled_manager_program.read_plan.sources
            for field in source.fields
        )

    def warmup(self) -> ManagerWarmupResult:
        if self._state is None:
            self.init_state()
        assert self._state is not None
        assert self._task_program is not None
        assert self._compiled_manager_program is not None
        assert self._kernel_buffers is not None
        assert self._kernel_outputs is not None

        self._refresh_sim_reads()
        inputs = self._kernel_inputs
        self._validate_kernel_context(inputs)
        warmup_args = tuple(
            clone_kernel_value(value)
            for value in (inputs, self._task_program.reward_weights, self._kernel_buffers, self._kernel_outputs)
        )
        observe_args = (clone_kernel_value(inputs), clone_kernel_value(self._kernel_outputs))
        started = perf_counter()
        self._compile_manager_specializations(inputs)
        kernel_compile_seconds = perf_counter() - started

        started = perf_counter()
        result = self._task_program.evaluate_kernel(*warmup_args)
        observe_result = self._task_program.observe_kernel(*observe_args)
        first_execution_seconds = perf_counter() - started
        if result is not None or observe_result is not None:
            raise TypeError(f"{type(self).__name__} manager kernels must return None.")
        term_compile_seconds = 0.0

        signatures = (
            tuple(getattr(self._task_program.evaluate_kernel, "nopython_signatures", ()))
            + tuple(getattr(self._task_program.observe_kernel, "nopython_signatures", ()))
            + tuple(getattr(self._task_program.reset_kernel, "nopython_signatures", ()))
        )
        if not signatures:
            raise RuntimeError(f"{type(self).__name__} manager kernels did not produce nopython signatures.")
        warmup = ManagerWarmupResult(
            compile_seconds=term_compile_seconds + kernel_compile_seconds,
            signatures=tuple(str(signature) for signature in signatures),
            threading_layer=get_threading_layer(),
            num_threads=get_num_threads(),
            term_compile_seconds=term_compile_seconds,
            kernel_compile_seconds=kernel_compile_seconds,
            first_execution_seconds=first_execution_seconds,
        )
        return warmup
