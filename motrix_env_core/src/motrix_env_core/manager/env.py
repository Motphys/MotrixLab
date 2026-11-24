# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core.array.env import ArrayEnv, ArrayEnvState, EnvCfgType, EnvObs, FinalObs
from motrix_env_core.base import EnvCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.sim_reset import ManagerResetCfg
from motrix_env_core.manager.actions import ActionCfg, ActionTerm, ManagerActionsCfg
from motrix_env_core.manager.commands import CommandCfg, CommandTerm, ManagerCommandsCfg
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.kernel_data import validate_kernel_data

if TYPE_CHECKING:
    from motrix_env_core.manager.resolver import ManagerLayout, ResolvedManagerProgram
from motrix_env_core.manager.kernel import EvaluationOutputs
from motrix_env_core.manager.metrics import collect_metrics, materialize_metrics
from motrix_env_core.manager.observation_manager import ObservationManager
from motrix_env_core.manager.observations import (
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ObservationGroupEntry,
    ObservationTermCfg,
    create_observation_groups,
)
from motrix_env_core.manager.rewards import ManagerRewardsCfg, RewardTermCfg, create_reward_terms
from motrix_env_core.manager.sim_reset import SimResetRuntime
from motrix_env_core.manager.terminations import (
    ManagerTerminationsCfg,
    TerminationManager,
    TerminationTermCfg,
)
from motrix_env_core.observations import ObservationSpec
from motrix_env_core.sim import (
    ModelQuery,
    PhysicsReadProgram,
    SimDataQuery,
    SimQueriesCfg,
)
from motrix_env_core.sim.backend import RecordConfig, RenderConfig, SimBackend, SimRenderer

if TYPE_CHECKING:
    from motrix_env_core.torch_manager.runtime import TorchTaskProgram
from motrix_env_core.sim.model import ActuatorSpec
from motrix_env_core.sim.write import CtrlTargetsWrite, SimWrite

_CfgT = TypeVar("_CfgT")
_Q = TypeVar("_Q")

logger = logging.getLogger(__name__)


def _validate_reset_ids(env_ids: torch.Tensor, label: str) -> torch.Tensor:
    """Validate one public reset id tensor before it enters the reset pipeline."""
    if not isinstance(env_ids, torch.Tensor):
        raise TypeError(
            f"ManagerEnv.reset {label} must be a torch integer tensor, got {type(env_ids).__name__}; "
            "use torch.tensor([...], dtype=torch.long)."
        )
    if env_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(
            f"ManagerEnv.reset {label} must have an integer dtype, got {env_ids.dtype}; "
            "use torch.tensor([...], dtype=torch.long)."
        )
    return env_ids


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

    backend: str | None = None
    sim_reset: ManagerResetCfg = ManagerResetCfg()
    commands: dict[str, CommandCfg] = field(default_factory=dict)
    actions: dict[str, ActionCfg] = field(default_factory=dict)
    queries: SimQueriesCfg = SimQueriesCfg()
    observations: ManagerObservationsCfg | dict[str, ManagerObservationGroupCfg | dict[str, ObservationTermCfg]] = (
        field(default_factory=dict)
    )
    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)

    def command_cfgs(self) -> dict[str, CommandCfg]:
        return _manager_group_to_dict(self.commands, ManagerCommandsCfg, CommandCfg, label="command")

    def action_cfgs(self) -> dict[str, ActionCfg]:
        return _manager_group_to_dict(self.actions, ManagerActionsCfg, ActionCfg, label="action")

    def model_query_cfgs(self) -> dict[str, ModelQuery]:
        """Return the task-declared model queries."""
        self.queries.validate()
        return dict(self.queries.model)

    def observation_group_cfgs(self) -> dict[str, ManagerObservationGroupCfg | dict[str, ObservationTermCfg]]:
        if isinstance(self.observations, dict):
            groups = self.observations
        elif isinstance(self.observations, ManagerObservationsCfg):
            groups = self.observations.to_dict()
        else:
            raise TypeError(
                "Manager observation config must be a dict or "
                f"ManagerObservationsCfg, got {type(self.observations).__name__}."
            )

        return groups

    def observation_cfgs(self) -> dict[str, dict[str, ObservationTermCfg]]:
        return {
            name: group.to_dict()
            if isinstance(group, ManagerObservationGroupCfg)
            else _manager_group_to_dict(
                group, ManagerObservationGroupCfg, ObservationTermCfg, label=f"observation group {name}"
            )
            for name, group in self.observation_group_cfgs().items()
        }

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
    """Simulator read program backing the manager context's sim views.

    ``read`` executes the compiled read program; the simulator arena the
    context's sim entries view (as zero-copy tensors) is rewritten in place,
    so the tensor views stay valid without rebuilding.
    """

    sim_data: PhysicsReadProgram

    def read(
        self,
        env: ManagerEnv,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        with env.perf.scope("sim_read"):
            self.sim_data.execute(env_ids)


class ManagerEnv(ArrayEnv[EnvCfgType]):
    """Torch-frontend environment whose task program is generated from manager config."""

    def __init__(
        self,
        cfg: EnvCfgType,
        num_envs: int = 1,
        backend: str | None = None,
        seed: int | None = None,
        *,
        headless: bool = False,
        record: RecordConfig | None = None,
    ) -> None:
        """Construct a manager environment using the selected simulator backend.

        The numerical task program runs on the generic Torch executor
        (:class:`motrix_env_core.torch_manager.runtime.TorchManagerRuntime`).
        """
        if not isinstance(cfg, ManagerBasedEnvCfg):
            raise TypeError(f"{type(cfg).__name__} must inherit ManagerBasedEnvCfg.")
        super().__init__(cfg, num_envs)
        self._build_seed = 1 if seed is None else seed
        from motrix_env_core.manager.random import RandomSource

        # Root Torch seeding belongs to the training entry point, not env construction.
        self._random = RandomSource()
        from motrix_env_core.sim.registry import create_sim_backend, default_sim_backend_name

        factory = create_sim_backend(backend or cfg.backend or default_sim_backend_name())
        # Construction compiles the scene inside the backend: no compiled
        # artifact crosses the boundary.
        render = RenderConfig(headless=headless, record=record)
        self.sim: SimBackend = factory(cfg.scene, cfg.sim, num_envs, render=render)
        self.model = self.sim.compile_model(cfg.model_query_cfgs())
        # Full-width ctrl buffer in canonical actuator order, owned by the
        # compiled ctrl write program; routing happens here, the backend only
        # receives the merged targets.
        # BuildContext availability follows this fixed order (see its docstring):
        # commands -> sim resets -> actions -> rewards -> terminations -> observations.
        # Commands see preceding commands; resets/actions see all commands but no
        # actions. Only the last three groups see both complete collections.
        self._action_terms = {}
        self._command_cfgs = cfg.command_cfgs()
        self._command_terms: dict[str, CommandTerm] = {}
        for name, command_cfg in self._command_cfgs.items():
            self._command_terms[name] = command_cfg(self.build_context())
        for name in list(self._command_terms):
            term = self._command_terms[name]
            self._command_terms[name] = validate_kernel_data(
                materialize_metrics(term, self.num_envs, context=f"Manager command {name!r}"),
                context=f"Manager command {name!r}",
            )
        self._sim_reset_runtime = SimResetRuntime.create(self.build_context(), cfg.sim_reset, self.sim)

        self._task_program: TorchTaskProgram | None = None
        self._kernel_buffers: tuple[Any, ...] | None = None
        self.metrics: dict[str, torch.Tensor] = {}
        # Persistent kernel-bound sim-reset-request buffer
        # (``ctx.sim_reset_requested``), cleared before each physics step and
        # consumed by _reset_done_envs.
        self._sim_reset_requested = torch.zeros((self.num_envs, 1), dtype=torch.bool)
        self._compiled_manager_program: ResolvedManagerProgram | None = None
        self._action_cfgs = cfg.action_cfgs()
        if not self._action_cfgs:
            raise ValueError("Manager environment config requires at least one action config.")
        self._action_actuators = self._resolve_action_actuators()
        self._action_writes = self.sim.compile_writes(
            {
                name: CtrlTargetsWrite(action_cfg.actuator_names)
                for name, action_cfg in self._action_cfgs.items()
                if action_cfg.actuator_names != ()
            }
        )
        self._action_terms: dict[str, ActionTerm] = {
            name: action_cfg(
                self.build_context(),
                self._action_actuators[name],
            )
            for name, action_cfg in self._action_cfgs.items()
        }
        for name, term in self._action_terms.items():
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Manager action {name!r} must return an ActionTerm.")
            queue = term.state.action_queue
            if queue.ndim != 3 or queue.shape[0] != self.num_envs or queue.shape[1] < 2:
                raise ValueError(f"Manager action {name!r} state.action_queue must have shape (N, W, A), W >= 2.")
            if queue.shape[2] != term.action_space.shape[0]:
                raise ValueError(f"Manager action {name!r} action dimension must match action space.")
        self._action_space, self._action_slices = self._build_action_space()
        self._reward_terms = create_reward_terms(cfg.reward_cfgs(), self.build_context())
        self.termination_manager = TerminationManager(cfg.termination_cfgs(), self)
        self.observation_manager = ObservationManager(
            create_observation_groups(cfg, self.build_context()), self.num_envs
        )
        # Terms are built before the read program: any SimDataQuery passed as
        # a term argument is collected here, merged with task-declared
        # queries, and compiled in one batch.
        self._term_query_keys: dict[Any, str] = {}
        declared = dict(cfg.queries.data)
        collected_index = 0
        for terms in (
            (term for term in self._reward_terms.values()),
            (term for term in self.termination_manager.terms.values()),
            (entry.term for group in self.observation_manager.groups.values() for entry in group.terms),
        ):
            for term in terms:
                for arg in term.args:
                    if isinstance(arg, SimDataQuery) and arg not in self._term_query_keys:
                        key = f"terms.args[{collected_index}]"
                        collected_index += 1
                        self._term_query_keys[arg] = key
                        existing = declared.get(key)
                        if existing is not None and existing != arg:
                            raise ValueError(f"Unequal declarations under term-arg key {key!r}.")
                        declared[key] = arg
        merged = _merge_query_declarations(declared, (), label="simulator data")
        self.sim_data: PhysicsReadProgram = self.sim.compile_reads(merged)
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

    def build_context(self) -> BuildContext:
        return BuildContext(
            cfg=self.cfg,
            actions=self._action_terms,
            commands=self._command_terms,
            model=self.model,
            num_envs=self.num_envs,
            seed=self._build_seed,
        )

    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        return self.sim.create_renderer(
            config,
            num_envs=self.num_envs,
            render_spacing=self.render_spacing,
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
        self.sim.step(self._cfg.sim_substeps, result=self.engine_error_codes)

    def _new_state(
        self,
        obs: EnvObs,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        episode_steps: torch.Tensor,
    ) -> ArrayEnvState:
        """Assemble the state over the context-bound termination buffer.

        The resolver's context is already live (the task program builds
        before the base lifecycle runs), so ``ctx.terminated``, the evaluate
        kernel's output, and ``state.terminated`` are one tensor from birth —
        the manager state is the only named holder on the env side.
        """
        context = self._compiled_manager_program.context
        context.terminated.fill_(True)  # keep the seed-all-lanes init contract
        return super()._new_state(
            self.observation_manager.observation, context.reward, context.terminated, truncated, episode_steps
        )

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
        logger.info("Manager env %r startup complete in %.3fs", env_name, perf_counter() - startup_started)
        return state

    def _prev_physics_step(self) -> None:
        self._sim_reset_requested.fill_(False)
        super()._prev_physics_step()
        self._state = self._state.replace(obs=self.observation_manager.begin_step())

    def _execute_evaluate_kernel(self) -> None:
        """Evaluate rewards and terminations against the current command."""
        assert self._task_program is not None
        assert self._kernel_buffers is not None
        self._task_program.evaluate_kernel(
            self._task_program.reward_weights,
            self._kernel_buffers,
            EvaluationOutputs(reward=self._state.reward, terminated=self._state.terminated),
        )

    def _capture_final_obs(self, env_ids: torch.Tensor) -> FinalObs:
        return self.observation_manager.preview(env_ids, self._task_program.observe_kernel)

    def _resolve_action_actuators(self) -> dict[str, tuple[ActuatorSpec, ...] | None]:
        routes = {}
        owners = {}
        by_name = {spec.name: spec for spec in self.model.actuators}
        for term_name, cfg in self._action_cfgs.items():
            if cfg.actuator_names == ():
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
            space = term.action_space
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
    def observation_spec(self) -> ObservationSpec:
        return self.observation_manager.spec

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        expected_shape = (self.num_envs, self.action_space.shape[0])
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {tuple(actions.shape)}.")
        for name, term in self._action_terms.items():
            controls = term.process(actions[:, self._action_slices[name]])
            actuators = self._action_actuators[name]
            if actuators is None:
                assert controls is None
                continue
            assert controls is not None
            # Buffers are torch tensors aliasing the compiled write program's
            # storage; the copy_ is the single handoff into the backend.
            self._action_writes.buffer(name).copy_(controls)
        self._action_writes.execute()
        return state

    def _reset_sim_rows(self, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        """Run the simulator-only stages of the reset pipeline for selected rows.

        Clears frontend ctrl targets to mirror the backend's reset-row ctrl
        clearing, reruns the reset kernel (command ``reset`` hooks followed
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
            self._sim_reset_runtime.apply(self._task_program, env_ids, episode_reset)
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
            sim_reset_mask = self._sim_reset_requested[:, 0]
            sim_reset_ids = torch.nonzero(sim_reset_mask & ~done, as_tuple=False).flatten()
            env_ids = torch.nonzero(done, as_tuple=False).flatten()
        if not env_ids.numel() and not sim_reset_ids.numel():
            return
        if env_ids.numel():
            self._clear_engine_error_rewards()
            with self.perf.scope("select_done"):
                state.episode_steps[env_ids] = 0
        with self.perf.scope("reset_envs"):
            self._reset_rows(
                env_ids,
                None if not sim_reset_ids.numel() else sim_reset_ids,
            )

    def reset(self, env_ids: torch.Tensor, sim_reset_ids: torch.Tensor | None = None) -> None:
        """Reset episode lanes fully and sim-reset the rest in one pass.

        ``env_ids`` and ``sim_reset_ids`` are integer torch tensors (the
        manager frontend's public currency).

        Episode lanes (``env_ids``) run the host lifecycle resets (action
        resets); sim-reset lanes (``ctx.sim_reset_requested`` set
        mid-transition) skip them — with no lane terminating, they would not
        change any state. Both sets join one reset-kernel run: command
        ``reset`` hooks (with ``episode_reset`` marking episode lanes)
        resample lane state, the configured sim reset terms rewrite simulator
        rows, and kernel inputs are reread once for the union. Simulator-only
        resets do not publish a new observation at the same control time;
        ordinary compute calls retain the published window until the next step
        appends its endpoint. Episode-reset lanes instead publish a first frame
        on the next compute call.
        """
        env_ids = _validate_reset_ids(env_ids, "env_ids")
        if sim_reset_ids is not None:
            sim_reset_ids = _validate_reset_ids(sim_reset_ids, "sim_reset_ids")
        if self._state.final_obs is not None:
            self._state = self._state.replace(final_obs=None)
        self._reset_rows(env_ids, sim_reset_ids)
        self._state.episode_steps[env_ids] = 0

    def _reset_rows(self, env_ids: torch.Tensor, sim_reset_ids: torch.Tensor | None = None) -> None:
        """Torch-control-plane reset shared by the public wrapper and the
        internal auto-reset flow."""
        assert self._state is not None
        if env_ids.numel():
            self.observation_manager.reset(env_ids)
            with self.perf.scope("action_reset"):
                for action_term in self._action_terms.values():
                    action_term.reset(env_ids)
        # Bool (K,) flags align with the merged env_ids: True for full episode
        # resets (also initialization/explicit restarts), False for sim-only
        # resets that continue the episode. They do not encode termination.
        # Context remains whole-batch and never carries reset-pass scratch.
        episode_reset = torch.ones_like(env_ids, dtype=torch.bool)
        if sim_reset_ids is not None and sim_reset_ids.numel():
            if env_ids.numel():
                # Episode reset wins when the public caller supplies overlapping sets.
                sim_reset_ids = sim_reset_ids[~torch.isin(sim_reset_ids, env_ids)]
            episode_reset = torch.cat([episode_reset, torch.zeros_like(sim_reset_ids, dtype=torch.bool)])
            env_ids = torch.cat([env_ids, sim_reset_ids])
        if env_ids.numel():
            self._reset_sim_rows(env_ids, episode_reset)

    def _make_metrics_view(self) -> dict[str, Any]:
        """Assemble the persistent live metrics view for the current state.

        Values are views into the kernel reward/termination buffers and the
        term-owned CPU tensor arrays, so kernels write them in place every
        step and nothing needs rebuilding per transition. Host-side writers
        may add batch-level scalar entries (e.g. sampler statistics at reset
        time). Consumers retain values across steps through
        ``ArrayEnvState.process_metrics()``.
        """
        assert self._compiled_manager_program is not None
        assert self._kernel_buffers is not None
        termination_masks = self._kernel_buffers[2]
        metrics = {
            term.name: termination_masks[:, term.index] for term in self._compiled_manager_program.layout.terminations
        }
        # Single-column term metrics export as ``(N,)`` per-env vectors.
        metrics.update(
            {
                name: (value[:, 0] if value.ndim == 2 and value.shape[1] == 1 else value)
                for name, value in self.metrics.items()
            }
        )
        return metrics

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        with self.perf.scope("read_inputs"):
            self._refresh_sim_reads()
        with self.perf.scope("evaluate"):
            self._execute_evaluate_kernel()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        """Publish a new whole-batch frame and append observation history."""
        with self.perf.scope("observe"):
            self.observation_manager.commit(self._task_program.observe_kernel)
        return state

    def reset_observation(self, state: ArrayEnvState, env_ids: torch.Tensor) -> ArrayEnvState:
        """Repeat-fill explicit episode-reset rows without advancing other lanes."""
        with self.perf.scope("observe"):
            self.observation_manager.publish_reset(env_ids, self._task_program.observe_kernel)
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

    def _build_task_program(self) -> TorchTaskProgram:
        from motrix_env_core.manager.resolver import ManagerProgramResolver
        from motrix_env_core.torch_manager.runtime import TorchManagerRuntime

        self._compiled_manager_program = ManagerProgramResolver(self).build()
        self._torch_runtime = TorchManagerRuntime(self, self._compiled_manager_program)
        return self._torch_runtime.task_program()

    @property
    def observation_groups(self) -> dict[str, ObservationGroupEntry]:
        """Return the environment-local runtime observation groups."""
        return self.observation_manager.groups

    def _make_kernel_buffers(self, state: ArrayEnvState) -> tuple[torch.Tensor, ...]:
        assert self._compiled_manager_program is not None
        layout = self._compiled_manager_program.layout
        reward_terms = torch.empty((self.num_envs, len(layout.rewards)), dtype=torch.float32)
        weighted_reward_terms = torch.empty_like(reward_terms)
        termination_masks = torch.empty((self.num_envs, len(layout.terminations)), dtype=torch.bool)
        state.reward_terms = {term.name: weighted_reward_terms[:, term.index] for term in layout.rewards}
        buffers = (
            reward_terms,
            weighted_reward_terms,
            termination_masks,
        )
        self._kernel_buffers = buffers
        return buffers

    def _refresh_sim_reads(self, env_ids: torch.Tensor | None = None) -> None:
        """Refresh the simulator-backed kernel inputs in place.

        One native batch read rewrites the simulator-owned arena; ``env_ids``
        refreshes only the selected rows (post-reset rereads).
        """
        assert self._compiled_manager_program is not None
        self._compiled_manager_program.read_plan.read(self, env_ids)
