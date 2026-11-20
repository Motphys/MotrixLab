# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Single-joint Manager task with simulator-free analytic TaskMDP heads.

The small CPU backend implements real Manager read arenas, control writes and
row-scoped reset writes; it does not depend on an installed physics engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import gymnasium as gym
import torch

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.config.sim_reset import ManagerResetCfg, ResetTermCfg
from motrix_env_core.manager.actions import ActionCfg, ActionState, ActionTerm
from motrix_env_core.manager.commands import CommandCfg, CommandTerm
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.observations import ManagerObservationGroupCfg, ObservationTermCfg, ObsTerm
from motrix_env_core.manager.rewards import RewardTerm, RewardTermCfg
from motrix_env_core.manager.sim_reset import ResetTerm
from motrix_env_core.manager.terminations import TerminationTerm, TerminationTermCfg
from motrix_env_core.sim.backend import RenderConfig, SimBackend
from motrix_env_core.sim.model import ActuatorSpec, ActuatorType, ModelQuery, SimModel
from motrix_env_core.sim.read import DofPositionQuery, DofVelocityQuery, PhysicsReadProgram, SimDataQuery
from motrix_env_core.sim.write import CtrlTargetsWrite, SimWrite, WriteProgram
from motrix_rl_supermimic.imagination import ImaginationBindings, ImaginationCfg, ImaginationInputs, ImaginationTerm
from motrix_rl_supermimic.mbrl_env import MbrlEnv, MbrlEnvCfg
from motrix_rl_supermimic.reference_command import ReferenceCommand
from motrix_rl_supermimic.world_state import EuclideanState, WorldStateTerm, WorldStateTermCfg

TARGETS = (0.0, 0.5, -0.5, 1.0)


@dataclass(frozen=True, slots=True)
class ToyReference:
    frame: torch.Tensor
    context: torch.Tensor


def _policy_term(ctx: ManagerContext, out: torch.Tensor) -> None:
    theta, velocity = ctx.sim["position"], ctx.sim["velocity"]
    out.copy_(torch.cat((theta.sin(), theta.cos(), velocity, ctx.actions["velocity"].current()), -1))


def _value_term(ctx: ManagerContext, out: torch.Tensor, targets: torch.Tensor) -> None:
    theta = ctx.sim["position"]
    target = targets[ctx.commands["reference"].frame % len(TARGETS)][:, None]
    out.copy_(torch.cat((theta, ctx.sim["velocity"], target - theta), -1))


def _reward_term(ctx: ManagerContext, targets: torch.Tensor) -> torch.Tensor:
    action = ctx.actions["velocity"]
    target = targets[ctx.commands["reference"].frame % len(TARGETS)]
    error = (target - ctx.sim["position"][:, 0]).square()
    rate = (action.current() - action.previous()).square().sum(-1)
    return (torch.exp(-error / 0.25) - 0.1 * rate) / ctx.dt


def _soft_term(ctx: ManagerContext) -> torch.Tensor:
    return torch.sigmoid(ctx.sim["position"][:, 0].abs() - 1.5)


def _goal(frame: torch.Tensor, horizon: int, targets: torch.Tensor) -> torch.Tensor:
    return targets[frame % len(TARGETS)][:, None].expand(-1, horizon)


def _noise(value: torch.Tensor) -> torch.Tensor:
    return value + torch.randn_like(value) * 1e-4


@dataclass(kw_only=True)
class ToyImaginationTerm(ImaginationTerm):
    def bind(self, inputs: ImaginationInputs) -> ImaginationBindings:
        state = inputs.state["joint"]
        return ImaginationBindings(
            commands=Map({"reference": ToyReference(inputs.frame, state.new_empty((inputs.frame.numel(), 0)))}),
            sim=Map({"position": state[:, 0:1], "velocity": state[:, 1:2]}),
        )

    def bind_reference(self, frame: torch.Tensor) -> ImaginationBindings:
        return ImaginationBindings(
            commands=Map({"reference": ToyReference(frame, torch.empty(frame.numel(), 0, device=frame.device))}),
            sim=Map({}),
        )

    def valid_starts(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        return torch.ones_like(frame, dtype=torch.bool)


@configclass
class ToyImaginationCfg(ImaginationCfg):
    def __call__(self, ctx: BuildContext) -> ToyImaginationTerm:
        return ToyImaginationTerm()


def _flat_state(position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    return torch.cat((position, velocity), -1)


def _capture_state(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(_flat_state(ctx.sim["position"], ctx.sim["velocity"]))


@configclass
class ToyWorldStateCfg(WorldStateTermCfg):
    def __call__(self, ctx: BuildContext) -> WorldStateTerm:
        return WorldStateTerm(_capture_state, EuclideanState(size=2))


class _ToyReads(PhysicsReadProgram):
    def __init__(self, backend: _ToyBackend, queries: Mapping[str, SimDataQuery]) -> None:
        self.backend = backend
        self.queries = dict(queries)
        self.views = {key: torch.zeros_like(backend.position) for key in queries}
        for query in queries.values():
            if not isinstance(query, (DofPositionQuery, DofVelocityQuery)):
                raise ValueError(f"Unsupported Manager toy read: {query!r}")

    @property
    def arena_bytes(self) -> int:
        return sum(view.numel() * view.element_size() for view in self.views.values())

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.queries)

    def query(self, key: str) -> SimDataQuery:
        return self.queries[key]

    def view(self, key: str) -> torch.Tensor:
        return self.views[key]

    def execute(self, env_ids: torch.Tensor | None = None) -> None:
        rows = slice(None) if env_ids is None else env_ids
        for key, query in self.queries.items():
            source = self.backend.position if isinstance(query, DofPositionQuery) else self.backend.velocity
            self.views[key][rows] = source[rows]


class _ToyWrites(WriteProgram):
    def __init__(self, backend: _ToyBackend, writes: Mapping[str, SimWrite], reset: bool) -> None:
        self.backend = backend
        self.reset = reset
        self.buffers = {}
        for key, write in writes.items():
            if not isinstance(write, CtrlTargetsWrite) or write.actuators not in (None, ("joint",)):
                raise ValueError(f"Unsupported Manager toy write: {write!r}")
            self.buffers[key] = torch.zeros_like(backend.velocity)

    def buffer(self, name: str) -> torch.Tensor:
        return self.buffers[name]

    def execute(self, env_ids: torch.Tensor | None = None) -> None:
        rows = slice(None) if env_ids is None else env_ids
        if self.reset:
            self.backend.position[rows] = 0.0
            self.backend.velocity[rows] = 0.0
            self.backend.reset_rows.append(torch.arange(self.backend.position.shape[0])[rows].clone())
        for buffer in self.buffers.values():
            self.backend.velocity[rows] = buffer[rows]


class _ToyBackend(SimBackend):
    name = "supermimic-toy-sim"

    def __init__(self, scene: SceneCfg, sim: SimCfg, num_envs: int, *, render: RenderConfig | None = None) -> None:
        super().__init__(scene, sim, num_envs, render=render)
        self.position = torch.zeros(num_envs, 1)
        self.velocity = torch.zeros(num_envs, 1)
        self.reset_rows: list[torch.Tensor] = []

    @property
    def num_dof_pos(self) -> int:
        return 1

    @property
    def num_dof_vel(self) -> int:
        return 1

    @property
    def num_actuators(self) -> int:
        return 1

    @property
    def model_compiler(self) -> _ToyBackend:
        return self

    def compile(self, scene: SceneCfg, queries: Mapping[str, ModelQuery]) -> SimModel:
        if queries:
            raise ValueError("Manager toy has no model queries")
        return SimModel(
            actuators=(ActuatorSpec("joint", ActuatorType.VELOCITY, "joint", (-1.0, 1.0), None),),
            init_dof_pos=torch.zeros(1),
        )

    def compile_reads(self, queries: Mapping[str, SimDataQuery]) -> PhysicsReadProgram:
        return _ToyReads(self, queries)

    def compile_writes(
        self, writes: Mapping[str, SimWrite], *, reset: bool = False, forward_kinematics: bool = True
    ) -> WriteProgram:
        return _ToyWrites(self, writes, reset)

    def step(self, substeps: int, result: torch.Tensor | None = None) -> None:
        self.position.add_(self.velocity * substeps)
        if result is not None:
            result.zero_()


class _VelocityAction(ActionTerm):
    def _process(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def _reset(self, env_ids: torch.Tensor) -> None:
        pass


@configclass(kw_only=True)
class _VelocityActionCfg(ActionCfg):
    def __call__(self, ctx: BuildContext, actuators: tuple[ActuatorSpec, ...] | None) -> ActionTerm:
        return _VelocityAction(
            gym.spaces.Box(-1.0, 1.0, (1,)),
            ActionState(
                torch.zeros(ctx.num_envs, 2, 1),
                torch.zeros(1, dtype=torch.long),
            ),
        )


@dataclass(frozen=True, slots=True)
class _ToyCommand(ReferenceCommand):
    context: torch.Tensor
    targets: torch.Tensor

    def update(self, ctx: ManagerContext) -> None:
        # Evaluate endpoint rewards at the advanced reference frame.
        self.frame.copy_((self.frame + 1) % len(TARGETS))
        self.command.copy_(self.targets[self.frame].unsqueeze(-1))

    def advance(self, ctx: ManagerContext) -> None:
        pass

    def reset(self, ctx: ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        self.frame[env_ids] = torch.where(episode_reset, 0, self.frame[env_ids])
        self.context[env_ids] = 0.0
        self.command[env_ids] = self.targets[self.frame[env_ids]].unsqueeze(-1)


@configclass(kw_only=True)
class _ToyCommandCfg(CommandCfg):
    def __call__(self, ctx: BuildContext) -> CommandTerm:
        n = ctx.num_envs
        targets = torch.tensor(TARGETS)
        return _ToyCommand(
            command=torch.zeros(n, 1),
            frame=torch.zeros(n, dtype=torch.long),
            context=torch.zeros(n, 0),
            targets=targets,
        )


@configclass(kw_only=True)
class _ToyObservationCfg(ObservationTermCfg):
    critic: bool = False
    noise: bool = True

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        reference = ctx.commands["reference"]
        if not isinstance(reference, _ToyCommand):
            raise TypeError("Toy observations require a toy reference command")
        if self.critic:
            return ObsTerm(3, _value_term, reference.targets)
        if not self.noise:
            return ObsTerm(4, _policy_term)

        def observe(ctx: ManagerContext, out: torch.Tensor) -> None:
            _policy_term(ctx, out)
            out.copy_(_noise(out))

        return ObsTerm(4, observe)


def _goal_term(ctx: ManagerContext, out: torch.Tensor, horizon: int, targets: torch.Tensor) -> None:
    out.copy_(_goal(ctx.commands["reference"].frame, horizon, targets))


@configclass(kw_only=True)
class _ToyGoalCfg(ObservationTermCfg):
    def __call__(self, ctx: BuildContext) -> ObsTerm:
        reference = ctx.commands["reference"]
        if not isinstance(reference, _ToyCommand):
            raise TypeError("Toy goals require a toy reference command")
        return ObsTerm(ctx.cfg.goal_frames, _goal_term, ctx.cfg.goal_frames, reference.targets)


@configclass(kw_only=True)
class _ToyRewardCfg(RewardTermCfg):
    def __call__(self, ctx: BuildContext) -> RewardTerm:
        reference = ctx.commands["reference"]
        if not isinstance(reference, _ToyCommand):
            raise TypeError("Toy reward requires a toy reference command")
        return RewardTerm(_reward_term, reference.targets)


def _terminated(ctx: ManagerContext) -> torch.Tensor:
    return ctx.sim["position"][:, 0].abs() > 2.0


@configclass(kw_only=True)
class _ToyTerminationCfg(TerminationTermCfg):
    def __call__(self, ctx: BuildContext) -> TerminationTerm:
        return TerminationTerm(_terminated)


@configclass
class _ToyResetCfg(ResetTermCfg):
    def __call__(self, ctx: BuildContext) -> ResetTerm:
        def reset(ctx: ManagerContext, env_ids: torch.Tensor, sim_writes: Map[torch.Tensor]) -> None:
            sim_writes["velocity"].zero_()

        return ResetTerm(reset, writes={"velocity": CtrlTargetsWrite(actuators=("joint",))})


@configclass
class _ToyResetsCfg(ManagerResetCfg):
    state: ResetTermCfg = _ToyResetCfg()


@configclass(kw_only=True)
class _ToySoftCfg(TerminationTermCfg):
    def __call__(self, ctx: BuildContext) -> TerminationTerm:
        return TerminationTerm(_soft_term)


@configclass
class _ToyPolicyGroupCfg(ManagerObservationGroupCfg):
    history_length: int = 2
    task: ObservationTermCfg = _ToyObservationCfg()


@configclass
class ToyEnvCfg(MbrlEnvCfg):
    backend: str | None = _ToyBackend.name
    scene: SceneCfg = SceneCfg()
    sim_reset: ManagerResetCfg = _ToyResetsCfg()
    world_model_state: dict[str, WorldStateTermCfg] = {"joint": ToyWorldStateCfg()}
    imagination: ToyImaginationCfg = ToyImaginationCfg()
    soft_termination: TerminationTermCfg = _ToySoftCfg()

    def __post_init__(self) -> None:
        self.commands = {"reference": _ToyCommandCfg()}
        self.actions = {"velocity": _VelocityActionCfg()}
        self.queries.data = {"position": DofPositionQuery(), "velocity": DofVelocityQuery()}
        self.observations = {
            "policy": _ToyPolicyGroupCfg(),
            "value": {"task": _ToyObservationCfg(critic=True)},
            "goal": {"reference": _ToyGoalCfg()},
        }
        self.rewards = {"task": _ToyRewardCfg(weight=1.0)}
        self.terminations = {"limit": _ToyTerminationCfg()}


_registered = False


def register() -> None:
    """Register the CPU backend and Manager task exactly once."""
    global _registered
    if not _registered:
        from motrix_env_core.sim.registry import register_sim_backend

        register_sim_backend(_ToyBackend.name, lambda: _ToyBackend)
        registry.envcfg("supermimic-toy")(ToyEnvCfg)
        registry.env("supermimic-toy")(MbrlEnv)
        _registered = True
