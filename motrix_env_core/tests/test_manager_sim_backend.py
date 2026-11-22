# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Fake-backend tests for the manager SimBackend boundary (issue #222)."""

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.config.sim_reset import ManagerResetCfg, ResetTermCfg
from motrix_env_core.manager.actions import ActionCfg, ActionState, ActionTerm, ManagerActionsCfg
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.env import ManagerBasedEnvCfg, ManagerEnv
from motrix_env_core.manager.observations import (
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ObservationTermCfg,
    ObsTerm,
)
from motrix_env_core.manager.rewards import ManagerRewardsCfg, RewardTerm, RewardTermCfg
from motrix_env_core.manager.sim_reset import ResetTerm
from motrix_env_core.manager.terminations import (
    ManagerTerminationsCfg,
    TerminationTerm,
    TerminationTermCfg,
)
from motrix_env_core.sim import (
    BodyJointPositionQuery,
    ModelQuery,
    PhysicsReadProgram,
    SimQueriesCfg,
)
from motrix_env_core.sim.backend import SimBackend
from motrix_env_core.sim.model import ActuatorSpec, ActuatorType, SimModel
from motrix_env_core.sim.registry import register_sim_backend
from motrix_env_core.sim.write import CtrlTargetsWrite, WriteProgram

_ACTUATORS = (
    ActuatorSpec(
        name="unrouted",
        actuator_type=ActuatorType.POSITION,
        target_name="unrouted",
        ctrl_range=None,
        force_range=None,
    ),
    ActuatorSpec(
        name="routed",
        actuator_type=ActuatorType.POSITION,
        target_name="routed",
        ctrl_range=None,
        force_range=None,
    ),
)


def _core_model() -> SimModel:
    """The required core layout every backend must surface as ``env.model``."""
    return SimModel(
        actuators=_ACTUATORS,
        init_dof_pos=torch.tensor([0.5, -0.5], dtype=torch.float32),
    )


class _FakeWriteProgram(WriteProgram):
    def __init__(self, backend: "_FakeBackend", writes, reset: bool) -> None:
        self._backend = backend
        self._reset = reset
        all_names = tuple(spec.name for spec in _ACTUATORS)
        self._routes = {}
        self._buffers = {}
        for name, write in writes.items():
            if isinstance(write, CtrlTargetsWrite):
                route = (
                    tuple(range(len(all_names)))
                    if write.actuators is None
                    else tuple(all_names.index(target) for target in write.actuators)
                )
                self._routes[name] = route
                self._buffers[name] = torch.zeros((backend.num_envs, len(route)), dtype=torch.float32)

    def buffer(self, name: str) -> torch.Tensor:
        return self._buffers[name]

    def execute(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self._backend.num_envs, dtype=torch.int64) if env_ids is None else env_ids
        if self._reset:
            self._backend.dof_pos[ids] = torch.tensor([0.5, -0.5], dtype=torch.float32)
            self._backend.dof_vel[ids] = 0.0
            for name, buffer in self._buffers.items():
                if name.endswith("_position"):
                    self._backend.dof_pos[ids] = buffer[ids]
                elif name.endswith("_velocity"):
                    self._backend.dof_vel[ids] = buffer[ids]
            self._backend.reset_calls.append((ids.clone(), self._backend.dof_pos[ids].clone()))
            self._backend.ctrl_targets[ids] = 0.0
        for name, route in self._routes.items():
            route_index = torch.tensor(route, dtype=torch.int64)
            self._backend.ctrl_targets[ids[:, None], route_index] = self.buffer(name)[ids]
        if self._routes:
            self._backend.ctrl_write_count += 1


class _FakeWriteCompiler:
    def __init__(self, backend: "_FakeBackend") -> None:
        self._backend = backend

    def compile(self, writes, *, reset: bool = False, forward_kinematics: bool = True) -> WriteProgram:
        del forward_kinematics
        return _FakeWriteProgram(self._backend, writes, reset)


class _FakeReadProgram(PhysicsReadProgram):
    """Reads the fake runtime's in-memory DOF state into a program-owned arena."""

    def __init__(self, backend: "_FakeBackend", queries: dict) -> None:
        self._backend = backend
        self._queries = dict(queries)
        keys = tuple(queries)
        width = 2
        self._arena = torch.zeros((backend.num_envs * width,), dtype=torch.float32)
        self._output = self._arena.reshape(backend.num_envs, width)
        self._views = {key: self._arena.reshape(backend.num_envs, width) for key in keys}

    @property
    def arena_bytes(self) -> int:
        return self._arena.numel() * self._arena.element_size()

    def view(self, key: str) -> torch.Tensor:
        return self._views[key]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._views)

    def query(self, key: str):
        return self._queries[key]

    def execute(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._output.copy_(self._backend.dof_pos)
        else:
            self._output[env_ids] = self._backend.dof_pos[env_ids]


class _FakeBackend(SimBackend):
    """In-memory backend that records every behavioral call."""

    name = "fake"

    last: "_FakeBackend | None" = None

    def __init__(self, scene, sim, num_envs: int, *, render=None) -> None:
        super().__init__(scene, sim, num_envs, render=render)
        self.num_envs = num_envs
        self.dof_pos = torch.zeros((num_envs, self.num_dof_pos), dtype=torch.float32)
        self.dof_vel = torch.zeros((num_envs, self.num_dof_vel), dtype=torch.float32)
        self.ctrl_targets = torch.zeros((num_envs, self.num_actuators), dtype=torch.float32)
        self.step_calls: list[int] = []
        self.status_codes = torch.zeros(num_envs, dtype=torch.int32)
        self.reset_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        self.ctrl_write_count = 0
        self._write_compiler = _FakeWriteCompiler(self)
        _FakeBackend.last = self

    @property
    def num_dof_pos(self) -> int:
        return 2

    @property
    def num_dof_vel(self) -> int:
        return 2

    @property
    def num_actuators(self) -> int:
        return 2

    def step(self, substeps: int, result: torch.Tensor | None = None) -> None:
        self.step_calls.append(substeps)
        self.dof_pos += 0.25 * substeps
        if result is not None:
            result.copy_(self.status_codes)

    @property
    def model_compiler(self):
        return self

    def compile(self, scene, queries: Mapping[str, ModelQuery]) -> SimModel:
        del scene, queries
        return _core_model()

    def compile_reads(self, queries) -> PhysicsReadProgram:
        return _FakeReadProgram(self, queries)

    @property
    def write_compiler(self):
        return self._write_compiler


@dataclass(frozen=True, slots=True)
class _CtrlActionState(ActionState):
    source: torch.Tensor


class _CtrlActionTerm(ActionTerm):
    def __init__(self, action_space: gym.spaces.Box, state: _CtrlActionState) -> None:
        super().__init__(action_space, state)

    def _process(self, actions: torch.Tensor) -> torch.Tensor:
        self.state.source.copy_(actions)
        return actions * 2.0

    def _reset(self, env_ids: np.ndarray) -> None:
        self.state.source[env_ids] = 0.0


def _ctrl_action_space(ctx, actuator_indices) -> gym.spaces.Box:
    del ctx, actuator_indices
    return gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)


@configclass(kw_only=True)
class _CtrlActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] | None = ("routed",)

    def __call__(self, ctx, actuator_indices):
        return _CtrlActionTerm(
            _ctrl_action_space(ctx, actuator_indices),
            _CtrlActionState(
                torch.from_numpy(np.zeros((ctx.num_envs, 2, 1), dtype=np.float32)),
                torch.from_numpy(np.zeros(1, dtype=np.int64)),
                torch.from_numpy(np.zeros((ctx.num_envs, 1), dtype=np.float32)),
            ),
        )


def _sim_reading_observation(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.sim["dof_pos"])


@configclass(kw_only=True)
class _SimReadingObservationCfg(ObservationTermCfg):
    def __call__(self, env):
        return ObsTerm(2, _sim_reading_observation)


@configclass
class _FakePolicyObsCfg(ManagerObservationGroupCfg):
    sim_read: _SimReadingObservationCfg = _SimReadingObservationCfg()


@configclass
class _FakeObservationsCfg(ManagerObservationsCfg):
    policy: _FakePolicyObsCfg = _FakePolicyObsCfg()


def _unit_reward(ctx: ManagerContext) -> torch.Tensor:
    return torch.ones_like(ctx.env_id, dtype=torch.float32)


@configclass(kw_only=True)
class _UnitRewardCfg(RewardTermCfg):
    weight: float = 1.0

    def __call__(self, env):
        return RewardTerm(_unit_reward)


@configclass
class _FakeRewardsCfg(ManagerRewardsCfg):
    unit: _UnitRewardCfg = _UnitRewardCfg()


def _never_termination(ctx: ManagerContext) -> torch.Tensor:
    return torch.zeros_like(ctx.env_id, dtype=torch.bool)


@configclass(kw_only=True)
class _NeverTerminationCfg(TerminationTermCfg):
    def __call__(self, env):
        return TerminationTerm(_never_termination)


@configclass
class _FakeTerminationsCfg(ManagerTerminationsCfg):
    never: _NeverTerminationCfg = _NeverTerminationCfg()


@configclass
class _FakeActionsCfg(ManagerActionsCfg):
    ctrl: _CtrlActionCfg = _CtrlActionCfg()


register_sim_backend("fake-manager", lambda: _FakeBackend)


@configclass
class _FakeManagerCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = SceneCfg()
    actions: ManagerActionsCfg = _FakeActionsCfg()
    observations: ManagerObservationsCfg = _FakeObservationsCfg()
    rewards: ManagerRewardsCfg = _FakeRewardsCfg()
    terminations: ManagerTerminationsCfg = _FakeTerminationsCfg()
    queries: SimQueriesCfg = SimQueriesCfg(data={"dof_pos": BodyJointPositionQuery(body="robot")})


def _make_env(num_envs: int = 3) -> ManagerEnv:
    return ManagerEnv(_FakeManagerCfg(), num_envs=num_envs, backend="fake-manager")


def test_config_backend_selection_and_explicit_override() -> None:
    cfg = _FakeManagerCfg(backend="fake-manager")
    env = ManagerEnv(cfg, num_envs=2)
    assert isinstance(env.sim, _FakeBackend)
    override = _FakeManagerCfg(backend="not-registered")
    env = ManagerEnv(override, num_envs=2, backend="fake-manager")
    assert isinstance(env.sim, _FakeBackend)


def test_manager_env_steps_on_fake_backend() -> None:
    env = _make_env()
    runtime = _FakeBackend.last

    state = env.step(torch.full((3, 1), 0.5, dtype=torch.float32))

    assert runtime.step_calls == [env.cfg.sim_substeps]
    # ctrl routing: only the "routed" actuator receives the doubled action.
    assert runtime.ctrl_write_count == 1
    np.testing.assert_allclose(runtime.ctrl_targets[:, 0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(runtime.ctrl_targets[:, 1], np.full(3, 1.0, dtype=np.float32))
    # observations reflect the sim snapshot read through the fake provider.
    np.testing.assert_allclose(state.obs.policy, runtime.dof_pos)
    # Manager rewards are scaled by the control dt.
    np.testing.assert_allclose(state.reward, np.full(3, env.cfg.ctrl_dt, dtype=np.float32))


@pytest.mark.parametrize("failed_reward", [1.0, float("nan"), float("inf")])
def test_engine_error_terminates_and_resets_only_failed_rows(failed_reward: float, monkeypatch) -> None:
    env = _make_env()
    runtime = _FakeBackend.last
    env.init_state()
    compute_transition = env.compute_transition

    def transition_with_failed_reward(state):
        state = compute_transition(state)
        state.reward[1] = failed_reward
        return state

    monkeypatch.setattr(env, "compute_transition", transition_with_failed_reward)
    status = env.engine_error_codes
    runtime.status_codes[1] = 7
    runtime.reset_calls.clear()

    state = env.step(torch.full((3, 1), 0.5))

    assert state.terminated.tolist() == [False, True, False]
    assert not bool(state.truncated.any())
    assert state.episode_steps.tolist() == [1, 0, 1]
    assert env.engine_error_mask.tolist() == [False, True, False]
    assert status is env.engine_error_codes
    assert status.tolist() == [0, 7, 0]
    torch.testing.assert_close(state.reward, torch.tensor([env.cfg.ctrl_dt, 0.0, env.cfg.ctrl_dt]))
    assert len(runtime.reset_calls) == 1
    assert runtime.reset_calls[0][0].tolist() == [1]
    torch.testing.assert_close(state.obs.policy, runtime.dof_pos)
    torch.testing.assert_close(state.obs.policy[1], torch.tensor([0.5, -0.5]))
    torch.testing.assert_close(state.obs.policy[0], torch.tensor([0.5, -0.5]) + 0.25 * env.cfg.sim_substeps)

    monkeypatch.setattr(env, "compute_transition", compute_transition)
    runtime.status_codes.zero_()
    runtime.reset_calls.clear()
    state = env.step(torch.zeros(3, 1))

    assert not bool(state.done.any())
    assert not bool(env.engine_error_mask.any())
    assert not bool(status.any())
    assert state.episode_steps.tolist() == [2, 1, 2]
    assert not runtime.reset_calls


def test_public_reset_restarts_only_episode_lane_counters() -> None:
    env = _make_env()
    env.init_state()
    env._state.episode_steps.copy_(torch.tensor([5, 6, 7]))
    env.reset(torch.tensor([0]), sim_reset_ids=torch.tensor([1]))
    assert env._state.episode_steps.tolist() == [0, 6, 7]


def test_partial_reset_hands_canonical_rows_to_runtime_and_clears_ctrl() -> None:
    env = _make_env()
    runtime = _FakeBackend.last

    env.init_state()
    runtime.reset_calls.clear()
    env.step(torch.full((3, 1), 0.5, dtype=torch.float32))
    assert runtime.ctrl_targets[1, 1] == np.float32(1.0)

    env._state.terminated[:] = torch.tensor([False, True, False])
    env._reset_done_envs()

    assert len(runtime.reset_calls) == 1
    env_ids, rows = runtime.reset_calls[0]
    np.testing.assert_array_equal(env_ids, [1])
    # The default reset term filled the canonical rows with model defaults.
    np.testing.assert_allclose(rows, np.asarray([[0.5, -0.5]], dtype=np.float32))
    np.testing.assert_allclose(runtime.dof_pos[1], [0.5, -0.5])
    # The frontend ctrl buffer mirrors the backend's reset-time clearing.
    np.testing.assert_allclose(env._action_writes.buffer("ctrl")[1], np.zeros(1, dtype=np.float32))


@pytest.mark.parametrize("compiled", [False, True])
def test_reset_dispatch_gathers_explicit_ids_and_scatters_compact_outputs(
    compiled: bool, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    # Keep these fullgraph variants independent of other manager schedules.
    torch._dynamo.reset()
    request.addfinalizer(torch._dynamo.reset)
    compile_torch = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: (
            compile_torch(fn, backend="eager", fullgraph=True, dynamic=kwargs.get("dynamic")) if compiled else fn
        ),
    )

    def reset_position(
        context: ManagerContext, selected: torch.Tensor, outputs: Mapping[str, torch.Tensor], offset: float
    ) -> None:
        outputs["joint_position"].copy_(context.sim["dof_pos"][selected] + offset)

    @configclass
    class PositionResetCfg(ResetTermCfg):
        def __call__(self, ctx: BuildContext) -> ResetTerm:
            return ResetTerm(reset_position, 10.0, writes={"joint_position": CtrlTargetsWrite()})

    @configclass
    class ResetsCfg(ManagerResetCfg):
        position: PositionResetCfg = PositionResetCfg()

    env = ManagerEnv(_FakeManagerCfg(sim_reset=ResetsCfg()), num_envs=4, backend="fake-manager")
    env.init_state()
    before = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    env.sim.dof_pos.copy_(before)
    env._refresh_sim_reads()

    env.reset(torch.tensor([3, 0]), sim_reset_ids=torch.tensor([2]))
    expected = before.clone()
    expected[[3, 0, 2]] += 10.0
    torch.testing.assert_close(env.sim.dof_pos, expected)

    env.reset(torch.tensor([1]))
    expected[1] += 10.0
    torch.testing.assert_close(env.sim.dof_pos, expected)


def test_reset_dynamic_rows_preserve_mapping_writes_and_frozen_state(request: pytest.FixtureRequest) -> None:
    """Reset compact staging keeps selected rows across changing K values.

    The reset term intentionally receives an ordinary ``Mapping`` and a nested
    frozen dataclass carrying caller-owned tensors, with an in-memory backend.
    """
    torch._dynamo.reset()
    request.addfinalizer(torch._dynamo.reset)

    @dataclass(frozen=True, slots=True)
    class _NestedState:
        values: torch.Tensor

    @dataclass(frozen=True, slots=True)
    class _ResetState:
        nested: _NestedState

    state = _ResetState(_NestedState(torch.tensor([[100.0, 101.0], [200.0, 201.0], [300.0, 301.0], [400.0, 401.0]])))

    def reset_selected(
        context: ManagerContext,
        selected: torch.Tensor,
        outputs: Mapping[str, torch.Tensor],
        frozen_state: _ResetState,
    ) -> None:
        # Keep the dynamic staging axis live across a deliberate graph break.
        torch._dynamo.graph_break()
        outputs["joint_position"].copy_(frozen_state.nested.values[selected])

    @configclass(kw_only=True)
    class DynamicResetCfg(ResetTermCfg):
        def __call__(self, ctx: BuildContext) -> ResetTerm:
            return ResetTerm(reset_selected, state, writes={"joint_position": CtrlTargetsWrite()})

    @configclass
    class DynamicResetsCfg(ManagerResetCfg):
        selected: DynamicResetCfg = DynamicResetCfg()

    cfg = _FakeManagerCfg(sim_reset=DynamicResetsCfg())
    env = ManagerEnv(cfg, num_envs=4, backend="fake-manager")
    env.init_state()
    runtime = _FakeBackend.last
    runtime.reset_calls.clear()

    for ids in (torch.tensor([3, 0, 2]), torch.tensor([1]), torch.tensor([2, 0])):
        before = runtime.dof_pos.clone()
        env.reset(ids)
        expected = state.nested.values[ids]
        torch.testing.assert_close(runtime.dof_pos[ids], expected)
        torch.testing.assert_close(runtime.reset_calls[-1][1], expected)
        untouched = torch.ones(4, dtype=torch.bool)
        untouched[ids] = False
        torch.testing.assert_close(runtime.dof_pos[untouched], before[untouched])


def test_manager_frontend_does_not_import_motrixsim() -> None:
    code = "import sys\nimport motrix_env_core.manager.env\nsys.exit(0 if 'motrixsim' not in sys.modules else 1)\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
