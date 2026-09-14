# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Fake-backend tests for the manager SimBackend boundary (issue #222)."""

import subprocess
import sys
from collections.abc import Mapping

import gymnasium as gym
import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.numba.kernel_data import kernel_data
from motrix_env_core.numba.manager.actions import ActionCfg, ActionTerm, ManagerActionsCfg
from motrix_env_core.numba.manager.context import ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.manager.env import ManagerBasedEnvCfg, ManagerEnv
from motrix_env_core.numba.manager.observations import (
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ObservationTermCfg,
    ObsTerm,
)
from motrix_env_core.numba.manager.rewards import ManagerRewardsCfg, RewardTerm, RewardTermCfg
from motrix_env_core.numba.manager.terminations import (
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
from motrix_env_core.sim.write import CtrlTargetsWrite, DofPositionWrite, DofVelocityWrite, WriteProgram

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
        init_dof_pos=np.asarray([0.5, -0.5], dtype=np.float32),
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
                self._buffers[name] = np.zeros((backend.num_envs, len(route)), dtype=np.float32)
            elif isinstance(write, DofPositionWrite):
                self._buffers[name] = np.zeros_like(backend.dof_pos)
            elif isinstance(write, DofVelocityWrite):
                self._buffers[name] = np.zeros_like(backend.dof_vel)

    def buffer(self, name: str) -> np.ndarray:
        return self._buffers[name]

    def execute(self, env_ids=None) -> None:
        ids = np.arange(self._backend.num_envs, dtype=np.int64) if env_ids is None else env_ids
        if self._reset:
            self._backend.dof_pos[ids] = [0.5, -0.5]
            self._backend.dof_vel[ids] = 0.0
            for name, buffer in self._buffers.items():
                if name.endswith("_position"):
                    self._backend.dof_pos[ids] = buffer[ids]
                elif name.endswith("_velocity"):
                    self._backend.dof_vel[ids] = buffer[ids]
            self._backend.reset_calls.append((ids.copy(), self._backend.dof_pos[ids].copy()))
            self._backend.ctrl_targets[ids] = 0.0
        for name, route in self._routes.items():
            self._backend.ctrl_targets[ids[:, None], route] = self.buffer(name)[ids]
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
        self._arena = np.zeros((backend.num_envs * width,), dtype=np.float32)
        self._output = self._arena.reshape(backend.num_envs, width)
        self._views = {key: self._arena.reshape(backend.num_envs, width) for key in keys}
        for value in self._views.values():
            value.flags.writeable = False

    @property
    def arena_bytes(self) -> int:
        return int(self._arena.nbytes)

    def view(self, key: str) -> np.ndarray:
        return self._views[key]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._views)

    def query(self, key: str):
        return self._queries[key]

    def execute(self, env_ids=None) -> None:
        if env_ids is None:
            self._output[:] = self._backend.dof_pos
        else:
            self._output[env_ids] = self._backend.dof_pos[env_ids]


class _FakeBackend(SimBackend):
    """In-memory backend that records every behavioral call."""

    name = "fake"

    last: "_FakeBackend | None" = None

    def __init__(self, scene, sim, num_envs: int) -> None:
        super().__init__(scene, sim, num_envs)
        self.num_envs = num_envs
        self.dof_pos = np.zeros((num_envs, self.num_dof_pos), dtype=np.float32)
        self.dof_vel = np.zeros((num_envs, self.num_dof_vel), dtype=np.float32)
        self.ctrl_targets = np.zeros((num_envs, self.num_actuators), dtype=np.float32)
        self.step_calls: list[int] = []
        self.reset_calls: list[tuple[np.ndarray, np.ndarray | None]] = []
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

    def step(self, substeps: int) -> None:
        self.step_calls.append(substeps)
        self.dof_pos += np.float32(0.25 * substeps)

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


@kernel_data
class _CtrlAction(ActionTerm):
    source: np.ndarray

    def action_space(self, env, actuator_indices) -> gym.spaces.Box:
        del env, actuator_indices
        return gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def process(self, actions: np.ndarray) -> np.ndarray:
        self.source[...] = actions
        return actions * np.float32(2.0)

    def reset(self, env_ids: np.ndarray) -> None:
        self.source[env_ids] = 0.0


@configclass(kw_only=True)
class _CtrlActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] | None = ("routed",)

    def __call__(self, env, actuator_indices):
        del actuator_indices
        return _CtrlAction(np.zeros((env.num_envs, 1), dtype=np.float32))


@dispatch
def _sim_reading_observation(ctx: ManagerContext, out: np.ndarray) -> None:
    dof_pos = ctx.sim["dof_pos"]
    out[0] = dof_pos[0]
    out[1] = dof_pos[1]


@configclass(kw_only=True)
class _SimReadingObservationCfg(ObservationTermCfg):
    def __call__(self, env):
        del env
        return ObsTerm(2, _sim_reading_observation)


@configclass
class _FakePolicyObsCfg(ManagerObservationGroupCfg):
    sim_read: _SimReadingObservationCfg = _SimReadingObservationCfg()


@configclass
class _FakeObservationsCfg(ManagerObservationsCfg):
    policy: _FakePolicyObsCfg = _FakePolicyObsCfg()


@dispatch
def _unit_reward(ctx: ManagerContext) -> float:
    return 1.0


@configclass(kw_only=True)
class _UnitRewardCfg(RewardTermCfg):
    weight: float = 1.0

    def __call__(self, env):
        del env
        return RewardTerm(_unit_reward)


@configclass
class _FakeRewardsCfg(ManagerRewardsCfg):
    unit: _UnitRewardCfg = _UnitRewardCfg()


@dispatch
def _never_termination(ctx: ManagerContext) -> bool:
    return False


@configclass(kw_only=True)
class _NeverTerminationCfg(TerminationTermCfg):
    def __call__(self, env):
        del env
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


def test_manager_env_steps_on_fake_backend() -> None:
    env = _make_env()
    runtime = _FakeBackend.last

    state = env.step(np.full((3, 1), 0.5, dtype=np.float32))

    assert runtime.step_calls == [env.cfg.sim_substeps]
    # ctrl routing: only the "routed" actuator receives the doubled action.
    assert runtime.ctrl_write_count == 1
    np.testing.assert_allclose(runtime.ctrl_targets[:, 0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(runtime.ctrl_targets[:, 1], np.full(3, 1.0, dtype=np.float32))
    # observations reflect the sim snapshot read through the fake provider.
    np.testing.assert_allclose(state.obs.policy, runtime.dof_pos)
    # Manager rewards are scaled by the control dt.
    np.testing.assert_allclose(state.reward, np.full(3, env.cfg.ctrl_dt, dtype=np.float32))


def test_partial_reset_hands_canonical_rows_to_runtime_and_clears_ctrl() -> None:
    env = _make_env()
    runtime = _FakeBackend.last

    env.init_state()
    runtime.reset_calls.clear()
    env.step(np.full((3, 1), 0.5, dtype=np.float32))
    assert runtime.ctrl_targets[1, 1] == np.float32(1.0)

    env._state.terminated[:] = [False, True, False]
    env._reset_done_envs()

    assert len(runtime.reset_calls) == 1
    env_ids, rows = runtime.reset_calls[0]
    np.testing.assert_array_equal(env_ids, [1])
    # The default reset term filled the canonical rows with model defaults.
    np.testing.assert_allclose(rows, np.asarray([[0.5, -0.5]], dtype=np.float32))
    np.testing.assert_allclose(runtime.dof_pos[1], [0.5, -0.5])
    # The frontend ctrl buffer mirrors the backend's reset-time clearing.
    np.testing.assert_allclose(env._action_writes.buffer("ctrl")[1], np.zeros(1, dtype=np.float32))


def test_manager_frontend_does_not_import_motrixsim() -> None:
    code = (
        "import sys\nimport motrix_env_core.numba.manager.env\nsys.exit(0 if 'motrixsim' not in sys.modules else 1)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
