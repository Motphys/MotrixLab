# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Fake-backend tests for the DirectEnv SimBackend boundary."""

import subprocess
import sys
from collections.abc import Mapping

import gymnasium as gym
import numpy as np
import pytest

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import ArrayEnvState, DirectEnv, DirectEnvCfg
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    DofPositionQuery,
    DofPositionWrite,
    DofVelocityQuery,
    ModelQuery,
    PhysicsReadProgram,
)
from motrix_env_core.sim.backend import (
    ActuatorSpec,
    ActuatorType,
    SimBackend,
    SimModel,
)
from motrix_env_core.sim.registry import register_sim_backend
from motrix_env_core.sim.write import CtrlTargetsWrite, DofVelocityWrite, WriteProgram


def _core_model() -> SimModel:
    """The required core layout every backend must surface as ``env.model``."""
    return SimModel(
        actuators=(
            ActuatorSpec("a", ActuatorType.POSITION, "a", None, None),
            ActuatorSpec("b", ActuatorType.POSITION, "b", None, None),
        ),
        init_dof_pos=np.asarray([0.5, -0.5], dtype=np.float32),
    )


class _FakeWriteProgram(WriteProgram):
    def __init__(self, backend: "_FakeBackend", writes, reset: bool) -> None:
        self._backend = backend
        self._reset = reset
        self._buffers = {}
        for name, write in writes.items():
            if isinstance(write, CtrlTargetsWrite):
                self._buffers[name] = np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)
            elif isinstance(write, DofPositionWrite):
                self._buffers[name] = np.zeros_like(backend.dof_pos)
            elif isinstance(write, DofVelocityWrite):
                self._buffers[name] = np.zeros_like(backend.dof_vel)

    def buffer(self, name: str) -> np.ndarray:
        return self._buffers[name]

    def execute(self, env_ids=None) -> None:
        ids = np.arange(self._backend.num_envs, dtype=np.int64) if env_ids is None else env_ids
        if self._reset:
            self._backend.reset_calls.append((ids.copy(), self.buffer("state_position")[ids].copy()))
            self._backend.dof_pos[ids] = self.buffer("state_position")[ids]
            self._backend.dof_vel[ids] = self.buffer("state_velocity")[ids]
            self._backend.actuator_ctrls[ids] = 0.0
        if "ctrl" in self._buffers:
            self._backend.ctrl_writes += 1
            self._backend.actuator_ctrls[ids] = self.buffer("ctrl")[ids]


class _FakeWriteCompiler:
    def __init__(self, backend: "_FakeBackend") -> None:
        self._backend = backend

    def compile(self, writes, *, reset: bool = False, forward_kinematics: bool = True) -> WriteProgram:
        del forward_kinematics
        return _FakeWriteProgram(self._backend, writes, reset)


class _FakeReadProgram(PhysicsReadProgram):
    """Owns one contiguous buffer with one region per whole-space runtime key."""

    def __init__(self, backend: "_FakeBackend", queries: dict) -> None:
        self._backend = backend
        self._keys = tuple(queries)
        self._queries = dict(queries)
        keys = self._keys
        widths = tuple(getattr(backend, key).shape[1] for key in keys)
        self.storage = np.zeros((backend.num_envs, sum(widths)), dtype=np.float32)
        self._views: dict[str, np.ndarray] = {}
        offset = 0
        for key, width in zip(keys, widths):
            view = self.storage[:, offset : offset + width]
            view.flags.writeable = False
            self._views[key] = view
            offset += width

    @property
    def arena_bytes(self) -> int:
        return int(self.storage.nbytes)

    def view(self, key: str) -> np.ndarray:
        return self._views[key]

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    def query(self, key: str):
        return self._queries[key]

    def execute(self, env_ids=None) -> None:
        offset = 0
        for key in self._keys:
            values = getattr(self._backend, key)
            columns = slice(offset, offset + values.shape[1])
            if env_ids is None:
                self.storage[:, columns] = values
            else:
                self.storage[env_ids, columns] = values[env_ids]
            offset += values.shape[1]


class _FakeBackend(SimBackend):
    name = "fake"

    last: "_FakeBackend | None" = None

    def __init__(self, scene, sim, num_envs: int) -> None:
        del scene, sim  # the fake compiles nothing
        self.num_envs = num_envs
        self.dof_pos = np.zeros((num_envs, self.num_dof_pos), dtype=np.float32)
        self.dof_vel = np.zeros((num_envs, self.num_dof_vel), dtype=np.float32)
        self.actuator_ctrls = np.zeros((num_envs, self.num_actuators), dtype=np.float32)
        self.step_calls: list[int] = []
        self.reset_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.ctrl_writes = 0
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
        self.dof_pos += self.dof_vel * np.float32(substeps)

    @property
    def model_query_compiler(self):
        return self

    def compile(self, queries: Mapping[str, ModelQuery]) -> SimModel:
        del queries
        return _core_model()

    def compile_reads(self, queries) -> PhysicsReadProgram:
        return _FakeReadProgram(self, queries)

    @property
    def write_compiler(self):
        return self._write_compiler


register_sim_backend("fake-direct", lambda: _FakeBackend)


@configclass
class _FakeDirectCfg(DirectEnvCfg):
    scene: SceneCfg = SceneCfg()


class _FakeDirectEnv(DirectEnv[_FakeDirectCfg]):
    def __init__(self, cfg: _FakeDirectCfg, num_envs: int, backend: str | None = None) -> None:
        super().__init__(cfg, num_envs, backend=backend)
        self.sim_data = self.sim.compile_reads(
            {
                "dof_pos": DofPositionQuery(),
                "dof_vel": DofVelocityQuery(),
                "actuator_ctrls": ActuatorCtrlQuery(),
            }
        )
        self._ctrl_writes = self.sim.compile_writes({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.compile_writes(
            {"state_position": DofPositionWrite(), "state_velocity": DofVelocityWrite()}, reset=True
        )
        self._action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = actions.astype(np.float32, copy=False)
        self._ctrl_writes.execute()
        return state

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        state.obs.policy[:] = np.concatenate([inputs["dof_pos"], inputs["dof_vel"], inputs["actuator_ctrls"]], axis=-1)
        state.reward = np.ones((self.num_envs,), dtype=np.float32)
        state.terminated = np.zeros((self.num_envs,), dtype=bool)
        self.sim_data.execute()
        return state

    def reset(self, env_ids: np.ndarray):
        rows = len(env_ids)
        dof_pos = np.full((rows, self.num_dof_pos), 2.0, dtype=np.float32)
        dof_vel = np.full((rows, self.num_dof_vel), 1.0, dtype=np.float32)
        self._reset_program.buffer("state_position")[env_ids] = dof_pos
        self._reset_program.buffer("state_velocity")[env_ids] = dof_vel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
        return {}


def _make_env(num_envs: int = 3) -> _FakeDirectEnv:
    return _FakeDirectEnv(_FakeDirectCfg(), num_envs, backend="fake-direct")


def test_direct_env_step_writes_ctrl_and_reads_inputs() -> None:
    env = _make_env()
    runtime = _FakeBackend.last

    state = env.step(np.full((3, 2), 0.5, dtype=np.float32))

    assert runtime.step_calls == [env.cfg.sim_substeps]
    assert runtime.ctrl_writes == 1
    np.testing.assert_allclose(runtime.actuator_ctrls, 0.5)
    # The fake read program refreshes the buffer used by the transition.
    np.testing.assert_allclose(state.obs.policy[:, 0:2], 2.0)
    np.testing.assert_allclose(state.obs.policy[:, 2:4], 1.0)
    np.testing.assert_allclose(state.obs.policy[:, 4:6], 0.0)


def test_direct_env_compiles_empty_read_program() -> None:
    class EmptyReadEnv(_FakeDirectEnv):
        def __init__(self, cfg, num_envs, backend: str | None = None):
            DirectEnv.__init__(self, cfg, num_envs, backend=backend)
            self.sim_data = self.sim.compile_reads({})

    env = EmptyReadEnv(_FakeDirectCfg(), num_envs=2, backend="fake-direct")

    assert env.sim_data.keys == ()
    assert env.sim_data.arena_bytes == 0
    env.sim_data.execute()


def test_direct_env_partial_reset_passes_canonical_rows() -> None:
    env = _make_env()
    runtime = _FakeBackend.last

    env.init_state()
    runtime.reset_calls.clear()
    env._state.terminated[:] = [False, True, False]
    env._reset_done_envs()

    assert len(runtime.reset_calls) == 1
    env_ids, dof_pos = runtime.reset_calls[0]
    np.testing.assert_array_equal(env_ids, [1])
    np.testing.assert_allclose(dof_pos, [[2.0, 2.0]])
    # reset 后部分 read 已刷新 sim inputs
    np.testing.assert_allclose(env.sim_data["dof_pos"][1], [2.0, 2.0])
    # reset 清 ctrl
    np.testing.assert_allclose(runtime.actuator_ctrls[1], 0.0)


def test_direct_env_renderer_delegates_to_backend() -> None:
    env = _make_env()
    with pytest.raises(NotImplementedError, match="does not provide rendering"):
        env.create_renderer(None)


def test_direct_frontend_does_not_import_motrixsim() -> None:
    code = "import sys\nimport motrix_env_core.direct.env\nsys.exit(0 if 'motrixsim' not in sys.modules else 1)\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
