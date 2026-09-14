# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compile-only MuJoCo adapter for the backend-neutral sim boundary."""

from __future__ import annotations

from motrix_env_core.config.sim import SimCfg
from motrix_env_core.sim.backend import SimBackend
from motrix_env_core.sim.model import SimModelCompiler
from motrix_env_core.sim.read import PhysicsReadProgram
from motrix_env_mujoco.compiler import MuJoCoSceneCompiler


class MuJoCoSimBackend(SimBackend):
    """Compile-only backend: builds MuJoCo scene models, no live simulation.

    Constructing compiles the scene into an MuJoCo model; every behavior and
    translation member reports its gap loudly. ``num_envs`` carries no
    meaning here and is ignored.
    """

    name = "mujoco"

    _GAP = "MuJoCo only compiles scene models; it provides no live simulation."

    def __init__(self, scene, sim: SimCfg, num_envs: int) -> None:
        super().__init__(scene, sim, num_envs)
        self._mujoco_model = MuJoCoSceneCompiler().compile(scene, sim)

    @property
    def num_dof_pos(self) -> int:
        raise NotImplementedError(self._GAP)

    @property
    def num_dof_vel(self) -> int:
        raise NotImplementedError(self._GAP)

    @property
    def num_actuators(self) -> int:
        raise NotImplementedError(self._GAP)

    def step(self, substeps: int) -> None:
        raise NotImplementedError(self._GAP)

    def compile_reads(self, queries) -> PhysicsReadProgram:
        raise NotImplementedError(self._GAP)

    @property
    def model_compiler(self) -> SimModelCompiler:
        raise NotImplementedError(self._GAP)
