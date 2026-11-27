# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Batched CUDA physics on a Warp stream ordered with the calling Torch stream."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch
import warp as wp
from motrixsim import gpu

from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.mdp.terrain import HeightFieldGrid, heightfield_lookup
from motrix_env_core.sim.backend import RenderConfig, SimBackend, SimRenderer
from motrix_env_core.sim.model import ModelQuery, SimModel
from motrix_env_core.sim.read import PhysicsReadProgram, SimDataQuery
from motrix_env_core.sim.write import CtrlTargetsWrite, WriteProgram
from motrix_env_motrixsim._gpu_primitives import GpuParameterOverrides, GpuSolverResetter, GpuStepStatus
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
from motrix_env_motrixsim.gpu_reads import CudaReadCompiler
from motrix_env_motrixsim.gpu_writes import CudaWriteCompiler
from motrix_env_motrixsim.runtime import MotrixSimModelCompiler, _heightfield_grid, _named_geom


class MotrixCudaBackend(SimBackend):
    name = "motrixsim"
    supports_native_masked_reset = True

    def __init__(self, scene: SceneCfg, sim: SimCfg, num_envs: int, *, render: RenderConfig | None = None) -> None:
        super().__init__(scene, sim, num_envs, render=render)
        device = torch.device(sim.device)
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        self._warp_device = wp.get_device(str(self.device))
        self._stream = wp.Stream(self._warp_device)
        self._stream.wait_stream(wp.get_stream(self._warp_device))
        self._caller_ready = wp.Event(self._warp_device)
        self._torch_stream = torch.cuda.ExternalStream(self._stream.cuda_stream, device=self.device)
        host_model = MotrixSimSceneCompiler().compile(scene, sim, render=render)
        self.host_model = host_model
        self._model_compiler = MotrixSimModelCompiler(host_model)
        with self.physics_stream():
            self.model = gpu.SceneModel.from_cmodel(
                host_model, max_contacts=sim.max_contacts, max_contact_constraints=sim.max_contact_constraints
            )
            self.data = gpu.SceneData(self.model, batch=num_envs)
            self.position = wp.to_torch(self.data.dof_pos)
            self.velocity = wp.to_torch(self.data.dof_vel)
            self.control = wp.to_torch(self.data.actuator_ctrls)
            self._resetter = GpuSolverResetter(host_model, self.data, self._warp_device)
            self._step_status = GpuStepStatus(self)
            self._all_worlds = self.position.new_ones(num_envs, dtype=torch.bool)
        self._terrain: dict[str, float | HeightFieldGrid] = {}
        self._gain_tables: dict[str, torch.Tensor] = {}
        self._gain_defaults: dict[str, torch.Tensor] = {}
        self._overrides: GpuParameterOverrides | None = None
        self._mocap_defaults = []
        with self.physics_stream():
            for body in self.model.bodies:
                if body.is_mocap:
                    mocap = body.mocap
                    self._mocap_defaults.append((mocap, wp.to_torch(mocap.get_pose(self.data))[0].clone()))
        self._write_compiler = CudaWriteCompiler(self)
        self._default_reset = self._write_compiler.compile({}, reset=True)

    @contextmanager
    def physics_stream(self) -> Iterator[None]:
        current = torch.cuda.current_stream(self.device)
        with wp.ScopedStream(self._stream, sync_enter=False, sync_exit=False):
            if current.cuda_stream == self._stream.cuda_stream:
                yield
                return
            caller = wp.stream_from_torch(current)
            self._stream.wait_stream(caller, event=self._caller_ready)
            try:
                with torch.cuda.stream(self._torch_stream):
                    yield
            finally:
                caller.wait_stream(self._stream)

    step_context = physics_stream

    @property
    def model_compiler(self) -> MotrixSimModelCompiler:
        return self._model_compiler

    @property
    def write_compiler(self) -> CudaWriteCompiler:
        return self._write_compiler

    def compile_action_writes(self, writes: Mapping[str, CtrlTargetsWrite]) -> WriteProgram:
        return CudaWriteCompiler(self, direct_ctrl=True).compile(writes, forward_kinematics=False)

    @property
    def num_dof_pos(self) -> int:
        return self.model.num_dof_pos

    @property
    def num_dof_vel(self) -> int:
        return self.model.num_dof_vel

    @property
    def num_actuators(self) -> int:
        return self.model.num_actuators

    def create_renderer(self, config: RenderConfig, *, num_envs: int, render_spacing: float) -> SimRenderer:
        from motrix_env_motrixsim.renderer import MotrixSimRenderer

        return MotrixSimRenderer(
            self.model,
            lambda: self.data,
            config,
            num_envs=num_envs,
            render_spacing=render_spacing,
            system_camera=self._scene.system_camera,
            render_app=gpu.render.RenderApp(headless=config.headless),
        )

    def world_mask(self, ids: torch.Tensor | None) -> torch.Tensor:
        if ids is None:
            return self._all_worlds
        return torch.zeros_like(self._all_worlds).index_fill_(0, ids, True)

    def gain_table(self, attribute: str) -> torch.Tensor:
        if attribute not in self._gain_tables:
            table = wp.to_torch(getattr(self.data, attribute))
            self._gain_tables[attribute] = table
            self._gain_defaults[attribute] = table.clone()
        return self._gain_tables[attribute]

    @property
    def overrides(self) -> GpuParameterOverrides:
        if self._overrides is None:
            self._overrides = GpuParameterOverrides(self)
        return self._overrides

    def reset_masked(self, world_mask: torch.Tensor) -> None:
        self._default_reset.execute_masked(world_mask)

    def reset_inputs(self, mask: torch.Tensor, warp_mask, *, recompute_inertia: bool = True) -> None:
        self._resetter.execute(warp_mask)
        for attribute, table in self._gain_tables.items():
            table.copy_(torch.where(mask[:, None], self._gain_defaults[attribute], table))
        if self._overrides is not None:
            self._overrides.restore_defaults(mask, warp_mask, recompute_inertia=recompute_inertia)
        for mocap, default in self._mocap_defaults:
            poses = wp.to_torch(mocap.get_pose(self.data))
            poses.copy_(torch.where(mask[:, None], default, poses))
            mocap.set_pose(self.data, wp.from_torch(poses), world_mask=warp_mask)

    def sample_terrain_height(self, geom_name: str, env_ids: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        from motrixsim import GeomHField

        if geom_name not in self._terrain:
            geom = _named_geom(self.host_model, geom_name)
            self._terrain[geom_name] = (
                _heightfield_grid(geom, geom_name, self.device)
                if isinstance(geom, GeomHField)
                else float(geom.local_pose[2])
            )
        xy = xy.to(self.device)
        terrain = self._terrain[geom_name]
        if isinstance(terrain, float):
            return xy.new_full(xy.shape[:-1], terrain, dtype=torch.float32)
        return heightfield_lookup(terrain, xy[..., 0].double(), xy[..., 1].double()).float()

    def compile_model(self, queries: Mapping[str, ModelQuery]) -> SimModel:
        return self.model_compiler.compile(self._scene, queries, device=self.device)

    def compile_reads(self, queries: Mapping[str, SimDataQuery]) -> PhysicsReadProgram:
        return CudaReadCompiler(self, queries).build()

    def step(self, substeps: int, result: torch.Tensor | None = None) -> None:
        with self.physics_stream():
            self._step_status.step(substeps, result)
