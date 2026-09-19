# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim backend: scene compilation, translation surface and live behavior."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import motrixsim as mtx
import numpy as np
import numpy.typing as npt

from motrix_env_core.config import SimCfg
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.config.scene.base import BodyCfg, RobotCfg
from motrix_env_core.sim.backend import RenderConfig, SimBackend, SimRenderer
from motrix_env_core.sim.body import assemble_body_model, resolved_key_pose
from motrix_env_core.sim.model import (
    ActuatorSpec,
    ActuatorType,
    BodyModel,
    GeomSpec,
    SimModel,
    SimModelCompiler,
)
from motrix_env_core.sim.read import PhysicsReadProgram, SimDataQuery
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
from motrix_env_motrixsim.renderer import MotrixSimRenderer
from motrix_env_motrixsim.sim_data import compile_read_program
from motrix_env_motrixsim.write_compiler import MotrixSimWriteCompiler

FloatArray: TypeAlias = npt.NDArray[np.float32]
IntArray: TypeAlias = npt.NDArray[np.int64]


class MotrixSimModelCompiler(SimModelCompiler):
    """Assemble the neutral model surface against one MotrixSim scene model."""

    def __init__(self, model: mtx.SceneModel) -> None:
        self._model = model
        self._others: dict[str, Any] = {}

    def _begin_compile(self) -> None:
        self._others = {}

    def _build_model(self, scene: SceneCfg) -> SimModel:
        actuators = _actuator_specs(self._model.actuators)
        init_dof_pos = np.asarray(self._model.compute_init_dof_pos(), dtype=np.float32)
        bodies = _build_body_models(self._model, scene, actuators, init_dof_pos)
        return SimModel(actuators=actuators, init_dof_pos=init_dof_pos, others=self._others, bodies=bodies)

    def compile_geom_specs(self, key: str, geom_names: tuple[str, ...]) -> None:
        self._others[key] = _geom_specs(self._model, geom_names)

    def compile_height_field_data(self, key: str, geom_name: str) -> None:
        geom = _named_geom(self._model, geom_name)
        if not isinstance(geom, mtx.GeomHField) or geom.hfield is None:
            raise ValueError(
                f"HeightFieldDataQuery requires geom {geom_name!r} to carry a height field, got {type(geom).__name__}."
            )
        hfield = geom.hfield
        pose = np.asarray(geom.local_pose, dtype=np.float32).reshape(-1)
        quat_ijkw = pose[3:7]
        if abs(float(quat_ijkw[3])) < 1.0 - 1e-5:
            raise ValueError(
                f"HeightFieldDataQuery requires geom {geom_name!r} to be world-aligned "
                f"(identity rotation), got quaternion {tuple(quat_ijkw)}."
            )
        heights = np.ascontiguousarray(np.asarray(hfield.height_matrix, dtype=np.float32))
        nrow, ncol = heights.shape
        bound = np.asarray(hfield.bound, dtype=np.float32)
        extent_x, extent_y = float(bound[3]), float(bound[4])
        spacing = np.asarray(
            [2.0 * extent_x / max(ncol - 1, 1), 2.0 * extent_y / max(nrow - 1, 1)],
            dtype=np.float32,
        )
        origin = np.asarray([float(pose[0]) - extent_x, float(pose[1]) - extent_y], dtype=np.float32)
        self._others[key] = {
            "heights": heights,
            "origin": origin,
            "spacing": spacing,
            "z0": np.asarray([pose[2]], dtype=np.float32),
        }

    def compile_body_joint_position_limits(self, key: str, body: str) -> None:
        self._others[key] = _body_joint_position_limits(self._model, body)

    def compile_dof_position_limits(self, key: str) -> None:
        self._others[key] = _dof_position_limits(self._model)

    def compile_actuator_kp(self, key: str, actuator_names: tuple[str, ...] | None) -> None:
        self._others[key] = _nominal_actuator_kp(self._model, actuator_names)

    def compile_actuator_kd(self, key: str, actuator_names: tuple[str, ...] | None) -> None:
        self._others[key] = _nominal_actuator_kd(self._model, actuator_names)

    def compile_body_mass(self, key: str, body: str) -> None:
        self._others[key] = float(_named_link(self._model, body).mass)

    def compile_body_center_of_mass(self, key: str, body: str) -> None:
        self._others[key] = np.asarray(_named_link(self._model, body).center_of_mass, dtype=np.float32).reshape(3)

    def compile_geom_friction(self, key: str, geom: str) -> None:
        self._others[key] = np.asarray(_named_geom(self._model, geom).friction, dtype=np.float32)


def _nominal_actuator_kp(model: mtx.SceneModel, actuator_names: tuple[str, ...] | None) -> FloatArray:
    values = []
    actuators = model.actuators if actuator_names is None else (_named_actuator(model, name) for name in actuator_names)
    for actuator in actuators:
        if not isinstance(actuator, mtx.PositionActuator):
            raise ValueError(
                "ActuatorKpQuery requires every actuator to carry kp, "
                f"but {actuator.name!r} ({type(actuator).__name__}) does not."
            )
        values.append(float(actuator.kp))
    return np.asarray(values, dtype=np.float32)


def _nominal_actuator_kd(model: mtx.SceneModel, actuator_names: tuple[str, ...] | None) -> FloatArray:
    values = []
    actuators = model.actuators if actuator_names is None else (_named_actuator(model, name) for name in actuator_names)
    for actuator in actuators:
        if not isinstance(actuator, mtx.PositionActuator):
            raise ValueError(
                "ActuatorKdQuery requires every actuator to carry kd, "
                f"but {actuator.name!r} ({type(actuator).__name__}) does not."
            )
        if actuator.kd is None:
            raise ValueError(f"ActuatorKdQuery requires actuator {actuator.name!r} to define kd.")
        values.append(float(actuator.kd))
    return np.asarray(values, dtype=np.float32)


def _named_link(model: mtx.SceneModel, link_name: str) -> mtx.Link:
    link = model.get_link(link_name)
    if link is None:
        raise KeyError(f"Unknown link {link_name!r}.")
    return link


def _named_geom(model: mtx.SceneModel, geom_name: str) -> mtx.Geom:
    geom = model.get_geom(geom_name)
    if geom is None:
        raise KeyError(f"Unknown geom {geom_name!r}.")
    return geom


def _as_pair(values: Iterable[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    lo, hi = (float(value) for value in values)
    return (lo, hi)


def _actuator_specs(actuators: Iterable[mtx.Actuator]) -> tuple[ActuatorSpec, ...]:
    specs = []
    for actuator in actuators:
        if actuator.name is None:
            raise ValueError("Every actuator must have a name.")
        specs.append(
            ActuatorSpec(
                name=actuator.name,
                actuator_type=ActuatorType(actuator.typ),
                target_name=actuator.target_name,
                ctrl_range=_as_pair(actuator.ctrl_range),
                force_range=_as_pair(actuator.force_range),
            )
        )
    return tuple(specs)


@dataclass(frozen=True)
class _BodyFacts:
    """Engine-side facts extracted for one body, before the shared FK pass."""

    name: str
    cfg: BodyCfg
    body: mtx.Body
    joint_names: tuple[str, ...]
    joint_pos_limits: tuple[FloatArray, FloatArray] | None
    link_names: tuple[str, ...]
    link_indices: list[int]
    joint_dof: IntArray
    init_joint_pos: FloatArray | None


def _body_facts(model: mtx.SceneModel, name: str, cfg: BodyCfg) -> _BodyFacts:
    body = _named_body(model, cfg.resolved_base_link_name)
    joints = tuple(body.joints)
    for joint in joints:
        if joint.num_dof_pos != 1:
            raise ValueError(
                f"BodyModel requires single-dof joints, but joint {joint.name!r} of body {name!r} "
                f"has {joint.num_dof_pos} position DOFs."
            )
    joint_names = tuple(joint.name for joint in joints)
    robot_cfg = cfg if isinstance(cfg, RobotCfg) else None
    init_joint_pos = (
        resolved_key_pose(name, robot_cfg, robot_cfg.init_key_pose, joint_names)
        if robot_cfg is not None and robot_cfg.key_pose.poses
        else None
    )
    return _BodyFacts(
        name=name,
        cfg=cfg,
        body=body,
        joint_names=joint_names,
        joint_pos_limits=_body_joint_position_limits(model, cfg.resolved_base_link_name) if joints else None,
        link_names=tuple(link.name for link in body.links),
        link_indices=[link.index for link in body.links],
        joint_dof=np.asarray(body.get_dof_pos_indices(include_floatingbase=False), dtype=np.int64),
        init_joint_pos=init_joint_pos,
    )


def _build_body_models(
    model: mtx.SceneModel,
    scene: SceneCfg,
    scene_actuators: tuple[ActuatorSpec, ...],
    default_dof: FloatArray,
) -> dict[str, BodyModel]:
    """Assemble one ``BodyModel`` per ``BodyCfg`` declared in the scene."""
    facts = [_body_facts(model, name, cfg) for name, cfg in scene.iter_objs() if isinstance(cfg, BodyCfg)]
    if not facts:
        return {}

    # One FK evaluation on batch-1 data: each body's key-pose override
    # touches only its own joints, so all overrides compose into a single
    # init configuration.
    init_dof = default_dof.copy()
    for fact in facts:
        if fact.init_joint_pos is not None:
            init_dof[fact.joint_dof] = fact.init_joint_pos
    data = mtx.SceneData(model, batch=[1])
    data.reset(
        model,
        dof_pos=init_dof,
        dof_vel=np.zeros((model.num_dof_vel,), dtype=np.float32),
        forward_kinematic=True,
    )
    link_poses = np.asarray(model.get_link_poses(data), dtype=np.float32)[0]

    bodies = {}
    for fact in facts:
        robot_cfg = fact.cfg if isinstance(fact.cfg, RobotCfg) else None
        poses = link_poses[fact.link_indices]
        base_pose = np.asarray(fact.body.get_pose(data), dtype=np.float32).reshape(7)
        bodies[fact.name] = assemble_body_model(
            name=fact.name,
            base_link_name=fact.cfg.resolved_base_link_name,
            link_names=fact.link_names,
            joint_names=fact.joint_names,
            joint_pos_limits=fact.joint_pos_limits,
            scene_actuators=scene_actuators,
            init_base_position=base_pose[:3],
            init_base_quat=base_pose[3:],
            init_link_positions=poses[:, :3],
            init_link_quats=poses[:, 3:7],
            robot_cfg=robot_cfg,
        )
    return bodies


def _dof_position_limits(model: mtx.SceneModel) -> tuple[FloatArray, FloatArray]:
    lower = np.full((model.num_dof_pos,), -np.inf, dtype=np.float32)
    upper = np.full((model.num_dof_pos,), np.inf, dtype=np.float32)
    for joint in model.joints:
        position_slice = slice(joint.dof_pos_index, joint.dof_pos_index + joint.num_dof_pos)
        limits = np.asarray(joint.range, dtype=np.float32).reshape(-1, 2)
        if limits.shape[0] != joint.num_dof_pos:
            raise RuntimeError(
                f"Joint {joint.name!r} limits contain {limits.shape[0]} position DOFs, expected {joint.num_dof_pos}."
            )
        lower[position_slice] = limits[:, 0]
        upper[position_slice] = limits[:, 1]
    return lower, upper


def _body_joint_position_limits(model: mtx.SceneModel, body_name: str) -> tuple[FloatArray, FloatArray]:
    body = _named_body(model, body_name)
    ranges = [np.asarray(joint.range, dtype=np.float32).reshape(-1, 2) for joint in body.joints]
    limits = np.concatenate(ranges, axis=0) if ranges else np.empty((0, 2), dtype=np.float32)
    if limits.shape[0] != body.num_joint_dof_pos:
        raise RuntimeError(
            f"Body {body_name!r} joint limits contain {limits.shape[0]} position DOFs, "
            f"expected {body.num_joint_dof_pos}."
        )
    return limits[:, 0].copy(), limits[:, 1].copy()


def _geom_specs(model: mtx.SceneModel, names: tuple[str, ...]) -> dict[str, GeomSpec]:
    specs = {}
    for name in names:
        geom = _named_geom(model, name)
        specs[name] = GeomSpec(
            size=tuple(float(value) for value in np.asarray(geom.size, dtype=np.float32).reshape(-1)),
            local_pose=tuple(float(value) for value in np.asarray(geom.local_pose, dtype=np.float32).reshape(-1)),
        )
    return specs


class MotrixSimBackend(SimBackend):
    """MotrixSim backend: scene compilation at construction plus live behavior."""

    name = "motrixsim"

    def __init__(self, scene: SceneCfg, sim: SimCfg, num_envs: int) -> None:
        super().__init__(scene, sim, num_envs)
        self._model: mtx.SceneModel = MotrixSimSceneCompiler().compile(scene, sim)
        self._data: mtx.SceneData = mtx.SceneData(self._model, batch=[num_envs])
        self._num_envs = num_envs
        self._model_compiler = MotrixSimModelCompiler(self._model)
        self._write_compiler = MotrixSimWriteCompiler(self._model, self._data)

    @property
    def model_compiler(self) -> SimModelCompiler:
        return self._model_compiler

    def compile_reads(self, queries: Mapping[str, SimDataQuery]) -> PhysicsReadProgram:
        return compile_read_program(self._model, self._data, queries)

    @property
    def num_dof_pos(self) -> int:
        return self._model.num_dof_pos

    @property
    def num_dof_vel(self) -> int:
        return self._model.num_dof_vel

    @property
    def num_actuators(self) -> int:
        return self._model.num_actuators

    def create_renderer(
        self,
        config: RenderConfig,
        *,
        num_envs: int,
        render_spacing: float,
        system_camera: SystemCameraCfg,
    ) -> SimRenderer:
        return MotrixSimRenderer(
            self._model,
            lambda: self._data,
            config,
            num_envs=num_envs,
            render_spacing=render_spacing,
            system_camera=system_camera,
        )

    def step(self, substeps: int) -> None:
        self._model.step_n(self._data, substeps)

    def sample_terrain_height(self, geom_name: str, env_ids: IntArray, xy: FloatArray) -> FloatArray:
        geom = self._model.get_geom(geom_name)
        if geom is None:
            raise KeyError(f"Unknown terrain geom {geom_name!r}.")
        if hasattr(geom, "sample_height"):
            mask = np.zeros((self._data.shape[0],), dtype=bool)
            mask[env_ids] = True
            points = np.ascontiguousarray(xy, dtype=np.float32)
            return np.asarray(geom.sample_height(self._data[mask], points), dtype=np.float32)
        return np.full(xy.shape[:-1], float(geom.local_pose[2]), dtype=np.float32)

    @property
    def write_compiler(self) -> MotrixSimWriteCompiler:
        return self._write_compiler


def _named_body(model: mtx.SceneModel, body_name: str) -> mtx.Body:
    body = model.get_body(body_name)
    if body is None:
        raise KeyError(f"Unknown body {body_name!r}.")
    return body


def _named_actuator(model: mtx.SceneModel, actuator_name: str) -> mtx.Actuator:
    for actuator in model.actuators:
        if actuator.name == actuator_name:
            return actuator
    raise KeyError(f"Unknown actuator {actuator_name!r}.")


__all__ = ["MotrixSimBackend"]
