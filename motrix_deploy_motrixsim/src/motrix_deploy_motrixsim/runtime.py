# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Native world owner: scene compilation, lifecycle and physics advancement."""

from copy import deepcopy
from typing import TYPE_CHECKING

import motrixsim as mtx
import numpy as np

from motrix_deploy.contracts import RobotSpec
from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.base import SimulationRuntime
from motrix_deploy.runtime.config import SimulationRuntimeConfig
from motrix_deploy.runtime.scheduler import FixedStepScheduler, RealtimeScheduler
from motrix_deploy_motrixsim.robot import Robot
from motrix_deploy_motrixsim.viewer import MotrixSimViewer
from motrix_env_core.config.scene.base import RobotCfg
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler

if TYPE_CHECKING:
    from motrix_deploy.runtime.result import SimulationRolloutResult


class MotrixSimRuntime(SimulationRuntime):
    """Own the native world; expose robot I/O before session assembly."""

    def __init__(self, config: SimulationRuntimeConfig, *, viewer_factory=None) -> None:
        super().__init__()
        self.config = config
        self.scene = deepcopy(config.scene)
        robot = self.scene.objs.robot
        if not isinstance(robot, RobotCfg):
            raise ValidationError("backend.scene.objs.robot", "a RobotCfg", robot)
        compiler = MotrixSimSceneCompiler()
        self.world = compiler.build_world(self.scene)
        compiler.configure_world(self.world, config.physics)
        source_model = self.world.build()
        derived, actuator_names = self._derive_spec(source_model, robot)
        self._prepare_motors(source_model, derived, actuator_names)
        self.model = self.world.build()
        self.data: mtx.SceneData | None = None
        self.simulation_time_s = 0.0
        self.physics_substeps: int | None = None
        self.realtime = config.realtime
        factory = viewer_factory or MotrixSimViewer
        self.viewer = factory(self.scene.system_camera) if config.render else None
        self._robot = Robot(self, derived, actuator_names)

    @property
    def robot(self):
        return self._robot

    @staticmethod
    def _derive_spec(model: mtx.SceneModel, robot: RobotCfg) -> tuple[RobotSpec, tuple[str, ...]]:
        names = tuple(robot.key_pose.joint_names)
        lower, upper, torque = [], [], []
        actuator_names = []
        base = model.get_body(robot.resolved_base_link_name)
        if base is None or base.floatingbase is None:
            raise ValidationError(
                "backend.robot.base_link_name", "a floating-base robot", robot.resolved_base_link_name
            )
        for name in names:
            resolved_name = robot.resolve_name(name)
            joint = model.get_joint(resolved_name)
            if joint is None or joint.num_dof_pos != 1 or joint.num_dof_vel != 1:
                raise ValidationError(f"backend.joints.{name}", "an existing one-DoF joint", joint)
            matches = [
                actuator
                for actuator in model.actuators
                if actuator.target_type == "joint" and actuator.target_name == resolved_name
            ]
            if len(matches) != 1:
                raise ValidationError(f"backend.actuators.{name}", "exactly one joint actuator", len(matches))
            actuator = matches[0]
            if not isinstance(actuator, (mtx.PositionActuator, mtx.MotorActuator)):
                raise ValidationError(f"backend.actuators.{name}", "a position servo or torque motor", actuator.typ)
            limits = actuator.ctrl_range if isinstance(actuator, mtx.PositionActuator) else joint.range
            limits = np.asarray(limits, dtype=np.float32).reshape(-1)
            forces = np.asarray(actuator.force_range, dtype=np.float32).reshape(-1)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValidationError(f"backend.actuators.{name}.position_range", "finite increasing limits", limits)
            if (
                forces.shape != (2,)
                or not np.isfinite(forces).all()
                or forces[1] <= 0
                or not np.isclose(forces[0], -forces[1])
            ):
                raise ValidationError(f"backend.actuators.{name}.force_range", "finite symmetric torque limits", forces)
            physical_range = np.asarray(joint.range).reshape(2)
            if limits[0] < physical_range[0] - 1e-6 or limits[1] > physical_range[1] + 1e-6:
                raise ValidationError(
                    f"backend.joints.{name}.range", "session limits inside physical joint range", limits
                )
            actuator_names.append(actuator.name)
            lower.append(limits[0])
            upper.append(limits[1])
            torque.append(forces[1])
        spec = RobotSpec(
            base_link_name=robot.base_link_name,
            joint_names=names,
            default_joint_position=np.asarray(robot.key_pose.poses[robot.init_key_pose], dtype=np.float32),
            position_lower=np.asarray(lower, dtype=np.float32),
            position_upper=np.asarray(upper, dtype=np.float32),
            torque_limit=np.asarray(torque, dtype=np.float32),
        )
        return spec, tuple(actuator_names)

    def _prepare_motors(self, source_model: mtx.SceneModel, spec: RobotSpec, actuator_names: tuple[str, ...]) -> None:
        selected = dict(zip(actuator_names, spec.torque_limit))
        for actuator in self.world.actuators:
            if actuator.name not in selected:
                continue
            if actuator.target.variant != "joint" or not np.allclose(actuator.gear, [1, 0, 0, 0, 0, 0]):
                raise ValidationError(f"backend.actuators.{actuator.name}", "unit-gear joint transmission", actuator)
            if actuator.actuator_type.variant not in ("position", "motor"):
                raise ValidationError(f"backend.actuators.{actuator.name}", "position or motor", actuator.actuator_type)
            limit = selected[actuator.name]
            native = source_model.get_actuator(actuator.name)
            if isinstance(native, mtx.MotorActuator) and (
                native.ctrl_range is None or not np.allclose(native.ctrl_range, [-limit, limit], atol=1e-6, rtol=0)
            ):
                raise ValidationError(
                    f"backend.actuators.{actuator.name}.ctrl_range", [-limit, limit], native.ctrl_range
                )
            actuator.actuator_type = mtx.msd.ActuatorType.motor()
            actuator.ctrlrange = mtx.msd.Range(-limit, limit)
            actuator.forcerange = mtx.msd.Range(-limit, limit)
        # Imported keyframe servo targets must not become motor torque defaults.
        for keyframe in self.world.keyframes:
            keyframe.ctrl = [0.0] * len(self.world.actuators)

    def _configure_session_period(self, period_s: float) -> None:
        ratio = period_s / self.config.physics.dt
        if not np.isfinite(ratio) or not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0) or round(ratio) < 1:
            raise ValidationError("session.period_s", "an integer multiple of physics.dt", period_s)
        self.physics_substeps = round(ratio)

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("MotrixSim simulation is already open")
        if self.model is None:
            self.model = self.world.build()
        try:
            self._robot.bind()
            self._opened = True
            if self.viewer is not None:
                self.viewer.open(self.model)
                self.sync_viewer()
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        """Restore compiled placement without stale solver state or sensor reports."""
        if self.model is None:
            raise RuntimeError("MotrixSim model is not available")
        position = np.asarray(self.model.compute_init_dof_pos(), dtype=np.float32).copy()
        position[self._robot.joint_dof_pos_indices] = self.robot.spec.default_joint_position
        self.data = mtx.SceneData(self.model, batch=[1])
        self.data.reset(
            self.model,
            dof_pos=position,
            dof_vel=np.zeros(self.model.num_dof_vel, dtype=np.float32),
            forward_kinematic=True,
        )
        self.simulation_time_s = 0.0
        if self._opened:
            self._robot.refresh_reads()
            self._robot.stop()

    def _step(self) -> None:
        self._robot.apply_control()
        self.model.step(self.data)
        self.simulation_time_s += self.config.physics.dt

    def advance_control_period(self) -> None:
        if not self._opened:
            raise RuntimeError("MotrixSim simulation is not open")
        if self.physics_substeps is None:
            raise RuntimeError("Bind a session before advancing a session period")
        self.check_viewer_running()
        for _ in range(self.physics_substeps):
            self._step()
        self.sync_viewer()

    def _physics_time_s(self) -> float:
        return self.simulation_time_s

    def run(self, *, realtime: bool | None = None) -> "SimulationRolloutResult":
        if not self._opened:
            raise RuntimeError("MotrixSim simulation is not open")
        if self.session is None:
            raise ValidationError("runtime.session", "a bound control session", None)
        paced = self.realtime if realtime is None else realtime
        if paced is None:
            paced = self.viewer is not None
        scheduler = (RealtimeScheduler if paced else FixedStepScheduler)(self.session.period_s)
        return self._run_session(scheduler)

    def check_viewer_running(self) -> None:
        if self.viewer is not None and not self.viewer.is_running():
            raise KeyboardInterrupt("MotrixSim viewer was closed")

    def sync_viewer(self) -> None:
        if self.viewer is not None:
            self.check_viewer_running()
            self.viewer.sync(self.data)

    def get_keyboard_device(self):
        if self.viewer is None:
            raise RuntimeError("MotrixSim keyboard input requires render=true")
        return self.viewer.keyboard_device

    def close(self) -> None:
        try:
            if self.viewer is not None:
                self.viewer.close()
        finally:
            self._robot.invalidate()
            self.data = None
            self.model = None
            self._opened = False
