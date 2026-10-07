# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Explicit external owner of MuJoCo resources, robot bindings and physics."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar, cast

from motrix_deploy.contracts import RobotSpec
from motrix_deploy.robot.interface import RobotInterface
from motrix_deploy.runtime.base import SimulationRuntime
from motrix_deploy.runtime.config import SimulationRuntimeConfig
from motrix_env_core.input import KeyboardDevice

if TYPE_CHECKING:
    import mujoco

    from motrix_deploy.runtime.control import ControlSession

import numpy as np
from numpy.typing import NDArray

from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.result import RolloutResult
from motrix_deploy.runtime.scheduler import FixedStepScheduler, RealtimeScheduler
from motrix_deploy_mujoco.robot import Robot
from motrix_deploy_mujoco.viewer import MujocoGlfwViewer
from motrix_env_core.config.scene import SystemCameraCfg
from motrix_env_core.config.scene.base import RobotCfg
from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

CommandT = TypeVar("CommandT")

logger = logging.getLogger(__name__)


class MujocoRuntime(SimulationRuntime):
    """Own a scene and expose independently controlled robot I/O and physics."""

    def __init__(
        self,
        config: SimulationRuntimeConfig,
        control_period_s: float | None = None,
        viewer_factory: Callable[[ModuleType, SystemCameraCfg], MujocoGlfwViewer] = MujocoGlfwViewer,
    ) -> None:
        super().__init__()
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("MuJoCo backend support requires the 'motrix-deploy-mujoco' package") from error
        self.scene = deepcopy(config.scene)
        if not isinstance(self.scene.objs.robot, RobotCfg):
            raise ValidationError("backend.scene.objs.robot", "a RobotCfg", self.scene.objs.robot)
        self.mj = mujoco
        self.sim = config.physics
        self.physics_substeps: int | None = None
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.viewer = viewer_factory(mujoco, self.scene.system_camera) if config.render else None
        self.config = config
        self.control_period_s: float | None = None
        if control_period_s is not None:
            self._configure_control_period(control_period_s)
        self.realtime = config.realtime
        self._closed = False
        self._source_spec: mujoco.MjSpec | None = None
        robot_spec = self._derive_robot_spec()
        self._robot = Robot(self, robot_spec)

    @property
    def robot(self) -> RobotInterface:
        return self._robot

    def _configure_control_period(self, period_s: float) -> None:
        ratio = period_s / self.sim.dt
        if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValidationError("control.period_s", "an integer multiple of physics.dt", period_s)
        self.control_period_s = period_s
        self.physics_substeps = round(ratio)

    def bind_control_session(self, control: ControlSession[CommandT]) -> None:
        """Bind control ownership and derive physics cadence from the session."""
        # Resolve identity conflicts before changing the runtime's timing.
        self._resolve_control(control)
        self._configure_control_period(control.period_s)
        super().bind_control_session(control)

    def _derive_robot_spec(self) -> RobotSpec:
        """Inspect physical limits, never controller gains, before binding robot I/O."""
        robot = cast(RobotCfg, self.scene.objs.robot)
        self._source_spec = MuJoCoSceneCompiler().create_spec(self.scene, self.sim)
        model = self._source_spec.compile()
        joint_names = tuple(robot.key_pose.joint_names)
        actuators = []
        for name in joint_names:
            joint_id = model.joint(robot.resolve_name(name)).id
            matches = np.flatnonzero(
                (model.actuator_trntype == self.mj.mjtTrn.mjTRN_JOINT) & (model.actuator_trnid[:, 0] == joint_id)
            )
            if matches.size != 1:
                raise ValidationError(f"backend.actuators.{name}", "exactly one joint actuator", matches.tolist())
            actuators.append(int(matches[0]))
        indices = np.asarray(actuators, dtype=np.int64)
        joint_ids = np.asarray([model.joint(robot.resolve_name(name)).id for name in joint_names])
        motors = model.actuator_biastype[indices] == self.mj.mjtBias.mjBIAS_NONE
        position_range = np.where(motors[:, None], model.jnt_range[joint_ids], model.actuator_ctrlrange[indices])
        return RobotSpec(
            base_link_name=robot.base_link_name,
            joint_names=joint_names,
            default_joint_position=np.asarray(robot.key_pose.poses[robot.init_key_pose], dtype=np.float32),
            position_lower=position_range[:, 0].astype(np.float32),
            position_upper=position_range[:, 1].astype(np.float32),
            torque_limit=model.actuator_forcerange[indices, 1].astype(np.float32),
        )

    def get_keyboard_device(self) -> KeyboardDevice:
        """Return keyboard input owned by this simulation's viewer."""
        if self.viewer is None:
            raise RuntimeError("MuJoCo keyboard input requires viewer=true")
        return self.viewer.keyboard_device

    def build_model(self) -> mujoco.MjModel:
        """Validate and transform registered robot servos before final compilation."""
        model_spec = self._source_spec or MuJoCoSceneCompiler().create_spec(self.scene, self.sim)
        self._source_spec = None
        source_model = model_spec.compile()
        self.model = source_model
        self._robot.prepare(source_model, model_spec)
        model = model_spec.compile()
        logger.info("Built MuJoCo deployment model from SceneCfg %s", type(self.scene).__name__)
        return model

    def open(self) -> None:
        """Compile, prepare, bind, and reset the registered robot."""
        if self._opened:
            raise RuntimeError("MuJoCo simulation is already open")
        self._closed = False
        try:
            self.model = self.build_model()
            self.data = self.mj.MjData(self.model)
            self._robot.bind()
            if self.viewer is not None:
                self.viewer.open(self.model, self.data)
            self.sync_viewer()
            self._opened = True
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("Cleanup failed after MuJoCo session open failure")
            raise

    def reset(
        self,
        joint_qpos_indices: NDArray[np.int64],
        joint_position: NDArray[np.float32],
    ) -> None:
        assert self.model is not None and self.data is not None
        self.mj.mj_resetData(self.model, self.data)
        qpos = self.data.qpos
        # qpos0 already composes RobotCfg's attach transform with the imported
        # model root pose. Keep it intact instead of applying placement twice.
        qpos[joint_qpos_indices] = joint_position
        self.data.qvel.fill(0.0)
        self.data.ctrl.fill(0.0)
        self.mj.mj_forward(self.model, self.data)

    def run(
        self,
        control: ControlSession[CommandT] | None = None,
        *,
        steps: int | None = None,
        realtime: bool | None = None,
    ) -> RolloutResult:
        """Drive shared control at physics boundaries inside an already-open world.

        Control writes cached commands only. This owner re-evaluates robot PD on
        every physics substep, and paces complete control intervals independently
        of the shared control lifecycle. Stopping control leaves this world open.
        """
        if not self._opened:
            raise RuntimeError("MuJoCo simulation with a registered robot is not open")
        control = self._resolve_control(control)
        if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0):
            raise ValidationError("steps", "a positive integer", steps)
        self._configure_control_period(control.period_s)
        if realtime is None:
            realtime = self.realtime
        paced = self.viewer is not None if realtime is None else realtime
        assert self.physics_substeps is not None and self.control_period_s is not None
        scheduler = (RealtimeScheduler if paced else FixedStepScheduler)(self.control_period_s)
        assert self.data is not None and self.model is not None
        initial_time_s = float(self.data.time)
        simulation_time_s = 0.0
        try:
            if control.start():
                # Establish relative simulation time after control initialization.
                initial_time_s = float(self.data.time)
                scheduler.reset()
                substep = 0
                while control.active:
                    if substep % self.physics_substeps == 0:
                        if steps is not None and control.completed_steps >= steps:
                            break
                        self.check_viewer_running()
                        if not control.tick(elapsed_time_s=float(self.data.time) - initial_time_s):
                            break
                    self._robot.apply_control()
                    self.mj.mj_step(self.model, self.data)
                    substep += 1
                    if substep % self.physics_substeps == 0:
                        self.sync_viewer()
                        # Absolute deadlines avoid accumulated wall-clock drift,
                        # and include the final interval without an extra tick.
                        scheduler.wait(substep // self.physics_substeps)
        except (Exception, KeyboardInterrupt) as error:
            control.fail(error)
        finally:
            simulation_time_s = float(self.data.time) - initial_time_s
            control.stop()
        result = control.result(steps=steps, overrun_count=scheduler.overrun_count)
        return replace(
            result,
            simulation_time_s=simulation_time_s,
            real_time_factor=simulation_time_s / result.wall_time_s if result.wall_time_s > 0 else 0.0,
        )

    def advance_control_period(self) -> None:
        """Advance physics, re-evaluating cached robot PD at every substep."""
        if not self._opened:
            raise RuntimeError("MuJoCo simulation with a registered robot is not open")
        if self.physics_substeps is None:
            raise RuntimeError("Bind a control session before advancing a control period")
        self.check_viewer_running()
        for _ in range(self.physics_substeps):
            self._robot.apply_control()
            self.mj.mj_step(self.model, self.data)
        self.sync_viewer()

    def check_viewer_running(self) -> None:
        if self.viewer is not None and not self.viewer.is_running():
            raise KeyboardInterrupt("MuJoCo viewer was closed")

    def sync_viewer(self) -> None:
        if self.viewer is not None:
            self.check_viewer_running()
            self.viewer.sync()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.control is not None and self.control.active:
                self.control.stop()
            if self.viewer is not None:
                self.viewer.close()
                deadline = time.monotonic() + 2.0
                while self.viewer.is_running() and time.monotonic() < deadline:
                    time.sleep(0.001)
                if self.viewer.is_running():
                    raise RuntimeError("MuJoCo viewer did not stop within 2 seconds")
        finally:
            self._robot.invalidate()
            self.data = None
            self.model = None
            self._opened = False
            self._closed = True
