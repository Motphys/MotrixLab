# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Explicit external owner of MuJoCo resources, robot bindings and physics."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from copy import deepcopy
from types import ModuleType
from typing import TYPE_CHECKING, cast

from motrix_deploy.contracts import RobotSpec
from motrix_deploy.robot.interface import RobotInterface
from motrix_deploy.runtime.base import SimulationRuntime
from motrix_deploy.runtime.config import SimulationRuntimeConfig
from motrix_env_core.input import KeyboardDevice

if TYPE_CHECKING:
    import mujoco


import numpy as np
from numpy.typing import NDArray

from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.result import SimulationRolloutResult
from motrix_deploy.runtime.scheduler import FixedStepScheduler, RealtimeScheduler
from motrix_deploy_mujoco.robot import Robot
from motrix_deploy_mujoco.viewer import MujocoGlfwViewer
from motrix_env_core.config.scene import SystemCameraCfg
from motrix_env_core.config.scene.base import RobotCfg
from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

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
            self._configure_session_period(control_period_s)
        self.realtime = config.realtime
        self._closed = False
        self._source_spec: mujoco.MjSpec | None = None
        self._speed_binding: object | None = None
        robot_spec = self._derive_robot_spec()
        self._robot = Robot(self, robot_spec)

    @property
    def robot(self) -> RobotInterface:
        return self._robot

    def _configure_session_period(self, period_s: float) -> None:
        ratio = period_s / self.sim.dt
        if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValidationError("session.period_s", "an integer multiple of physics.dt", period_s)
        self.control_period_s = period_s
        self.physics_substeps = round(ratio)

    def bind_session(self, session: ControlSession) -> None:
        super().bind_session(session)
        self._speed_binding = self._find_speed_binding(session.command_binding)

    def _find_speed_binding(self, binding: object | None) -> object | None:
        visited: set[int] = set()
        while binding is not None and id(binding) not in visited:
            visited.add(id(binding))
            if hasattr(binding, "axis_speed_scales"):
                return binding
            binding = getattr(binding, "_binding", None)
        return None

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
            default_joint_position=np.asarray(robot.key_pose.poses["default"], dtype=np.float32),
            position_lower=position_range[:, 0].astype(np.float32),
            position_upper=position_range[:, 1].astype(np.float32),
            torque_limit=model.actuator_forcerange[indices, 1].astype(np.float32),
        )

    def initial_joint_position(self, spec: RobotSpec) -> NDArray[np.float32]:
        robot = cast(RobotCfg, self.scene.objs.robot)
        pose = np.asarray(robot.key_pose.poses[robot.init_key_pose], dtype=np.float32)
        return np.clip(pose, spec.position_lower, spec.position_upper)

    @property
    def keyboard(self) -> KeyboardDevice | None:
        """Return viewer keyboard input, or None when rendering is disabled."""
        if self.viewer is None:
            return None
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

    def _physics_time_s(self) -> float:
        return float(self.data.time)

    def run(self, *, realtime: bool | None = None) -> SimulationRolloutResult:
        if not self._opened:
            raise RuntimeError("MuJoCo simulation with a registered robot is not open")
        if self.session is None:
            raise ValidationError("runtime.session", "a bound control session", None)
        paced = self.realtime if realtime is None else realtime
        if paced is None:
            paced = self.viewer is not None
        scheduler = (RealtimeScheduler if paced else FixedStepScheduler)(self.session.period_s)
        return self._run_session(scheduler)

    def advance_control_period(self) -> None:
        """Advance physics, re-evaluating cached robot PD at every substep."""
        if not self._opened:
            raise RuntimeError("MuJoCo simulation with a registered robot is not open")
        if self.physics_substeps is None:
            raise RuntimeError("Bind a session before advancing a session period")
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
            self._update_viewer_overlay()
            self.viewer.sync()

    def _update_viewer_overlay(self) -> None:
        session = self.session
        if session is None:
            return
        state = session.last_state
        command = session.last_command
        if state is None or command is None or not hasattr(command, "values"):
            return

        target = np.asarray(command.values[0, :3], dtype=np.float32)
        actual = np.zeros(3, dtype=np.float32)
        if state.base_linear_velocity is not None:
            world_velocity = np.asarray(state.base_linear_velocity, dtype=np.float64)
            x, y, z, w = np.asarray(state.base_orientation_xyzw, dtype=np.float64)
            body_to_world = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )
            actual[:2] = (body_to_world.T @ world_velocity)[:2]
        actual[2] = float(np.asarray(state.base_angular_velocity, dtype=np.float64)[2])
        binding = self._speed_binding
        axis_scales = getattr(binding, "axis_speed_scales", None) if binding is not None else None
        levels = getattr(self._speed_binding, "speed_levels", None) if self._speed_binding is not None else None
        if axis_scales is not None and levels is not None:
            scale_text = (
                f"all {axis_scales[0]:.0%}  |  W/S {axis_scales[0]:.0%}  "
                f"A/D {axis_scales[1]:.0%}  Q/E {axis_scales[2]:.0%}"
            )
            level_text = "levels " + "/".join(f"{value:g}" for value in levels)
        else:
            scale_text = "speed N/A"
            level_text = "levels N/A"
        primary = (
            "Go2 speed control  "
            f"{scale_text}\n{level_text}\n"
            f"target  vx={target[0]:+.3f} m/s  vy={target[1]:+.3f} m/s  yaw={target[2]:+.3f} rad/s\n"
            f"measured vx={actual[0]:+.3f} m/s  vy={actual[1]:+.3f} m/s  yaw={actual[2]:+.3f} rad/s"
        )
        secondary = "UP/DOWN all  1/2 W/S speed  3/4 A/D speed  5/6 Q/E speed"
        self.viewer.set_overlay(primary, secondary)

    def close(self) -> None:
        if self._closed:
            return
        try:
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
