# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Test-owned controller for backend lifecycle and cadence contracts."""

import numpy as np

from motrix_deploy.contracts import FloatArray, JointServoCommand, RobotSpec, RobotState
from motrix_deploy.runtime.context import PolicyContext
from motrix_deploy.task import DeployTask
from motrix_env_core.input import PlanarVelocityCommand


class DummyServoTask(DeployTask[PlanarVelocityCommand]):
    """Keep generic driver tests independent of examples and trained artifacts."""

    def __init__(self, robot: RobotSpec, kp: FloatArray, kd: FloatArray) -> None:
        self.robot = robot
        self.kp = kp
        self.kd = kd
        self._zeros = np.zeros(robot.joint_count, dtype=np.float32)

    def reset(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> None:
        pass

    def build_observation(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> FloatArray:
        return self._zeros.copy()

    def process_action(self, action: FloatArray) -> JointServoCommand:
        return JointServoCommand(
            joint_position=self.robot.default_joint_position + action,
            joint_velocity=self._zeros.copy(),
            feedforward_torque=self._zeros.copy(),
            kp=self.kp,
            kd=self.kd,
        )

    def validate_command(self, command: PlanarVelocityCommand) -> None:
        if command.batch_size != 1 or np.any(command.values != 0.0):
            raise ValueError("DummyServoTask accepts one zero velocity command")
