# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Lightweight test-owned task spec plugin; never installed in production."""

from typing import ClassVar

from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy.task import DeployTask


class TestTaskSpec(TaskSpec):
    task_name: ClassVar[str] = "test/v1"


class TestDeployTask(DeployTask[object]):
    spec_type = TestTaskSpec

    def __init__(self, spec, robot, policy, steps=None, artifact=None):
        self.spec = spec
        self.robot = robot
        self.policy = policy
        self.steps = steps
        self.states = []
        self.stop_requested = False

    def reset(self, state):
        self.states = [state]
        self.stop_requested = False

    def step(self, state, context):
        self.states.append(state)
        return ControllerStep(complete=True)

    def request_stop(self):
        self.stop_requested = True

    def validate_command(self, command):
        if command is not None:
            raise ValueError("fixture task accepts no external input")
