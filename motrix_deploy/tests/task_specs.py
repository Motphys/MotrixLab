# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Lightweight test-owned task spec plugin; never installed in production."""

from typing import ClassVar

from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.task import DeployTask


class TestTaskSpec(TaskSpec):
    task_name: ClassVar[str] = "test/v1"


class TestDeployTask(DeployTask[object]):
    spec_type = TestTaskSpec

    def __init__(self, spec, robot):
        self.spec = spec
        self.robot = robot

    def reset(self, state, context):
        pass

    def build_observation(self, state, context):
        raise NotImplementedError

    def process_action(self, action):
        raise NotImplementedError

    def validate_command(self, command):
        pass
