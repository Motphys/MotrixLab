# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared artifact fixtures."""

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from task_specs import TestDeployTask, TestTaskSpec

from motrix_deploy.artifact import (
    ControlSpec,
    DeploymentManifest,
    PolicySpec,
    SourceSpec,
    sha256_bytes,
)
from motrix_deploy.contracts import RobotSpec, TensorSpec

POLICY_BYTES = b"deterministic ONNX placeholder"


@pytest.fixture
def manifest_factory(monkeypatch) -> Callable[[], DeploymentManifest]:
    from motrix_deploy import task

    original = task.metadata.entry_points

    def entry_points(*, group):
        if group == task.TASK_ENTRY_POINT_GROUP:
            return [SimpleNamespace(name="test/v1", load=lambda: TestDeployTask)]
        return original(group=group)

    monkeypatch.setattr(task.metadata, "entry_points", entry_points)

    def create() -> DeploymentManifest:
        return DeploymentManifest(
            schema_version="motrix-deploy/v1",
            source=SourceSpec(framework="test", run_id="fixture", checkpoint="fixture.pt"),
            policy=PolicySpec(
                component_version="onnx/v1",
                payload_path="policy/model.onnx",
                sha256=sha256_bytes(POLICY_BYTES),
                input=TensorSpec(name="observation", shape=(1, 4)),
                output=TensorSpec(name="action", shape=(1, 2)),
            ),
            robot=RobotSpec(
                base_link_name="base",
                joint_names=("left_joint", "right_joint"),
                default_joint_position=np.array([0.0, 0.0], dtype=np.float32),
                position_lower=np.array([-1.0, -1.0], dtype=np.float32),
                position_upper=np.array([1.0, 1.0], dtype=np.float32),
                torque_limit=np.array([3.0, 3.0], dtype=np.float32),
            ),
            task=TestTaskSpec(),
            control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
        )

    return create
