# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-agnostic deployment export service tests."""

from importlib import metadata
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from motrix_deploy.artifact import ControlSpec, TaskSpec
from motrix_deploy.contracts import RobotSpec
from motrix_deploy.profile import DeploymentProfile
from motrix_deploy.task import DeployTask
from motrix_rl import checkpoints, runs
from motrix_rl.deploy.api import OnnxExportReport, OnnxModelArtifact, OnnxParityMetrics, PolicyTensorSpec
from motrix_rl.deploy.service import export_deploy_run


class TestTaskSpec(TaskSpec):
    __test__ = False
    task_name: ClassVar[str] = "test/v1"


class TestDeployTask(DeployTask):
    __test__ = False
    spec_type = TestTaskSpec

    def reset(self, state) -> None:
        pass

    def step(self, state, context):
        raise NotImplementedError

    def request_stop(self) -> None:
        pass

    def validate_command(self, command) -> None:
        pass


@pytest.fixture(autouse=True)
def test_task_spec_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    installed_entry_points = metadata.entry_points
    entry = SimpleNamespace(name=TestTaskSpec.task_name, load=lambda: TestDeployTask)

    def entry_points(**kwargs):
        entries = installed_entry_points(**kwargs)
        if kwargs.get("group") == "motrix_deploy.tasks":
            return (*entries, entry)
        return entries

    monkeypatch.setattr(metadata, "entry_points", entry_points)


def _profile() -> DeploymentProfile:
    return DeploymentProfile(
        robot=RobotSpec(
            base_link_name="base",
            joint_names=("left", "right"),
            default_joint_position=np.zeros(2, dtype=np.float32),
            position_lower=np.full(2, -1.0, dtype=np.float32),
            position_upper=np.full(2, 1.0, dtype=np.float32),
            torque_limit=np.full(2, 3.0, dtype=np.float32),
        ),
        task=TestTaskSpec(),
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
        payloads={"payloads/motion.npz": b"motion-clip-npz-bytes"},
        observation_size=4,
        action_size=2,
    )


@pytest.mark.parametrize(
    ("input_size", "output_size", "invalid_kind"),
    [(4, 2, None), (5, 2, "input"), (4, 3, "output")],
)
def test_deployment_export_injects_profile_builder_selected_by_run_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    input_size: int,
    output_size: int,
    invalid_kind: str | None,
) -> None:
    run = runs.create_run_context(
        env_name="cartpole",
        rllib="skrl",
        train_backend="torch",
        algo="ppo",
        seed=1,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )
    checkpoint = run.checkpoint_dir / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    checkpoints.record_checkpoint_artifact(
        run.run_dir,
        checkpoints.BEST_POLICY,
        checkpoint,
        checkpoints.POLICY,
        checkpoint_format="pt",
    )
    model = OnnxModelArtifact(
        model_bytes=b"validated ONNX payload",
        report=OnnxExportReport(
            input_spec=PolicyTensorSpec(name="obs", shape=(None, input_size), dtype="float32"),
            output_spec=PolicyTensorSpec(name="actions", shape=(None, output_size), dtype="float32"),
            parity=OnnxParityMetrics(samples=8, max_abs_error=1e-6, max_rel_error=2e-6),
        ),
    )
    monkeypatch.setattr("motrix_rl.deploy.service.export_onnx_model", lambda *args, **kwargs: model)
    selected: list[str] = []

    if invalid_kind is not None:
        with pytest.raises(ValueError, match=f"deployment policy {invalid_kind} shape"):
            export_deploy_run(
                run.run_dir,
                tmp_path / "test.deploy",
                profile_builder=lambda env_name: selected.append(env_name) or _profile(),
                validation_samples=8,
            )
        assert not (tmp_path / "test.deploy").exists()
        return

    result = export_deploy_run(
        run.run_dir,
        tmp_path / "test.deploy",
        profile_builder=lambda env_name: selected.append(env_name) or _profile(),
        validation_samples=8,
    )

    assert selected == ["cartpole"]
    assert isinstance(result.artifact.manifest.task, TestTaskSpec)
    manifest = result.artifact.manifest.to_dict()
    assert manifest["task"] == {"name": "test/v1", "config": {}}
    assert result.artifact.manifest.policy.input.shape == (1, _profile().observation_size)
    assert result.artifact.manifest.policy.output.shape == (1, _profile().action_size)
    assert result.artifact.manifest.source.framework == "skrl.ppo/torch"
    # Task payloads are declared in the manifest and written next to the policy.
    (payload,) = result.artifact.manifest.payloads
    assert payload.path == "payloads/motion.npz"
    assert (result.artifact.root / "payloads" / "motion.npz").read_bytes() == b"motion-clip-npz-bytes"
    assert result.artifact.policy_path.read_bytes() == model.model_bytes
    assert result.validation_samples == 8
    assert not checkpoint.with_name("policy.onnx").exists()
