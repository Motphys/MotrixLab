# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Artifact schema, safety and checksum tests."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from motrix_deploy.artifact import (
    ControlSpec,
    DeploymentManifest,
    inspect_artifact,
    read_artifact,
    sha256_bytes,
    write_artifact,
)
from motrix_deploy.contracts import JointControlMode
from motrix_deploy.errors import ArtifactError, ValidationError

POLICY_BYTES = b"deterministic ONNX placeholder"


def test_artifact_round_trip_and_inspect(
    tmp_path: Path,
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    output = tmp_path / "fixture.deploy"
    artifact = write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})
    summary = inspect_artifact(output)

    assert artifact.manifest.to_dict() == read_artifact(output).manifest.to_dict()
    assert summary["valid"] is True
    assert summary["task"] == {"name": "test/v1", "config": {}}
    assert summary["observation_size"] == artifact.manifest.policy.input.shape[1]
    assert summary["action_size"] == artifact.manifest.policy.output.shape[1]
    assert summary["robot"]["joint_names"] == ["left_joint", "right_joint"]
    assert summary["policy"]["input"] == {"name": "observation", "shape": [1, 4], "dtype": "float32"}


def test_task_payload_round_trip_and_tamper_rejection(
    tmp_path: Path,
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    import dataclasses

    from motrix_deploy.artifact.schema import PayloadSpec

    payload = b"motion-clip-npz-bytes"
    manifest = dataclasses.replace(
        manifest_factory(),
        payloads=(PayloadSpec(path="payloads/motion.npz", sha256=sha256_bytes(payload)),),
    )
    output = tmp_path / "fixture.deploy"
    artifact = write_artifact(output, manifest, {"policy/model.onnx": POLICY_BYTES, "payloads/motion.npz": payload})
    assert (output / "payloads" / "motion.npz").read_bytes() == payload
    assert artifact.manifest.payloads[0].path == "payloads/motion.npz"
    # A modified task payload is rejected by its checksum.
    (output / "payloads" / "motion.npz").write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="payload sha256 mismatch"):
        read_artifact(output)
    # Missing and unexpected payload keys are rejected at the write boundary.
    with pytest.raises(ArtifactError, match="payload keys must be exactly"):
        write_artifact(tmp_path / "short.deploy", manifest, {"policy/model.onnx": POLICY_BYTES})


def test_cli_inspect_outputs_json(tmp_path: Path, manifest_factory: Callable[[], DeploymentManifest]) -> None:
    output = tmp_path / "fixture.deploy"
    write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})

    # A subprocess cannot inherit monkeypatched discovery. Install only test
    # metadata on its temporary import path, not a production fixture plugin.
    dist_info = tmp_path / "test_task_spec-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: test-task-spec\nVersion: 1.0\n", encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(
        "[motrix_deploy.tasks]\ntest/v1 = task_specs:TestDeployTask\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).parent), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-m", "motrix_deploy.cli", "inspect", f"artifact={output}"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["valid"] is True


def test_corrupt_checksum_is_rejected(tmp_path: Path, manifest_factory: Callable[[], DeploymentManifest]) -> None:
    output = tmp_path / "fixture.deploy"
    write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})
    (output / "policy/model.onnx").write_bytes(b"corrupt")

    with pytest.raises(ArtifactError, match="policy.sha256 mismatch"):
        read_artifact(output)


@pytest.mark.parametrize("payload_path", ["../model.onnx", "/tmp/model.onnx", "policy\\model.onnx"])
def test_unsafe_payload_path_is_rejected(
    tmp_path: Path,
    manifest_factory: Callable[[], DeploymentManifest],
    payload_path: str,
) -> None:
    manifest = manifest_factory().to_dict()
    manifest["policy"]["payload_path"] = payload_path
    output = tmp_path / "unsafe.deploy"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValidationError, match="policy.payload_path"):
        read_artifact(output)


def test_unknown_schema_version_is_rejected(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory().to_dict()
    manifest["schema_version"] = "motrix-deploy/v999"

    with pytest.raises(ValidationError, match="schema_version"):
        DeploymentManifest.from_dict(manifest)


@pytest.mark.parametrize("name", ["test", "test/v0", "test/v01", "test/v1/extra", 1])
def test_invalid_task_name_is_rejected(
    manifest_factory: Callable[[], DeploymentManifest],
    name: object,
) -> None:
    manifest = manifest_factory().to_dict()
    manifest["task"]["name"] = name

    with pytest.raises(ValidationError, match="task.name"):
        DeploymentManifest.from_dict(manifest)


@pytest.mark.parametrize("field_name", ["input", "output"])
@pytest.mark.parametrize("shape", [[4], [2, 4], [1, 2, 3], [1, 0], [1, -1], [1, 1.5]])
def test_policy_requires_positive_batch_one_vector(
    manifest_factory: Callable[[], DeploymentManifest],
    field_name: str,
    shape: list[int | float],
) -> None:
    manifest = manifest_factory().to_dict()
    manifest["policy"][field_name]["shape"] = shape

    with pytest.raises(ValidationError, match=rf"policy\.{field_name}\.shape"):
        DeploymentManifest.from_dict(manifest)


@pytest.mark.parametrize("field_name", ["input", "output"])
def test_policy_vector_size_is_independent_of_robot_joint_count(
    manifest_factory: Callable[[], DeploymentManifest],
    field_name: str,
) -> None:
    manifest = manifest_factory().to_dict()
    manifest["policy"][field_name]["shape"] = [1, 7]

    parsed = DeploymentManifest.from_dict(manifest)

    assert getattr(parsed.policy, field_name).shape == (1, 7)
    assert parsed.task.to_dict() == {"name": "test/v1", "config": {}}


@pytest.mark.parametrize("mode", list(JointControlMode))
def test_control_mode_serializes_explicitly(mode: JointControlMode) -> None:
    control = ControlSpec(period_s=0.02, state_timeout_s=0.1, mode=mode)
    assert control.to_dict()["mode"] == mode.value
    assert ControlSpec.from_dict(control.to_dict()).mode is mode
    assert ControlSpec(period_s=0.02, state_timeout_s=0.1).mode is JointControlMode.SERVO


def test_control_mode_is_required_in_manifest(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory().to_dict()
    del manifest["control"]["mode"]
    with pytest.raises(ValidationError, match="control.*mode"):
        DeploymentManifest.from_dict(manifest)


@pytest.mark.parametrize("mode", ["invalid", None, 1])
def test_invalid_control_mode_is_rejected(mode: object) -> None:
    with pytest.raises(ValidationError, match="control.mode"):
        ControlSpec(period_s=0.02, state_timeout_s=0.1, mode=mode)


def test_joint_count_mismatch_is_rejected(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory().to_dict()
    manifest["robot"]["joint_names"] = ["left_joint"]

    with pytest.raises(ValidationError, match="robot.default_joint_position.shape"):
        DeploymentManifest.from_dict(manifest)


def test_reversed_limit_is_rejected(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory().to_dict()
    manifest["robot"]["position_lower"][0] = 2.0

    with pytest.raises(ValidationError, match="robot.position_range"):
        DeploymentManifest.from_dict(manifest)


def test_writer_is_create_only(tmp_path: Path, manifest_factory: Callable[[], DeploymentManifest]) -> None:
    output = tmp_path / "fixture.deploy"
    write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})

    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        write_artifact(output, manifest_factory(), {"policy/model.onnx": POLICY_BYTES})
