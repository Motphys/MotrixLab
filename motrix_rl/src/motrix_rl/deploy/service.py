# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compose exported policies and injected task profiles into deployment artifacts."""

from collections.abc import Callable
from pathlib import Path

from motrix_deploy.artifact import (
    DeploymentManifest,
    PolicySpec,
    SourceSpec,
    sha256_bytes,
    write_artifact,
)
from motrix_deploy.contracts import TensorSpec
from motrix_deploy.profile import DeploymentProfile
from motrix_rl import checkpoints
from motrix_rl.deploy.api import DeploymentExportResult, PolicyTensorSpec
from motrix_rl.deploy.onnx import export_onnx_model
from motrix_rl.runs import open_run_context

ProfileBuilder = Callable[[str], DeploymentProfile]


def export_deploy_run(
    run_dir: str | Path,
    output: str | Path,
    *,
    profile_builder: ProfileBuilder,
    validation_seed: int = 1,
    validation_samples: int = 32,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> DeploymentExportResult:
    """Export one metadata-selected policy with its task deployment profile."""
    if validation_samples <= 0:
        raise ValueError(f"validation_samples must be positive, got {validation_samples}")
    context = open_run_context(run_dir)
    metadata = context.metadata
    profile = profile_builder(metadata.env_name)
    checkpoint_path = checkpoints.best_policy(metadata, context.run_dir)
    exported = export_onnx_model(
        context.run_dir,
        validation_seed=validation_seed,
        validation_samples=validation_samples,
        atol=atol,
        rtol=rtol,
    )
    policy_bytes = exported.model_bytes
    input_spec = _deployment_tensor_spec(
        exported.report.input_spec,
        expected_size=profile.observation_size,
        kind="input",
    )
    output_spec = _deployment_tensor_spec(
        exported.report.output_spec,
        expected_size=profile.action_size,
        kind="output",
    )

    try:
        checkpoint_source = str(checkpoint_path.relative_to(context.run_dir))
    except ValueError:
        checkpoint_source = str(checkpoint_path)
    manifest = DeploymentManifest(
        schema_version="motrix-deploy/v1",
        source=SourceSpec(
            framework=f"{metadata.rllib}.{metadata.algo}/{metadata.train_backend}",
            run_id=context.run_dir.name,
            checkpoint=checkpoint_source,
        ),
        policy=PolicySpec(
            component_version="onnx/v1",
            payload_path="policy/model.onnx",
            sha256=sha256_bytes(policy_bytes),
            input=input_spec,
            output=output_spec,
        ),
        robot=profile.robot,
        task=profile.task,
        control=profile.control,
    )
    artifact = write_artifact(output, manifest, {"policy/model.onnx": policy_bytes})
    return DeploymentExportResult(
        artifact=artifact,
        validation_samples=exported.report.parity.samples,
        max_abs_error=exported.report.parity.max_abs_error,
    )


def _deployment_tensor_spec(spec: PolicyTensorSpec, *, expected_size: int, kind: str) -> TensorSpec:
    expected_shape = (None, expected_size)
    if spec.shape != expected_shape:
        raise ValueError(f"deployment policy {kind} shape must be {expected_shape}, got {spec.shape}")
    return TensorSpec(name=spec.name, shape=(1, expected_size), dtype=spec.dtype)
