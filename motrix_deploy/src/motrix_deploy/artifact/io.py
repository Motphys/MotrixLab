# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Safe and atomic artifact filesystem I/O."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from motrix_deploy.artifact.schema import DeploymentManifest
from motrix_deploy.errors import ArtifactError, ValidationError

MANIFEST_NAME = "manifest.json"


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of payload bytes."""
    return hashlib.sha256(payload).hexdigest()


def _safe_relative_path(value: str, *, field_path: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValidationError(field_path, "a non-empty POSIX relative path", value)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValidationError(field_path, "a normalized relative path without '..'", value)
    return path


def _payload_path(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactError(f"artifact payload must not contain symlinks: {relative}")
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ArtifactError(f"artifact payload escapes root: {relative}")
    return candidate


@dataclass(frozen=True)
class Artifact:
    """A validated artifact root and its parsed manifest."""

    root: Path
    manifest: DeploymentManifest

    @property
    def policy_path(self) -> Path:
        relative = _safe_relative_path(self.manifest.policy.payload_path, field_path="policy.payload_path")
        return _payload_path(self.root, relative)


def write_artifact(
    output: str | Path,
    manifest: DeploymentManifest,
    payloads: Mapping[str, bytes],
) -> Artifact:
    """Validate and atomically write a new artifact directory."""
    target = Path(output).absolute()
    if target.exists():
        raise ArtifactError(f"refusing to overwrite existing artifact: {target}")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    expected_path = _safe_relative_path(manifest.policy.payload_path, field_path="policy.payload_path")
    expected_payloads = {str(expected_path): manifest.policy.sha256}
    for payload in manifest.payloads:
        relative = _safe_relative_path(payload.path, field_path="payloads.path")
        expected_payloads[str(relative)] = payload.sha256
    if set(payloads) != set(expected_payloads):
        raise ArtifactError(f"payload keys must be exactly {sorted(expected_payloads)}, got {sorted(payloads)!r}")
    for relative, expected_sha in expected_payloads.items():
        if not isinstance(payloads[relative], bytes):
            raise ArtifactError(f"payload {relative} must be bytes")
        digest = sha256_bytes(payloads[relative])
        if digest != expected_sha:
            raise ArtifactError(f"sha256 mismatch for payload {relative}: expected {expected_sha}, got {digest}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
    try:
        for relative in sorted(expected_payloads):
            payload_path = _payload_path(staging, PurePosixPath(relative))
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(payloads[relative])
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read_artifact(target)


def read_artifact(root: str | Path) -> Artifact:
    """Read and fully validate an artifact without loading its policy runtime."""
    artifact_root = Path(root).absolute()
    if not artifact_root.is_dir():
        raise ArtifactError(f"artifact root is not a directory: {artifact_root}")
    manifest_path = artifact_root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArtifactError(f"artifact manifest is missing or unsafe: {manifest_path}")
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot parse artifact manifest {manifest_path}: {error}") from error
    manifest = DeploymentManifest.from_dict(raw_manifest)
    for payload in manifest.payloads:
        relative = _safe_relative_path(payload.path, field_path="payloads.path")
        payload_file = _payload_path(artifact_root, relative)
        if not payload_file.is_file():
            raise ArtifactError(f"artifact payload is missing: {relative}")
        digest = hashlib.sha256(payload_file.read_bytes()).hexdigest()
        if digest != payload.sha256:
            raise ArtifactError(f"payload sha256 mismatch for {relative}: expected {payload.sha256}, got {digest}")
    relative = _safe_relative_path(manifest.policy.payload_path, field_path="policy.payload_path")
    policy_path = _payload_path(artifact_root, relative)
    if not policy_path.is_file():
        raise ArtifactError(f"artifact payload is missing: {relative}")
    digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if digest != manifest.policy.sha256:
        raise ArtifactError(f"policy.sha256 mismatch: expected {manifest.policy.sha256}, got {digest}")
    return Artifact(root=artifact_root, manifest=manifest)


def inspect_artifact(root: str | Path) -> dict[str, Any]:
    """Return a machine-readable static summary without opening a backend or policy runtime."""
    artifact = read_artifact(root)
    manifest = artifact.manifest
    return {
        "valid": True,
        "schema_version": manifest.schema_version,
        "source": manifest.source.to_dict(),
        "policy": {
            "component_version": manifest.policy.component_version,
            "input": _tensor_summary(
                manifest.policy.input.name,
                manifest.policy.input.shape,
                manifest.policy.input.dtype,
            ),
            "output": _tensor_summary(
                manifest.policy.output.name,
                manifest.policy.output.shape,
                manifest.policy.output.dtype,
            ),
            "sha256": manifest.policy.sha256,
        },
        "robot": {
            "base_link_name": manifest.robot.base_link_name,
            "joint_count": manifest.robot.joint_count,
            "joint_names": list(manifest.robot.joint_names),
        },
        "task": manifest.task.to_dict(),
        "observation_size": manifest.policy.input.shape[1],
        "action_size": manifest.policy.output.shape[1],
        "control_period_s": manifest.control.period_s,
    }


def _tensor_summary(name: str, shape: tuple[int, ...], dtype: str) -> dict[str, Any]:
    return {"name": name, "shape": list(shape), "dtype": dtype}
