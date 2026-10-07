# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Schema v1 for the deployment manifest."""

import re
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from motrix_deploy.contracts import JointControlMode, RobotSpec, TensorSpec
from motrix_deploy.errors import ValidationError

SCHEMA_VERSION = "motrix-deploy/v1"
POLICY_COMPONENT_VERSION = "onnx/v1"
_TASK_NAME_PATTERN = re.compile(r"[^/\s]+/v[1-9][0-9]*")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(path, "an object", type(value).__name__)
    return value


def _keys(value: Mapping[str, Any], *, path: str, required: set[str]) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise ValidationError(path, f"required fields {sorted(required)}", f"missing {missing}")
    unknown = sorted(value.keys() - required)
    if unknown:
        raise ValidationError(path, f"only fields {sorted(required)}", f"unknown {unknown}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(path, "a non-empty string", value)
    return value


def _validate_positive_number(value: object, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValidationError(path, "a positive finite number", value)


def _tensor_to_dict(value: TensorSpec) -> dict[str, Any]:
    return {"name": value.name, "shape": list(value.shape), "dtype": value.dtype}


def _tensor_from_dict(value: object, path: str) -> TensorSpec:
    data = _mapping(value, path)
    _keys(data, path=path, required={"name", "shape", "dtype"})
    shape = data["shape"]
    if not isinstance(shape, list):
        raise ValidationError(f"{path}.shape", "an array of positive integers", shape)
    try:
        return TensorSpec(name=data["name"], shape=tuple(shape), dtype=data["dtype"])
    except ValidationError as error:
        suffix = error.path.removeprefix("tensor")
        raise ValidationError(f"{path}{suffix}", error.expected, error.actual) from error


@dataclass(frozen=True)
class SourceSpec:
    """Traceability information for the training source."""

    framework: str
    run_id: str
    checkpoint: str

    def __post_init__(self) -> None:
        for field_name in ("framework", "run_id", "checkpoint"):
            _string(getattr(self, field_name), f"source.{field_name}")

    def to_dict(self) -> dict[str, Any]:
        return {"framework": self.framework, "run_id": self.run_id, "checkpoint": self.checkpoint}

    @classmethod
    def from_dict(cls, value: object) -> "SourceSpec":
        data = _mapping(value, "source")
        _keys(data, path="source", required={"framework", "run_id", "checkpoint"})
        return cls(framework=data["framework"], run_id=data["run_id"], checkpoint=data["checkpoint"])


@dataclass(frozen=True)
class PolicySpec:
    """One self-contained deterministic policy payload."""

    component_version: str
    payload_path: str
    sha256: str
    input: TensorSpec
    output: TensorSpec

    def __post_init__(self) -> None:
        if self.component_version != POLICY_COMPONENT_VERSION:
            raise ValidationError("policy.component_version", POLICY_COMPONENT_VERSION, self.component_version)
        _string(self.payload_path, "policy.payload_path")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValidationError("policy.sha256", "64 lowercase hexadecimal characters", self.sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_version": self.component_version,
            "payload_path": self.payload_path,
            "sha256": self.sha256,
            "input": _tensor_to_dict(self.input),
            "output": _tensor_to_dict(self.output),
        }

    @classmethod
    def from_dict(cls, value: object) -> "PolicySpec":
        data = _mapping(value, "policy")
        required = {"component_version", "payload_path", "sha256", "input", "output"}
        _keys(data, path="policy", required=required)
        return cls(
            component_version=data["component_version"],
            payload_path=data["payload_path"],
            sha256=data["sha256"],
            input=_tensor_from_dict(data["input"], "policy.input"),
            output=_tensor_from_dict(data["output"], "policy.output"),
        )


def _robot_to_dict(value: RobotSpec) -> dict[str, Any]:
    return {
        "base_link_name": value.base_link_name,
        "joint_names": list(value.joint_names),
        "default_joint_position": value.default_joint_position.tolist(),
        "position_lower": value.position_lower.tolist(),
        "position_upper": value.position_upper.tolist(),
        "torque_limit": value.torque_limit.tolist(),
    }


def _robot_from_dict(value: object) -> RobotSpec:
    data = _mapping(value, "robot")
    required = {
        "base_link_name",
        "joint_names",
        "default_joint_position",
        "position_lower",
        "position_upper",
        "torque_limit",
    }
    _keys(data, path="robot", required=required)
    joint_names = data["joint_names"]
    if not isinstance(joint_names, list):
        raise ValidationError("robot.joint_names", "an array of strings", joint_names)
    try:
        return RobotSpec(
            base_link_name=data["base_link_name"],
            joint_names=tuple(joint_names),
            default_joint_position=np.asarray(data["default_joint_position"], dtype=np.float32),
            position_lower=np.asarray(data["position_lower"], dtype=np.float32),
            position_upper=np.asarray(data["position_upper"], dtype=np.float32),
            torque_limit=np.asarray(data["torque_limit"], dtype=np.float32),
        )
    except ValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise ValidationError("robot", "numeric arrays matching joint_names", str(error)) from error


def _task_validation_error(error: PydanticValidationError) -> ValidationError:
    """Translate model diagnostics into the deployment artifact field namespace."""
    detail = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in detail["loc"])
    path = f"task.config.{location}" if location else "task.config"
    if detail["type"] == "missing":
        return ValidationError(
            path, "a required task field (re-export artifacts using an older task schema)", "missing"
        )
    return ValidationError(path, detail["msg"], detail.get("input"))


class TaskSpec(BaseModel, ABC):
    """Typed task configuration, resolved through an installed versioned task class.

    Concrete implementations declare ``task_name`` as a ClassVar and use
    Pydantic fields and validators for configuration semantics. YAML composition
    and interpolation resolution belong to the caller, not the artifact codec.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False, validate_default=True)
    task_name: ClassVar[str]

    @property
    def name(self) -> str:
        return self.task_name

    def to_dict(self) -> dict[str, Any]:
        # Frozen models may still contain mutable lists. Revalidate a fresh
        # mapping so mutations cannot bypass the artifact serialization boundary.
        try:
            spec = type(self).model_validate(self.model_dump())
        except PydanticValidationError as error:
            raise _task_validation_error(error) from error
        return {"name": self.name, "config": spec.model_dump(mode="json")}

    @classmethod
    def from_dict(cls, value: object) -> "TaskSpec":
        data = _mapping(value, "task")
        _keys(data, path="task", required={"name", "config"})
        name = data["name"]
        if not isinstance(name, str) or _TASK_NAME_PATTERN.fullmatch(name) is None:
            raise ValidationError("task.name", "a versioned task identifier such as go2_walk/v1", name)
        config = _mapping(data["config"], "task.config")
        # Lazy import: the runtime contract itself depends on TaskSpec.
        from motrix_deploy.task import load_task_type

        try:
            spec_type = getattr(load_task_type(name), "spec_type", None)
        except Exception as error:
            raise ValidationError("task.name", "an available installed task class plugin", str(error)) from error
        if (
            not isinstance(spec_type, type)
            or not issubclass(spec_type, TaskSpec)
            or getattr(spec_type, "task_name", None) != name
        ):
            raise ValidationError("task.name", "a TaskSpec BaseModel with matching task_name", name)
        try:
            return spec_type.model_validate(config)
        except PydanticValidationError as error:
            raise _task_validation_error(error) from error


@dataclass(frozen=True)
class ControlSpec:
    """Timing and coordinate conventions shared by runtime and backend."""

    period_s: float
    state_timeout_s: float
    quaternion_order: str = "xyzw"
    base_orientation: str = "body_to_world"
    angular_velocity_frame: str = "body"
    mode: JointControlMode = JointControlMode.SERVO

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "mode", JointControlMode(self.mode))
        except (TypeError, ValueError) as error:
            raise ValidationError("control.mode", "joint_servo or joint_torque", self.mode) from error
        _validate_positive_number(self.period_s, "control.period_s")
        _validate_positive_number(self.state_timeout_s, "control.state_timeout_s")
        if self.quaternion_order != "xyzw":
            raise ValidationError("control.quaternion_order", "xyzw", self.quaternion_order)
        if self.base_orientation != "body_to_world":
            raise ValidationError("control.base_orientation", "body_to_world", self.base_orientation)
        if self.angular_velocity_frame != "body":
            raise ValidationError("control.angular_velocity_frame", "body", self.angular_velocity_frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "period_s": self.period_s,
            "state_timeout_s": self.state_timeout_s,
            "quaternion_order": self.quaternion_order,
            "base_orientation": self.base_orientation,
            "angular_velocity_frame": self.angular_velocity_frame,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ControlSpec":
        data = _mapping(value, "control")
        required = {
            "mode",
            "period_s",
            "state_timeout_s",
            "quaternion_order",
            "base_orientation",
            "angular_velocity_frame",
        }
        _keys(data, path="control", required=required)
        return cls(**data)


@dataclass(frozen=True)
class DeploymentManifest:
    """Top-level deployment manifest with strict v1 component versions."""

    schema_version: str
    source: SourceSpec
    policy: PolicySpec
    robot: RobotSpec
    task: TaskSpec
    control: ControlSpec

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError("schema_version", SCHEMA_VERSION, self.schema_version)
        for field_name in ("input", "output"):
            shape = getattr(self.policy, field_name).shape
            if len(shape) != 2 or shape[0] != 1:
                raise ValidationError(f"policy.{field_name}.shape", "(1, size): a single batch-1 vector", shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "policy": self.policy.to_dict(),
            "robot": _robot_to_dict(self.robot),
            "task": self.task.to_dict(),
            "control": self.control.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "DeploymentManifest":
        data = _mapping(value, "manifest")
        required = {"schema_version", "source", "policy", "robot", "task", "control"}
        _keys(data, path="manifest", required=required)
        return cls(
            schema_version=data["schema_version"],
            source=SourceSpec.from_dict(data["source"]),
            policy=PolicySpec.from_dict(data["policy"]),
            robot=_robot_from_dict(data["robot"]),
            task=TaskSpec.from_dict(data["task"]),
            control=ControlSpec.from_dict(data["control"]),
        )
