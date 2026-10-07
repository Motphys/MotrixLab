# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed task configuration codec and installed-plugin boundary."""

import json
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as PydanticValidationError
from task_specs import TestDeployTask as FixtureDeployTask

from motrix_deploy import task as task_module
from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.errors import ValidationError


class NumericSpec(TaskSpec):
    task_name: ClassVar[str] = "numeric/v1"
    gain: float
    values: list[float]
    bounds: list[list[float]]
    threshold: float | None = None
    enabled: bool = False
    count: int = 1
    label: str = "literal"

    @field_validator("gain")
    @classmethod
    def positive_gain(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("gain must be positive")
        return value


class NumericTask(FixtureDeployTask):
    spec_type = NumericSpec


@pytest.fixture
def codec_entries(monkeypatch):
    selected = SimpleNamespace(name="numeric/v1", load=Mock(return_value=NumericTask))
    other = SimpleNamespace(name="other/v1", load=Mock())
    entries = [selected, other]

    def entry_points(*, group):
        assert group == task_module.TASK_ENTRY_POINT_GROUP
        return entries

    monkeypatch.setattr(task_module.metadata, "entry_points", entry_points)
    return entries


@pytest.fixture
def encoded():
    return NumericSpec(gain=2.0, values=[1.0, 2.0], bounds=[[0.0, 1.0]]).to_dict()


def test_roundtrip_frozen_concrete_spec(codec_entries, encoded):
    parsed = TaskSpec.from_dict(encoded)
    assert type(parsed) is NumericSpec
    assert parsed.to_dict() == encoded
    codec_entries[0].load.assert_called_once_with()
    codec_entries[1].load.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value", "path"),
    [
        ("gain", True, "gain"),
        ("gain", "2.0", "gain"),
        ("gain", "${oc.env:HOME}", "gain"),
        ("values", [True], "values.0"),
        ("values", ["1.0"], "values.0"),
        ("values", (1.0, 2.0), "values"),
        ("values", "1,2", "values"),
        ("bounds", [[False]], "bounds.0.0"),
        ("bounds", [["${gain}"]], "bounds.0.0"),
        ("threshold", True, "threshold"),
        ("enabled", 1, "enabled"),
        ("count", True, "count"),
        ("count", "1", "count"),
        ("enabled", "true", "enabled"),
        ("label", 1, "label"),
        ("gain", float("nan"), "gain"),
        ("gain", float("inf"), "gain"),
        ("values", [float("-inf")], "values.0"),
    ],
)
def test_invalid_types_are_rejected_before_coercion(codec_entries, encoded, field, value, path):
    encoded["config"][field] = value
    with pytest.raises(ValidationError) as exc:
        TaskSpec.from_dict(encoded)
    assert exc.value.path == f"task.config.{path}"


def test_numeric_integers_and_optional_float(codec_entries, encoded):
    encoded["config"].update(gain=2, threshold=3)
    parsed = TaskSpec.from_dict(encoded)
    assert parsed.gain == 2.0
    assert parsed.threshold == 3.0


@pytest.mark.parametrize("kind", ["missing", "unknown", "semantic"])
def test_config_validation_paths(codec_entries, encoded, kind):
    if kind == "missing":
        del encoded["config"]["gain"]
    elif kind == "unknown":
        encoded["config"]["extra"] = 1
    else:
        encoded["config"]["gain"] = -1
    with pytest.raises(ValidationError) as exc:
        TaskSpec.from_dict(encoded)
    assert exc.value.path == ("task.config.extra" if kind == "unknown" else "task.config.gain")
    if kind == "missing":
        assert "required task field" in str(exc.value)
        assert "re-export" in str(exc.value)


@pytest.mark.parametrize("config", [None, [], "not an object"])
def test_config_must_be_mapping(codec_entries, encoded, config):
    encoded["config"] = config
    with pytest.raises(ValidationError) as exc:
        TaskSpec.from_dict(encoded)
    assert exc.value.path == "task.config"


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "unavailable", "invalid", "missing_spec", "invalid_spec", "wrong_name"]
)
def test_spec_plugin_failures(codec_entries, encoded, kind):
    selected = codec_entries[0]
    if kind == "missing":
        codec_entries.remove(selected)
    elif kind == "duplicate":
        codec_entries.append(SimpleNamespace(name="numeric/v1", load=Mock()))
    elif kind == "unavailable":
        selected.load.side_effect = ImportError("plugin dependency unavailable")
    elif kind == "invalid":
        selected.load.return_value = dict
    elif kind == "missing_spec":
        selected.load.return_value = type("MissingSpecTask", (NumericTask,), {"spec_type": None})
    elif kind == "invalid_spec":
        selected.load.return_value = type("InvalidSpecTask", (NumericTask,), {"spec_type": dict})
    else:
        wrong_spec = type("WrongName", (NumericSpec,), {"task_name": "wrong/v1"})
        selected.load.return_value = type("WrongTask", (NumericTask,), {"spec_type": wrong_spec})
    with pytest.raises(ValidationError) as exc:
        TaskSpec.from_dict(encoded)
    assert exc.value.path == "task.name"
    if kind == "missing":
        codec_entries[0].load.assert_not_called()
    if kind == "duplicate":
        selected.load.assert_not_called()


def test_artifact_cannot_select_arbitrary_import(codec_entries, encoded):
    encoded["name"] = "os:system"
    with pytest.raises(ValidationError) as exc:
        TaskSpec.from_dict(encoded)
    assert exc.value.path == "task.name"
    for entry in codec_entries:
        entry.load.assert_not_called()


def test_taskspec_model_contract():
    spec = NumericSpec(gain=2.0, values=[1.0], bounds=[[0.0, 1.0]])
    assert isinstance(spec, BaseModel)
    assert spec.name == "numeric/v1"
    assert "name" not in spec.model_dump()
    assert "task_name" not in spec.model_dump()
    with pytest.raises(PydanticValidationError) as exc:
        spec.gain = 3.0
    assert exc.value.errors()[0]["type"] == "frozen_instance"


@pytest.mark.parametrize("mutation", ["nested_type", "nested_nonfinite", "semantic"])
def test_serialization_revalidates_mutable_values(mutation):
    spec = NumericSpec(gain=2.0, values=[1.0], bounds=[[0.0, 1.0]])
    if mutation == "nested_type":
        spec.bounds[0][0] = True
        path = "task.config.bounds.0.0"
    elif mutation == "nested_nonfinite":
        spec.values.append(float("nan"))
        path = "task.config.values.1"
    else:
        # model_construct is an explicit escape hatch; serialization still
        # establishes the public artifact contract for such instances.
        spec = NumericSpec.model_construct(gain=-1.0, values=[], bounds=[])
        path = "task.config.gain"
    with pytest.raises(ValidationError) as exc:
        spec.to_dict()
    assert exc.value.path == path


def test_invalid_plugin_defaults_are_validated(codec_entries, encoded):
    class InvalidDefaultSpec(NumericSpec):
        tolerance: float = Field(default=-1.0, gt=0)

    codec_entries[0].load.return_value = type("InvalidDefaultTask", (NumericTask,), {"spec_type": InvalidDefaultSpec})
    with pytest.raises(ValidationError) as exc:
        TaskSpec.from_dict(encoded)
    assert exc.value.path == "task.config.tolerance"


def test_serialization_uses_json_primitives(codec_entries):
    spec = NumericSpec(gain=2, values=[1, 2], bounds=[[0, 1]], enabled=True)
    wire = spec.to_dict()
    json_wire = json.loads(json.dumps(wire, allow_nan=False))
    assert json_wire == wire
    assert json_wire["config"]["bounds"] == [[0.0, 1.0]]
    assert TaskSpec.from_dict(json_wire) == spec


def test_literal_interpolation_string_is_not_resolved(codec_entries, encoded):
    encoded["config"]["label"] = "${oc.env:HOME}"
    parsed = TaskSpec.from_dict(encoded)
    assert parsed.label == "${oc.env:HOME}"
    assert parsed.to_dict() == encoded


def test_yaml_composition_resolves_before_model_validation(tmp_path):
    source = tmp_path / "task.yaml"
    source.write_text(
        "gain: 2.0\nvalues: [1.0, 2.0]\nbounds: [[0.0, 1.0]]\nthreshold: ${gain}\n",
        encoding="utf-8",
    )
    composed = OmegaConf.merge(OmegaConf.load(source), OmegaConf.create({"gain": 3.0, "enabled": True}))
    config = OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)
    spec = NumericSpec.model_validate(config)
    assert spec.gain == 3.0
    assert spec.threshold == spec.gain
    assert spec.enabled is True
    assert spec.to_dict()["config"] == spec.model_dump()
