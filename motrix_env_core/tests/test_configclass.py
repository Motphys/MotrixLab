# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import InitVar, asdict, dataclass, field, fields, is_dataclass, replace
from typing import ClassVar

import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, ReadonlyConfigError

from motrix_env_core.config import configclass


@dataclass
class NestedCfg:
    values: list[int] = field(default_factory=lambda: [1, 2])


@configclass
class StructuredNestedCfg:
    value: int = 1


@configclass
class StructuredCfg:
    scalar: int = 2
    nested: StructuredNestedCfg = StructuredNestedCfg()
    items: list[int] = [3]
    mapping: dict[str, int] = {"left": 4}


@configclass(frozen=True)
class FrozenStructuredCfg:
    value: int = 1
    nested: StructuredNestedCfg = StructuredNestedCfg()


def test_mutable_and_nested_defaults_are_isolated():
    @configclass
    class DemoCfg:
        items: list[int] = [1]
        mapping: dict[str, list[int]] = {"value": [2]}
        labels: set[str] = {"demo"}
        nested: NestedCfg = NestedCfg()
        tuple_value: tuple[str, list[int]] = ("value", [3])

    first = DemoCfg()
    second = DemoCfg()

    first.items.append(4)
    first.mapping["value"].append(5)
    first.labels.add("first")
    first.nested.values.append(6)
    first.tuple_value[1].append(7)

    assert second.items == [1]
    assert second.mapping == {"value": [2]}
    assert second.labels == {"demo"}
    assert second.nested.values == [1, 2]
    assert second.tuple_value == ("value", [3])

    assert first.items is not second.items
    assert first.mapping["value"] is not second.mapping["value"]
    assert first.labels is not second.labels
    assert first.nested is not second.nested
    assert first.tuple_value[1] is not second.tuple_value[1]


def test_callable_dataclass_defaults_are_isolated():
    @configclass
    class CallableCfg:
        values: list[int] = [1]

        def __call__(self) -> int:
            return sum(self.values)

    template = CallableCfg()

    @configclass
    class BaseCfg:
        callback: CallableCfg = template

    @configclass
    class ChildCfg(BaseCfg):
        pass

    first = ChildCfg()
    second = ChildCfg()
    first.callback.values.append(2)

    assert first.callback() == 3
    assert second.callback() == 1
    assert template.values == [1]
    assert first.callback is not second.callback
    assert second.callback is not template


def test_function_and_class_defaults_keep_identity():
    def callback() -> int:
        return 1

    @configclass
    class DemoCfg:
        function: object = callback
        config_type: type = NestedCfg

    cfg = DemoCfg()

    assert cfg.function is callback
    assert cfg.config_type is NestedCfg


def test_ndarray_defaults_are_isolated():
    @configclass
    class DemoCfg:
        values: np.ndarray = np.array([1.0, 2.0], dtype=np.float32)
        tuple_value: tuple[str, np.ndarray] = ("value", np.array([3.0], dtype=np.float32))

    first = DemoCfg()
    second = DemoCfg()

    first.values[0] = 4.0
    first.tuple_value[1][0] = 5.0

    np.testing.assert_array_equal(second.values, [1.0, 2.0])
    np.testing.assert_array_equal(second.tuple_value[1], [3.0])
    assert first.values is not second.values
    assert first.tuple_value[1] is not second.tuple_value[1]


def test_scalar_classvar_initvar_and_explicit_field_keep_dataclass_semantics():
    @configclass
    class DemoCfg:
        shared: ClassVar[dict[str, int]] = {"value": 1}
        source: InitVar[int] = 2
        value: int = 3
        cache: dict[str, int] = field(default_factory=dict, repr=False)

        def __post_init__(self, source: int):
            self.value += source

    first = DemoCfg(source=4)
    second = DemoCfg()

    assert first.value == 7
    assert second.value == 5
    assert first.cache is not second.cache
    assert DemoCfg.shared == {"value": 1}
    assert [item.name for item in fields(DemoCfg)] == ["value", "cache"]


def test_forward_reference_failure_falls_back_to_raw_annotations():
    @configclass
    class DemoCfg:
        shared: "ClassVar[dict[str, int]]" = {"value": 1}
        unresolved: "UnavailableType" = []  # noqa: F821

    first = DemoCfg()
    second = DemoCfg()

    assert DemoCfg.shared is first.shared is second.shared
    assert first.unresolved is not second.unresolved


def test_inheritance_replace_fields_and_asdict():
    @dataclass
    class BaseCfg:
        name: str = "base"

    @configclass
    class ChildCfg(BaseCfg):
        nested: NestedCfg = NestedCfg(values=[3])

    @configclass
    class OverrideCfg(ChildCfg):
        nested: NestedCfg = NestedCfg(values=[4])

    cfg = OverrideCfg(name="demo")
    replaced = replace(cfg, name="other")

    assert is_dataclass(OverrideCfg)
    assert [item.name for item in fields(OverrideCfg)] == ["name", "nested"]
    assert asdict(cfg) == {"name": "demo", "nested": {"values": [4]}}
    assert replaced == OverrideCfg(name="other")
    assert replaced.nested is cfg.nested


def test_dataclass_options_are_forwarded():
    @configclass(frozen=True, kw_only=True, slots=True, order=True)
    class DemoCfg:
        value: int = 1
        items: list[int] = [2]

    first = DemoCfg(value=2)
    second = DemoCfg(value=3)

    assert first < second
    assert not hasattr(first, "__dict__")
    assert first.items is not second.items
    with pytest.raises(AttributeError):
        first.value = 4


def test_deepcopy_error_contains_field_context():
    @dataclass
    class BadDefault:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy")

    @configclass
    class DemoCfg:
        bad: BadDefault = BadDefault()

    with pytest.raises(TypeError, match=r"DemoCfg\.bad"):
        DemoCfg()


def test_already_decorated_dataclass_is_rejected():
    with pytest.raises(TypeError, match="already a dataclass"):

        @configclass
        @dataclass
        class DemoCfg:
            value: int = 1


def test_omegaconf_structured_accepts_configclass_type_and_instance():
    from_type = OmegaConf.structured(StructuredCfg)
    from_instance = OmegaConf.structured(
        StructuredCfg(
            scalar=5,
            nested=StructuredNestedCfg(value=6),
            items=[7],
            mapping={"right": 8},
        )
    )

    typed_from_type = OmegaConf.to_object(from_type)
    typed_from_instance = OmegaConf.to_object(from_instance)

    assert isinstance(typed_from_type, StructuredCfg)
    assert isinstance(typed_from_type.nested, StructuredNestedCfg)
    assert typed_from_type == StructuredCfg()
    assert isinstance(typed_from_instance, StructuredCfg)
    assert isinstance(typed_from_instance.nested, StructuredNestedCfg)
    assert typed_from_instance == StructuredCfg(
        scalar=5,
        nested=StructuredNestedCfg(value=6),
        items=[7],
        mapping={"right": 8},
    )


def test_configstore_compose_supports_scalar_nested_list_and_dict_overrides():
    schema_name = "configclass_test_schema"
    ConfigStore.instance().store(name=schema_name, node=StructuredCfg)

    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name=schema_name,
            overrides=[
                "scalar=10",
                "nested.value=11",
                "items=[12,13]",
                "mapping.left=14",
            ],
        )

    typed_cfg = OmegaConf.to_object(cfg)

    assert isinstance(typed_cfg, StructuredCfg)
    assert isinstance(typed_cfg.nested, StructuredNestedCfg)
    assert typed_cfg.scalar == 10
    assert typed_cfg.nested.value == 11
    assert typed_cfg.items == [12, 13]
    assert typed_cfg.mapping == {"left": 14}


def test_frozen_configclass_produces_readonly_structured_config():
    cfg_from_type = OmegaConf.structured(FrozenStructuredCfg)
    cfg_from_instance = OmegaConf.structured(FrozenStructuredCfg(value=2))

    assert OmegaConf.is_readonly(cfg_from_type)
    assert OmegaConf.is_readonly(cfg_from_instance)

    with pytest.raises(ReadonlyConfigError):
        cfg_from_type.value = 3
    with pytest.raises(ReadonlyConfigError):
        cfg_from_instance.nested.value = 4
