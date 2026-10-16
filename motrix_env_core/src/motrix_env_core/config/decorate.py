# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import Field, InitVar, dataclass, field, is_dataclass
from typing import Any, ClassVar, TypeVar, get_origin, get_type_hints, overload

import numpy as np
from typing_extensions import dataclass_transform

_T = TypeVar("_T")


def _requires_copy(value: object) -> bool:
    if isinstance(value, (list, dict, set, np.ndarray)):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return True
    if isinstance(value, tuple):
        return any(_requires_copy(item) for item in value)
    return False


def _copy_factory(owner: str, name: str, template: object) -> Callable[[], object]:
    def factory() -> object:
        try:
            return deepcopy(template)
        except Exception as exc:
            raise TypeError(f"Failed to deepcopy default value for {owner}.{name}") from exc

    return factory


def _normalized_annotation_name(annotation: str) -> str:
    return annotation.replace("typing.", "").replace("dataclasses.", "").replace(" ", "")


def _is_class_var(annotation: object) -> bool:
    if get_origin(annotation) is ClassVar:
        return True
    if isinstance(annotation, str):
        name = _normalized_annotation_name(annotation)
        return name == "ClassVar" or name.startswith("ClassVar[")
    return False


def _is_init_var(annotation: object) -> bool:
    if annotation is InitVar or isinstance(annotation, InitVar):
        return True
    if isinstance(annotation, str):
        name = _normalized_annotation_name(annotation)
        return name == "InitVar" or name.startswith("InitVar[")
    return False


def _resolved_annotations(cls: type[Any], annotations: dict[str, object]) -> dict[str, object]:
    try:
        resolved = get_type_hints(cls, include_extras=True)
    except (NameError, TypeError):
        return annotations
    return {name: resolved.get(name, annotation) for name, annotation in annotations.items()}


def _wrap_configclass(cls: type[_T], **dataclass_options: Any) -> type[_T]:
    if "__dataclass_fields__" in cls.__dict__:
        raise TypeError(f"{cls.__qualname__} is already a dataclass; use either @configclass or @dataclass")

    annotations = dict(cls.__dict__.get("__annotations__", {}))
    resolved_annotations = _resolved_annotations(cls, annotations)

    for name, annotation in resolved_annotations.items():
        if name not in cls.__dict__ or _is_class_var(annotation) or _is_init_var(annotation):
            continue

        value = cls.__dict__[name]
        if isinstance(value, Field) or isinstance(value, type) or not _requires_copy(value):
            continue

        setattr(cls, name, field(default_factory=_copy_factory(cls.__qualname__, name, value)))

    return dataclass(cls, **dataclass_options)


@overload
def configclass(cls: type[_T], /) -> type[_T]: ...


@overload
def configclass(
    cls: None = None,
    /,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
) -> Callable[[type[_T]], type[_T]]: ...


@dataclass_transform(field_specifiers=(field,))
def configclass(
    cls: type[_T] | None = None,
    /,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
) -> type[_T] | Callable[[type[_T]], type[_T]]:
    """Create a dataclass whose declared mutable defaults are copied per instance."""
    options = {
        "init": init,
        "repr": repr,
        "eq": eq,
        "order": order,
        "unsafe_hash": unsafe_hash,
        "frozen": frozen,
        "match_args": match_args,
        "kw_only": kw_only,
        "slots": slots,
    }

    if cls is None:
        return lambda decorated_cls: _wrap_configclass(decorated_cls, **options)
    return _wrap_configclass(cls, **options)


__all__ = ["configclass"]
