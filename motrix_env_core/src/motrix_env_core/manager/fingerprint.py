# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Function dependency fingerprints shared by Numba compilation paths."""

import hashlib
import inspect
from collections.abc import Callable
from types import CodeType
from typing import Any

import numpy as np


def _stable_constant(value: Any) -> tuple[Any, ...]:
    if isinstance(value, CodeType):
        return ("code", _code_fingerprint_parts(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_stable_constant(item) for item in value))
    if isinstance(value, frozenset):
        items = sorted((_stable_constant(item) for item in value), key=repr)
        return ("frozenset", tuple(items))
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return (type(value).__name__, value)
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _code_fingerprint_parts(code: CodeType) -> tuple[Any, ...]:
    # marshal includes string interning/reference flags that can change when
    # Numba inspects the same code object. Hash immutable code fields instead.
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        tuple(_stable_constant(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_filename,
        code.co_name,
        code.co_firstlineno,
        code.co_lnotab,
    )


def function_fingerprint(function: Callable[..., Any]) -> str:
    """Hash function code, defaults, closures, and referenced global helpers.

    Dispatch entries and kernel-data methods use the same dependency traversal.
    Numba dispatchers are unwrapped to their Python functions. Code objects include
    constants and nested code, even when source is unavailable. Cycles are visited
    once; cheaply representable global/closure constants are hashed by value.
    Callers remain responsible for discovering member-method dependencies and
    incorporating their fingerprints into the appropriate compilation cache key.
    """
    hasher = hashlib.sha256()
    seen: set[int] = set()
    queue = [function]
    while queue:
        current = queue.pop()
        current = getattr(current, "py_func", current)
        if not inspect.isfunction(current) or id(current) in seen:
            continue
        seen.add(id(current))
        code = current.__code__
        hasher.update(f"{current.__module__}.{current.__qualname__}\0".encode())
        hasher.update(repr(_code_fingerprint_parts(code)).encode())
        references = [(name, current.__globals__[name]) for name in code.co_names if name in current.__globals__]
        references.extend((f"default[{i}]", value) for i, value in enumerate(current.__defaults__ or ()))
        references.extend(sorted((current.__kwdefaults__ or {}).items()))
        if current.__closure__:
            references.extend(zip(code.co_freevars, (cell.cell_contents for cell in current.__closure__)))
        for name, referenced in references:
            helper = getattr(referenced, "py_func", referenced)
            if inspect.isfunction(helper):
                queue.append(helper)
            elif isinstance(referenced, (int, float, bool, str, complex, np.generic, tuple, type(None))):
                hasher.update(f"{name}={referenced!r}\0".encode())
    return hasher.hexdigest()
