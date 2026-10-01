# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Function dependency fingerprints shared by Numba compilation paths."""

import hashlib
import inspect
import marshal
from collections.abc import Callable
from typing import Any

import numpy as np


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
        hasher.update(marshal.dumps(code))
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
