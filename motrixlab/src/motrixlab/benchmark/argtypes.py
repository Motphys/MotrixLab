# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable argparse value validators for benchmarks."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def constrained_type(converter: Callable[[str], T], predicate: Callable[[T], bool], message: str) -> Callable[[str], T]:
    def parse(value: str) -> T:
        try:
            parsed = converter(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid value: {value!r}") from error
        if not predicate(parsed):
            raise argparse.ArgumentTypeError(message)
        return parsed

    return parse


positive_int = constrained_type(int, lambda value: value > 0, "must be positive")
nonnegative_int = constrained_type(int, lambda value: value >= 0, "must be non-negative")
positive_float = constrained_type(
    float,
    lambda value: math.isfinite(value) and value > 0,
    "must be positive and finite",
)
