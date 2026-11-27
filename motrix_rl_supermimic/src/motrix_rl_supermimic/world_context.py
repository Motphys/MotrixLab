# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned named context capture and affine world-model normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext, ManagerContext


@dataclass(frozen=True)
class WorldContextTerm:
    size: int
    function: Callable[[ManagerContext, torch.Tensor], None]
    center: tuple[float, ...]
    scale: tuple[float, ...]


@configclass
class WorldContextTermCfg(ABC):
    @abstractmethod
    def __call__(self, ctx: BuildContext) -> WorldContextTerm:
        raise NotImplementedError


class WorldContext:
    """Ordered context terms with fixed affine normalization, including no terms.

    Normalization tensors are built on Torch's current default device in float32,
    alongside the task that owns this context. Capture dispatches into named
    trailing-dimension slices without changing leading batch dimensions.
    """

    def __init__(self, terms: Mapping[str, WorldContextTerm]) -> None:
        self.terms = dict(terms)
        self.slices: dict[str, slice] = {}
        self.size = 0
        centers: list[float] = []
        scales: list[float] = []
        for name, term in self.terms.items():
            if term.size <= 0:
                raise ValueError(f"World context term {name!r} requires a positive size")
            if len(term.center) != term.size or len(term.scale) != term.size:
                raise ValueError(f"World context term {name!r} center and scale lengths must equal its size")
            if not all(isfinite(value) for value in term.center):
                raise ValueError(f"World context term {name!r} requires finite center values")
            if not all(isfinite(value) and value > 0 for value in term.scale):
                raise ValueError(f"World context term {name!r} requires finite positive scale values")
            self.slices[name] = slice(self.size, self.size + term.size)
            self.size += term.size
            centers.extend(term.center)
            scales.extend(term.scale)
        self.center = torch.tensor(centers, dtype=torch.float32)
        self.scale = torch.tensor(scales, dtype=torch.float32)

    def capture(self, ctx: ManagerContext, out: torch.Tensor) -> None:
        for name, term in self.terms.items():
            term.function(ctx, out[..., self.slices[name]])

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.center) / self.scale
