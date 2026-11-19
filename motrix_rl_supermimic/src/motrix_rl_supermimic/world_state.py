# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned flat state capture and per-term dynamics representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext, ManagerContext


class StateRepresentation(ABC):
    """Encode one flat state term while preserving its leading batch dimensions."""

    size: int

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def prediction_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def features(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def increments(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def integrate(
        self, state: torch.Tensor, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class EuclideanState(StateRepresentation):
    """Additive dynamics for task state with no robot-specific geometry."""

    size: int
    error_scale: float = 1.0

    @property
    def feature_dim(self) -> int:
        return self.size

    @property
    def prediction_dim(self) -> int:
        return self.size

    def features(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def increments(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return next_state - state

    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return self.increments(state, next_state)

    def integrate(
        self, state: torch.Tensor, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return state + prediction

    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return pred - truth

    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        return torch.full((self.size,), self.error_scale, device=device)


@dataclass(frozen=True)
class WorldStateTerm:
    function: Callable[[ManagerContext, torch.Tensor], None]
    representation: StateRepresentation

    @property
    def size(self) -> int:
        return self.representation.size


@configclass
class WorldStateTermCfg(ABC):
    @abstractmethod
    def __call__(self, ctx: BuildContext) -> WorldStateTerm:
        raise NotImplementedError


class WorldState:
    """Ordered state capture whose terms own their numerical representations.

    State slices address captured tensors; prediction slices may have different
    widths, such as rotation vectors replacing quaternions. All numerical methods
    concatenate only the trailing feature dimension and retain arbitrary batches.
    """

    def __init__(self, terms: Mapping[str, WorldStateTerm]) -> None:
        if not terms:
            raise ValueError("World state requires at least one term")
        self.terms = dict(terms)
        self.slices: dict[str, slice] = {}
        self._prediction_slices: dict[str, slice] = {}
        self.size = 0
        self.feature_dim = 0
        self.prediction_dim = 0
        for name, term in self.terms.items():
            representation = term.representation
            if min(representation.size, representation.feature_dim, representation.prediction_dim) <= 0:
                raise ValueError(f"World state term {name!r} requires positive representation widths")
            self.slices[name] = slice(self.size, self.size + term.size)
            self._prediction_slices[name] = slice(
                self.prediction_dim, self.prediction_dim + representation.prediction_dim
            )
            self.size += term.size
            self.feature_dim += representation.feature_dim
            self.prediction_dim += representation.prediction_dim

    def capture(self, ctx: ManagerContext, out: torch.Tensor) -> None:
        for name, term in self.terms.items():
            term.function(ctx, out[..., self.slices[name]])

    def features(self, state: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [term.representation.features(state[..., self.slices[name]]) for name, term in self.terms.items()],
            -1,
        )

    def increments(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                term.representation.increments(state[..., self.slices[name]], next_state[..., self.slices[name]])
                for name, term in self.terms.items()
            ],
            -1,
        )

    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return torch.cat(
            [
                term.representation.targets(
                    state[..., self.slices[name]],
                    next_state[..., self.slices[name]],
                    structured=structured,
                    dt=dt,
                )
                for name, term in self.terms.items()
            ],
            -1,
        )

    def integrate(
        self, state: torch.Tensor, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return torch.cat(
            [
                term.representation.integrate(
                    state[..., self.slices[name]],
                    prediction[..., self._prediction_slices[name]],
                    structured=structured,
                    dt=dt,
                )
                for name, term in self.terms.items()
            ],
            -1,
        )

    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                term.representation.error(pred[..., self.slices[name]], truth[..., self.slices[name]])
                for name, term in self.terms.items()
            ],
            -1,
        )

    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        return torch.cat([term.representation.error_floor(device) for term in self.terms.values()], -1)
