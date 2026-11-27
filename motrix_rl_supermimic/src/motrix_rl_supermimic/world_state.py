# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned flat state capture and per-term dynamics representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext


class StateStatistics(NamedTuple):
    """World-model normalization samples for one observed transition."""

    features: torch.Tensor
    targets: torch.Tensor
    delta: torch.Tensor


class WorldStateTerm(ABC):
    """One named state slice: its capture dispatch and numerical semantics.

    :meth:`capture` fills the term's trailing slice of ``out`` from simulator
    reads alone; the remaining methods define the slice's dynamics encoding
    while preserving arbitrary leading batch dimensions.

    Three widths describe one term and may differ: ``size`` is the term's
    width inside a captured state tensor, ``feature_dim`` the width of its
    world-model input encoding, and ``prediction_dim`` the width of its
    predicted delta (for example rotation vectors instead of quaternions).
    ``delta``, ``targets`` output, and ``error_floor`` share one width so
    the world-model loss can normalize residuals by running statistics clamped
    to the floor.
    """

    size: int

    @abstractmethod
    def capture(self, sim: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
        """Fill this term's trailing slice of ``out`` from simulator reads."""
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Width of the decoded state features fed to the world model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def prediction_dim(self) -> int:
        """Width of one predicted increment; may differ from ``size``."""
        raise NotImplementedError

    @abstractmethod
    def features(self, state: torch.Tensor) -> torch.Tensor:
        """Encode one flat state slice as world-model input features."""
        raise NotImplementedError

    @abstractmethod
    def integrate(
        self, state: torch.Tensor, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        """Apply predicted delta to one flat state slice."""
        raise NotImplementedError

    @abstractmethod
    def delta(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Encode the relative change between flat state slices."""
        raise NotImplementedError

    @abstractmethod
    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        """Return the predictor target for two flat state slices."""
        raise NotImplementedError

    def statistics(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> StateStatistics:
        return StateStatistics(
            self.features(state),
            self.targets(state, next_state, structured=structured, dt=dt),
            self.delta(state, next_state),
        )

    @abstractmethod
    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Return prediction-space residuals between flat state slices."""
        raise NotImplementedError

    @abstractmethod
    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        """Return a strictly positive per-dimension loss-normalization floor.

        Prevents near-zero increment variance from amplifying the loss on
        barely-moving dimensions. Width matches :meth:`error` output.
        """
        raise NotImplementedError

    def views(self, state: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Present one flat state slice as simulator views for imagination.

        The default empty mapping serves tasks whose terms read the canonical
        snapshot (``ReferenceView.physical_state``); tasks whose numerical
        terms read ``ctx.sim`` keys override this with the inverse of
        :meth:`capture`.
        """
        return {}


@dataclass(frozen=True)
class EuclideanState(WorldStateTerm):
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

    def integrate(
        self, state: torch.Tensor, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return state + prediction

    def delta(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return next_state - state

    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return self.delta(state, next_state)

    def statistics(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> StateStatistics:
        delta = self.delta(state, next_state)
        return StateStatistics(self.features(state), delta, delta)

    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return pred - truth

    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        return torch.full((self.size,), self.error_scale, device=device)


@configclass
class WorldStateTermCfg(ABC):
    @abstractmethod
    def __call__(self, ctx: BuildContext) -> WorldStateTerm:
        raise NotImplementedError


class WorldStateManager:
    """Ordered state capture whose terms own their numerical representations.

    State slices address captured tensors; prediction slices may have different
    widths, such as rotation vectors replacing quaternions. All numerical methods
    concatenate only the trailing feature dimension and retain arbitrary batches.

    Terms receive only their own flat tensor slice. Numerical methods return
    tensors; the manager composes those results without shared decoded state.
    """

    def __init__(self, terms: Mapping[str, WorldStateTerm]) -> None:
        if not terms:
            raise ValueError("World-state manager requires at least one term")
        self.terms = dict(terms)
        self.slices: dict[str, slice] = {}
        self._prediction_slices: dict[str, slice] = {}
        self.size = 0
        self.feature_dim = 0
        self.prediction_dim = 0
        for name, term in self.terms.items():
            if min(term.size, term.feature_dim, term.prediction_dim) <= 0:
                raise ValueError(f"World state term {name!r} requires positive representation widths")
            self.slices[name] = slice(self.size, self.size + term.size)
            self._prediction_slices[name] = slice(self.prediction_dim, self.prediction_dim + term.prediction_dim)
            self.size += term.size
            self.feature_dim += term.feature_dim
            self.prediction_dim += term.prediction_dim

    def capture(self, sim: Mapping[str, torch.Tensor], out: torch.Tensor) -> None:
        for name, term in self.terms.items():
            term.capture(sim, out[..., self.slices[name]])

    def views(self, state: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Merge the per-term simulator views of one flat state (capture's inverse).

        Each term owns a disjoint set of sim keys; a duplicate key means two
        terms claim the same view over different state slices, which capture
        never produced — reject it instead of letting the later term win.
        """
        merged: dict[str, torch.Tensor] = {}
        for name, term in self.terms.items():
            for key, view in term.views(state[..., self.slices[name]]).items():
                if key in merged:
                    raise ValueError(f"World-state terms declare duplicate sim view {key!r}.")
                merged[key] = view
        return merged

    def features(self, state: torch.Tensor) -> torch.Tensor:
        return torch.cat([term.features(state[..., self.slices[name]]) for name, term in self.terms.items()], -1)

    def integrate(
        self, state: torch.Tensor, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return torch.cat(
            [
                term.integrate(state[..., self.slices[name]], prediction[..., part], structured=structured, dt=dt)
                for (name, term), part in zip(self.terms.items(), self._prediction_slices.values())
            ],
            -1,
        )

    def statistics(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> StateStatistics:
        samples = [
            term.statistics(
                state[..., self.slices[name]], next_state[..., self.slices[name]], structured=structured, dt=dt
            )
            for name, term in self.terms.items()
        ]
        return StateStatistics(
            torch.cat([sample.features for sample in samples], -1),
            torch.cat([sample.targets for sample in samples], -1),
            torch.cat([sample.delta for sample in samples], -1),
        )

    def delta(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                term.delta(state[..., self.slices[name]], next_state[..., self.slices[name]])
                for name, term in self.terms.items()
            ],
            -1,
        )

    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return torch.cat(
            [
                term.targets(
                    state[..., self.slices[name]], next_state[..., self.slices[name]], structured=structured, dt=dt
                )
                for name, term in self.terms.items()
            ],
            -1,
        )

    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                term.error(pred[..., self.slices[name]], truth[..., self.slices[name]])
                for name, term in self.terms.items()
            ],
            -1,
        )

    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        return torch.cat([term.error_floor(device) for term in self.terms.values()], -1)
