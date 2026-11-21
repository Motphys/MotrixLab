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
from motrix_env_core.manager.kernel_data import Map


class StateComputation(ABC):
    """Numerical evaluation bound to borrowed state tensors for one step."""

    @abstractmethod
    def features(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02) -> torch.Tensor:
        raise NotImplementedError


class StateStatistics(NamedTuple):
    """World-model normalization samples for one observed transition."""

    features: torch.Tensor
    targets: torch.Tensor
    increments: torch.Tensor


class WorldStateTerm(ABC):
    """One named state slice: its capture dispatch and numerical semantics.

    :meth:`capture` fills the term's trailing slice of ``out`` from simulator
    reads alone; the remaining methods define the slice's dynamics encoding
    while preserving arbitrary leading batch dimensions.

    Three widths describe one term and may differ: ``size`` is the term's
    width inside a captured state tensor, ``feature_dim`` the width of its
    world-model input encoding, and ``prediction_dim`` the width of its
    predicted increments (for example rotation vectors instead of quaternions).
    ``increments``, ``targets`` output, and ``error_floor`` share one width so
    the world-model loss can normalize residuals by running statistics clamped
    to the floor.
    """

    size: int

    @abstractmethod
    def capture(self, sim: Map[torch.Tensor], out: torch.Tensor) -> None:
        """Fill this term's trailing slice of ``out`` from simulator reads."""
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Width of :meth:`features` output fed to the world model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def prediction_dim(self) -> int:
        """Width of one predicted increment; may differ from ``size``."""
        raise NotImplementedError

    @abstractmethod
    def prepare(self, state: torch.Tensor) -> StateComputation:
        """Bind numerical evaluation to one borrowed captured state slice."""
        raise NotImplementedError

    def features(self, state: torch.Tensor) -> torch.Tensor:
        """Encode one captured state slice into world-model input features."""
        return self.prepare(state).features()

    @abstractmethod
    def increments(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Return the raw state-space delta of one observed transition.

        Output width matches ``prediction_dim``; also used as the unstructured
        training target and for increment statistics.
        """
        raise NotImplementedError

    @abstractmethod
    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        """Return the predictor's regression target for one transition.

        With ``structured=False`` this equals :meth:`increments`. With
        ``structured=True`` the passive drift accumulated over ``dt`` seconds
        is removed first, so the model only learns the residual dynamics.
        """
        raise NotImplementedError

    def integrate(
        self, state: torch.Tensor, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        """Apply one predicted increment (width ``prediction_dim``) to a state.

        Inverse of :meth:`targets`: returns the next state in raw state space
        and re-adds the passive drift when ``structured=True``.
        """
        return self.prepare(state).integrate(prediction, structured=structured, dt=dt)

    def statistics(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> StateStatistics:
        """Collect feature, target, and increment normalization samples."""
        return StateStatistics(
            self.features(state),
            self.targets(state, next_state, structured=structured, dt=dt),
            self.increments(state, next_state),
        )

    @abstractmethod
    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Return per-dimension residuals between an integrated state and truth.

        Width matches :meth:`increments` output; the world-model loss divides
        these by increment statistics clamped to :meth:`error_floor`.
        """
        raise NotImplementedError

    @abstractmethod
    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        """Return a strictly positive per-dimension loss-normalization floor.

        Prevents near-zero increment variance from amplifying the loss on
        barely-moving dimensions. Width matches :meth:`error` output.
        """
        raise NotImplementedError


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

    def prepare(self, state: torch.Tensor) -> StateComputation:
        return _EuclideanStateComputation(state)

    def increments(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return next_state - state

    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        return self.increments(state, next_state)

    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return pred - truth

    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        return torch.full((self.size,), self.error_scale, device=device)


@dataclass(frozen=True)
class _EuclideanStateComputation(StateComputation):
    state: torch.Tensor

    def features(self) -> torch.Tensor:
        return self.state

    def integrate(self, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02) -> torch.Tensor:
        return self.state + prediction


@dataclass(frozen=True)
class WorldStateComputation(StateComputation):
    """One step's ordered term evaluations, with no persistent state cache."""

    terms: tuple[StateComputation, ...]
    prediction_slices: tuple[slice, ...]

    def features(self) -> torch.Tensor:
        return torch.cat([term.features() for term in self.terms], -1)

    def integrate(self, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02) -> torch.Tensor:
        return torch.cat(
            [
                term.integrate(prediction[..., prediction_slice], structured=structured, dt=dt)
                for term, prediction_slice in zip(self.terms, self.prediction_slices)
            ],
            -1,
        )


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
            if min(term.size, term.feature_dim, term.prediction_dim) <= 0:
                raise ValueError(f"World state term {name!r} requires positive representation widths")
            self.slices[name] = slice(self.size, self.size + term.size)
            self._prediction_slices[name] = slice(self.prediction_dim, self.prediction_dim + term.prediction_dim)
            self.size += term.size
            self.feature_dim += term.feature_dim
            self.prediction_dim += term.prediction_dim

    def capture(self, sim: Map[torch.Tensor], out: torch.Tensor) -> None:
        for name, term in self.terms.items():
            term.capture(sim, out[..., self.slices[name]])

    def prepare(self, state: torch.Tensor) -> WorldStateComputation:
        return WorldStateComputation(
            tuple(term.prepare(state[..., self.slices[name]]) for name, term in self.terms.items()),
            tuple(self._prediction_slices.values()),
        )

    def features(self, state: torch.Tensor) -> torch.Tensor:
        return self.prepare(state).features()

    def statistics(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> StateStatistics:
        statistics = [
            term.statistics(
                state[..., self.slices[name]],
                next_state[..., self.slices[name]],
                structured=structured,
                dt=dt,
            )
            for name, term in self.terms.items()
        ]
        return StateStatistics(
            torch.cat([sample.features for sample in statistics], -1),
            torch.cat([sample.targets for sample in statistics], -1),
            torch.cat([sample.increments for sample in statistics], -1),
        )

    def increments(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                term.increments(state[..., self.slices[name]], next_state[..., self.slices[name]])
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
        return self.prepare(state).integrate(prediction, structured=structured, dt=dt)

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
