# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Static observation contracts and differentiable time-major window operations."""

from dataclasses import dataclass
from numbers import Integral

import gymnasium as gym
import numpy as np
import torch


@dataclass(frozen=True)
class ObservationGroupSpec:
    """One flat frame and its oldest-to-newest, repeat-filled history."""

    frame_space: gym.spaces.Box
    history_length: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.history_length, Integral) or isinstance(self.history_length, bool):
            raise TypeError("Observation history_length must be an integer.")
        if self.history_length < 1:
            raise ValueError("Observation history_length must be at least 1.")
        if len(self.frame_space.shape) != 1:
            raise ValueError("Observation frame_space must be one-dimensional.")

    @property
    def space(self) -> gym.spaces.Box:
        """Derive the final flat space without mutating the static description."""
        space = self.frame_space
        if self.history_length == 1:
            return space
        return gym.spaces.Box(
            np.tile(space.low, self.history_length),
            np.tile(space.high, self.history_length),
            dtype=space.dtype,
        )


@dataclass(frozen=True)
class ObservationSpec:
    """Named observation definitions with a required policy group."""

    groups: dict[str, ObservationGroupSpec]

    def __post_init__(self) -> None:
        if "policy" not in self.groups:
            raise ValueError("ObservationSpec requires a 'policy' observation group.")

    def __getitem__(self, name: str) -> ObservationGroupSpec:
        return self.groups[name]

    @property
    def policy(self) -> ObservationGroupSpec:
        return self.groups["policy"]

    @property
    def value(self) -> ObservationGroupSpec | None:
        return self.groups.get("value")

    @property
    def value_or_policy(self) -> ObservationGroupSpec:
        return self.policy if self.value is None else self.value


def repeat_observation(frame: torch.Tensor, history_length: int) -> torch.Tensor:
    """Repeat the same actual first frame as a ``[B,H,D]`` window."""
    return frame.unsqueeze(1).expand(-1, history_length, -1)


def append_observation(
    history: torch.Tensor, frame: torch.Tensor, reset_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Functionally append a frame, optionally repeat-filling episode-reset rows.

    No input is mutated or detached, so imagined observations retain gradients
    through both the retained prefix and the newly appended frame.
    """
    appended = torch.cat((history[:, 1:], frame.unsqueeze(1)), dim=1)
    if reset_mask is None:
        return appended
    return torch.where(reset_mask[:, None, None], repeat_observation(frame, history.shape[1]), appended)
