# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed real transitions shared by transport consumers, learner and replay."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Transitions:
    """Tensor views of transitions, without taking ownership of their storage.

    Batches have leading (steps, environments) dimensions; ``at_step`` removes
    the time dimension. Ring-backed views are only valid until reads are committed.
    Dtypes follow the transition schema and are decoded at the transport boundary.
    """

    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    context: torch.Tensor
    reward: torch.Tensor
    frame: torch.Tensor
    episode_step: torch.Tensor
    done: torch.Tensor

    @classmethod
    def from_dict(cls, values: Mapping[str, torch.Tensor]) -> "Transitions":
        """Wrap schema-named tensors without copying or converting them."""
        return cls(
            state=values["s"],
            next_state=values["s1"],
            action=values["a"],
            previous_action=values["a_prev"],
            context=values["xi"],
            reward=values["r"],
            frame=values["f"],
            episode_step=values["ep_step"],
            done=values["done"],
        )

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Expose schema-named tensor views for field-oriented storage."""
        return {
            "s": self.state,
            "s1": self.next_state,
            "a": self.action,
            "a_prev": self.previous_action,
            "xi": self.context,
            "r": self.reward,
            "f": self.frame,
            "ep_step": self.episode_step,
            "done": self.done,
        }

    def to(self, device: str | torch.device) -> "Transitions":
        """Move tensors to a device, reusing storage when already on that device."""
        return Transitions(
            state=self.state.to(device),
            next_state=self.next_state.to(device),
            action=self.action.to(device),
            previous_action=self.previous_action.to(device),
            context=self.context.to(device),
            reward=self.reward.to(device),
            frame=self.frame.to(device),
            episode_step=self.episode_step.to(device),
            done=self.done.to(device),
        )

    def at_step(self, index: int) -> "Transitions":
        """Select one time step of a batch, retaining the environment dimension."""
        return Transitions(
            state=self.state[index],
            next_state=self.next_state[index],
            action=self.action[index],
            previous_action=self.previous_action[index],
            context=self.context[index],
            reward=self.reward[index],
            frame=self.frame[index],
            episode_step=self.episode_step[index],
            done=self.done[index],
        )
