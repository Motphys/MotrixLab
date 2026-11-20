# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager observation publication, history ownership, and terminal previews."""

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core.array.env import EnvObs, FinalObs
from motrix_env_core.manager.observations import ObservationGroupEntry
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec, append_observation, repeat_observation


@dataclass
class _SwapObsBuffer:
    """Alternate borrowed outputs so a transition can retain its starting obs."""

    slots: tuple[EnvObs, EnvObs]
    index: int = 0

    @property
    def current(self) -> EnvObs:
        return self.slots[self.index]

    def swap(self) -> None:
        self.index = 1 - self.index


class ObservationManager:
    """Own static groups, observation buffers, and episode-local history.

    Construction allocates all output storage and history from resolved groups
    and batch width. Numerical execution is supplied explicitly at publication.
    Public environment results remain stacked; no separate frame API is exposed.
    """

    def __init__(self, groups: dict[str, ObservationGroupEntry], num_envs: int) -> None:
        self.groups = groups
        self.num_envs = num_envs
        specs = {
            name: ObservationGroupSpec(
                gym.spaces.Box(-np.inf, np.inf, (group.size,), dtype=np.float32), group.history_length
            )
            for name, group in groups.items()
        }
        self.spec = ObservationSpec(specs)
        self._frame_spec = ObservationSpec(
            {name: ObservationGroupSpec(group.frame_space) for name, group in self.spec.groups.items()}
        )
        self._history_reset = torch.ones(num_envs, dtype=torch.bool)

        self._stack_buffer = _SwapObsBuffer((self._zeros(self.spec), self._zeros(self.spec)))
        has_history = any(group.history_length > 1 for group in self.groups.values())
        self._frame_buffer = (
            _SwapObsBuffer(
                (
                    self._zeros(self._frame_spec),
                    self._zeros(self._frame_spec),
                )
            )
            if has_history
            else self._stack_buffer
        )
        self._scratch = self._zeros(self._frame_spec)
        self._history = {
            name: torch.zeros((self.num_envs, group.history_length, group.size), dtype=torch.float32)
            for name, group in self.groups.items()
            if group.history_length > 1
        }

    def _zeros(self, spec: ObservationSpec) -> EnvObs:
        return EnvObs(
            {
                name: torch.zeros((self.num_envs, *group.space.shape), dtype=torch.float32)
                for name, group in spec.groups.items()
            }
        )

    @property
    def observation(self) -> EnvObs:
        return self._stack_buffer.current

    @property
    def frame_output(self) -> EnvObs:
        """Current single-frame kernel write target, not a published read view."""
        return self._frame_buffer.current

    def begin_step(self) -> EnvObs:
        """Select the next borrowed output without advancing any history."""
        self._frame_buffer.swap()
        if self._stack_buffer is not self._frame_buffer:
            self._stack_buffer.swap()
        return self._stack_buffer.current

    def reset(self, env_ids: torch.Tensor) -> None:
        """Invalidate only episode-reset histories, not simulator-only resets."""
        self._history_reset[env_ids] = True

    def publish_reset(self, env_ids: torch.Tensor, observe: Callable[[EnvObs], None]) -> None:
        """Repeat-fill explicit episode-reset lanes, leaving other rows untouched."""
        reset = env_ids
        observe(self._scratch)
        frame = self._scratch
        output = self._stack_buffer.current
        for name, group in self.groups.items():
            values = frame[name][reset]
            destination = output[name]
            if group.history_length == 1:
                destination[reset] = values
            else:
                initialized = repeat_observation(values, group.history_length)
                self._history[name][reset] = initialized
                destination[reset] = initialized.flatten(1)
        self._history_reset[env_ids] = False

    def commit(self, observe: Callable[[EnvObs], None]) -> None:
        """Publish exactly one control-step endpoint, repeat-filling new episodes."""
        observe(self.frame_output)
        frame = self._frame_buffer.current
        output = self._stack_buffer.current
        for name, group in self.groups.items():
            values = frame[name]
            destination = output[name]
            if group.history_length == 1:
                if destination is not values:
                    destination.copy_(values)
            else:
                history = self._history[name]
                history.copy_(append_observation(history, values, self._history_reset))
                destination.copy_(history.flatten(1))
        self._history_reset.fill_(False)

    def preview(self, env_ids: torch.Tensor, observe: Callable[[EnvObs], None]) -> FinalObs:
        """Return owned sparse endpoint windows without committing history."""
        observe(self._scratch)
        frame = self._scratch
        stacked = {}
        for name, group in self.groups.items():
            values = frame[name][env_ids]
            stacked[name] = (
                values
                if group.history_length == 1
                else append_observation(self._history[name][env_ids], values).flatten(1)
            )
        return FinalObs(env_ids.clone(), EnvObs(stacked))
