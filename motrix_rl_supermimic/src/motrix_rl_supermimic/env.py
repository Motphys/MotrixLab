# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task configuration and collector composition for ordinary Manager environments."""

from __future__ import annotations

import torch

from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.config import configclass
from motrix_env_core.manager.actions import ActionState
from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.env import ManagerBasedEnvCfg, ManagerEnv
from motrix_env_core.manager.terminations import TerminationTermCfg
from motrix_rl_supermimic.imagination import ImaginationCfg
from motrix_rl_supermimic.recorder import TransitionRecorder
from motrix_rl_supermimic.reference_command import ReferenceCommand
from motrix_rl_supermimic.world_context import WorldContext, WorldContextTermCfg
from motrix_rl_supermimic.world_state import WorldStateManager, WorldStateTermCfg


@configclass
class SuperMimicEnvCfg(ManagerBasedEnvCfg):
    """Manager task declarations with a simulator-free analytic execution path.

    The first version supports the existing +1 reference-frame progression;
    physical derived inputs (e.g. FK) are explicit pure functions in head code.
    No arbitrary simulator query is silently synthesized for imagination.
    """

    reference_command: str = "reference"
    world_model_context: dict[str, WorldContextTermCfg] = {}
    world_model_state: dict[str, WorldStateTermCfg] = {}
    imagination: ImaginationCfg | None = None
    soft_termination: TerminationTermCfg | None = None
    goal_frames: int = 10

    def validate_goal_frames(self) -> None:
        """Validate the task horizon before real or imagined term construction."""
        if type(self.goal_frames) is not int or self.goal_frames <= 0:
            raise ValueError("MBRL goal_frames must be a positive integer")

    def reference_action(self) -> torch.nn.Module | None:
        """Build the optional reference policy without loading task runtime data."""
        return None

    def for_imagined(self) -> SuperMimicEnvCfg:
        """Return the config variant used to assemble the imagined program."""
        return self


class TransitionCapture:
    """Compiled transition-endpoint capture attached to one Manager task build."""

    def __init__(
        self,
        ctx: ManagerContext,
        world_state: WorldStateManager,
        world_context: WorldContext,
        recorder: TransitionRecorder,
        reference: ReferenceCommand,
        action_states: tuple[tuple[ActionState, slice], ...],
    ) -> None:
        self._ctx = ctx
        self._world_state = world_state
        self._world_context = world_context
        self._recorder = recorder
        self._reference = reference
        self._action_states = action_states
        self.begin = torch.compile(self._capture_begin, dynamic=False)
        self.actions = torch.compile(self._capture_actions, dynamic=False)
        self.end = torch.compile(self._capture_end, dynamic=False)

    def _capture_begin(self) -> None:
        self._world_state.capture(self._ctx.sim, self._recorder.s)
        self._world_context.capture(self._ctx, self._recorder.context)
        self._recorder.frame.copy_(self._reference.frame)

    def _capture_actions(self) -> None:
        for state, action_slice in self._action_states:
            self._recorder.action[:, action_slice].copy_(state.current())
            self._recorder.previous[:, action_slice].copy_(state.previous())

    def _capture_end(self) -> None:
        self._world_state.capture(self._ctx.sim, self._recorder.s1)


class SuperMimicEnv(ManagerEnv):
    """Manager environment owning transition capture and dispatch to task terms."""

    def __init__(
        self,
        cfg: SuperMimicEnvCfg | None = None,
        num_envs: int = 1,
        backend: str | None = None,
        seed: int | None = None,
    ) -> None:
        if cfg is None or not isinstance(cfg, SuperMimicEnvCfg):
            raise TypeError("SuperMimicEnv requires a Manager task configuration inheriting SuperMimicEnvCfg")
        cfg.validate_goal_frames()
        super().__init__(cfg, num_envs, backend, seed)
        if not cfg.world_model_state:
            raise TypeError("SuperMimicEnv requires world-model state terms")
        goal = self.observation_spec["goal"]
        if goal.history_length != 1:
            raise ValueError("MBRL goal observation must have history_length=1")
        reference = self.command_terms[cfg.reference_command]
        if not isinstance(reference, ReferenceCommand):
            raise TypeError("MBRL reference_command must select a ReferenceCommand")
        if reference.frame.shape != (num_envs,) or reference.frame.dtype != torch.long:
            raise ValueError("ReferenceCommand.frame must be an int64 tensor with shape (num_envs,)")
        self._reference_command = reference
        build = self.build_context()
        self._world_state = WorldStateManager({name: term(build) for name, term in cfg.world_model_state.items()})
        self._world_context = WorldContext({name: term(build) for name, term in cfg.world_model_context.items()})
        self._transition_capture: TransitionCapture | None = None

    @property
    def world_state(self) -> WorldStateManager:
        return self._world_state

    @property
    def world_context(self) -> WorldContext:
        return self._world_context

    def _init_owned_state(self) -> None:
        super()._init_owned_state()
        self._recorder = TransitionRecorder.create(
            self.num_envs,
            device=self.device,
            state_dim=self.world_state.size,
            context_dim=self.world_context.size,
            action_dim=self.action_space.shape[0],
        )
        self._transition_capture = TransitionCapture(
            self._compiled_manager_program.context,
            self.world_state,
            self.world_context,
            self._recorder,
            self._reference_command,
            tuple((term.state, self.action_slices[name]) for name, term in self.action_terms.items()),
        )

    def _pre_action(self, actions: torch.Tensor, state: ArrayEnvState) -> None:
        self._transition_capture.begin()
        self._recorder.episode_step.copy_(state.episode_steps)

    def _post_action(self, actions: torch.Tensor, state: ArrayEnvState) -> None:
        self._transition_capture.actions()

    def _post_physics(self, state: ArrayEnvState) -> None:
        self._transition_capture.end()

    def snapshot_transition(self, result: ArrayEnvState) -> dict[str, torch.Tensor]:
        """Export the last step's owned endpoints and final framework outputs."""
        return self._recorder.snapshot(result)
