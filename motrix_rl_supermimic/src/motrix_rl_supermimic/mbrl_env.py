# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task configuration and collector composition for ordinary Manager environments."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box

from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.config import configclass
from motrix_env_core.manager.commands import CommandTerm
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.env import ManagerBasedEnvCfg, ManagerEnv
from motrix_env_core.manager.observations import ObsTerm, create_observation_groups
from motrix_env_core.manager.rewards import RewardTerm, create_reward_terms
from motrix_env_core.manager.terminations import TerminationTermCfg
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec
from motrix_env_core.torch_manager.runtime import TorchTaskProgram
from motrix_rl_supermimic.imagination import ImaginationCfg
from motrix_rl_supermimic.manager_task import ManagerTaskMDP, resolve_manager_task
from motrix_rl_supermimic.recorder import TransitionRecorder
from motrix_rl_supermimic.reference_command import ReferenceCommand
from motrix_rl_supermimic.task_mdp import StateSpec
from motrix_rl_supermimic.world_context import WorldContext, WorldContextTermCfg
from motrix_rl_supermimic.world_state import WorldState, WorldStateTermCfg


@configclass
class MbrlEnvCfg(ManagerBasedEnvCfg):
    """Manager task declarations with a simulator-free analytic execution path.

    The first version supports the existing +1 reference-frame progression;
    physical derived inputs (e.g. FK) are explicit pure functions in head code.
    No arbitrary simulator query is silently synthesized for imagination.
    """

    # Name of the command whose reference timeline is captured for replay.
    reference_command: str = "reference"
    world_model_context: dict[str, WorldContextTermCfg] = {}
    world_model_state: dict[str, WorldStateTermCfg] = {}
    imagination: ImaginationCfg | None = None
    soft_termination: TerminationTermCfg | None = None
    # Fixed before construction so the ordinary goal group has a static width.
    goal_frames: int = 10

    def validate_goal_frames(self) -> None:
        """Validate the task horizon before real or imagined term construction."""
        if type(self.goal_frames) is not int or self.goal_frames <= 0:
            raise ValueError("MBRL goal_frames must be a positive integer")

    def reference_action(self) -> torch.nn.Module | None:
        """Build the optional reference policy without loading task runtime data.

        Collector, learner and playback use this same factory inside their policy
        initialization seed scope. Returned modules own their device-movable data.
        """
        return None

    def for_imagined(self) -> MbrlEnvCfg:
        """Return the config variant used to assemble the imagined world-model program.

        The same task config serves both the real environment and the
        simulator-free imagined build; the imagined assembly applies this
        override instead of passing a mode through term construction. The
        default is mode-agnostic (imagined terms behave exactly like real
        ones); tasks whose imagined behavior differs override this to return
        a modified copy (e.g. deterministic observations without noise).
        """
        return self


class MbrlEnv(ManagerEnv):
    """Manager environment owning transition capture and dispatch to task terms.

    The inherited lifecycle owns episode boundaries. Recording uses its existing
    action/evaluation phases without replacing step, reset or observation logic.
    """

    def __init__(
        self,
        cfg: MbrlEnvCfg | None = None,
        num_envs: int = 1,
        backend: str | None = None,
        seed: int | None = None,
    ) -> None:
        if cfg is None or not isinstance(cfg, MbrlEnvCfg):
            raise TypeError("MbrlEnv requires a Manager task configuration inheriting MbrlEnvCfg")
        cfg.validate_goal_frames()
        super().__init__(cfg, num_envs, backend, seed)
        if not cfg.world_model_state:
            raise TypeError("MbrlEnv requires world-model state terms")
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
        self._world_state = WorldState({name: term(build) for name, term in cfg.world_model_state.items()})
        self._world_context = WorldContext({name: term(build) for name, term in cfg.world_model_context.items()})
        self._state_spec = StateSpec(self._world_state.size, self._world_context.size)
        self._recorder = TransitionRecorder.create(num_envs, self.state_spec, action_dim=self.action_space.shape[0])

    @property
    def world_state(self) -> WorldState:
        return self._world_state

    @property
    def world_context(self) -> WorldContext:
        return self._world_context

    @property
    def state_spec(self) -> StateSpec:
        return self._state_spec

    def _build_task_program(self) -> TorchTaskProgram:
        program = super()._build_task_program()
        ctx = self._compiled_manager_program.context
        world_context = self.world_context
        world_state = self.world_state
        recorder = self._recorder
        reference = self._reference_command
        action_states = tuple((term.state, self.action_slices[name]) for name, term in self.action_terms.items())

        def begin() -> None:
            world_state.capture(ctx, recorder.s)
            world_context.capture(ctx, recorder.context)
            recorder.frame.copy_(reference.frame)

        def actions() -> None:
            for state, action_slice in action_states:
                recorder.action[:, action_slice].copy_(state.current())
                recorder.previous[:, action_slice].copy_(state.previous())

        def end() -> None:
            world_state.capture(ctx, recorder.s1)

        self._capture_begin = torch.compile(begin, dynamic=False)
        self._capture_actions = torch.compile(actions, dynamic=False)
        self._capture_end = torch.compile(end, dynamic=False)
        return program

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        # Freeze task inputs before action processing can mutate task state.
        self._capture_begin()
        self._recorder.episode_step.copy_(state.episode_steps)
        state = super().apply_action(actions, state)
        # ActionState now exposes this step's raw action and its predecessor.
        self._capture_actions()
        return state

    def _execute_evaluate_kernel(self) -> None:
        # Manager has refreshed sim reads, but no command has advanced/reset.
        self._capture_end()
        super()._execute_evaluate_kernel()

    def snapshot_transition(self, result: ArrayEnvState) -> dict[str, torch.Tensor]:
        """Export the last step's owned endpoints and final framework outputs."""
        return self._recorder.snapshot(result)


def assemble_task(cfg: MbrlEnvCfg, device: str) -> ManagerTaskMDP:
    """Assemble the analytic task from an explicit configuration variant.

    The caller chooses the variant (``cfg.for_imagined()`` for the learner's
    imagination program); assembly itself never applies one.
    """
    with torch.device(device):
        return _assemble_task(cfg)


def _assemble_task(task: MbrlEnvCfg) -> ManagerTaskMDP:
    task.validate_goal_frames()
    if not task.world_model_state or task.imagination is None or task.soft_termination is None:
        raise ValueError("MBRL configuration requires world_model_state, imagination and soft_termination")
    commands: dict[str, CommandTerm] = {}
    build = BuildContext(cfg=task, actions={}, commands=commands, model=None, num_envs=1, seed=0)
    for name, command_cfg in task.command_cfgs().items():
        commands[name] = command_cfg(build)
    actions = {}
    action_slices: dict[str, slice] = {}
    action_offset = 0
    for name, action in task.action_cfgs().items():
        build = BuildContext(cfg=task, actions=actions, commands=commands, model=None, num_envs=1, seed=0)
        # The imagined build has no compiled scene model: model-independent
        # factories receive no actuators, named actuator routes are not
        # supported.
        actuators = () if action.actuator_names == () else None
        term = action(build, actuators)
        actions[name] = term
        action_width = term.action_space.shape[0]
        action_slices[name] = slice(action_offset, action_offset + action_width)
        action_offset += action_width

    context = BuildContext(cfg=task, actions=actions, commands=commands, model=None, num_envs=1, seed=0)
    world_state = WorldState({name: term(context) for name, term in task.world_model_state.items()})
    world_context = WorldContext({name: term(context) for name, term in task.world_model_context.items()})
    spec = StateSpec(world_state.size, world_context.size)
    imagination = task.imagination(context)
    groups = create_observation_groups(task, context)
    for name in ("policy", "value", "goal"):
        if name not in groups:
            raise ValueError(f"MBRL requires observation group {name!r}")
    if groups["goal"].history_length != 1:
        raise ValueError("MBRL goal observation must have history_length=1")
    action_space = Box(
        np.concatenate([term.action_space.low for term in actions.values()]),
        np.concatenate([term.action_space.high for term in actions.values()]),
        dtype=np.float32,
    )
    observation_spec = ObservationSpec(
        groups={
            name: ObservationGroupSpec(Box(-float("inf"), float("inf"), (group.size,)), group.history_length)
            for name, group in groups.items()
        }
    )
    from motrix_env_core.manager.resolver import ManagerTermResolver

    resolver = ManagerTermResolver()

    def observation(group_name: str) -> ObsTerm:
        group = groups[group_name]
        calls = tuple(
            (resolver.resolve_observation(group_name, entry.name, entry.term), entry.size) for entry in group.terms
        )

        def dispatch(ctx: ManagerContext, out: torch.Tensor) -> None:
            offset = 0
            for call, size in calls:
                call.invoke(ctx, out[:, offset : offset + size])
                offset += size

        return ObsTerm(group.size, dispatch)

    rewards = create_reward_terms(task.reward_cfgs(), context)
    reward_cfgs = task.reward_cfgs()
    calls = tuple(
        (resolver.resolve_reward(name, index, term), reward_cfgs[name].weight)
        for index, (name, term) in enumerate(rewards.items())
    )

    def reward(ctx: ManagerContext) -> torch.Tensor:
        total = ctx.env_id.new_zeros(ctx.env_id.shape, dtype=torch.float32)
        for call, weight in calls:
            total = total + call.invoke(ctx).to(torch.float32) * weight
        return total

    return resolve_manager_task(
        spec,
        task.ctrl_dt,
        1.0,
        imagination,
        observation("policy"),
        observation("value"),
        RewardTerm(reward),
        task.soft_termination(context),
        goal=observation("goal"),
        goal_frames=task.goal_frames,
        world_context=world_context,
        observation_spec=observation_spec,
        action_space=action_space,
        action_slices=action_slices,
        world_state=world_state,
        reference_action=task.reference_action,
    )
