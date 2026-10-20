# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral persistent control states and command transitions."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from motrix_deploy.contracts import JointServoCommand, JointTorqueCommand, RobotCommand, RobotState
from motrix_deploy.errors import ValidationError
from motrix_deploy.operator import OperatorCommand
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.lifecycle import ControllerStep


class ControlState(Protocol):
    """One persistent state that produces commands in its declared joint order."""

    name: str
    joint_names: tuple[str, ...]

    def reset(self, state: RobotState) -> None: ...

    def step(self, state: RobotState, context: ControlContext[OperatorCommand]) -> ControllerStep: ...

    def request_stop(self) -> None: ...


class CommandTransition(Protocol):
    """Stateful blend from a frozen source command to live target commands."""

    def start(self, source: RobotCommand, target: RobotCommand) -> None: ...

    def step(self, target: RobotCommand, elapsed_time_s: float) -> tuple[RobotCommand, bool]: ...


TransitionFactory = Callable[[], CommandTransition]


@dataclass(frozen=True)
class TransitionRule:
    """One legal directed state edge and the command blend used on that edge."""

    source: str
    target: str
    create: TransitionFactory


def _quintic_smoothstep(value: float) -> np.float32:
    s = np.float32(np.clip(value, 0.0, 1.0))
    return np.float32(10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)


def _blend_array(source: np.ndarray, target: np.ndarray, alpha: np.float32) -> np.ndarray:
    return (source + alpha * (target - source)).astype(np.float32)


class SmoothServoTransition:
    """Blend position, velocity, feed-forward torque and gains with zero endpoint slope."""

    def __init__(self, duration_s: float) -> None:
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValidationError("transition.duration_s", "a positive finite value", duration_s)
        self.duration_s = float(duration_s)
        self._source: JointServoCommand | None = None

    def start(self, source: RobotCommand, target: RobotCommand) -> None:
        if not isinstance(source, JointServoCommand) or not isinstance(target, JointServoCommand):
            raise ValidationError("transition.command", "joint-servo source and target commands", (source, target))
        if source.joint_position.shape != target.joint_position.shape:
            raise ValidationError(
                "transition.command.shape", str(source.joint_position.shape), target.joint_position.shape
            )
        self._source = source

    def step(self, target: RobotCommand, elapsed_time_s: float) -> tuple[RobotCommand, bool]:
        if self._source is None:
            raise RuntimeError("transition must be started before stepping")
        if not isinstance(target, JointServoCommand):
            raise ValidationError("transition.target", "a joint-servo command", type(target).__name__)
        alpha = _quintic_smoothstep(elapsed_time_s / self.duration_s)
        source = self._source
        command = JointServoCommand(
            joint_position=_blend_array(source.joint_position, target.joint_position, alpha),
            joint_velocity=_blend_array(source.joint_velocity, target.joint_velocity, alpha),
            feedforward_torque=_blend_array(source.feedforward_torque, target.feedforward_torque, alpha),
            kp=_blend_array(source.kp, target.kp, alpha),
            kd=_blend_array(source.kd, target.kd, alpha),
        )
        return command, elapsed_time_s + 1e-9 >= self.duration_s


class AdaptiveServoTransition(SmoothServoTransition):
    """Choose duration from the largest commanded joint displacement."""

    def __init__(self, min_duration_s: float, max_duration_s: float, max_joint_speed_rad_s: float) -> None:
        if not 0 < min_duration_s <= max_duration_s:
            raise ValidationError(
                "transition.duration_range", "0 < min_duration_s <= max_duration_s", (min_duration_s, max_duration_s)
            )
        if not np.isfinite(max_joint_speed_rad_s) or max_joint_speed_rad_s <= 0:
            raise ValidationError("transition.max_joint_speed_rad_s", "a positive finite value", max_joint_speed_rad_s)
        self.min_duration_s = float(min_duration_s)
        self.max_duration_s = float(max_duration_s)
        self.max_joint_speed_rad_s = float(max_joint_speed_rad_s)
        super().__init__(min_duration_s)

    def start(self, source: RobotCommand, target: RobotCommand) -> None:
        super().start(source, target)
        assert isinstance(source, JointServoCommand) and isinstance(target, JointServoCommand)
        displacement = float(np.max(np.abs(target.joint_position - source.joint_position)))
        self.duration_s = float(
            np.clip(displacement / self.max_joint_speed_rad_s, self.min_duration_s, self.max_duration_s)
        )


class ImmediateTransition:
    """Select the target command without blending, for emergency paths."""

    def start(self, source: RobotCommand, target: RobotCommand) -> None:
        del source, target

    def step(self, target: RobotCommand, elapsed_time_s: float) -> tuple[RobotCommand, bool]:
        del elapsed_time_s
        return target, True


@dataclass
class _ActiveTransition:
    target: str
    transition: CommandTransition
    origin_s: float


class StateMachineController:
    """Run persistent states while keeping backend command order and lifecycle stable."""

    def __init__(
        self,
        *,
        joint_names: tuple[str, ...],
        states: Mapping[str, ControlState],
        initial_state: str,
        transitions: tuple[TransitionRule, ...],
        intent_targets: Mapping[str, str],
        completion_targets: Mapping[str, str] | None = None,
        emergency_intent: str | tuple[str, ...] = "request_emergency",
        emergency_state: str = "emergency_damping",
    ) -> None:
        if not joint_names or len(joint_names) != len(set(joint_names)):
            raise ValidationError("state_machine.joint_names", "unique joint names", joint_names)
        self.joint_names = joint_names
        self.states = dict(states)
        if set(self.states) != {state.name for state in self.states.values()}:
            raise ValidationError("state_machine.states", "mapping keys equal state names", tuple(self.states))
        if initial_state not in self.states:
            raise ValidationError("state_machine.initial_state", "a configured state", initial_state)
        for state in self.states.values():
            if set(state.joint_names) != set(joint_names):
                raise ValidationError(
                    f"state_machine.states.{state.name}.joint_names", "the canonical joint-name set", state.joint_names
                )
        self._rules = {(rule.source, rule.target): rule.create for rule in transitions}
        if len(self._rules) != len(transitions):
            raise ValidationError("state_machine.transitions", "unique directed edges", transitions)
        for source, target in self._rules:
            if source not in self.states or target not in self.states:
                raise ValidationError("state_machine.transitions", "edges between configured states", (source, target))
        self._intent_targets = dict(intent_targets)
        self._completion_targets = dict(completion_targets or {})
        for target in (*self._intent_targets.values(), *self._completion_targets.values(), emergency_state):
            if target not in self.states:
                raise ValidationError("state_machine.target", "a configured state", target)
        self._initial_state = initial_state
        self._current = initial_state
        self._emergency_intents = (
            frozenset({emergency_intent}) if isinstance(emergency_intent, str) else frozenset(emergency_intent)
        )
        self._emergency_state = emergency_state
        self._active: _ActiveTransition | None = None
        self._queued_targets: list[str] = []
        self._last_command: RobotCommand | None = None
        self._state_origin_s = 0.0
        self._state_step = 0
        self._stop_requested = False

    @property
    def current_state(self) -> str:
        return self._current

    @property
    def target_state(self) -> str | None:
        return None if self._active is None else self._active.target

    def reset(self, state: RobotState) -> None:
        self._current = self._initial_state
        self._active = None
        self._queued_targets.clear()
        self._last_command = None
        self._state_origin_s = 0.0
        self._state_step = 0
        self._stop_requested = False
        self.states[self._current].reset(state)

    def request_stop(self) -> None:
        self._stop_requested = True

    def _state_context(
        self, context: ControlContext[OperatorCommand], command: OperatorCommand | None = None
    ) -> ControlContext[OperatorCommand]:
        return ControlContext(
            step=self._state_step,
            elapsed_time_s=max(0.0, context.elapsed_time_s - self._state_origin_s),
            command=context.command if command is None else command,
            dt_s=context.dt_s,
        )

    def _canonical_command(self, command: RobotCommand, state_name: str) -> RobotCommand:
        source_names = self.states[state_name].joint_names
        indices = np.asarray([source_names.index(name) for name in self.joint_names], dtype=np.int64)
        if isinstance(command, JointServoCommand):
            return JointServoCommand(
                command.joint_position[indices].astype(np.float32),
                command.joint_velocity[indices].astype(np.float32),
                command.feedforward_torque[indices].astype(np.float32),
                command.kp[indices].astype(np.float32),
                command.kd[indices].astype(np.float32),
            )
        if isinstance(command, JointTorqueCommand):
            return JointTorqueCommand(command.torque[indices].astype(np.float32))
        raise ValidationError("state_machine.command", "a RobotCommand", type(command).__name__)

    def _target_from_intents(self, command: OperatorCommand) -> str | None:
        if self._emergency_intents.intersection(command.intents):
            return self._emergency_state
        targets = {self._intent_targets[intent] for intent in command.intents if intent in self._intent_targets}
        if len(targets) > 1:
            raise ValidationError("operator.intents", "at most one state transition request", command.intents)
        return next(iter(targets), None)

    def _completion_target(self, state_name: str) -> str | None:
        return self._completion_targets.get(state_name)

    def _path_to(self, source: str, target: str) -> list[str]:
        """Return the shortest configured route, preserving declaration order."""
        if source == target:
            return []
        frontier: deque[tuple[str, list[str]]] = deque([(source, [])])
        visited = {source}
        while frontier:
            node, path = frontier.popleft()
            for edge_source, edge_target in self._rules:
                if edge_source != node or edge_target in visited:
                    continue
                next_path = [*path, edge_target]
                if edge_target == target:
                    return next_path
                visited.add(edge_target)
                frontier.append((edge_target, next_path))
        raise ValidationError("state_machine.transition", "a configured route", (source, target))

    def _begin_routed_transition(
        self,
        target: str,
        state: RobotState,
        context: ControlContext[OperatorCommand],
        source_command: RobotCommand,
    ) -> ControllerStep:
        path = self._path_to(self._current, target)
        if not path:
            return ControllerStep(command=source_command)
        self._queued_targets = path[1:]
        return self._begin_transition(path[0], state, context, source_command)

    def _begin_transition(
        self,
        target: str,
        state: RobotState,
        context: ControlContext[OperatorCommand],
        source_command: RobotCommand,
    ) -> ControllerStep:
        if target == self._current:
            return ControllerStep(command=source_command)
        factory = self._rules.get((self._current, target))
        if factory is None:
            raise ValidationError("state_machine.transition", "a configured legal edge", (self._current, target))
        target_state = self.states[target]
        target_state.reset(state)
        self._state_origin_s = context.elapsed_time_s
        self._state_step = 0
        target_step = target_state.step(state, self._state_context(context))
        if target_step.error is not None:
            return target_step
        if target_step.command is None:
            if not target_step.complete:
                return target_step
            self._current = target
            completion_target = self._completion_target(target)
            if completion_target is None:
                return target_step
            return self._begin_transition(completion_target, state, context, source_command)
        target_command = self._canonical_command(target_step.command, target)
        transition = factory()
        transition.start(source_command, target_command)
        self._active = _ActiveTransition(target=target, transition=transition, origin_s=context.elapsed_time_s)
        blended, complete = transition.step(target_command, 0.0)
        if complete:
            self._current = target
            self._active = None
        self._state_step += 1
        return replace(target_step, command=blended, complete=False)

    def step(self, state: RobotState, context: ControlContext[OperatorCommand]) -> ControllerStep:
        if self._stop_requested:
            return ControllerStep(complete=True)
        operator = context.command
        if not isinstance(operator, OperatorCommand):
            raise ValidationError("state_machine.command", "an OperatorCommand", type(operator).__name__)
        requested = self._target_from_intents(operator)
        if self._active is not None:
            if requested == self._emergency_state and self._active.target != self._emergency_state:
                assert self._last_command is not None
                self._current = self._active.target
                self._active = None
                self._queued_targets.clear()
                return self._begin_routed_transition(requested, state, context, self._last_command)
            target = self._active.target
            decision = self.states[target].step(state, self._state_context(context))
            if decision.error is not None or decision.command is None:
                return decision
            target_command = self._canonical_command(decision.command, target)
            output, complete = self._active.transition.step(
                target_command, context.elapsed_time_s - self._active.origin_s
            )
            self._last_command = output
            self._state_step += 1
            if complete:
                self._current = target
                self._active = None
            return replace(decision, command=output, complete=False)

        decision = self.states[self._current].step(state, self._state_context(context))
        if decision.error is not None:
            return decision
        if decision.command is None:
            if decision.complete:
                target = requested if requested is not None else self._completion_target(self._current)
                if target is None:
                    return decision
                if self._last_command is None:
                    raise ValidationError("state_machine.command", "a command before state completion", None)
                return self._begin_transition(target, state, context, self._last_command)
            raise ValidationError("state_machine.command", "one command per active interval", None)
        command = self._canonical_command(decision.command, self._current)
        self._last_command = command
        self._state_step += 1
        target = requested
        if target == self._emergency_state:
            self._queued_targets.clear()
        elif target is None and self._queued_targets:
            target = self._queued_targets.pop(0)
        if target is None and decision.complete:
            target = self._completion_target(self._current)
        if target is not None:
            return self._begin_routed_transition(target, state, context, command)
        return replace(decision, command=command, complete=False)


__all__ = [
    "AdaptiveServoTransition",
    "CommandTransition",
    "ControlState",
    "ImmediateTransition",
    "SmoothServoTransition",
    "StateMachineController",
    "TransitionRule",
]
