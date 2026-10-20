# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral orchestration for baseline states and finite skill modules."""

from __future__ import annotations

from dataclasses import dataclass

from motrix_deploy.contracts import RobotState
from motrix_deploy.errors import ValidationError
from motrix_deploy.operator import OperatorCommand
from motrix_deploy.runtime.preparation import JointServoPreparationState, SkillPreparationSpec
from motrix_deploy.runtime.state_machine import (
    ControlState,
    ImmediateTransition,
    StateMachineController,
    TransitionFactory,
    TransitionRule,
)


@dataclass(frozen=True)
class BaselineStates:
    """Persistent passive, fixed-standing and locomotion states."""

    passive: ControlState
    fixstand: ControlState
    walk: ControlState


@dataclass(frozen=True)
class SkillModule:
    """One finite policy state and the preparation required before entering it."""

    name: str
    state: ControlState
    preparation: SkillPreparationSpec
    request_intent: str

    def __post_init__(self) -> None:
        if self.name != self.state.name:
            raise ValueError("skill module name must equal its control state name")


class RobotHotSwappingController(StateMachineController):
    """Run P/A/B/C/D with one shared preparation state for all skills."""

    def __init__(
        self,
        *,
        joint_names: tuple[str, ...],
        baseline: BaselineStates,
        preparation: JointServoPreparationState,
        skills: tuple[SkillModule, ...],
        initial_state: str,
        baseline_transitions: tuple[TransitionRule, ...],
        passive_transition: TransitionFactory,
        emergency_intents: tuple[str, ...] = ("request_passive", "request_emergency"),
    ) -> None:
        if not skills:
            raise ValidationError("hot_swapping.skills", "at least one skill module", skills)
        self._preparation = preparation
        self._skills = {module.name: module for module in skills}
        if len(self._skills) != len(skills):
            raise ValidationError("hot_swapping.skills", "unique skill names", tuple(self._skills))
        self._skill_intents = {module.request_intent: module for module in skills}
        if len(self._skill_intents) != len(skills):
            raise ValidationError("hot_swapping.skills", "unique request intents", tuple(self._skill_intents))
        self._pending_skill: str | None = None

        states = (
            baseline.passive,
            baseline.fixstand,
            baseline.walk,
            preparation,
            *(module.state for module in skills),
        )
        passive_name = baseline.passive.name
        transitions = (
            *baseline_transitions,
            TransitionRule(baseline.walk.name, preparation.name, ImmediateTransition),
            *(TransitionRule(preparation.name, module.name, ImmediateTransition) for module in skills),
            *(TransitionRule(module.name, baseline.walk.name, ImmediateTransition) for module in skills),
            *(
                TransitionRule(state.name, passive_name, passive_transition)
                for state in states
                if state.name != passive_name
                and not any(rule.source == state.name and rule.target == passive_name for rule in baseline_transitions)
            ),
        )
        self._walk_name = baseline.walk.name
        super().__init__(
            joint_names=joint_names,
            states={state.name: state for state in states},
            initial_state=initial_state,
            transitions=transitions,
            intent_targets={
                "request_fixstand": baseline.fixstand.name,
                "request_walk": baseline.walk.name,
            },
            emergency_intent=emergency_intents,
            emergency_state=passive_name,
        )

    @property
    def pending_skill(self) -> str | None:
        return self._pending_skill

    def reset(self, state: RobotState) -> None:
        self._pending_skill = None
        super().reset(state)

    def _target_from_intents(self, command: OperatorCommand) -> str | None:
        target = super()._target_from_intents(command)
        if target == self._emergency_state:
            self._pending_skill = None
            return target
        protected = {self._preparation.name, *self._skills}
        if self._current in protected or (self._active is not None and self._active.target in protected):
            return None
        requested_modules = [self._skill_intents[intent] for intent in command.intents if intent in self._skill_intents]
        if len(requested_modules) > 1 or (target is not None and requested_modules):
            raise ValidationError("operator.intents", "at most one transition request", command.intents)
        if not requested_modules:
            return target
        # Skills may only be armed from stable locomotion. Requests received in
        # P/A/C/D or during a transition are deliberately ignored.
        if self._current != self._walk_name or self._active is not None:
            return None
        module = requested_modules[0]
        self._pending_skill = module.name
        self._preparation.select(module.preparation)
        return self._preparation.name

    def _completion_target(self, state_name: str) -> str | None:
        if state_name == self._preparation.name:
            if self._pending_skill is None:
                raise RuntimeError("preparation completed without a pending skill")
            return self._pending_skill
        if state_name in self._skills:
            self._pending_skill = None
            return self._walk_name
        return super()._completion_target(state_name)


__all__ = ["BaselineStates", "RobotHotSwappingController", "SkillModule"]
