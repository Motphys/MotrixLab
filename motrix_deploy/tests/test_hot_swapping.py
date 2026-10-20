# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from motrix_deploy.contracts import JointServoCommand, RobotSpec, RobotState
from motrix_deploy.operator import OperatorCommand
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.hot_swapping import BaselineStates, RobotHotSwappingController, SkillModule
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy.runtime.preparation import JointServoPreparationState, SkillPreparationSpec
from motrix_deploy.runtime.state_machine import ImmediateTransition, TransitionRule


@dataclass
class _State:
    name: str
    joint_names: tuple[str, ...]
    target: float
    complete_after: int | None = None

    def reset(self, state: RobotState) -> None:
        del state
        self.steps = 0

    def request_stop(self) -> None:
        pass

    def step(self, state: RobotState, context: ControlContext[OperatorCommand]) -> ControllerStep:
        del state, context
        self.steps += 1
        position = np.full(2, self.target, dtype=np.float32)
        zeros = np.zeros(2, dtype=np.float32)
        return ControllerStep(
            command=JointServoCommand(position, zeros, zeros, np.ones(2, dtype=np.float32), zeros),
            complete=self.complete_after is not None and self.steps > self.complete_after,
        )


def _robot() -> RobotSpec:
    return RobotSpec(
        base_link_name="base",
        joint_names=("a", "b"),
        default_joint_position=np.zeros(2, dtype=np.float32),
        position_lower=np.full(2, -2, dtype=np.float32),
        position_upper=np.full(2, 2, dtype=np.float32),
        torque_limit=np.ones(2, dtype=np.float32),
    )


def _state() -> RobotState:
    return RobotState(
        sample_time_ns=0,
        receive_time_ns=0,
        joint_position=np.zeros(2, dtype=np.float32),
        joint_velocity=np.zeros(2, dtype=np.float32),
        base_orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
        base_angular_velocity=np.zeros(3, dtype=np.float32),
        base_linear_acceleration=np.zeros(3, dtype=np.float32),
    )


def _context(step: int, intent: str | None = None) -> ControlContext[OperatorCommand]:
    intents = frozenset() if intent is None else frozenset({intent})
    return ControlContext(step, step * 0.1, OperatorCommand(intents), 0.1)


def _machine(initial_state: str = "walk") -> RobotHotSwappingController:
    robot = _robot()
    passive = _State("passive", robot.joint_names, 0.0)
    fixstand = _State("fixstand", robot.joint_names, 0.1)
    walk = _State("walk", robot.joint_names, 0.2)
    wave = _State("wave", robot.joint_names, 0.3, complete_after=1)
    bow = _State("bow", robot.joint_names, 0.4, complete_after=1)
    preparation = JointServoPreparationState(robot)
    preparation_spec = SkillPreparationSpec(
        joint_position=np.zeros(2, dtype=np.float32),
        kp=np.ones(2, dtype=np.float32),
        kd=np.ones(2, dtype=np.float32),
        ramp_duration_s=0.1,
        settle_duration_s=0.0,
    )
    return RobotHotSwappingController(
        joint_names=robot.joint_names,
        baseline=BaselineStates(passive, fixstand, walk),
        preparation=preparation,
        skills=(
            SkillModule("wave", wave, preparation_spec, "request_wave"),
            SkillModule("bow", bow, preparation_spec, "request_bow"),
        ),
        initial_state=initial_state,
        baseline_transitions=(
            TransitionRule("passive", "fixstand", ImmediateTransition),
            TransitionRule("fixstand", "walk", ImmediateTransition),
            TransitionRule("walk", "fixstand", ImmediateTransition),
        ),
        passive_transition=ImmediateTransition,
    )


def test_skill_request_uses_one_shared_preparation_state_then_returns_to_walk():
    machine = _machine()
    state = _state()
    machine.reset(state)

    machine.step(state, _context(0))
    machine.step(state, _context(1, "request_wave"))
    assert machine.current_state == "skill_preparation"
    assert machine.pending_skill == "wave"

    machine.step(state, _context(2))
    assert machine.current_state == "wave"
    machine.step(state, _context(3))
    machine.step(state, _context(4))
    assert machine.current_state == "walk"
    assert machine.pending_skill is None


def test_multiple_skills_share_preparation_and_select_the_pending_destination():
    machine = _machine()
    state = _state()
    machine.reset(state)

    preparation = machine.states["skill_preparation"]
    machine.step(state, _context(0))
    machine.step(state, _context(1, "request_bow"))
    assert machine.states["skill_preparation"] is preparation
    assert machine.current_state == "skill_preparation"
    assert machine.pending_skill == "bow"

    machine.step(state, _context(2))
    assert machine.current_state == "bow"


def test_skill_request_is_ignored_until_walk_and_passive_preempts_preparation():
    machine = _machine(initial_state="fixstand")
    state = _state()
    machine.reset(state)

    machine.step(state, _context(0, "request_wave"))
    assert machine.current_state == "fixstand"
    assert machine.pending_skill is None

    machine.step(state, _context(1, "request_walk"))
    machine.step(state, _context(2, "request_wave"))
    assert machine.current_state == "skill_preparation"
    machine.step(state, _context(3, "request_passive"))
    assert machine.current_state == "passive"
    assert machine.pending_skill is None
