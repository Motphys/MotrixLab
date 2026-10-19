# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from motrix_deploy.contracts import JointServoCommand, RobotState
from motrix_deploy.operator import OperatorCommand
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy.runtime.state_machine import (
    ImmediateTransition,
    SmoothServoTransition,
    StateMachineController,
    TransitionRule,
)


@dataclass
class _State:
    name: str
    joint_names: tuple[str, ...]
    target: tuple[float, ...]
    kp: float
    complete_after: int | None = None

    def reset(self, state: RobotState) -> None:
        del state
        self.steps = 0

    def request_stop(self) -> None:
        pass

    def step(self, state, context):
        del state, context
        target = np.asarray(self.target, dtype=np.float32)
        zeros = np.zeros_like(target)
        self.steps += 1
        return ControllerStep(
            command=JointServoCommand(target, zeros, zeros, np.full_like(target, self.kp), np.full_like(target, 1.0)),
            complete=self.complete_after is not None and self.steps > self.complete_after,
            policy_tick=self.name == "walk",
        )


def _robot_state() -> RobotState:
    return RobotState(
        sample_time_ns=0,
        receive_time_ns=0,
        joint_position=np.zeros(2, dtype=np.float32),
        joint_velocity=np.zeros(2, dtype=np.float32),
        base_orientation_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float32),
        base_angular_velocity=np.zeros(3, dtype=np.float32),
        base_linear_acceleration=np.zeros(3, dtype=np.float32),
    )


def _context(step, elapsed, intent=None):
    command = OperatorCommand(frozenset() if intent is None else frozenset({intent}))
    return ControlContext(step, elapsed, command, 0.1)


def test_transition_blends_live_target_and_reorders_by_joint_name():
    passive = _State("passive", ("hip", "knee"), (0.0, 0.0), 0.0)
    walk = _State("walk", ("knee", "hip"), (2.0, 1.0), 10.0)
    machine = StateMachineController(
        joint_names=("hip", "knee"),
        states={state.name: state for state in (passive, walk)},
        initial_state="passive",
        transitions=(TransitionRule("passive", "walk", lambda: SmoothServoTransition(0.5)),),
        intent_targets={"request_walk": "walk"},
        emergency_state="passive",
    )
    state = _robot_state()
    machine.reset(state)
    first = machine.step(state, _context(0, 0.0))
    start = machine.step(state, _context(1, 0.1, "request_walk"))
    middle = machine.step(state, _context(2, 0.35))
    end = machine.step(state, _context(3, 0.6))
    np.testing.assert_array_equal(first.command.joint_position, [0, 0])
    np.testing.assert_array_equal(start.command.joint_position, [0, 0])
    np.testing.assert_allclose(middle.command.joint_position, [0.5, 1.0], atol=1e-6)
    np.testing.assert_array_equal(end.command.joint_position, [1, 2])
    assert middle.policy_tick and machine.current_state == "walk"


def test_emergency_preempts_smooth_transition_immediately():
    states = {
        "fixstand": _State("fixstand", ("a", "b"), (1.0, 1.0), 20.0),
        "walk": _State("walk", ("a", "b"), (2.0, 2.0), 10.0),
        "emergency": _State("emergency", ("a", "b"), (0.0, 0.0), 0.0),
    }
    machine = StateMachineController(
        joint_names=("a", "b"),
        states=states,
        initial_state="fixstand",
        transitions=(
            TransitionRule("fixstand", "walk", lambda: SmoothServoTransition(0.5)),
            TransitionRule("walk", "emergency", ImmediateTransition),
        ),
        intent_targets={"request_walk": "walk"},
        emergency_state="emergency",
    )
    state = _robot_state()
    machine.reset(state)
    machine.step(state, _context(0, 0.0))
    machine.step(state, _context(1, 0.1, "request_walk"))
    emergency = machine.step(state, _context(2, 0.2, "request_emergency"))
    np.testing.assert_array_equal(emergency.command.kp, [0, 0])
    assert machine.current_state == "emergency" and machine.target_state is None


def test_completed_policy_returns_to_safe_intermediate_state():
    fixstand = _State("fixstand", ("a", "b"), (1.0, 1.0), 20.0)
    motion = _State("motion", ("a", "b"), (2.0, 2.0), 10.0, complete_after=1)
    machine = StateMachineController(
        joint_names=("a", "b"),
        states={state.name: state for state in (fixstand, motion)},
        initial_state="motion",
        transitions=(TransitionRule("motion", "fixstand", lambda: SmoothServoTransition(0.5)),),
        intent_targets={},
        completion_targets={"motion": "fixstand"},
        emergency_state="fixstand",
    )
    state = _robot_state()
    machine.reset(state)
    assert not machine.step(state, _context(0, 0.0)).complete
    assert machine.step(state, _context(1, 0.1)).command is not None
    assert machine.target_state == "fixstand"


def test_policy_switch_is_routed_through_safe_intermediate_state():
    states = {
        name: _State(name, ("a", "b"), target, 10.0)
        for name, target in (("walk", (1.0, 1.0)), ("fixstand", (0.0, 0.0)), ("motion", (2.0, 2.0)))
    }
    machine = StateMachineController(
        joint_names=("a", "b"),
        states=states,
        initial_state="walk",
        transitions=(
            TransitionRule("walk", "fixstand", lambda: SmoothServoTransition(0.1)),
            TransitionRule("fixstand", "motion", lambda: SmoothServoTransition(0.1)),
        ),
        intent_targets={"request_motion": "motion"},
        emergency_state="fixstand",
    )
    state = _robot_state()
    machine.reset(state)
    machine.step(state, _context(0, 0.0))
    machine.step(state, _context(1, 0.1, "request_motion"))
    assert machine.target_state == "fixstand"
    machine.step(state, _context(2, 0.2))
    assert machine.current_state == "fixstand"
    machine.step(state, _context(3, 0.3))
    assert machine.target_state == "motion"
