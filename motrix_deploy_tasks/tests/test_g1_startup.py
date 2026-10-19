# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned G1 hardware startup gates and policy handoff contracts."""

import numpy as np
import pytest
from test_g1_wbt_task import _instant, _motion, _state, _task

from motrix_deploy.contracts import RobotSpec
from motrix_deploy.errors import ControlFailure
from motrix_deploy.operator import ButtonGestureCfg, OperatorProfileCfg
from motrix_deploy.policy import NoOpPolicy
from motrix_deploy.robot.g1_standing import G1_29DOF_STANDING_KP
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy_tasks.tasks.g1_startup import G1StartupController, G1StartupSettings
from motrix_deploy_tasks.tasks.g1_walk import G1WalkDeployTask, G1WalkTaskSpec
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtDeployTask
from motrix_env_core.input import GamePadDevice, KeyboardDevice


class _Pad(GamePadDevice):
    def __init__(self):
        self.buttons = set()
        self.previous = set()
        self.polls = 0

    def poll(self):
        self.down = self.buttons - self.previous
        self.previous = self.buttons.copy()
        self.polls += 1

    def is_button_down(self, name):
        return name in self.down

    def is_button_pressing(self, name):
        return name in self.buttons

    def is_button_up(self, name):
        return False

    def axis_value(self, name):
        return 0.0


@pytest.mark.parametrize("button", [None, "", " ", 42])
def test_startup_rejects_invalid_button_names(button):
    with pytest.raises(ValueError, match="start_button"):
        G1StartupSettings(start_button=button)


def _robot():
    return RobotSpec(
        "pelvis",
        tuple(G1_29DOF_STANDING_KP),
        np.zeros(29, np.float32),
        np.full(29, -1, np.float32),
        np.ones(29, np.float32),
        np.full(29, 100, np.float32),
    )


def _context(tick, command=None):
    return ControlContext(tick, tick * 0.02, command, 0.02)


def _step(controller, state, tick):
    decision = controller.step(state, _context(tick))
    if decision.command is not None:
        state.joint_position[:] = decision.command.joint_position
    return decision


def test_startup_requires_start_ramp_then_new_policy_confirmation():
    pad = _Pad()
    controller = G1StartupController(_robot(), G1StartupSettings(default_transition_duration_s=0.04), pad, None)
    state = _state()
    state.joint_position[:] = 0.2
    controller.reset(state)
    passive = _step(controller, state, 0)
    assert not passive.complete and not np.any(passive.command.kp) and not np.any(passive.command.kd)
    pad.buttons = {"start", "A"}
    halfway = _step(controller, state, 1)
    np.testing.assert_allclose(halfway.command.joint_position, 0.1)
    assert not halfway.complete
    _step(controller, state, 2)
    assert controller.phase == "wait_policy"
    assert not _step(controller, state, 3).complete  # Holding early A cannot authorize takeover.
    pad.buttons = set()
    _step(controller, state, 4)
    pad.buttons = {"A"}
    assert _step(controller, state, 5).complete
    controller.reset(state)
    assert controller.phase == "wait_stand"
    assert not _step(controller, state, 6).complete


def test_startup_consumes_operator_intents_and_reset_discards_old_edges():
    profile = OperatorProfileCfg(
        intents={
            "request_stand": ButtonGestureCfg(held=["LT"], pressed="up"),
            "enable_policy": ButtonGestureCfg(held=["RB"], pressed="X"),
            "enter_passive": ButtonGestureCfg(held=["LT"], pressed="B"),
        }
    )
    pad = _Pad()
    controller = G1StartupController(_robot(), G1StartupSettings(default_transition_duration_s=0.02), pad, profile)
    state = _state()
    controller.reset(state)
    _step(controller, state, 0)
    pad.buttons = {"LT", "up"}
    _step(controller, state, 1)
    assert controller.phase == "wait_policy"
    pad.buttons = {"RB", "X"}
    assert _step(controller, state, 2).complete
    controller.reset(state)
    _step(controller, state, 3)
    assert controller.phase == "wait_stand"
    pad.buttons = {"LT", "B"}
    with pytest.raises(ControlFailure) as error:
        _step(controller, state, 4)
    assert error.value.reason == "passive_requested"


def _walk(*, settings=None):
    spec = G1WalkTaskSpec(
        action_scale=[0.5] * 29,
        kp=[10] * 29,
        kd=[1] * 29,
        action_lower=[-1] * 29,
        action_upper=[1] * 29,
        command_lower=[-1] * 3,
        command_upper=[1] * 3,
        base_angular_velocity_scale=1,
        joint_position_scale=1,
        joint_velocity_scale=1,
        gait_period_s=1,
        gait_phase_offsets_rad=[0, np.pi],
        standing_command_threshold=0.1,
    )
    return G1WalkDeployTask(spec, _robot(), NoOpPolicy(29), steps=2, settings=settings)


def _wbt(*, settings=None):
    source = _task(controller=True, automatic=True, steps=2, preparation=_instant)
    return G1WbtDeployTask(source.spec, _robot(), NoOpPolicy(29), steps=2, motion=_motion(), settings=settings)


@pytest.mark.parametrize("factory", [_walk, _wbt])
def test_hardware_task_does_not_run_policy_before_confirmation_and_can_stop(factory):
    task = factory(settings={"preparation": {"default_transition_duration_s": 0.02}})
    pad = _Pad()
    binding = task.prepare(simulation=False, device=pad)
    state = _state()
    task.reset(state)
    command = binding.read_command() if binding is not None else None
    waiting = task.step(state, _context(0, command))
    assert not waiting.policy_tick and not np.any(waiting.command.kp)
    task.request_stop()
    stopped = task.step(state, _context(1, command))
    assert stopped.complete and not stopped.success and stopped.command is None
    task.reset(state)
    waiting = task.step(state, _context(2, command))
    assert not waiting.complete and not waiting.policy_tick


def test_wbt_automatic_playback_waits_for_hardware_gates_then_keeps_held_reference_interval():
    task = _wbt(settings={"preparation": {"default_transition_duration_s": 0.02}})
    pad = _Pad()
    task.prepare(simulation=False, device=pad)
    state = _state()
    task.reset(state)
    _step(task, state, 0)
    pad.buttons = {"start"}
    assert not _step(task, state, 1).policy_tick
    pad.buttons = set()
    for tick in range(2, 10):
        assert not _step(task, state, tick).policy_tick
    pad.buttons = {"A"}
    assert not _step(task, state, 10).policy_tick  # Task-specific readiness still runs.
    hold = _step(task, state, 11)
    assert hold.policy_tick and task.phase == "policy_hold"
    assert task.playback_steps == 0
    playback = _step(task, state, 12)
    assert playback.policy_tick and task.phase == "playback"
    assert task.playback_steps == 1


@pytest.mark.parametrize("factory", [_walk, _wbt])
def test_headless_simulation_runs_shared_preparation_before_policy(factory):
    task = factory(settings={"preparation": {"default_transition_duration_s": 0.04}})
    binding = task.prepare(simulation=True)
    state = _state()
    task.reset(state)
    for tick in range(10):
        command = binding.read_command() if binding is not None else None
        decision = task.step(state, _context(tick, command))
        if tick == 0:
            assert not decision.policy_tick
        if decision.policy_tick:
            break
    assert decision.policy_tick
    assert task.policy_controller.completed_steps == 1


class _Keyboard(KeyboardDevice):
    def __init__(self):
        self.keys = set()
        self.previous = set()
        self.polls = 0

    def poll(self):
        self.down = self.keys - self.previous
        self.previous = self.keys.copy()
        self.polls += 1

    def is_key_down(self, key):
        return key in self.down

    def is_key_up(self, key):
        return False

    def is_pressing(self, key):
        return key in self.keys


@pytest.mark.parametrize("factory", [_walk, _wbt])
def test_keyboard_simulation_holds_fix_stand_until_policy_confirmation(factory):
    task = factory(settings={"preparation": {"default_transition_duration_s": 0.04}})
    keyboard = _Keyboard()
    binding = task.prepare(simulation=True, device=keyboard)
    state = _state()
    task.reset(state)
    for tick, keys in enumerate((set(), set(), {"r"}, set(), set(), {"p"})):
        keyboard.keys = keys
        command = binding.read_command() if binding is not None else None
        decision = task.step(state, _context(tick, command))
        assert keyboard.polls == tick + 1
        if tick < 5:
            assert not decision.policy_tick
            np.testing.assert_array_equal(decision.command.joint_position, task.robot.default_joint_position)
            assert np.any(decision.command.kp > 0)
            assert np.any(decision.command.kd > 0)
    if isinstance(task, G1WalkDeployTask):
        assert decision.policy_tick
    else:
        assert not decision.policy_tick
