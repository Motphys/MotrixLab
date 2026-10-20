# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from motrix_deploy.artifact.io import read_artifact
from motrix_deploy.robot.g1_standing import resolve_g1_standing_gains
from motrix_deploy.runtime.hot_swapping import RobotHotSwappingController
from motrix_deploy.runtime.preparation import JointServoPreparationState
from motrix_deploy.runtime.state_machine import ImmediateTransition
from motrix_deploy_tasks.programs.g1_hot_swapping import G1HotSwappingProgramCfg, create_program


def _config(**overrides):
    return G1HotSwappingProgramCfg(
        walk_artifact="artifacts/g1-flat.deploy",
        wbt_artifact="artifacts/g1-wbt-dance.deploy",
        **overrides,
    )


def test_g1_program_fixes_fixstand_gains_but_keeps_preparation_per_artifact():
    program = create_program(_config())
    order = program.robot.joint_names

    standing_kp, standing_kd = resolve_g1_standing_gains(order)
    fixstand = program.controller.states["fixstand"]
    np.testing.assert_array_equal(fixstand._kp, standing_kp)
    np.testing.assert_array_equal(fixstand._kd, standing_kd)

    # Preparation is a per-artifact stand-up recipe, not the fixed FixStand gains.
    spec = read_artifact("artifacts/g1-wbt-dance.deploy").manifest.task
    preparation = program.controller._skills["wbt"].preparation
    np.testing.assert_array_equal(preparation.kp, np.asarray(spec.preparation_kp, np.float32))
    np.testing.assert_array_equal(preparation.kd, np.asarray(spec.preparation_kd, np.float32))
    np.testing.assert_array_equal(preparation.joint_position, fixstand._target)
    assert preparation.ramp_duration_s == spec.preparation_ramp_duration_s


def test_g1_program_defaults_to_supported_simulation_startup_state():
    program = create_program(_config())
    assert program.controller.current_state == "fixstand"
    assert isinstance(program.controller, RobotHotSwappingController)
    assert isinstance(program.controller.states["skill_preparation"], JointServoPreparationState)


def test_g1_program_can_explicitly_use_passive_startup_for_hardware():
    program = create_program(_config(initial_state="passive"))
    assert program.controller.current_state == "passive"


def test_g1_program_routes_walk_through_shared_preparation_and_returns_wbt_to_walk():
    program = create_program(_config())

    assert ("fixstand", "wbt") not in program.controller._rules
    assert isinstance(program.controller._rules[("walk", "skill_preparation")](), ImmediateTransition)
    assert isinstance(program.controller._rules[("skill_preparation", "wbt")](), ImmediateTransition)
    transition = program.controller._rules[("wbt", "walk")]()
    assert isinstance(transition, ImmediateTransition)
    program.controller._pending_skill = "wbt"
    assert program.controller._completion_target("skill_preparation") == "wbt"
    assert program.controller._completion_target("wbt") == "walk"
    assert program.controller.pending_skill is None
