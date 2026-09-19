# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed SimWrite dispatch at the backend-neutral simulator boundary."""

import numpy as np

from motrix_env_core.sim.write import (
    ActuatorDampingWrite,
    ActuatorKpWrite,
    BodyAngularVelocityWrite,
    BodyComWrite,
    BodyJointPositionWrite,
    BodyJointVelocityWrite,
    BodyLinearVelocityWrite,
    BodyMassWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    CtrlTargetsWrite,
    GeomFrictionWrite,
    JointAngularVelocityWrite,
    JointPositionWrite,
    JointQuaternionWrite,
    JointVelocityWrite,
    KinematicBodyPositionWrite,
    KinematicBodyRotationWrite,
    SimWriteCompiler,
    WriteProgram,
)


class _RecordingProgram(WriteProgram):
    def buffer(self, name: str) -> np.ndarray:
        raise NotImplementedError

    def execute(self, env_ids: np.ndarray | None = None) -> None:
        raise NotImplementedError


class _DispatchCompiler(SimWriteCompiler):
    """Record the typed hook every write dispatches to."""

    def __init__(self) -> None:
        self.dispatched: list[str] = []

    def _begin_compile(self) -> None:
        self.dispatched = []

    def _build_program(self, *, reset: bool, forward_kinematics: bool) -> WriteProgram:
        del reset, forward_kinematics
        return _RecordingProgram()

    def compile_body_joint_position(self, name, write) -> None:
        del name, write
        self.dispatched.append("body_dof_position")

    def compile_body_joint_velocity(self, name, write) -> None:
        del name, write
        self.dispatched.append("body_dof_velocity")

    def compile_joint_position(self, name, write) -> None:
        del name, write
        self.dispatched.append("joint_position")

    def compile_joint_velocity(self, name, write) -> None:
        del name, write
        self.dispatched.append("joint_velocity")

    def compile_joint_quaternion(self, name, write) -> None:
        del name, write
        self.dispatched.append("joint_quaternion")

    def compile_joint_angular_velocity(self, name, write) -> None:
        del name, write
        self.dispatched.append("joint_angular_velocity")

    def compile_ctrl_targets(self, name, write) -> None:
        del name, write
        self.dispatched.append("ctrl")

    def compile_body_position(self, name, write) -> None:
        del name, write
        self.dispatched.append("body_position")

    def compile_body_rotation(self, name, write) -> None:
        del name, write
        self.dispatched.append("body_rotation")

    def compile_body_linear_velocity(self, name, write) -> None:
        del name, write
        self.dispatched.append("body_linear_velocity")

    def compile_body_angular_velocity(self, name, write) -> None:
        del name, write
        self.dispatched.append("body_angular_velocity")

    def compile_kinematic_body_position(self, name, write) -> None:
        del name, write
        self.dispatched.append("kinematic_body_position")

    def compile_kinematic_body_rotation(self, name, write) -> None:
        del name, write
        self.dispatched.append("kinematic_body_rotation")

    def compile_actuator_kp(self, name, write) -> None:
        del name, write
        self.dispatched.append("kp")

    def compile_actuator_damping(self, name, write) -> None:
        del name, write
        self.dispatched.append("damping")

    def compile_body_mass(self, name, write) -> None:
        del name, write
        self.dispatched.append("mass")

    def compile_body_com(self, name, write) -> None:
        del name, write
        self.dispatched.append("com")

    def compile_geom_friction(self, name, write) -> None:
        del name, write
        self.dispatched.append("friction")


def test_sim_write_compiler_dispatches_each_write_to_its_typed_compiler() -> None:
    compiler = _DispatchCompiler()

    BodyJointPositionWrite("body").compile_with(compiler, "write")
    BodyJointVelocityWrite("body").compile_with(compiler, "write")
    JointPositionWrite(("joint",)).compile_with(compiler, "write")
    JointVelocityWrite(("joint",)).compile_with(compiler, "write")
    JointQuaternionWrite(("joint",)).compile_with(compiler, "write")
    JointAngularVelocityWrite(("joint",)).compile_with(compiler, "write")
    CtrlTargetsWrite().compile_with(compiler, "write")
    BodyPositionWrite(("body",)).compile_with(compiler, "write")
    BodyRotationWrite(("body",)).compile_with(compiler, "write")
    BodyLinearVelocityWrite(("body",)).compile_with(compiler, "write")
    BodyAngularVelocityWrite(("body",)).compile_with(compiler, "write")
    KinematicBodyPositionWrite(("body",)).compile_with(compiler, "write")
    KinematicBodyRotationWrite(("body",)).compile_with(compiler, "write")
    ActuatorKpWrite(("actuator",)).compile_with(compiler, "write")
    ActuatorDampingWrite(("actuator",)).compile_with(compiler, "write")
    BodyMassWrite(("link",)).compile_with(compiler, "write")
    BodyComWrite(("link",)).compile_with(compiler, "write")
    GeomFrictionWrite(("geom",)).compile_with(compiler, "write")

    assert compiler.dispatched == [
        "body_dof_position",
        "body_dof_velocity",
        "joint_position",
        "joint_velocity",
        "joint_quaternion",
        "joint_angular_velocity",
        "ctrl",
        "body_position",
        "body_rotation",
        "body_linear_velocity",
        "body_angular_velocity",
        "kinematic_body_position",
        "kinematic_body_rotation",
        "kp",
        "damping",
        "mass",
        "com",
        "friction",
    ]


def test_compile_dispatches_every_write_in_order_and_builds_one_program() -> None:
    compiler = _DispatchCompiler()

    program = compiler.compile(
        {"a": JointQuaternionWrite(("joint",)), "b": CtrlTargetsWrite(), "c": KinematicBodyRotationWrite(("body",))},
        reset=True,
        forward_kinematics=False,
    )

    assert isinstance(program, WriteProgram)
    assert compiler.dispatched == ["joint_quaternion", "ctrl", "kinematic_body_rotation"]
