# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.sim import (
    ActuatorKdQuery,
    ActuatorKpQuery,
    BodyCenterOfMassQuery,
    BodyJointPositionLimitsQuery,
    BodyMassQuery,
    DofPositionLimitsQuery,
    GeomFrictionQuery,
    GeomSpecsQuery,
    SimModelCompiler,
)
from motrix_env_core.sim.model import SimModel


class _DispatchCompiler(SimModelCompiler):
    """Record the typed hook every query dispatches to."""

    def __init__(self) -> None:
        self.dispatched: dict[str, str] = {}

    def _begin_compile(self) -> None:
        self.dispatched = {}

    def _build_model(self, scene) -> SimModel:
        return SimModel(actuators=(), init_dof_pos=np.zeros(0, dtype=np.float32), others=dict(self.dispatched))

    def compile_geom_specs(self, key, geom_names) -> None:
        del geom_names
        self.dispatched[key] = "geoms"

    def compile_body_joint_position_limits(self, key, body) -> None:
        del body
        self.dispatched[key] = "body_limits"

    def compile_dof_position_limits(self, key) -> None:
        self.dispatched[key] = "dof_limits"

    def compile_actuator_kp(self, key, actuator_names) -> None:
        del actuator_names
        self.dispatched[key] = "kp"

    def compile_actuator_kd(self, key, actuator_names) -> None:
        del actuator_names
        self.dispatched[key] = "kd"

    def compile_body_mass(self, key, body) -> None:
        del body
        self.dispatched[key] = "mass"

    def compile_body_center_of_mass(self, key, body) -> None:
        del body
        self.dispatched[key] = "com"

    def compile_geom_friction(self, key, geom) -> None:
        del geom
        self.dispatched[key] = "friction"


def test_model_queries_dispatch_to_typed_compiler_methods() -> None:
    compiler = _DispatchCompiler()
    queries = {
        "geoms": GeomSpecsQuery(names=("first", "second")),
        "body_limits": BodyJointPositionLimitsQuery(body="body"),
        "dof_limits": DofPositionLimitsQuery(),
        "kp": ActuatorKpQuery(names=("first", "second")),
        "kd": ActuatorKdQuery(names=None),
        "mass": BodyMassQuery(name="body"),
        "com": BodyCenterOfMassQuery(name="body"),
        "friction": GeomFrictionQuery(name="geom"),
    }

    model = compiler.compile(SceneCfg(), queries)

    assert model.others == {name: name for name in queries}
