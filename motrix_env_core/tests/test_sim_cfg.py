# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest

from motrix_env_core.base import EnvCfg
from motrix_env_core.config.scene import SceneCfg, SceneCompiler
from motrix_env_core.config.sim import SimCfg


def test_scene_compiler_is_an_abstract_backend_boundary():
    class _DummySceneCompiler(SceneCompiler):
        def compile(self, scene, sim):
            raise NotImplementedError

    with pytest.raises(TypeError):
        SceneCompiler()

    assert isinstance(_DummySceneCompiler(), SceneCompiler)


def test_env_cfg_uses_sim_config_dt_for_substeps_and_validation():
    cfg = EnvCfg(scene=SceneCfg(), sim=SimCfg(dt=0.005), ctrl_dt=0.02)

    cfg.validate()
    assert cfg.sim_substeps == 4

    with pytest.raises(ValueError, match="sim.dt must be less than or equal to ctrl_dt"):
        EnvCfg(scene=SceneCfg(), sim=SimCfg(dt=0.02), ctrl_dt=0.01).validate()


def test_env_cfg_requires_scene_config():
    with pytest.raises(ValueError, match="EnvCfg.scene must be configured"):
        EnvCfg().validate()


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf"), -float("inf"), True, False])
def test_sim_cfg_rejects_invalid_timestep(dt):
    with pytest.raises(ValueError, match=r"sim\.dt must be positive and finite"):
        SimCfg(dt=dt).validate()


@pytest.mark.parametrize("iterations", [0, -1, 1.5, 1.0, True, False, "100"])
def test_sim_cfg_requires_positive_integer_solver_iterations(iterations):
    with pytest.raises(ValueError, match=r"sim\.solver_iterations must be a positive integer"):
        SimCfg(solver_iterations=iterations).validate()


@pytest.mark.parametrize("tolerance", [0.0, -0.1, float("nan"), float("inf"), -float("inf")])
def test_sim_cfg_requires_positive_finite_solver_tolerance(tolerance):
    with pytest.raises(ValueError, match=r"sim\.solver_tolerance must be positive and finite"):
        SimCfg(solver_tolerance=tolerance).validate()


def test_sim_cfg_supports_backend_defaults_and_explicit_physics():
    # None preserves model/backend solver and gravity defaults.
    SimCfg(solver_iterations=None, solver_tolerance=None, gravity=None).validate()
    SimCfg(dt=1, solver_iterations=10, solver_tolerance=1e-6, gravity=(0.0, 0.0, -9.81)).validate()


def test_sim_cfg_retains_gravity_length_validation():
    with pytest.raises(ValueError, match=r"sim\.gravity must contain 3 values"):
        SimCfg(gravity=(0.0, -9.81)).validate()
