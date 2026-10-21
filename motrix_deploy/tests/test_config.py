# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed deployment application configuration tests."""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from motrix_deploy.config import DeployRunConfig, HardwareRuntimeCfg, SimRuntimeCfg


def _sim_runtime(overrides: dict[str, object] | None = None) -> object:
    values = {"backend": "test", "deploy_env_id": "flat", "robot_id": "go2"}
    values.update(overrides or {})
    return OmegaConf.merge(OmegaConf.structured(SimRuntimeCfg), values)


def _typed_config(values: dict[str, object]) -> DeployRunConfig:
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DeployRunConfig), values))
    assert isinstance(config, DeployRunConfig)
    return config


def test_deploy_run_config_decodes_typed_simulation_runtime() -> None:
    config = _typed_config(
        {
            "artifact": "artifact.deploy",
            "runtime": _sim_runtime({"robot_init_key_pose": "lie_down", "viewer": False}),
            "duration_s": 0.1,
        }
    )

    assert isinstance(config.runtime, SimRuntimeCfg)
    assert config.runtime.backend == "test"
    assert config.runtime.viewer is False
    assert config.runtime.robot_init_key_pose == "lie_down"
    assert config.duration_s == pytest.approx(0.1)


def test_deploy_run_config_preserves_nested_task_settings_as_plain_dicts() -> None:
    settings = {"preparation": {"kp": [12.0, 15.0], "start_button": "X"}, "enabled": True}
    config = _typed_config({"artifact": "artifact.deploy", "runtime": _sim_runtime(), "task_settings": settings})

    assert type(config.task_settings) is dict
    assert type(config.task_settings["preparation"]) is dict
    assert config.task_settings == settings


def test_deploy_run_config_task_settings_defaults_are_independent() -> None:
    first = _typed_config({"artifact": "first.deploy", "runtime": _sim_runtime()})
    second = _typed_config({"artifact": "second.deploy", "runtime": _sim_runtime()})

    assert first.task_settings == second.task_settings == {}
    first.task_settings["preparation"] = {"start_button": "X"}
    assert second.task_settings == {}


def test_deploy_run_config_defaults_to_unbounded_duration() -> None:
    config = _typed_config({"artifact": "artifact.deploy", "runtime": _sim_runtime()})

    assert config.duration_s is None


@pytest.mark.parametrize("duration_s", [0, -0.1, float("nan"), float("inf")])
def test_cli_duration_rejects_nonpositive_or_nonfinite_budget(duration_s: float) -> None:
    from motrix_deploy.cli import _duration_steps

    with pytest.raises(ValueError, match="duration_s"):
        _duration_steps(duration_s, 0.02)


def test_sim_schema_rejects_unknown_runtime_fields() -> None:
    with pytest.raises(OmegaConfBaseException, match="robot_init_key_posee"):
        _sim_runtime({"robot_init_key_posee": "lie_down"})


def test_deploy_run_config_rejects_untyped_simulation_mapping() -> None:
    runtime = {"kind": "simulation", "backend": "test", "deploy_env_id": "flat", "robot_id": "go2"}

    with pytest.raises(ValueError, match="SimRuntimeCfg"):
        _typed_config({"artifact": "artifact.deploy", "runtime": runtime})


def test_deploy_run_config_parses_hardware_runtime() -> None:
    hardware = OmegaConf.merge(OmegaConf.structured(HardwareRuntimeCfg), {"backend": "test"})

    config = _typed_config({"artifact": "artifact.deploy", "runtime": hardware})

    assert isinstance(config.runtime, HardwareRuntimeCfg)
    assert config.runtime.backend == "test"
    assert config.duration_s is None


def test_deploy_run_config_rejects_untyped_hardware_mapping() -> None:
    runtime = {"kind": "hardware", "backend": "test", "network_interface": "enp3s0"}

    with pytest.raises(ValueError, match="HardwareRuntimeCfg"):
        _typed_config({"artifact": "artifact.deploy", "runtime": runtime})


def test_deploy_run_config_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(OmegaConfBaseException, match="unknown"):
        _typed_config({"artifact": "artifact.deploy", "runtime": _sim_runtime(), "unknown": True})
