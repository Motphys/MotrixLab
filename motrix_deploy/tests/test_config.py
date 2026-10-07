# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed deployment application configuration tests."""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from motrix_deploy.config import DeployRunConfig


def _run_mapping() -> dict[str, object]:
    return {
        "artifact": "artifact.deploy",
        "runtime": {
            "kind": "simulation",
            "backend": "test",
            "option": 1,
            "viewer": False,
            "realtime": None,
        },
        "duration_s": 0.1,
    }


def _typed_config(values: dict[str, object]) -> DeployRunConfig:
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DeployRunConfig), values))
    assert isinstance(config, DeployRunConfig)
    return config


def test_deploy_run_config_parses_resolved_application_tree() -> None:
    config = _typed_config(_run_mapping())

    assert config.artifact == "artifact.deploy"
    assert config.runtime == _run_mapping()["runtime"]
    assert config.backend_name == "test"
    assert config.runtime_options == {"option": 1}
    assert config.duration_s == pytest.approx(0.1)
    assert config.command is None


def test_deploy_run_config_defaults_to_unbounded_duration() -> None:
    values = _run_mapping()
    values.pop("duration_s")

    config = _typed_config(values)

    assert config.duration_s is None


def test_deploy_run_config_parses_hardware_runtime() -> None:
    runtime = {"kind": "hardware", "backend": "test", "network_interface": "enp3s0"}

    config = _typed_config({"artifact": "artifact.deploy", "runtime": runtime})

    assert config.runtime == runtime
    assert config.backend_name == "test"
    assert config.runtime_options == {"network_interface": "enp3s0"}
    assert config.duration_s is None
    assert config.command is None


def test_deploy_run_config_rejects_unknown_top_level_fields() -> None:
    values = _run_mapping()
    values["unknown"] = True

    with pytest.raises(OmegaConfBaseException, match="unknown"):
        _typed_config(values)
