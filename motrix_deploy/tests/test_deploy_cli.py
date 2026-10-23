# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment CLI configuration handoff without a policy engine or hardware SDK."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fake_robot import FakeRobotInterface
from task_specs import TestDeployTask as FixtureDeployTask

from motrix_deploy import cli
from motrix_deploy.config import DeployRunConfig, HardwareRuntimeCfg, SimRuntimeCfg
from motrix_deploy.runtime.hardware import HardwareRuntime


@pytest.fixture
def cli_artifact(monkeypatch, manifest_factory):
    artifact = SimpleNamespace(manifest=manifest_factory(), policy_path="unused.onnx")
    monkeypatch.setattr(cli, "read_artifact", lambda path: artifact)
    monkeypatch.setattr("motrix_deploy.policy.OnnxPolicy", Mock(return_value=object()))
    return artifact


def _runtime_config(kind):
    if kind == "simulation":
        return SimRuntimeCfg(backend="test", deploy_env_id="flat", robot_id="go2", viewer=False)
    return HardwareRuntimeCfg(backend="test")


@pytest.mark.parametrize("kind", ["simulation", "hardware"])
@pytest.mark.parametrize("settings", [{}, {"preparation": {"start_button": "X", "kp": [12.0, 15.0]}}])
def test_cli_configures_task_before_runtime_creation(monkeypatch, cli_artifact, capsys, kind, settings):
    """Both runtime routes must observe task-owned configuration before acquiring resources."""
    configured_tasks = []

    class ConfigurableTask(FixtureDeployTask):
        def __init__(self, *args, settings=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.execution_settings = settings
            configured_tasks.append(self)

    monkeypatch.setattr(
        "motrix_deploy.task.metadata.entry_points",
        lambda *, group: [SimpleNamespace(name=cli_artifact.manifest.task.name, load=lambda: ConfigurableTask)],
    )
    robot = FakeRobotInterface(cli_artifact.manifest.robot)
    runtime = HardwareRuntime(robot, realtime=False)

    def create_runtime(*args):
        assert len(configured_tasks) == 1
        assert configured_tasks[0].execution_settings == settings
        assert robot.events == []
        return runtime

    monkeypatch.setattr(cli, "create_hardware_runtime", create_runtime)
    monkeypatch.setattr(cli, "create_simulation_runtime", create_runtime)
    monkeypatch.setattr(SimRuntimeCfg, "build_runtime_config", lambda self: object())
    config = DeployRunConfig(
        artifact="unused.deploy", runtime=_runtime_config(kind), duration_s=0.02, task_settings=settings
    )

    assert cli.run(config) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["success"] is True
    assert runtime.session.controller is configured_tasks[0]
    assert configured_tasks[0].spec is cli_artifact.manifest.task
    assert robot.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize("kind", ["simulation", "hardware"])
def test_cli_rejected_execution_settings_do_not_create_runtime(monkeypatch, cli_artifact, kind):
    hardware_factory = Mock()
    simulation_factory = Mock()
    monkeypatch.setattr(cli, "create_hardware_runtime", hardware_factory)
    monkeypatch.setattr(cli, "create_simulation_runtime", simulation_factory)
    config = DeployRunConfig(
        artifact="unused.deploy",
        runtime=_runtime_config(kind),
        task_settings={"unsupported": True},
    )

    with pytest.raises(ValueError, match="does not support execution settings"):
        cli.run(config)

    hardware_factory.assert_not_called()
    simulation_factory.assert_not_called()


def test_cli_policy_load_failure_never_creates_hardware_runtime(monkeypatch, cli_artifact):
    factory = Mock()
    monkeypatch.setattr(cli, "create_hardware_runtime", factory)
    monkeypatch.setattr("motrix_deploy.policy.OnnxPolicy", Mock(side_effect=ValueError("invalid policy")))
    config = DeployRunConfig(artifact="unused.deploy", runtime=_runtime_config("hardware"))

    with pytest.raises(ValueError, match="invalid policy"):
        cli.run(config)

    factory.assert_not_called()


def test_cli_task_preparation_failure_never_runs_hardware_runtime(monkeypatch, cli_artifact):
    class RejectingTask(FixtureDeployTask):
        def prepare(self, **kwargs):
            raise ValueError("invalid task preparation")

    monkeypatch.setattr(
        "motrix_deploy.task.metadata.entry_points",
        lambda *, group: [SimpleNamespace(name=cli_artifact.manifest.task.name, load=lambda: RejectingTask)],
    )
    robot = FakeRobotInterface(cli_artifact.manifest.robot)
    runtime = HardwareRuntime(robot, realtime=False)
    run = Mock()
    monkeypatch.setattr(runtime, "run", run)
    monkeypatch.setattr(cli, "create_hardware_runtime", lambda *args: runtime)
    config = DeployRunConfig(artifact="unused.deploy", runtime=_runtime_config("hardware"))

    with pytest.raises(ValueError, match="invalid task preparation"):
        cli.run(config)

    run.assert_not_called()
    assert robot.events == []
    assert not runtime.opened
