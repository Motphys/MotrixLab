# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Installed task discovery and deployment CLI configuration tests."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import pytest
from hydra import compose, initialize_config_dir, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.object_type import ObjectType
from motrix_robots.unitree import UnitreeGo2Robot
from onnx import TensorProto, helper, numpy_helper

from motrix_deploy.artifact import (
    ControlSpec,
    DeploymentManifest,
    PolicySpec,
    SourceSpec,
    sha256_bytes,
    write_artifact,
)
from motrix_deploy.artifact.schema import PayloadSpec
from motrix_deploy.contracts import RobotSpec, TensorSpec
from motrix_deploy.env import assemble_deploy_scene, create_deploy_env
from motrix_deploy_tasks.tasks.go2_walk import Go2WalkTaskSpec
from motrix_env_core.config.scene import FlatTerrainCfg, HFieldTerrainCfg, SceneCfg


def _rollout_result(stdout: str) -> dict[str, object]:
    json_start = stdout.find("{")
    assert json_start >= 0, stdout
    return json.loads(stdout[json_start:])


@pytest.mark.parametrize("args", [[], ["--help"]])
def test_cli_general_help(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "motrix_deploy.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "artifact: ???" in result.stdout
    assert "runtime: ???" in result.stdout


def test_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from motrix_deploy.cli import main; raise SystemExit(main())",
            "task=go2-walk-rough/sim",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "artifact: ???" in result.stdout
    assert "deploy_env_id: rough" in result.stdout


def test_hardware_cli_help_selects_hardware_recipe() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from motrix_deploy.cli import main; raise SystemExit(main())",
            "task=go2-walk-flat/hardware",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "backend: unitree_go2" in result.stdout
    assert "kind: hardware" in result.stdout


@pytest.mark.parametrize(
    ("task", "runtime_kind", "backend"),
    [
        ("go2-walk-flat/sim", "simulation", "mujoco"),
        ("go2-walk-rough/sim", "simulation", "mujoco"),
        ("go2-walk-flat/hardware", "hardware", "unitree_go2"),
    ],
)
def test_installed_recipes_compose_outside_workspace(tmp_path, task, runtime_kind, backend) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "motrix_deploy.cli", f"task={task}", "--cfg", "job"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"backend: {backend}" in result.stdout
    assert "artifact: ???" in result.stdout
    assert f"kind: {runtime_kind}" in result.stdout
    if runtime_kind == "simulation":
        assert "sensor_bindings:" in result.stdout
        assert "viewer: true" in result.stdout
        assert "realtime: null" in result.stdout


def test_deployment_plugin_preserves_other_app_task_discovery(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    task_dir = config_dir / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "custom.yaml").write_text("value: custom\n")
    (config_dir / "app.yaml").write_text("defaults:\n  - task: custom\n")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        loader = GlobalHydra.instance().config_loader()
        assert loader.get_group_options("task", results_filter=ObjectType.CONFIG) == ["custom"]
        cfg = compose(config_name="app")
    assert cfg.task.value == "custom"


def test_installed_go2_task_is_available_without_preimport() -> None:
    script = "from motrix_deploy.task import available_tasks; assert 'go2_walk/v1' in available_tasks()"

    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("task", "deploy_env_id", "terrain_type"),
    [
        ("go2-walk-rough/sim", "rough", HFieldTerrainCfg),
        ("go2-walk-flat/sim", "flat", FlatTerrainCfg),
    ],
)
def test_mujoco_hydra_recipes_assemble_robot_and_selected_world(
    task: str,
    deploy_env_id: str,
    terrain_type: type,
) -> None:
    with initialize_config_module(version_base=None, config_module="motrix_deploy.config"):
        cfg = compose(config_name="deploy", overrides=[f"task={task}", "artifact=artifact.deploy"])

    assert cfg.runtime.kind == "simulation"

    scene = assemble_deploy_scene(cfg.runtime.deploy_env_id, cfg.runtime.robot_id)
    assert isinstance(scene, SceneCfg)
    assert isinstance(scene.objs.robot, UnitreeGo2Robot)
    assert isinstance(scene.objs.floor, terrain_type)
    assert cfg.runtime.deploy_env_id == deploy_env_id
    assert cfg.runtime.robot_id == "go2"
    assert cfg.artifact == "artifact.deploy"
    assert cfg.runtime.backend == "mujoco"
    assert cfg.runtime.physics.dt > 0
    assert cfg.runtime.physics.solver_iterations > 0
    assert len(cfg.runtime.robot_translation) == 3
    assert np.linalg.norm(cfg.runtime.robot_rotation) == pytest.approx(1.0)
    assert cfg.runtime.viewer is True
    assert cfg.runtime.realtime is None
    assert cfg.duration_s is None


@pytest.mark.parametrize("deploy_env_id", ["flat", "rough"])
def test_installed_world_factories_are_robot_free_and_fresh(deploy_env_id: str) -> None:
    first = create_deploy_env(deploy_env_id)
    second = create_deploy_env(deploy_env_id)
    assert isinstance(first, SceneCfg)
    assert first.objs.robot is None
    assert second.objs.robot is None
    assert first.objs.floor is not second.objs.floor
    first.objs.robot = UnitreeGo2Robot()
    assert second.objs.robot is None
    if isinstance(first.objs.floor, HFieldTerrainCfg):
        terrain = first.assets[first.objs.floor.hfield]
        other = second.assets[second.objs.floor.hfield]
        heights = terrain.generator.generate(terrain.size, terrain.shape)
        np.testing.assert_array_equal(heights, other.generator.generate(other.size, other.shape))
        assert np.ptp(heights) > 0


def test_installed_go2_task_can_be_created_without_preimport(tmp_path: Path) -> None:
    specs_path = tmp_path / "specs.json"
    robot = {
        key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in asdict(_robot_spec()).items()
    }
    specs_path.write_text(json.dumps({"task": _task_spec().to_dict(), "robot": robot}))
    script = (
        "import json, sys; import numpy as np; "
        "from motrix_deploy.artifact import TaskSpec; "
        "from motrix_deploy.contracts import RobotSpec; "
        "from motrix_deploy.task import create_task; "
        "from motrix_deploy.policy import NoOpPolicy; "
        "specs = json.load(open(sys.argv[1])); "
        "specs['robot']['joint_names'] = tuple(specs['robot']['joint_names']); "
        "specs['robot'].update({key: np.asarray(value, dtype=np.float32) "
        "for key, value in specs['robot'].items() if isinstance(value, list)}); "
        "spec = TaskSpec.from_dict(specs['task']); "
        "assert type(spec).__name__ == 'Go2WalkTaskSpec'; "
        "assert type(spec).__module__ == 'motrix_deploy_tasks.tasks.go2_walk'; "
        "assert not any(name in sys.modules for name in ('motrix_envs', 'motrixsim', 'mujoco', 'torch')); "
        "task = create_task(spec, RobotSpec(**specs['robot']), NoOpPolicy(12)); "
        "assert type(task).__module__ == 'motrix_deploy_tasks.tasks.go2_walk'; "
        "assert type(task).__name__ == 'Go2WalkDeployTask'"
    )

    result = subprocess.run(
        [sys.executable, "-c", script, str(specs_path)], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_hardware_hydra_config_selects_runtime_and_command_defaults() -> None:
    with initialize_config_module(version_base=None, config_module="motrix_deploy.config"):
        cfg = compose(
            config_name="deploy",
            overrides=[
                "task=go2-walk-flat/hardware",
                "artifact=artifact.deploy",
                "runtime.network_interface=enp3s0",
            ],
        )

    assert cfg.runtime.kind == "hardware"
    assert cfg.runtime.backend == "unitree_go2"
    assert cfg.duration_s is not None and cfg.duration_s > 0
    assert cfg.command.source == "gamepad"
    assert cfg.runtime.lie_down_button == "B"
    assert cfg.command.gamepad.deadman_button == "L1"
    assert cfg.command.gamepad.invert_linear_x is False
    assert cfg.command.gamepad.invert_linear_y is True
    assert cfg.command.gamepad.invert_yaw is True


def test_hardware_hydra_config_accepts_explicit_gain_overrides() -> None:
    with initialize_config_module(version_base=None, config_module="motrix_deploy.config"):
        cfg = compose(
            config_name="deploy",
            overrides=[
                "task=go2-walk-flat/hardware",
                "artifact=artifact.deploy",
                "runtime.kp=25.0",
                "runtime.kd=[0.4,0.4,0.4,0.5,0.5,0.5,0.6,0.6,0.6,0.7,0.7,0.7]",
            ],
        )

    assert cfg.runtime.kp == pytest.approx(25.0)
    assert list(cfg.runtime.kd) == pytest.approx([0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7])


@pytest.mark.parametrize("backend", ["mujoco", "motrixsim"])
@pytest.mark.parametrize("task", ["go2-walk-flat", "go2-walk-rough"])
def test_headless_simulation_cli_runs_deterministic_onnx_fixture(tmp_path: Path, task: str, backend: str) -> None:
    policy_path = tmp_path / "fixture.onnx"
    weight = numpy_helper.from_array(np.zeros((49, 12), dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weight"], ["actions"])],
        "zero_go2_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 49])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 12])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, policy_path)
    policy_bytes = policy_path.read_bytes()
    manifest = DeploymentManifest(
        schema_version="motrix-deploy/v1",
        source=SourceSpec(framework="test", run_id="mujoco-smoke", checkpoint="fixture"),
        policy=PolicySpec(
            component_version="onnx/v1",
            payload_path="policy/model.onnx",
            sha256=sha256_bytes(policy_bytes),
            input=TensorSpec(name="obs", shape=(1, 49)),
            output=TensorSpec(name="actions", shape=(1, 12)),
        ),
        robot=_robot_spec(),
        task=_task_spec(),
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
    )
    artifact_path = tmp_path / "fixture.deploy"
    write_artifact(artifact_path, manifest, {"policy/model.onnx": policy_bytes})
    command = [
        sys.executable,
        "-c",
        "from motrix_deploy.cli import main; raise SystemExit(main())",
        f"task={task}/sim",
        f"runtime.backend={backend}",
        f"artifact={artifact_path}",
        "command.velocity=[0.5,0.0,0.0]",
        "duration_s=0.1",
        "runtime.viewer=false",
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    first_result = _rollout_result(first.stdout)
    assert first_result["success"] is True
    assert first_result["completed_steps"] == 5

    # Both worlds smoke-test each backend; repeat only one world per backend.
    if task == "go2-walk-flat":
        second = subprocess.run(command, check=False, capture_output=True, text=True)
        assert second.returncode == 0, second.stderr
        assert first_result["trace_sha256"] == _rollout_result(second.stdout)["trace_sha256"]

    # Duration validation is shared, so cover it once rather than per world/backend.
    if task == "go2-walk-flat" and backend == "mujoco":
        for invalid_duration in ("0", "-0.1", "nan", "inf"):
            invalid_command = command.copy()
            invalid_command[invalid_command.index("duration_s=0.1")] = f"duration_s={invalid_duration}"
            invalid_bound = subprocess.run(
                invalid_command,
                check=False,
                capture_output=True,
                text=True,
            )
            assert invalid_bound.returncode == 2, invalid_bound.stderr
            assert "duration_s" in invalid_bound.stderr


@pytest.mark.parametrize("backend,solver_iterations", [("mujoco", 100), ("motrixsim", 3)])
def test_standard_cli_completes_four_phase_flow(tmp_path, backend, solver_iterations):
    """Standing preparation, frozen-reference hold, then the requested playback budget."""
    pytest.importorskip(backend)
    pytest.importorskip("onnxruntime")
    from motrix_envs.deploy.g1_wbt import build_g1_wbt_profile

    profile = build_g1_wbt_profile("g1-wbt-dance")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weight"], ["actions"])],
        "zero_wbt_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 154])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 29])],
        [numpy_helper.from_array(np.zeros((154, 29), dtype=np.float32), name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    payload = model.SerializeToString()
    manifest = DeploymentManifest(
        schema_version="motrix-deploy/v1",
        source=SourceSpec("test", "wbt-smoke", "fixture"),
        policy=PolicySpec(
            "onnx/v1",
            "policy/model.onnx",
            sha256_bytes(payload),
            TensorSpec("obs", (1, 154)),
            TensorSpec("actions", (1, 29)),
        ),
        robot=profile.robot,
        task=profile.task,
        control=profile.control,
        payloads=tuple(PayloadSpec(path, sha256_bytes(data)) for path, data in sorted(profile.payloads.items())),
    )
    artifact = tmp_path / "wbt.deploy"
    write_artifact(artifact, manifest, {"policy/model.onnx": payload, **profile.payloads})
    command = [
        sys.executable,
        "-c",
        "from motrix_deploy.cli import main; main()",
        "task=g1-wbt-dance/sim",
        f"artifact={artifact}",
        "runtime.viewer=false",
        f"runtime.backend={backend}",
        f"runtime.physics.solver_iterations={solver_iterations}",
        # The default zero hold plus the task's mandatory held-reference interval
        # make the takeover-to-playback distance deterministic.
        "duration_s=0.04",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    report = _rollout_result(result.stdout)
    assert report["success"]
    assert report["exit_reason"] == "completed"
    # One mandatory hold interval plus the two-tick playback budget.
    assert report["completed_steps"] == 3
    assert report["policy_simulation_time_s"] == pytest.approx(3 * profile.control.period_s)


def _robot_spec() -> RobotSpec:
    return RobotSpec(
        base_link_name="base",
        joint_names=(
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ),
        default_joint_position=np.array(
            [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
            dtype=np.float32,
        ),
        position_lower=np.tile(np.array([-0.9472, -1.4, -2.6227], dtype=np.float32), 4),
        position_upper=np.tile(np.array([0.9472, 2.5, -0.84776], dtype=np.float32), 4),
        torque_limit=np.full(12, 24.0, dtype=np.float32),
    )


def _task_spec() -> Go2WalkTaskSpec:
    return Go2WalkTaskSpec(
        action_scale=[0.25] * 12,
        command_lower=[0.5, 0.0, 0.0],
        command_upper=[0.5, 0.0, 0.0],
        command_scale=[1.0, 1.0, 1.0],
        feet_phase_offsets=[0.0, 0.5, 0.5, 0.0],
        gait_frequency_hz=2.0,
        standing_threshold=0.05,
        termination_min_up_z=0.25,
        kp=[35.0] * 12,
        kd=[0.5] * 12,
        action_lower=[-1.0] * 12,
        action_upper=[1.0] * 12,
    )
