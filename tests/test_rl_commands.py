# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Installed command composition from project directories, without simulation."""

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


def _task(config_dir: Path, seed: int) -> None:
    task_dir = config_dir / "task" / "external"
    task_dir.mkdir(parents=True)
    (task_dir / "ppo.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n  - /algo_base@algo: skrl.ppo\n  - _self_\n"
        "task:\n  env: external\n  rllib: skrl\n  algo: ppo\n"
        f"num_envs: 2\nplay_num_envs: 1\nseed: {seed}\nenv_modules: []\n",
        encoding="utf-8",
    )


def _compose(cwd: Path, *args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "motrix_rl.commands.train", "task=external/ppo", "--cfg", "job", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return yaml.safe_load(result.stdout)


def test_project_task_composes_with_packaged_algorithm_and_explicit_override(tmp_path):
    _task(tmp_path / "configs", seed=13)
    _task(tmp_path / "other", seed=29)

    local = _compose(tmp_path)
    explicit = _compose(tmp_path, "--config-dir", str(tmp_path / "other"))

    assert local["task"]["env"] == "external"
    assert local["env_modules"] == []
    assert local["algo"]["agent"]
    assert local["seed"] == 13
    assert explicit["seed"] == 29


def test_project_algorithm_defaults_override_package_and_explicit_dir_wins(tmp_path):
    base = yaml.safe_load(files("motrix_rl.configs").joinpath("algo_base/skrl.ppo.yaml").read_text())
    for directory, rate in ((tmp_path / "configs", 0.0123), (tmp_path / "other", 0.0456)):
        _task(directory, seed=1)
        algo_dir = directory / "algo_base"
        algo_dir.mkdir()
        base["agent"]["learning_rate"] = rate
        (algo_dir / "skrl.ppo.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")

    assert _compose(tmp_path)["algo"]["agent"]["learning_rate"] == 0.0123
    for args in (("--config-dir", str(tmp_path / "other")), (f"--config-dir={tmp_path / 'other'}",)):
        assert _compose(tmp_path, *args)["algo"]["agent"]["learning_rate"] == 0.0456


def test_train_help_does_not_require_task_but_training_does(tmp_path):
    command = [sys.executable, "-m", "motrix_rl.commands.train"]
    help_result = subprocess.run([*command, "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert help_result.returncode == 0, help_result.stderr
    assert "env_modules" in help_result.stdout

    train_result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert train_result.returncode != 0
    assert "MissingMandatoryValue" in train_result.stderr
    assert "task" in train_result.stderr


@pytest.mark.parametrize("format_name,validation", [("onnx", "parity"), ("deploy", "validation")])
def test_export_help_and_project_config_override(tmp_path, format_name, validation):
    command = [sys.executable, "-m", "motrix_rl.commands.export", format_name]
    help_result = subprocess.run([*command, "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert help_result.returncode == 0, help_result.stderr
    assert "run_dir" in help_result.stdout

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / f"export_{format_name}.yaml").write_text(
        f"defaults:\n  - export_{format_name}_schema\n  - _self_\nrun_dir: example\n{validation}:\n  samples: 7\n",
        encoding="utf-8",
    )
    result = subprocess.run([*command, "--cfg", "job"], cwd=tmp_path, capture_output=True, text=True, check=True)
    config = yaml.safe_load(result.stdout)
    assert config[validation]["samples"] == 7
    assert config["run_dir"] == "example"


def test_play_composes_without_project_config_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "motrix_rl.commands.play", "policy=/tmp/policy.pt", "--cfg", "job"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert yaml.safe_load(result.stdout)["policy"] == "/tmp/policy.pt"
