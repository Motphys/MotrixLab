# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Package boundary and command-line smoke tests."""

import subprocess
import sys

import pytest


@pytest.mark.integration
def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "motrix_deploy.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "artifact: ???" in result.stdout
    assert "Powered by Hydra" in result.stdout


@pytest.mark.integration
def test_import_does_not_load_optional_or_training_packages() -> None:
    forbidden = (
        "motrix_deploy_mujoco",
        "motrix_deploy_tasks",
        "motrix_env_core",
        "motrix_envs",
        "motrix_rl",
        "mujoco",
        "onnxruntime",
        "rsl_rl",
    )
    script = (
        "import sys; import motrix_deploy; "
        f"forbidden = {forbidden!r}; "
        "loaded = sorted(name for name in forbidden if name in sys.modules); "
        "print(','.join(loaded)); sys.exit(1 if loaded else 0)"
    )

    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout
