# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Package boundary and command-line smoke tests."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import motrix_deploy

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_package_version() -> None:
    """The boundary version mirrors the release version declared in pyproject.toml."""
    declared = re.search(r'^version = "(.+)"$', _PYPROJECT.read_text(), re.MULTILINE)
    assert declared is not None, f"Missing version in {_PYPROJECT}"
    assert motrix_deploy.__version__ == declared.group(1)


@pytest.mark.integration
def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "motrixlab.cli", "deploy", "--help"],
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
