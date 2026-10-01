# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Hydra configuration and composition tests for deployment export."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from motrix_rl.cli import to_typed_config
from motrix_rl.config import DeploymentExportConfig, OnnxParityConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(ROOT / "configs/deploy")
_SCRIPT_SPEC = importlib.util.spec_from_file_location("export_deploy_cli", ROOT / "scripts/export_deploy.py")
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
export_deploy = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(export_deploy)


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_export_config_composes_hydra_overrides() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="export",
            overrides=[
                "env=go2-walk-rough",
                "validation.samples=8",
            ],
        )

    typed = to_typed_config(cfg, DeploymentExportConfig)

    assert typed == DeploymentExportConfig(
        env="go2-walk-rough",
        run=None,
        output=None,
        validation=OnnxParityConfig(seed=1, samples=8, atol=1e-5, rtol=1e-5),
    )


def test_export_cli_forwards_typed_validation_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_path = tmp_path / "example.deploy"
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _export(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            artifact=SimpleNamespace(root=artifact_path),
            validation_samples=4,
            max_abs_error=2e-7,
        )

    monkeypatch.setattr(export_deploy, "export_deploy_run", _export)
    run_dir = Path("runs/example")
    monkeypatch.setattr(
        export_deploy,
        "open_run_context",
        lambda path: SimpleNamespace(run_dir=run_dir, metadata=SimpleNamespace(env_name="go2-walk-rough")),
    )
    cfg = DeploymentExportConfig(
        run=str(run_dir),
        output=str(artifact_path),
        validation=OnnxParityConfig(seed=7, samples=4, atol=2e-5, rtol=3e-5),
    )

    export_deploy.run(cfg)

    assert calls == [
        (
            (run_dir, artifact_path),
            {
                "profile_builder": export_deploy.build_deployment_profile,
                "validation_seed": 7,
                "validation_samples": 4,
                "atol": 2e-5,
                "rtol": 3e-5,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == {
        "artifact": str(artifact_path),
        "max_abs_error": 2e-7,
        "run": str(run_dir),
        "validation_samples": 4,
    }


def test_export_cli_selects_latest_environment_run_and_default_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = Path("runs/go2-walk-rough/latest")
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        export_deploy,
        "latest_metadata_run",
        lambda env: (run_dir, SimpleNamespace(env_name=env)),
    )
    monkeypatch.setattr(
        export_deploy,
        "export_deploy_run",
        lambda run, output, **kwargs: (
            calls.append((run, output))
            or SimpleNamespace(
                artifact=SimpleNamespace(root=output),
                validation_samples=32,
                max_abs_error=1e-7,
            )
        ),
    )

    export_deploy.run(DeploymentExportConfig(env="go2-walk-rough"))

    assert calls == [(run_dir, Path("artifacts/go2-walk-rough.deploy"))]


@pytest.mark.parametrize(
    "cfg",
    [
        DeploymentExportConfig(),
        DeploymentExportConfig(env="go2-walk-rough", run="runs/example"),
    ],
)
def test_export_cli_requires_exactly_one_run_selector(cfg: DeploymentExportConfig) -> None:
    with pytest.raises(ValueError, match="Exactly one of env and run"):
        export_deploy.run(cfg)
