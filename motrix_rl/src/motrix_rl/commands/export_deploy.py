# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Export a metadata-backed training run to a deployment artifact."""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from motrix_deploy.profile import build_deployment_profile
from motrix_rl.cli import run_cli, to_typed_config
from motrix_rl.config import DeploymentExportConfig
from motrix_rl.deploy import export_deploy_run
from motrix_rl.runs import latest_metadata_run, open_run_context


def run(cfg: DeploymentExportConfig) -> None:
    """Export one training run using its environment deployment profile."""
    run_dir, env_name = _resolve_run(cfg)
    output = Path(cfg.output) if cfg.output is not None else Path("artifacts") / f"{env_name}.deploy"
    result = export_deploy_run(
        run_dir,
        output,
        profile_builder=build_deployment_profile,
        validation_seed=cfg.validation.seed,
        validation_samples=cfg.validation.samples,
        atol=cfg.validation.atol,
        rtol=cfg.validation.rtol,
    )
    print(
        json.dumps(
            {
                "artifact": str(result.artifact.root),
                "run": str(run_dir),
                "validation_samples": result.validation_samples,
                "max_abs_error": result.max_abs_error,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _resolve_run(cfg: DeploymentExportConfig) -> tuple[Path, str]:
    if (cfg.env is None) == (cfg.run_dir is None):
        raise ValueError("Exactly one of env and run_dir must be configured")
    if cfg.run_dir is not None:
        context = open_run_context(cfg.run_dir)
        return context.run_dir, context.metadata.env_name
    assert cfg.env is not None
    selected = latest_metadata_run(cfg.env)
    if selected is None:
        raise FileNotFoundError(f"No metadata-backed training runs found for environment {cfg.env!r}")
    run_dir, metadata = selected
    return run_dir, metadata.env_name


@hydra.main(version_base=None, config_path="pkg://motrix_rl.configs", config_name="export_deploy")
def _hydra_main(cfg: DictConfig) -> None:
    run(to_typed_config(cfg, DeploymentExportConfig))


def main() -> None:
    run_cli(_hydra_main)


if __name__ == "__main__":
    main()
