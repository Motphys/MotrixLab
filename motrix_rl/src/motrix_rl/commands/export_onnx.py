# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Export a metadata-backed MotrixLab policy to ONNX."""

import json

import hydra
from omegaconf import DictConfig

from motrix_rl.cli import run_cli, to_typed_config
from motrix_rl.config import OnnxExportConfig
from motrix_rl.deploy import export_onnx


def run(cfg: OnnxExportConfig) -> None:
    """Export one metadata-backed run and print its persisted report."""
    result = export_onnx(
        cfg.run_dir,
        cfg.output,
        opset=cfg.opset,
        validation_seed=cfg.parity.seed,
        validation_samples=cfg.parity.samples,
        atol=cfg.parity.atol,
        rtol=cfg.parity.rtol,
    )
    print(
        json.dumps(
            {
                "output": str(result.path),
                "input": {
                    "name": result.report.input_spec.name,
                    "shape": result.report.input_spec.shape,
                    "dtype": result.report.input_spec.dtype,
                },
                "output_tensor": {
                    "name": result.report.output_spec.name,
                    "shape": result.report.output_spec.shape,
                    "dtype": result.report.output_spec.dtype,
                },
                "validation_samples": result.report.parity.samples,
                "max_abs_error": result.report.parity.max_abs_error,
                "max_rel_error": result.report.parity.max_rel_error,
            },
            indent=2,
            sort_keys=True,
        )
    )


@hydra.main(version_base=None, config_path="pkg://motrix_rl.configs", config_name="export_onnx")
def _hydra_main(cfg: DictConfig) -> None:
    run(to_typed_config(cfg, OnnxExportConfig))


def main() -> None:
    run_cli(_hydra_main)


if __name__ == "__main__":
    main()
