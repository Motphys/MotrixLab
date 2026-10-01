# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest

from motrix_rl.deploy import OnnxPolicyExporter, PolicyTensorSpec


def test_onnx_policy_exporter_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        OnnxPolicyExporter()


def test_policy_tensor_spec_rejects_non_tuple_shape() -> None:
    with pytest.raises(TypeError, match="shape must be a tuple, got list"):
        PolicyTensorSpec(name="obs", shape=[None, 3], dtype="float32")  # type: ignore[arg-type]


def test_import_does_not_load_optional_export_dependencies() -> None:
    script = (
        "import sys; import motrix_rl; import motrix_rl.deploy; "
        "assert 'torch' not in sys.modules; assert 'onnx' not in sys.modules; "
        "assert 'onnxruntime' not in sys.modules; assert 'skrl' not in sys.modules; "
        "assert 'rsl_rl' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
