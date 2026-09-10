# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Policy runtime contract and optional ONNX implementation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from motrix_deploy.contracts import FloatArray, TensorSpec, float32_array
from motrix_deploy.errors import ValidationError


class PolicyRuntime(ABC):
    """Framework-independent deterministic policy inference."""

    @property
    @abstractmethod
    def input_spec(self) -> TensorSpec:
        """Return the policy input contract."""

    @property
    @abstractmethod
    def output_spec(self) -> TensorSpec:
        """Return the policy output contract."""

    def reset(self) -> None:
        """Reset recurrent state; v1 policies are stateless."""

    @abstractmethod
    def infer(self, observation: FloatArray) -> FloatArray:
        """Infer one unbatched action."""


class OnnxPolicyRuntime(PolicyRuntime):
    """Single-input, single-output ONNX Runtime policy."""

    def __init__(
        self,
        path: str | Path,
        input_spec: TensorSpec,
        output_spec: TensorSpec,
        *,
        session_options: Any = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime is missing; it is a core dependency of motrix-deploy, so reinstall the environment"
            ) from error
        self._input_spec = input_spec
        self._output_spec = output_spec
        self._session = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._validate_session()

    @property
    def input_spec(self) -> TensorSpec:
        return self._input_spec

    @property
    def output_spec(self) -> TensorSpec:
        return self._output_spec

    def _validate_session(self) -> None:
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValidationError("policy.onnx.io", "one input and one output", (len(inputs), len(outputs)))
        self._validate_node(inputs[0], self.input_spec, "policy.input")
        self._validate_node(outputs[0], self.output_spec, "policy.output")

    @staticmethod
    def _validate_node(node: Any, spec: TensorSpec, path: str) -> None:
        if node.name != spec.name:
            raise ValidationError(f"{path}.name", spec.name, node.name)
        if node.type != "tensor(float)":
            raise ValidationError(f"{path}.dtype", "float32", node.type)
        actual_shape = tuple(node.shape)
        shape_matches = actual_shape == spec.shape or (
            len(actual_shape) == len(spec.shape)
            and spec.shape[0] == 1
            and not isinstance(actual_shape[0], int)
            and actual_shape[1:] == spec.shape[1:]
        )
        if not shape_matches:
            raise ValidationError(f"{path}.shape", str(spec.shape), tuple(node.shape))

    def infer(self, observation: FloatArray) -> FloatArray:
        expected_input = (self.input_spec.shape[1],)
        value = float32_array(observation, path="policy.input", shape=expected_input)
        output = self._session.run([self.output_spec.name], {self.input_spec.name: value[None, :]})[0]
        output = float32_array(output, path="policy.output", shape=self.output_spec.shape)
        return float32_array(output[0], path="policy.output", shape=(self.output_spec.shape[1],))


__all__ = ["OnnxPolicyRuntime", "PolicyRuntime"]
