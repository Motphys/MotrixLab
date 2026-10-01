# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SKRL Torch PPO checkpoint to ONNX adapter."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from motrix_rl.deploy.api import OnnxExportRequest, OnnxModelArtifact, OnnxPolicyExporter
from motrix_rl.deploy.onnx_validation import validate_onnx_policy
from motrix_rl_skrl.config import SkrlCfg


@dataclass(frozen=True)
class _PolicyWeights:
    hidden_weights: tuple[np.ndarray, ...]
    hidden_biases: tuple[np.ndarray, ...]
    hidden_activations: tuple[str, ...]
    output_weight: np.ndarray
    output_bias: np.ndarray
    observation_mean: np.ndarray
    observation_variance: np.ndarray
    normalizer_epsilon: float = 1e-8
    normalizer_clip: float = 5.0

    @property
    def observation_size(self) -> int:
        return int(self.hidden_weights[0].shape[0])

    @property
    def action_size(self) -> int:
        return int(self.output_bias.shape[0])


class SkrlPpoTorchOnnxExporter(OnnxPolicyExporter):
    """Extract SKRL PPO policy weights and build a framework-independent ONNX graph."""

    def export(self, request: OnnxExportRequest) -> OnnxModelArtifact:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("SKRL Torch ONNX export requires the 'motrix-rl[skrl-torch,onnx]' extras") from error

        algo_cfg = request.task_config.algo
        if not isinstance(algo_cfg, SkrlCfg):
            raise TypeError(f"SKRL exporter expects SkrlCfg, got {type(algo_cfg).__name__}")
        checkpoint = torch.load(request.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"SKRL checkpoint must contain a module mapping: {request.checkpoint}")
        policy_state = _module_state(checkpoint, "policy", request.checkpoint)
        normalizer_state = _module_state(checkpoint, "observation_preprocessor", request.checkpoint)
        weights = _extract_weights(algo_cfg, policy_state, normalizer_state)
        model = _build_onnx(weights, request)
        reference = _build_reference(weights)
        report = validate_onnx_policy(
            model,
            reference,
            observation_size=weights.observation_size,
            action_size=weights.action_size,
            config=request.parity,
            source="skrl/torch/ppo",
        )
        return OnnxModelArtifact(model_bytes=model, report=report)


def _module_state(checkpoint: Mapping, name: str, path) -> Mapping:
    state = checkpoint.get(name)
    if not isinstance(state, Mapping):
        raise ValueError(f"SKRL checkpoint has no {name!r} state_dict: {path}")
    return state


def _extract_weights(algo_cfg: SkrlCfg, policy_state: Mapping, normalizer_state: Mapping) -> _PolicyWeights:
    policy_cfg = algo_cfg.models.policy
    hidden_sizes = policy_cfg.hiddens
    if not hidden_sizes:
        raise ValueError("SKRL ONNX export requires at least one policy hidden layer")
    activations = policy_cfg.hidden_activation
    if len(activations) == 1:
        activations = activations * len(hidden_sizes)
    if len(activations) != len(hidden_sizes):
        raise ValueError(f"SKRL policy activation count must be 1 or {len(hidden_sizes)}, got {len(activations)}")

    hidden_weights = []
    hidden_biases = []
    previous_size = None
    for index, hidden_size in enumerate(hidden_sizes):
        weight = _float_array(policy_state, f"net.{index * 2}.weight", rank=2).T
        bias = _float_array(policy_state, f"net.{index * 2}.bias", rank=1)
        if weight.shape[1] != hidden_size or bias.shape != (hidden_size,):
            raise ValueError(
                f"SKRL policy layer {index} shape does not match configured hidden size {hidden_size}: "
                f"weight={weight.shape}, bias={bias.shape}"
            )
        if previous_size is not None and weight.shape[0] != previous_size:
            raise ValueError(f"SKRL policy layer {index} input size {weight.shape[0]} does not match {previous_size}")
        previous_size = hidden_size
        hidden_weights.append(weight)
        hidden_biases.append(bias)

    output_weight = _float_array(policy_state, "mean_layer.weight", rank=2).T
    output_bias = _float_array(policy_state, "mean_layer.bias", rank=1)
    if output_weight.shape[0] != hidden_sizes[-1] or output_weight.shape[1] != output_bias.shape[0]:
        raise ValueError(
            f"SKRL mean layer shapes are inconsistent: weight={output_weight.shape}, bias={output_bias.shape}"
        )
    observation_size = int(hidden_weights[0].shape[0])
    observation_mean = _float_array(normalizer_state, "running_mean", rank=1)
    observation_variance = _float_array(normalizer_state, "running_variance", rank=1)
    if observation_mean.shape != (observation_size,) or observation_variance.shape != (observation_size,):
        raise ValueError(
            f"SKRL observation normalizer must have shape {(observation_size,)}, got "
            f"mean={observation_mean.shape}, variance={observation_variance.shape}"
        )
    if np.any(observation_variance < 0):
        raise ValueError("SKRL observation normalizer contains negative variance")
    return _PolicyWeights(
        hidden_weights=tuple(hidden_weights),
        hidden_biases=tuple(hidden_biases),
        hidden_activations=tuple(str(value).lower() for value in activations),
        output_weight=output_weight,
        output_bias=output_bias,
        observation_mean=observation_mean,
        observation_variance=observation_variance,
    )


def _float_array(state: Mapping, name: str, *, rank: int) -> np.ndarray:
    if name not in state:
        raise ValueError(f"SKRL state_dict is missing {name!r}")
    value = state[name]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != rank:
        raise ValueError(f"SKRL state {name!r} must have rank {rank}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"SKRL state {name!r} contains NaN or Inf")
    return np.ascontiguousarray(array)


def _build_reference(weights: _PolicyWeights):
    try:
        import torch
        import torch.nn as nn
        from skrl.resources.preprocessors.torch import RunningStandardScaler
    except ImportError as error:
        raise RuntimeError("SKRL reference validation requires the 'motrix-rl[skrl-torch]' extra") from error

    activation_types = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
    }
    unknown = sorted(set(weights.hidden_activations) - activation_types.keys())
    if unknown:
        raise ValueError(f"SKRL ONNX export does not support activations {unknown}")

    normalizer = RunningStandardScaler(
        size=weights.observation_size,
        epsilon=weights.normalizer_epsilon,
        clip_threshold=weights.normalizer_clip,
        device="cpu",
    )
    normalizer.running_mean.copy_(torch.from_numpy(weights.observation_mean).to(torch.float64))
    normalizer.running_variance.copy_(torch.from_numpy(weights.observation_variance).to(torch.float64))
    layers = []
    for weight, bias, activation in zip(
        weights.hidden_weights,
        weights.hidden_biases,
        weights.hidden_activations,
    ):
        linear = nn.Linear(weight.shape[0], weight.shape[1])
        linear.weight.data.copy_(torch.from_numpy(weight.T))
        linear.bias.data.copy_(torch.from_numpy(bias))
        layers.extend((linear, activation_types[activation]()))
    network = nn.Sequential(*layers).eval()
    output = nn.Linear(weights.output_weight.shape[0], weights.output_weight.shape[1])
    output.weight.data.copy_(torch.from_numpy(weights.output_weight.T))
    output.bias.data.copy_(torch.from_numpy(weights.output_bias))
    output.eval()

    def reference(observations: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            value = normalizer(torch.from_numpy(observations))
            value = output(network(value))
        return value.numpy()

    return reference


def _build_onnx(weights: _PolicyWeights, request: OnnxExportRequest) -> bytes:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as error:
        raise RuntimeError("SKRL Torch ONNX export requires the 'motrix-rl[onnx]' extra") from error

    nodes = []
    initializers = []

    def initializer(name: str, value) -> None:
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name))

    initializer("observation_mean", weights.observation_mean)
    initializer("observation_variance", weights.observation_variance)
    initializer("normalizer_epsilon", np.asarray(weights.normalizer_epsilon, dtype=np.float32))
    initializer("normalizer_clip_min", np.asarray(-weights.normalizer_clip, dtype=np.float32))
    initializer("normalizer_clip_max", np.asarray(weights.normalizer_clip, dtype=np.float32))
    nodes.extend(
        (
            helper.make_node("Sub", ["obs", "observation_mean"], ["observation_centered"]),
            helper.make_node("Sqrt", ["observation_variance"], ["observation_std"]),
            helper.make_node("Add", ["observation_std", "normalizer_epsilon"], ["observation_denominator"]),
            helper.make_node("Div", ["observation_centered", "observation_denominator"], ["observation_scaled"]),
            helper.make_node(
                "Clip",
                ["observation_scaled", "normalizer_clip_min", "normalizer_clip_max"],
                ["observation_normalized"],
            ),
        )
    )
    current = "observation_normalized"
    activation_nodes = {
        "elu": ("Elu", {"alpha": 1.0}),
        "relu": ("Relu", {}),
        "tanh": ("Tanh", {}),
        "sigmoid": ("Sigmoid", {}),
        "leaky_relu": ("LeakyRelu", {"alpha": 0.01}),
        "selu": ("Selu", {}),
    }
    for index, (weight, bias, activation) in enumerate(
        zip(weights.hidden_weights, weights.hidden_biases, weights.hidden_activations)
    ):
        if activation not in activation_nodes:
            raise ValueError(f"SKRL ONNX export does not support activation {activation!r}")
        weight_name = f"hidden_{index}_weight"
        bias_name = f"hidden_{index}_bias"
        linear_name = f"hidden_{index}_linear"
        biased_name = f"hidden_{index}_biased"
        output_name = f"hidden_{index}"
        initializer(weight_name, weight)
        initializer(bias_name, bias)
        nodes.append(helper.make_node("MatMul", [current, weight_name], [linear_name]))
        nodes.append(helper.make_node("Add", [linear_name, bias_name], [biased_name]))
        operation, attributes = activation_nodes[activation]
        nodes.append(helper.make_node(operation, [biased_name], [output_name], **attributes))
        current = output_name

    initializer("output_weight", weights.output_weight)
    initializer("output_bias", weights.output_bias)
    nodes.append(helper.make_node("MatMul", [current, "output_weight"], ["output_linear"]))
    nodes.append(helper.make_node("Add", ["output_linear", "output_bias"], ["actions"]))
    graph = helper.make_graph(
        nodes,
        "motrix_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch_size", weights.observation_size])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, ["batch_size", weights.action_size])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", request.opset)])
    model.ir_version = 10
    metadata = request.run.metadata
    helper.set_model_props(
        model,
        {
            "env_name": metadata.env_name,
            "rllib": metadata.rllib,
            "train_backend": metadata.train_backend,
            "algo": metadata.algo,
            "obs_dim": str(weights.observation_size),
            "action_dim": str(weights.action_size),
            "source_checkpoint": str(request.checkpoint),
        },
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()
