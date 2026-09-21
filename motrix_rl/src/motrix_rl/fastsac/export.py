# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC Torch checkpoint to deterministic ONNX policy adapter."""

import io
from collections.abc import Mapping

import numpy as np

from motrix_rl.deploy.api import OnnxExportRequest, OnnxModelArtifact, OnnxPolicyExporter
from motrix_rl.deploy.onnx_validation import validate_onnx_policy
from motrix_rl.fastsac.config import FastSacCfg


class FastSacOnnxExporter(OnnxPolicyExporter):
    """Restore a FastSAC actor and export its deterministic inference path."""

    def export(self, request: OnnxExportRequest) -> OnnxModelArtifact:
        try:
            import onnx
            import torch
            from torch import nn
        except ImportError as error:
            raise RuntimeError("FastSAC ONNX export requires the 'motrix-rl[onnx]' extra") from error

        from motrix_rl.fastsac.buffer import EmpiricalNormalization
        from motrix_rl.fastsac.factory import resolve_policy_variant

        algo_cfg = request.task_config.algo
        if not isinstance(algo_cfg, FastSacCfg):
            raise TypeError(f"FastSAC exporter expects FastSacCfg, got {type(algo_cfg).__name__}")

        checkpoint = torch.load(request.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"FastSAC checkpoint must be a mapping: {request.checkpoint}")
        actor_state = _module_state(checkpoint, "actor", request.checkpoint)
        actor_cfg = algo_cfg.agent
        if isinstance(checkpoint.get("sonic"), Mapping):
            raise ValueError(
                "FastSAC checkpoint uses the pre-PolicyVariant SONIC format; "
                "re-export or retrain it with the current FastSAC policy variant"
            )
        variant_name = checkpoint.get("policy_variant", "default")
        variant_metadata = checkpoint.get("policy_variant_metadata", {})
        if not isinstance(variant_name, str) or not isinstance(variant_metadata, Mapping):
            raise ValueError("FastSAC checkpoint has invalid policy variant metadata")
        variant = resolve_policy_variant(variant_name)
        passthrough_dims = variant.passthrough_dims(algo_cfg)
        if variant_name == "default":
            observation_size, action_size = _actor_sizes(actor_state)
            variant_metadata = {**variant_metadata, "obs_dim": observation_size, "act_dim": action_size}
        actor = type(variant).build_from_checkpoint(
            variant_metadata,
            algo_cfg,
            device="cpu",
        )
        export_observation = getattr(actor, "export_observation", None)
        actor.load_state_dict(actor_state, strict=True)
        actor.eval()
        observation_size, action_size = actor.n_obs, actor.n_act

        if actor_cfg.obs_normalization:
            normalizer = EmpiricalNormalization(
                shape=observation_size,
                device="cpu",
                passthrough_dims=passthrough_dims,
            )
            normalizer.load_state_dict(_module_state(checkpoint, "obs_normalizer", request.checkpoint), strict=True)
            normalizer.eval()
        else:
            normalizer_state = _module_state(checkpoint, "obs_normalizer", request.checkpoint)
            if normalizer_state:
                raise ValueError("FastSAC checkpoint has observation normalizer state while obs_normalization=false")
            normalizer = nn.Identity()

        class DeterministicPolicy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.actor = actor
                self.normalizer = normalizer

            def forward(self, observations):
                if export_observation is not None:
                    observations = export_observation(observations)
                if isinstance(self.normalizer, EmpiricalNormalization):
                    observations = self.normalizer(observations, update=False)
                actions, _, _ = self.actor(observations)
                return actions

        policy = DeterministicPolicy().eval()
        buffer = io.BytesIO()
        torch.onnx.export(
            policy,
            _export_observation(torch.zeros(1, observation_size, dtype=torch.float32), export_observation),
            buffer,
            export_params=True,
            opset_version=request.opset,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
            dynamo=False,
        )
        model_proto = onnx.load_model_from_string(buffer.getvalue())
        onnx.helper.set_model_props(model_proto, _metadata(request, observation_size, action_size))
        onnx.checker.check_model(model_proto)
        model_bytes = model_proto.SerializeToString()

        def reference(observations: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                return policy(torch.from_numpy(observations)).numpy()

        report = validate_onnx_policy(
            model_bytes,
            reference,
            observation_size=observation_size,
            action_size=action_size,
            config=request.parity,
            source="motrix/torch/fastsac",
        )
        return OnnxModelArtifact(model_bytes=model_bytes, report=report)


def _module_state(checkpoint: Mapping, name: str, path) -> Mapping:
    state = checkpoint.get(name)
    if not isinstance(state, Mapping):
        raise ValueError(f"FastSAC checkpoint has no {name!r} state_dict: {path}")
    return state


def _export_observation(observations, export_observation):
    return export_observation(observations) if export_observation is not None else observations


def _actor_sizes(actor_state: Mapping) -> tuple[int, int]:
    first_weight = actor_state.get("net.0.weight")
    output_bias = actor_state.get("fc_mu.bias")
    if getattr(first_weight, "ndim", None) != 2 or getattr(output_bias, "ndim", None) != 1:
        raise ValueError("FastSAC actor state_dict has invalid input or output layer weights")
    return int(first_weight.shape[1]), int(output_bias.shape[0])


def _metadata(request: OnnxExportRequest, observation_size: int, action_size: int) -> dict[str, str]:
    metadata = request.run.metadata
    return {
        "env_name": metadata.env_name,
        "rllib": metadata.rllib,
        "train_backend": metadata.train_backend,
        "algo": metadata.algo,
        "obs_dim": str(observation_size),
        "action_dim": str(action_size),
        "source_checkpoint": str(request.checkpoint),
    }
