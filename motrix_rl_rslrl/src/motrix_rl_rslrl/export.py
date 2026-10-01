# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL Torch PPO ONNX exporter."""

import io
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np

from motrix_rl.deploy.api import OnnxExportRequest, OnnxModelArtifact, OnnxPolicyExporter
from motrix_rl.deploy.onnx_validation import validate_onnx_policy
from motrix_rl_rslrl.cfg import RslrlCfg


class RslrlPpoOnnxExporter(OnnxPolicyExporter):
    """Restore an RSL-RL MLP actor and export its deterministic path."""

    def export(self, request: OnnxExportRequest) -> OnnxModelArtifact:
        try:
            import onnx
            import torch
            from rsl_rl.models import MLPModel
            from tensordict import TensorDict
        except ImportError as error:
            raise RuntimeError("RSL-RL ONNX export requires the 'motrix-rl[rslrl,onnx]' extras") from error

        algo_cfg = request.task_config.algo
        if not isinstance(algo_cfg, RslrlCfg):
            raise TypeError(f"RSL-RL exporter expects RslrlCfg, got {type(algo_cfg).__name__}")
        if algo_cfg.obs_groups.get("actor") != ["policy"]:
            raise ValueError(
                "RSL-RL ONNX export currently requires obs_groups.actor=['policy'] for a single ONNX input"
            )
        actor_config = asdict(algo_cfg.actor)
        class_name = actor_config.pop("class_name")
        if class_name != "MLPModel":
            raise ValueError(f"RSL-RL ONNX export supports actor.class_name=MLPModel, got {class_name!r}")
        if actor_config.get("state_dependent_std"):
            raise ValueError("RSL-RL ONNX export does not yet support state-dependent actor standard deviation")

        checkpoint = torch.load(request.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping) or "actor_state_dict" not in checkpoint:
            raise ValueError(f"RSL-RL checkpoint has no actor_state_dict: {request.checkpoint}")
        actor_state = checkpoint["actor_state_dict"]
        if not isinstance(actor_state, Mapping):
            raise ValueError(f"RSL-RL actor_state_dict must be a mapping: {request.checkpoint}")
        observation_size, action_size = _mlp_sizes(actor_state)

        observation = TensorDict(
            {"policy": torch.zeros(1, observation_size, dtype=torch.float32)},
            batch_size=[1],
        )
        actor = MLPModel(
            observation,
            algo_cfg.obs_groups,
            "actor",
            output_dim=action_size,
            **actor_config,
        )
        actor.load_state_dict(actor_state, strict=True)
        actor.cpu().eval()
        exportable = actor.as_onnx(verbose=False).cpu().eval()
        buffer = io.BytesIO()
        torch.onnx.export(
            exportable,
            exportable.get_dummy_inputs(),
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
        model = model_proto.SerializeToString()

        def reference(observations: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                result = actor(
                    TensorDict(
                        {"policy": torch.from_numpy(observations)},
                        batch_size=[observations.shape[0]],
                    )
                )
            return result.cpu().numpy()

        report = validate_onnx_policy(
            model,
            reference,
            observation_size=observation_size,
            action_size=action_size,
            config=request.parity,
            source="rslrl/torch/ppo",
        )
        return OnnxModelArtifact(model_bytes=model, report=report)


def _mlp_sizes(actor_state: Mapping) -> tuple[int, int]:
    weights = []
    for name, value in actor_state.items():
        parts = name.split(".")
        if len(parts) == 3 and parts[0] == "mlp" and parts[1].isdigit() and parts[2] == "weight":
            weights.append((int(parts[1]), value))
    if not weights:
        raise ValueError("RSL-RL actor_state_dict has no MLP weights")
    weights.sort(key=lambda item: item[0])
    first = weights[0][1]
    last = weights[-1][1]
    if getattr(first, "ndim", None) != 2 or getattr(last, "ndim", None) != 2:
        raise ValueError("RSL-RL actor MLP weights must be rank-2 tensors")
    return int(first.shape[1]), int(last.shape[0])


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
