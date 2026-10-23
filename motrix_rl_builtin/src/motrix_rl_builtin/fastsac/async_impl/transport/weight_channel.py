# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC inference snapshot adapter over the shared tensor channel."""

from collections.abc import Mapping

import torch
from torch import nn

from motrix_rl.transport.snapshot import (
    GpuIpcSnapshotReceiver,
    GpuIpcSnapshotSender,
    HostSnapshotReceiver,
    HostSnapshotSender,
    SnapshotReceiver,
    SnapshotSender,
    SnapshotTensorSpec,
    TensorSnapshotShared,
    flatten_params,
)

_NORM_KEYS = ("_mean", "_std", "_var", "count")


def _normalizer_state(normalizer: nn.Module) -> Mapping[str, torch.Tensor]:
    if isinstance(normalizer, nn.Identity):
        return {}
    if not all(hasattr(normalizer, name) for name in _NORM_KEYS):
        raise ValueError("normalizer must provide all FastSAC statistics or be nn.Identity (disabled)")
    return {name: getattr(normalizer, name) for name in _NORM_KEYS}


class WeightChannelShared(TensorSnapshotShared):
    """FastSAC normalizer layout, fixed for the channel's lifetime.

    With ``obs_normalization=False`` the schema is empty and both endpoints
    must use ``nn.Identity``. Enabled channels require every statistic on every
    publish/load; omitting them cannot pair new parameters with old statistics.
    """

    def __init__(self, obs_dim: int, *, obs_normalization: bool = True):
        super().__init__(
            {
                "_mean": SnapshotTensorSpec((1, obs_dim)),
                "_std": SnapshotTensorSpec((1, obs_dim)),
                "_var": SnapshotTensorSpec((1, obs_dim)),
                "count": SnapshotTensorSpec((), torch.int64, host_scalar=True),
            }
            if obs_normalization
            else {}
        )
        if obs_normalization:
            self.mean = self.auxiliary["_mean"]
            self.std = self.auxiliary["_std"]
            self.var = self.auxiliary["_var"]
            self.count = self.auxiliary["count"]


class WeightSender(SnapshotSender):
    def publish(self, actor: nn.Module, obs_normalizer: nn.Module) -> None:
        # Keep preparation outside the seqlock's odd window, exactly as before.
        device_flat = flatten_params(actor)
        self.publish_flat(device_flat, _normalizer_state(obs_normalizer))


class HostWeightSender(WeightSender, HostSnapshotSender):
    pass


class GpuIpcWeightSender(WeightSender, GpuIpcSnapshotSender):
    pass


class WeightReceiver(SnapshotReceiver):
    @property
    def _norm_staging(self):
        if self._aux_staging is None:
            return None
        return tuple(self._aux_staging[name] for name in _NORM_KEYS[:3])

    def maybe_load(
        self, actor: nn.Module, obs_normalizer: nn.Module, flat_params: torch.Tensor | None = None
    ) -> tuple[int, float, float, float]:
        return super().maybe_load(actor, _normalizer_state(obs_normalizer), flat_params)


class HostWeightReceiver(WeightReceiver, HostSnapshotReceiver):
    pass


class GpuIpcWeightReceiver(WeightReceiver, GpuIpcSnapshotReceiver):
    pass


def weight_receiver_for(shared: WeightChannelShared, params: list[torch.Tensor]) -> WeightReceiver:
    """Build the receiver matching a shipped slot pair (dispatch on device)."""
    if params[0].is_cuda:
        return GpuIpcWeightReceiver(shared, params)
    return HostWeightReceiver(shared, params)
