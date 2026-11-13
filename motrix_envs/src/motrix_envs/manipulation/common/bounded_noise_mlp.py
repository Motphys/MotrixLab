# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Actor model with the exploration noise std clamped to a fixed band.

The stock ``rsl_rl`` scalar std is a bare ``nn.Parameter`` with no clamp.
Two failure modes seen in practice:

* collapse — with sparse rewards + action penalty, PPO squeezes sigma toward
  zero to farm the penalty-free region (1.00 -> 0.0368 within 300 iters);
* blow-up — once a reward seam is found, entropy bonus + clipped-saturation
  gradients inflate sigma without bound (1.00 -> 1.72 by iter 1500).

This subclass clamps std into [min_noise_std, max_noise_std] on every
distribution update. Wire from the training recipe::

    algo:
      actor:
        class_name: motrix_envs.manipulation.piper_insert_peg.bounded_noise_mlp:BoundedNoiseMLPModel
        min_noise_std: 0.20
        max_noise_std: 0.80
"""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel


class BoundedNoiseMLPModel(MLPModel):
    """``MLPModel`` whose state-independent noise std stays in a fixed band."""

    def __init__(self, *args, min_noise_std: float = 0.25, max_noise_std: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if min_noise_std <= 0.0:
            raise ValueError(f"min_noise_std must be positive, got {min_noise_std}")
        if max_noise_std is not None and max_noise_std < min_noise_std:
            raise ValueError(f"max_noise_std ({max_noise_std}) < min_noise_std ({min_noise_std})")
        self.min_noise_std = float(min_noise_std)
        self.max_noise_std = None if max_noise_std is None else float(max_noise_std)
        with torch.no_grad():
            if self.noise_std_type == "scalar" and hasattr(self, "std"):
                hi = self.max_noise_std if self.max_noise_std is not None else float("inf")
                self.std.clamp_(min=self.min_noise_std, max=hi)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.noise_std_type == "scalar" and hasattr(self, "std"):
            with torch.no_grad():
                hi = self.max_noise_std if self.max_noise_std is not None else float("inf")
                self.std.clamp_(min=self.min_noise_std, max=hi)
        super()._update_distribution(obs)
