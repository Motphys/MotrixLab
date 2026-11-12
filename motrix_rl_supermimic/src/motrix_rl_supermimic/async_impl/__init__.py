# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Async training topology: CPU collector process ‖ device learner.

The data plane reuses the shared transport primitives: a schema-driven
``HostTransitionRing`` carries collector→learner transition batches
(bounded, ordered, non-dropping), and the ``InferenceBundle`` weight
channel carries learner→collector policy snapshots (latest-only, lagging
allowed). Governance mirrors the reference implementation: warmup, strict
post-warmup update credit, and bounded policy lag.
"""

from motrix_rl_supermimic.async_impl.train import AsyncTrainer

__all__ = ["AsyncTrainer"]
