# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared white-box mocks for the async FastSAC learner's process-free tests.

``make_mock_learner`` builds a ``Learner`` without running ``__init__`` (no
agent build, no CUDA): the drain path only needs the rings, a capturing
replay-buffer stub and the generation assembler. Used by
test_fastsac_async_multi.py and test_fastsac_pipeline_equivalence.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from motrix_rl.fastsac.async_impl.learner import GenerationAssembler, Learner
from motrix_rl.fastsac.async_impl.transport import Control


def make_mock_learner(rings: list, extends: list | None) -> Learner:
    """A drain-capable Learner over ``rings``; ``extends`` captures extend_batch columns."""
    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(
        device=torch.device("cpu"),
        rb=SimpleNamespace(extend_batch=lambda *c: extends.append([x.clone() for x in c])),
        cfg=SimpleNamespace(learning_starts=0),
    )
    learner.async_options = SimpleNamespace(max_ingest_per_iter=8)
    learner.control = Control(num_collectors=len(rings))
    learner.rings = rings
    learner.weights = []
    learner._pending = GenerationAssembler(num_rings=len(rings))
    learner._staging = None
    learner._pending_copy = False
    return learner
