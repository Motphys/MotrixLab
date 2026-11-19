# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Stable purpose seeds and construction-only model RNG isolation."""

import hashlib
import json
import secrets
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def resolve_seed(seed: int | None) -> int:
    """Resolve an unspecified root from system entropy, not the Torch stream."""
    return secrets.randbits(63) if seed is None else seed


def derive_seed(root: int, purpose: str, worker_id: int = 0) -> int:
    """Derive a process-stable 63-bit seed from a versioned purpose domain."""
    payload = json.dumps(["motrix-supermimic-rng-v1", root, purpose, worker_id], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


@contextmanager
def model_initialization(seed: int) -> Iterator[None]:
    """Isolate model/reference factories, including any CUDA random allocation.

    All visible CUDA generators are preserved: task-owned reference factories
    may allocate on a device other than the policy target. This construction-only
    scope can initialize CUDA contexts; it is never used in numerical kernels.
    """
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed)
        yield
