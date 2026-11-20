# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Per-row numerical status transport without a real simulator or GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motrix_env_core.config import SimCfg
from motrix_env_core.config.scene import SceneCfg
from motrix_env_motrixsim import runtime


class _EngineModel:
    def __init__(self):
        self.statuses = np.array([0, 1, 2], dtype=np.int32)

    def step_n(self, data, substeps, *, result=None):
        # Mirror the engine boundary: healthy rows finish all substeps, failed
        # rows may be partially advanced. Every result element is overwritten.
        status = self.statuses if substeps else np.zeros_like(self.statuses)
        data.steps += np.where(status == 0, substeps, min(substeps, 1))
        if result is None:
            if status.any():
                raise RuntimeError("simulation step failed")
        else:
            assert result.dtype == np.int32
            assert result.shape == self.statuses.shape
            assert result.flags.c_contiguous
            result[:] = status


def _backend(monkeypatch):
    model = _EngineModel()
    data = SimpleNamespace(steps=np.zeros(3, dtype=np.int64))
    monkeypatch.setattr(runtime.MotrixSimSceneCompiler, "compile", lambda self, scene, sim, *, render: model)
    monkeypatch.setattr(runtime.mtx, "SceneData", lambda *args, **kwargs: data)
    return runtime.MotrixSimBackend(SceneCfg(), SimCfg(), 3), model, data


def test_step_writes_mixed_status_and_preserves_strict_default(monkeypatch):
    backend, model, data = _backend(monkeypatch)
    status = torch.empty(3, dtype=torch.int32)
    backend.step(4, status)

    assert status.device.type == "cpu"
    assert status.tolist() == [0, 1, 2]
    assert data.steps.tolist() == [4, 1, 1]
    with pytest.raises(RuntimeError, match="simulation step failed"):
        backend.step(4)
    assert data.steps.tolist() == [8, 2, 2]


@pytest.mark.parametrize("stride", [1, 2])
def test_step_writes_into_caller_storage_each_call(monkeypatch, stride):
    backend, model, data = _backend(monkeypatch)
    storage = torch.full((3 * stride,), -1, dtype=torch.int32)
    status = storage[::stride]
    backend.step(2, status)
    held = status.clone()
    model.statuses[:] = 0
    backend.step(3, status)
    assert status.tolist() == [0, 0, 0]
    assert held.tolist() == [0, 1, 2]

    model.statuses[:] = [1, 2, 3]
    backend.step(1, status)
    assert status.tolist() == [1, 2, 3]
    before = data.steps.copy()
    backend.step(0, status)
    assert status.tolist() == [0, 0, 0]
    np.testing.assert_array_equal(data.steps, before)
    if stride == 2:
        assert storage[1::2].tolist() == [-1, -1, -1]
