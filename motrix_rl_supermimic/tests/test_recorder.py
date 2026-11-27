# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Exported replay snapshots own their data and use final framework boundaries."""

import pytest
import torch
from motrix_rl_supermimic.recorder import TransitionRecorder

from motrix_env_core.array.env import ArrayEnvState, EnvObs


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.integration)])
def test_recorder_allocates_and_snapshots_on_explicit_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    recorder = TransitionRecorder.create(2, device=device, state_dim=3, context_dim=1, action_dim=2)
    target = torch.device(device)
    result = ArrayEnvState(
        obs=EnvObs({"policy": torch.zeros(2, 1, device=device)}),
        reward=torch.ones(2, device=device),
        terminated=torch.zeros(2, dtype=torch.bool, device=device),
        truncated=torch.zeros(2, dtype=torch.bool, device=device),
        episode_steps=torch.zeros(2, dtype=torch.long, device=device),
    )
    for value in recorder.snapshot(result).values():
        assert value.device == target


def test_snapshot_owns_fields_and_combines_final_termination_and_truncation() -> None:
    recorder = TransitionRecorder.create(3, device="cpu", state_dim=2, context_dim=2, action_dim=1)
    values = {
        "s": torch.arange(3 * 2, dtype=torch.float32).reshape(3, -1),
        "s1": torch.arange(3 * 2, dtype=torch.float32).reshape(3, -1) + 100,
        "frame": torch.tensor([12, 18, 24]),
        "context": torch.tensor([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
        "action": torch.tensor([[0.1], [0.2], [0.3]]),
        "previous": torch.tensor([[-0.1], [-0.2], [-0.3]]),
        "episode_step": torch.tensor([4, 7, 9], dtype=torch.int32),
    }
    for name, value in values.items():
        getattr(recorder, name).copy_(value)
    result = ArrayEnvState(
        obs=EnvObs(groups={"policy": torch.empty(3, 4)}),
        reward=torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
        terminated=torch.tensor([True, False, False]),
        truncated=torch.tensor([False, True, False]),
        episode_steps=torch.tensor([0, 0, 10]),
    )
    snapshot = recorder.snapshot(result)
    for exported, field in (
        ("s", "s"),
        ("s1", "s1"),
        ("f", "frame"),
        ("xi", "context"),
        ("a", "action"),
        ("a_prev", "previous"),
        ("ep_step", "episode_step"),
    ):
        torch.testing.assert_close(snapshot[exported], values[field])
    torch.testing.assert_close(snapshot["r"], result.reward)
    assert snapshot["done"].tolist() == [True, True, False]
    assert not snapshot["r"].requires_grad
    expected = {name: value.clone() for name, value in snapshot.items()}
    for name in values:
        getattr(recorder, name).zero_()
    with torch.no_grad():
        result.reward.zero_()
    result.terminated.zero_()
    result.truncated.zero_()
    for name, value in expected.items():
        torch.testing.assert_close(snapshot[name], value)
