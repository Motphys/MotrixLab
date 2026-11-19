# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.array.env import EnvObs
from motrix_env_core.observations import (
    ObservationGroupSpec,
    ObservationSpec,
    append_observation,
    repeat_observation,
)


def test_static_spec_derives_time_major_bounds_and_optional_value() -> None:
    frame = gym.spaces.Box(np.array([-1, -2], dtype=np.float32), np.array([3, 4], dtype=np.float32))
    group = ObservationGroupSpec(frame, 3)
    np.testing.assert_array_equal(group.space.low, [-1, -2, -1, -2, -1, -2])
    np.testing.assert_array_equal(group.space.high, [3, 4, 3, 4, 3, 4])
    assert ObservationGroupSpec(frame).space is frame
    assert ObservationSpec({"policy": group}).value_or_policy is group
    with pytest.raises(FrozenInstanceError):
        group.history_length = 2


def test_named_groups_and_semantic_conveniences_share_storage() -> None:
    group = ObservationGroupSpec(gym.spaces.Box(-1, 1, (2,)))
    spec = ObservationSpec({"policy": group, "goal": group})
    tensor = torch.zeros(2, 2)
    obs = EnvObs({"policy": tensor, "goal": tensor})
    assert spec["goal"] is spec.groups["goal"]
    assert spec.policy is spec["policy"]
    assert spec.value is None
    assert spec.value_or_policy is spec.policy
    assert obs["goal"] is obs.groups["goal"]
    assert obs.policy is obs["policy"]
    assert obs.value is None
    assert obs.value_or_policy is obs.policy
    value = torch.ones(2, 2)
    obs.groups["value"] = value
    assert obs.value is value
    assert obs.value_or_policy is value
    value_spec = ObservationSpec({"policy": group, "value": group})
    assert value_spec.value is group
    assert value_spec.value_or_policy is group
    with pytest.raises(AttributeError):
        obs.policy = value
    with pytest.raises(KeyError):
        obs["missing"]


@pytest.mark.parametrize("constructor", [ObservationSpec, EnvObs])
def test_named_observations_require_policy(constructor: type[ObservationSpec] | type[EnvObs]) -> None:
    with pytest.raises(ValueError, match="'policy'"):
        constructor({})


@pytest.mark.parametrize("length,error", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_spec_rejects_invalid_history_length(length: int, error: type[Exception]) -> None:
    with pytest.raises(error, match="history_length"):
        ObservationGroupSpec(gym.spaces.Box(-1, 1, (2,)), length)


@pytest.mark.parametrize("compile_helper", [False, True])
def test_functional_append_preserves_prefix_gradient_and_repeat_fill(compile_helper: bool) -> None:
    append = (
        torch.compile(append_observation, backend="eager", fullgraph=True) if compile_helper else append_observation
    )
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    second = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    history = repeat_observation(first, 3)
    updated = append(history, second, torch.tensor([False, True]))
    torch.testing.assert_close(updated[0], torch.tensor([[1.0, 2.0], [1.0, 2.0], [5.0, 6.0]]))
    torch.testing.assert_close(updated[1], second[1].expand(3, -1))
    torch.testing.assert_close(history, first[:, None].expand(-1, 3, -1))
    updated.sum().backward()
    torch.testing.assert_close(first.grad, torch.tensor([[2.0, 2.0], [0.0, 0.0]]))
    torch.testing.assert_close(second.grad, torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
