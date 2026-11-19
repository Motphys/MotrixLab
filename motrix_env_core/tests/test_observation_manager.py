# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from motrix_env_core.array.env import EnvObs
from motrix_env_core.manager.observation_manager import ObservationManager
from motrix_env_core.manager.observations import ObservationGroupEntry


@pytest.mark.parametrize("history_length", [1, 3])
def test_observation_manager_owns_commit_reset_and_preview(history_length: int) -> None:
    manager = ObservationManager({"policy": ObservationGroupEntry("policy", (), 2, history_length)}, 2)
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    evaluations = 0

    def observe(outputs: EnvObs) -> None:
        nonlocal evaluations
        evaluations += 1
        outputs.policy.copy_(source)

    initial = manager.observation
    manager.publish_reset(torch.arange(2), observe)
    expected = source[:, None].expand(-1, history_length, -1).clone()
    torch.testing.assert_close(initial.policy.reshape(2, history_length, 2), expected)
    assert evaluations == 1
    source.add_(10)
    preview = manager.preview(torch.tensor([1]), observe)
    torch.testing.assert_close(preview.obs.policy.reshape(1, history_length, 2)[:, -1], source[[1]])
    torch.testing.assert_close(initial.policy.reshape(2, history_length, 2), expected)
    saved = preview.obs.policy.clone()
    output = manager.begin_step()
    manager.reset(torch.tensor([0]))
    manager.commit(observe)
    expected = torch.cat((expected[:, 1:], source[:, None]), dim=1)
    expected[0] = source[0].expand(history_length, -1)
    torch.testing.assert_close(output.policy.reshape(2, history_length, 2), expected)
    # The previous borrowed output survives the next control step; previews
    # remain owned even after their scratch and normal slots are reused.
    if history_length > 1:
        torch.testing.assert_close(initial.policy[:, -2:], source - 10)
    source.add_(10)
    manager.preview(torch.tensor([1]), observe)
    manager.begin_step()
    manager.commit(observe)
    torch.testing.assert_close(preview.obs.policy, saved)
