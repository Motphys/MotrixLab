# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed transition views preserve schema fields, slicing and replay ownership."""

import torch
from motrix_rl_supermimic.replay import ReplayBuffer, transition_schema
from motrix_rl_supermimic.transitions import Transitions


def test_transition_views_and_replay_ownership() -> None:
    schema = transition_schema(state_dim=2, context_dim=1, action_dim=1)
    values = {
        field.name: torch.full((3, 2, *field.per_env_shape), index + 1, dtype=field.dtype)
        for index, field in enumerate(schema.fields)
    }
    batch = Transitions.from_dict(values)
    assert batch.state is values["s"]
    assert batch.next_state is values["s1"]
    assert batch.previous_action is values["a_prev"]
    assert batch.context is values["xi"]
    assert batch.frame is values["f"]
    assert batch.episode_step is values["ep_step"]
    restored = batch.to("cpu").to_dict()
    assert restored.keys() == values.keys()
    for name, tensor in values.items():
        assert restored[name] is tensor

    step = batch.at_step(1)
    replay = ReplayBuffer(4, 2, state_dim=2, context_dim=1, action_dim=1)
    replay.add(step)
    expected = {name: tensor[1].clone() for name, tensor in values.items()}
    for tensor in values.values():
        tensor.zero_()
    for name, tensor in expected.items():
        torch.testing.assert_close(replay.data[name][0], tensor)
        torch.testing.assert_close(step.to_dict()[name], torch.zeros_like(tensor))
