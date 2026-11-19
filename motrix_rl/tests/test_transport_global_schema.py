# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""A global transition field occupies one native-shape tensor per slot."""

import pytest
import torch

from motrix_rl.transport.ring import HostTransitionRing
from motrix_rl.transport.schema import TransitionField, TransitionSchema


@pytest.mark.parametrize("shape", [(), (5,), (2, 3)])
def test_legacy_per_environment_fields_keep_shape_and_feature_width(shape: tuple[int, ...]) -> None:
    field = TransitionField("state", shape, torch.float32)
    schema = TransitionSchema((field,))
    assert field.scope == "per-env"
    assert schema.slot_shape(field, num_envs=7) == (7, *shape)
    assert schema.feature_width == field.per_env_numel
    assert schema.global_feature_width == 0


@pytest.mark.parametrize("num_envs", [1, 17])
def test_mixed_schema_separates_global_storage_from_environment_width(num_envs: int) -> None:
    state = TransitionField("state", (3,), torch.float32)
    counter = TransitionField("counter", (), torch.int64, scope="global")
    histogram = TransitionField("histogram", (2, 3), torch.float32, scope="global")
    reward = TransitionField("reward", (), torch.float32)
    schema = TransitionSchema((state, counter, histogram, reward))
    assert schema.field_names == ("state", "counter", "histogram", "reward")
    assert schema.slot_shape(state, num_envs) == (num_envs, 3)
    assert schema.slot_shape(reward, num_envs) == (num_envs,)
    assert schema.slot_shape(counter, num_envs) == ()
    assert schema.slot_shape(histogram, num_envs) == (2, 3)
    assert schema.feature_width == 4
    assert schema.global_feature_width == 7
    # Storage scales with the environment batch only for per-env fields.
    total = sum(torch.Size(schema.slot_shape(field, num_envs)).numel() for field in schema.fields)
    assert total == num_envs * schema.feature_width + schema.global_feature_width


def test_global_only_schema_has_no_per_environment_feature_width() -> None:
    field = TransitionField("global", (2, 3), torch.float64, "global")
    schema = TransitionSchema((field,))
    assert schema.feature_width == 0
    assert schema.global_feature_width == 6
    assert schema.slot_shape(field, 100) == (2, 3)


def test_field_rejects_unknown_scope_at_declaration_boundary() -> None:
    with pytest.raises(ValueError, match="scope"):
        TransitionField("state", (), torch.float32, "batch")


@pytest.mark.parametrize("num_envs", [1, 7])
def test_host_ring_mixed_global_fields_wrap_backpressure_and_snapshot_lifetime(num_envs: int) -> None:
    schema = TransitionSchema(
        (
            TransitionField("state", (2,), torch.float32),
            TransitionField("global_matrix", (2, 3), torch.float64, scope="global"),
            TransitionField("global_scalar", (), torch.int64, scope="global"),
        )
    )
    ring = HostTransitionRing(schema, capacity=3, num_envs=num_envs)
    assert ring.data["global_matrix"].shape == (3, 2, 3)
    assert ring.data["global_scalar"].shape == (3,)
    retained = []
    expected = 0

    def batch(step: int) -> dict[str, torch.Tensor]:
        return {
            "state": torch.full((num_envs, 2), float(step)),
            "global_matrix": torch.arange(6, dtype=torch.float64).reshape(2, 3) + step,
            "global_scalar": torch.tensor(step, dtype=torch.int64),
        }

    def drain(limit: int) -> None:
        nonlocal expected
        count, views = ring.read_span()
        count = min(count, limit)
        assert views["global_matrix"].shape[1:] == (2, 3)
        assert views["global_scalar"].ndim == 1
        for offset in range(count):
            snapshot = {name: view[offset].clone() for name, view in views.items()}
            reference = batch(expected + offset)
            for name in reference:
                torch.testing.assert_close(snapshot[name], reference[name])
            retained.append((expected + offset, snapshot))
        ring.commit_reads(count)
        expected += count

    for start in (0, 3, 6):
        for step in range(start, start + 3):
            payload = batch(step)
            assert ring.push(payload)
            for tensor in payload.values():
                tensor.zero_()
        assert ring.is_full()
        assert not ring.push(batch(999))
        drain(2)
        drain(3)
    assert expected == 9
    assert ring.read_span() == (0, {})
    for step, snapshot in retained:
        for name, reference in batch(step).items():
            torch.testing.assert_close(snapshot[name], reference)


def test_field_names_remain_unique_across_scopes() -> None:
    with pytest.raises(ValueError, match="unique"):
        TransitionSchema(
            (
                TransitionField("state", (2,), torch.float32),
                TransitionField("state", (2,), torch.float32, scope="global"),
            )
        )
