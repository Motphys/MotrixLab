# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Purpose streams isolate model construction from sampling and exploration."""

from pathlib import Path

import pytest
import torch
from _helpers import make_toy_task
from motrix_rl_supermimic.config import AgentCfg, SuperMimicCfg
from motrix_rl_supermimic.learner import Learner
from motrix_rl_supermimic.randomness import derive_seed, model_initialization, resolve_seed


def test_seed_derivation_is_stable_and_separates_purposes_and_workers() -> None:
    first = derive_seed(19, "exploration", 2)
    assert first == derive_seed(19, "exploration", 2)
    assert 0 <= first < 2**63
    assert len({first, derive_seed(19, "environment", 2), derive_seed(19, "exploration", 3)}) == 3


def test_unspecified_seed_uses_system_entropy_without_resetting_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import motrix_rl_supermimic.randomness as randomness

    before = torch.get_rng_state()
    monkeypatch.setattr(randomness.secrets, "randbits", lambda bits: 123)
    assert resolve_seed(None) == 123
    assert resolve_seed(42) == 42
    torch.testing.assert_close(torch.get_rng_state(), before)


def test_model_initialization_restores_cpu_stream_and_is_order_independent() -> None:
    before = torch.get_rng_state()
    with model_initialization(derive_seed(7, "model/policy")):
        expected = torch.nn.Linear(3, 4)
    torch.testing.assert_close(torch.get_rng_state(), before)
    with model_initialization(derive_seed(7, "model/unrelated")):
        torch.nn.Linear(17, 23)
    with model_initialization(derive_seed(7, "model/policy")):
        actual = torch.nn.Linear(3, 4)
    torch.testing.assert_close(actual.weight, expected.weight)
    torch.testing.assert_close(actual.bias, expected.bias)
    torch.testing.assert_close(torch.get_rng_state(), before)


def test_auxiliary_model_does_not_shift_environment_or_exploration_stream() -> None:
    environment_seed = derive_seed(43, "environment")
    exploration_seed = derive_seed(43, "exploration")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(environment_seed)
        expected_environment = torch.rand(12)
        torch.random.default_generator.manual_seed(environment_seed)
        exploration = torch.Generator(device="cpu").manual_seed(exploration_seed)
        expected_exploration = torch.randn(12, generator=exploration)
        exploration.manual_seed(exploration_seed)
        with model_initialization(derive_seed(43, "model/auxiliary")):
            torch.nn.Linear(57, 91)
        torch.testing.assert_close(torch.rand(12), expected_environment)
        torch.rand(37)  # Environment noise consumes only the default stream.
        torch.testing.assert_close(torch.randn(12, generator=exploration), expected_exploration)


def test_model_initialization_restores_stream_when_factory_raises() -> None:
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="factory"):
        with model_initialization(7):
            torch.rand(20)
            raise RuntimeError("factory")
    torch.testing.assert_close(torch.get_rng_state(), before)


def test_learner_model_roles_do_not_consume_default_rng_or_couple_widths(tmp_path: Path) -> None:
    task = make_toy_task()
    before = torch.get_rng_state()
    first = Learner(SuperMimicCfg(agent=AgentCfg(hidden=8)), task, 1, 0.02, "cpu", tmp_path, seed=17)
    torch.testing.assert_close(torch.get_rng_state(), before)
    second = Learner(SuperMimicCfg(agent=AgentCfg(hidden=8, wm_hidden=16)), task, 1, 0.02, "cpu", tmp_path, seed=17)
    torch.testing.assert_close(torch.get_rng_state(), before)
    for name in ("pi", "critic"):
        for actual, expected in zip(getattr(first, name).parameters(), getattr(second, name).parameters(), strict=True):
            torch.testing.assert_close(actual, expected)


def test_unseeded_learner_model_construction_preserves_default_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import motrix_rl_supermimic.randomness as randomness

    task = make_toy_task()
    before = torch.get_rng_state()
    monkeypatch.setattr(randomness.secrets, "randbits", lambda bits: 51)
    learner = Learner(SuperMimicCfg(agent=AgentCfg(hidden=8)), task, 1, 0.02, "cpu", tmp_path)
    assert learner.root_seed == 51
    torch.testing.assert_close(torch.get_rng_state(), before)


def test_checkpoint_restores_root_and_learner_default_rng(tmp_path: Path) -> None:
    task = make_toy_task()
    cfg = SuperMimicCfg(agent=AgentCfg(hidden=8))
    first = Learner(cfg, task, 1, 0.02, "cpu", tmp_path, seed=17)
    first.save(0)
    expected_state = torch.get_rng_state()
    torch.rand(50)
    restored = Learner(cfg, task, 1, 0.02, "cpu", tmp_path, seed=99)
    restored.load(str(tmp_path / "last.pt"))
    assert restored.root_seed == 17
    torch.testing.assert_close(torch.get_rng_state(), expected_state)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_model_initialization_restores_all_visible_cuda_streams() -> None:
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all()
    with model_initialization(31):
        torch.rand(3)
        for device in range(torch.cuda.device_count()):
            torch.rand(3, device=f"cuda:{device}")
    torch.testing.assert_close(torch.get_rng_state(), cpu_before)
    for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_before, strict=True):
        torch.testing.assert_close(actual, expected)
