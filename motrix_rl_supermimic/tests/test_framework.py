# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic provider contracts independent of application registration."""


def test_total_env_steps():
    from motrix_rl_supermimic.config import SuperMimicCfg, TrainerCfg
    from motrix_rl_supermimic.provider import SuperMimicProvider

    cfg = SuperMimicCfg(trainer=TrainerCfg(iterations=100))
    assert SuperMimicProvider().total_env_steps(cfg, 8) == 800
