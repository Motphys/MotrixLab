# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest
from motrix_rl_builtin.fastsac.config import FastSacCfg
from motrix_rl_builtin.fastsac.framework import FastSacProvider

from motrix_rl import frameworks


def test_builtin_plugin_registers_fastsac():
    assert frameworks.supported_agents("motrix") == ("fastsac",)
    assert frameworks.get_config_type("motrix", "fastsac") is FastSacCfg
    assert frameworks.supported_train_backends("motrix", "fastsac") == ("torch",)
    assert isinstance(frameworks.get_agent_provider("motrix", "fastsac", "torch"), FastSacProvider)


@pytest.mark.parametrize(("iterations", "num_envs", "expected"), [(1, 1, 1), (100, 64, 6400), (0, 8, 0)])
def test_fastsac_total_env_steps_counts_one_batch_per_iteration(iterations, num_envs, expected):
    cfg = FastSacCfg()
    cfg.trainer.num_learning_iterations = iterations

    assert FastSacProvider().total_env_steps(cfg, num_envs) == expected
