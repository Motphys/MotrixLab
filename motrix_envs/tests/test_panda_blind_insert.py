# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch

from motrix_env_core import registry

TASK_DIR = Path(__file__).parents[1] / "src" / "motrix_envs" / "manipulation" / "panda_blind_insert"


def _make(num_envs: int = 2):
    return registry.make("panda-blind-insertion", num_envs=num_envs)


def test_panda_blind_insert_asset_tree_is_open() -> None:
    required = (
        "scene.xml",
        "mjx_panda.xml",
        "assets",
        "PANDA_MODEL_LICENSE",
    )
    assert all((TASK_DIR / path).exists() for path in required)


def test_panda_blind_insert_socket_is_kinematic() -> None:
    import xml.etree.ElementTree as ET

    root = ET.parse(TASK_DIR / "scene.xml").getroot()
    socket = root.find("./worldbody/body[@name='blind_socket']")
    assert socket is not None
    assert socket.attrib.get("mocap") == "true"
    walls = [g for g in socket.findall("geom") if g.attrib.get("name", "").startswith("socket_wall_")]
    assert len(walls) == 4  # NIST-style square hole


def test_panda_blind_insert_direct_env_contract() -> None:
    env = _make()
    state = env.init_state()
    assert tuple(state.obs.policy.shape) == (2, 1260)  # 30 x 42
    assert state.obs.value is not None
    assert tuple(state.obs.value.shape) == (2, 52)  # 42 + 10 privileged

    state = env.step(torch.zeros((2, 7), dtype=torch.float32))
    assert tuple(state.obs.policy.shape) == (2, 1260)
    assert torch.isfinite(state.obs.policy).all()
    assert torch.isfinite(state.obs.value).all()
    assert torch.isfinite(state.reward).all()


def test_panda_blind_insert_reset_places_peg_near_hole() -> None:
    env = _make(num_envs=8)
    env.init_state()
    metrics = env._metrics()
    peg = env.sim_data["peg_pos"].numpy()
    gap_mm = (peg[:, 2] - 0.040 - 0.040) * 1000.0
    # Settled airborne start: near-contact but not inserted, no ground contact.
    assert float(np.median(gap_mm)) < 8.0
    assert not metrics["contact"].any()


def test_panda_blind_insert_closed_hold_survives_one_hundred_steps() -> None:
    env = _make(num_envs=4)
    env.init_state()
    for _ in range(10):  # settle completes inside reset; hold a beat
        env.step(torch.zeros((4, 7), dtype=torch.float32))
    initial = env.sim_data["peg_pos"].numpy().copy()
    for _ in range(100):
        env.step(torch.zeros((4, 7), dtype=torch.float32))
    final = env.sim_data["peg_pos"].numpy()
    drift_mm = np.abs(final[:, 2] - initial[:, 2]) * 1000.0
    assert float(np.percentile(drift_mm, 95)) < 2.5
    assert env._grasped.all()
