# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Per-environment primitive dimensions and their physical effects."""

from pathlib import Path

import numpy as np
import pytest
import torch

from motrix_env_core.config.scene.base import SceneCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.sim.model import GeomSpecsQuery, GeomType
from motrix_env_core.sim.read import GeomPairCollidingQuery
from motrix_env_core.sim.write import GeomSizeWrite
from motrix_env_motrixsim.runtime import MotrixSimBackend

pytestmark = pytest.mark.integration


@pytest.fixture
def backend(tmp_path: Path) -> MotrixSimBackend:
    scene_file = tmp_path / "scene.xml"
    scene_file.write_text("""<mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="0 0 0.1"/>
        <body name="object" pos="0 0 2"><freejoint/>
          <geom name="box" type="box" size="0.1 0.2 0.3" mass="1"/>
          <geom name="finger" type="capsule" size="0.02 0.1" pos="1 0 0"/>
        </body>
      </worldbody>
    </mujoco>""")
    return MotrixSimBackend(SceneCfg(file=scene_file), SimCfg(gravity=(0, 0, 0)), 3)


def test_size_scales_nominal_geometry_only_in_selected_environments(backend: MotrixSimBackend) -> None:
    names = ("finger", "box")
    metadata = backend.model_compiler.compile(SceneCfg(), {"geoms": GeomSpecsQuery(names=names)}).others["geoms"]
    assert metadata["finger"].geom_type is GeomType.CAPSULE
    assert metadata["box"].geom_type is GeomType.BOX
    nominal = torch.tensor([metadata[name].size for name in names])
    link = backend._model.get_link("object")
    mass = link.get_mass_override(backend._data).copy()
    inertia = link.get_inertia_override(backend._data).copy()
    program = backend.write_compiler.compile({"size": GeomSizeWrite(names)})
    assert program.buffer("size").shape == (3, 2, 3)
    scales = torch.tensor([[0.97, 0.85], [1.0, 1.0], [1.03, 1.15]])
    program.buffer("size").copy_(nominal * scales.unsqueeze(-1))
    program.execute(torch.tensor([2, 0]))

    for col, name in enumerate(names):
        geom = backend._model.get_geom(name)
        effective = geom.get_size_override(backend._data)
        np.testing.assert_allclose(
            effective,
            nominal[col, : effective.shape[-1]].numpy() * scales[:, col, None].numpy(),
        )
    box = backend._model.get_geom("box")
    bounds = box.get_local_aabb(backend._data)
    np.testing.assert_allclose(bounds[:, 1], nominal[1].numpy() * scales[:, 1, None].numpy())
    np.testing.assert_array_equal(link.get_mass_override(backend._data), mass)
    np.testing.assert_array_equal(link.get_inertia_override(backend._data), inertia)
    # Model queries continue to describe defaults, not the last randomized row.
    after = backend.model_compiler.compile(SceneCfg(), {"geoms": GeomSpecsQuery(names=names)})
    assert after.others["geoms"] == metadata

    # Reset only selected rows; unselected randomized dimensions survive.
    reset = backend.write_compiler.compile({"size": GeomSizeWrite(names)}, reset=True)
    reset.buffer("size").copy_(nominal.expand(3, -1, -1))
    reset.execute(torch.tensor([0]))
    expected = nominal[1].expand(3, -1).clone()
    expected[2] *= 1.15
    np.testing.assert_allclose(box.get_size_override(backend._data), expected)


def test_size_changes_collision_detection(tmp_path: Path) -> None:
    scene_file = tmp_path / "collision.xml"
    scene_file.write_text("""<mujoco><worldbody>
      <geom name="obstacle" type="sphere" size="0.1" pos="0.25 0 0"/>
      <body name="object"><freejoint/>
        <geom name="box" type="box" size="0.1 0.1 0.1" mass="1"/>
      </body>
    </worldbody></mujoco>""")
    backend = MotrixSimBackend(SceneCfg(file=scene_file), SimCfg(gravity=(0, 0, 0)), 2)
    program = backend.write_compiler.compile({"size": GeomSizeWrite(("box",))})
    program.buffer("size").copy_(torch.tensor([[[0.1, 0.1, 0.1]], [[0.2, 0.1, 0.1]]]))
    program.execute()
    backend.step(1)
    reads = backend.compile_reads({"collision": GeomPairCollidingQuery(pairs=(("box", "obstacle"),))})
    reads.execute()
    torch.testing.assert_close(reads["collision"], torch.tensor([[0.0], [1.0]]))


@pytest.mark.parametrize("names", [(), ("box", "box"), ("missing",), ("floor",)])
def test_size_write_rejects_invalid_targets(backend: MotrixSimBackend, names: tuple[str, ...]) -> None:
    with pytest.raises((ValueError, KeyError)):
        backend.write_compiler.compile({"size": GeomSizeWrite(names)})
