# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Randomize named primitive sizes without modifying mass or inertia.

Run with an MJCF scene and its primitive geom names::

    python examples/geom_size_randomization.py scene.xml cube finger --num-envs 4

Each environment/geom receives one uniform scale shared by its active size
components. Compile once, then fill buffers again to resample; sizes always
scale the nominal model dimensions rather than accumulating random scales.
"""

import argparse
from pathlib import Path

import torch

from motrix_env_core.config.scene.base import SceneCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.sim.model import GeomSpecsQuery
from motrix_env_core.sim.write import GeomSizeWrite
from motrix_env_motrixsim.runtime import MotrixSimBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("geoms", nargs="+")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--scale-min", type=float, default=0.85)
    parser.add_argument("--scale-max", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_envs <= 0 or not 0 < args.scale_min <= args.scale_max < float("inf"):
        parser.error("Require num-envs > 0 and finite 0 < scale-min <= scale-max")

    torch.manual_seed(args.seed)
    scene = SceneCfg(file=args.scene)
    names = tuple(args.geoms)
    backend = MotrixSimBackend(scene, SimCfg(), args.num_envs)
    model = backend.model_compiler.compile(scene, {"geoms": GeomSpecsQuery(names=names)})
    nominal = torch.tensor([model.others["geoms"][name].size for name in names])
    program = backend.write_compiler.compile({"size": GeomSizeWrite(names)})
    scales = torch.empty(args.num_envs, len(names), 1).uniform_(args.scale_min, args.scale_max)
    program.buffer("size").copy_(nominal * scales)
    program.execute()
    print("Geom order:", names)
    print("Sampled sizes (env, geom, component):\n", program.buffer("size"))


if __name__ == "__main__":
    main()
