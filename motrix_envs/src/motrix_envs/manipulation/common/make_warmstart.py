# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Prepare a stage-2 warm-start checkpoint from a stage-1 run.

The RSL-RL trainer cannot resume (``Resume is not implemented``), and a plain
warm start would inherit two stage-1 artifacts we do NOT want:

* ``optimizer_state_dict`` — Adam moments tuned to the stage-1 loss surface;
* the inflated noise std (measured 1.72 at iter 1500).

This script loads a stage-1 ``latest.pt``, keeps ONLY actor/critic weights,
resets the actor std parameter to ``--std`` (default 0.5), and writes a fresh
checkpoint the trainer can start from. Import it into the new run via the
``BoundedNoiseMLPModel`` band clamp (0.20-0.80) which keeps it there.

Usage:
    python make_warmstart.py STAGE1_CKPT OUT_CKPT [std]
"""

from __future__ import annotations

import sys

import torch


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    std = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    ck = torch.load(src, map_location="cpu", weights_only=False)

    actor = dict(ck["actor_state_dict"])
    # Reset the scalar std parameter (state-independent noise).
    if "std" in actor:
        n = actor["std"].numel()
        actor["std"] = torch.full((n,), std, dtype=actor["std"].dtype)
        print(f"actor std reset -> {std}")
    if "log_std" in actor:
        n = actor["log_std"].numel()
        actor["log_std"] = torch.full((n,), float(torch.log(torch.tensor(std))), dtype=actor["log_std"].dtype)
        print(f"actor log_std reset -> log({std})")

    out = {
        "actor_state_dict": actor,
        "critic_state_dict": ck["critic_state_dict"],
        "iter": 0,
        "infos": {"warm_start_from": src, "std_reset": std, "optimizer_discarded": True},
    }
    torch.save(out, dst)
    print(f"wrote {dst}")
    print("optimizer_state_dict: DISCARDED (Adam re-initialized by the trainer)")


if __name__ == "__main__":
    main()
