# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Layer-3 pipeline equivalence: the DDP update path must match single-process.

Two ranks on gloo (CPU), each holding the SAME parameter clone and feeding
half of a fixed batch, run the real ``FastSacAgent`` update methods with
gradient averaging. A single-process reference runs the same methods on the
full batch. After every step the parameters must agree to float tolerance —
this pins down _allreduce_module_grads (parameter coverage, ordering),
log_alpha averaging and optimizer stepping, without any environment.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp

OBS, COBS, ACT = 6, 4, 2


def _make_agent(world_size: int, seed: int) -> object:
    from pathlib import Path

    from omegaconf import OmegaConf

    from motrix_rl.fastsac.agent import FastSacAgent
    from motrix_rl.fastsac.config import FastSacAgentCfg

    yaml_path = Path(__file__).resolve().parents[2] / "configs" / "algo_base" / "motrix.fastsac.yaml"
    base = OmegaConf.merge(OmegaConf.structured(FastSacAgentCfg), OmegaConf.load(yaml_path)["agent"])
    cfg = OmegaConf.to_object(
        OmegaConf.merge(
            base,
            {
                "actor_hidden_dim": 32,
                "critic_hidden_dim": 32,
                "num_atoms": 11,
                "compile": False,
                "learning_starts": 0,
            },
        )
    )
    torch.manual_seed(seed)
    agent = FastSacAgent(
        obs_dim=OBS,
        critic_obs_dim=COBS,
        act_dim=ACT,
        num_envs=8,
        cfg=cfg,
        device=torch.device("cpu"),
        action_scale=torch.ones(ACT),
        action_bias=torch.zeros(ACT),
        world_size=world_size,
    )
    return agent


def _fixed_batch(rows: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(rows, OBS, generator=g)
    return {
        "obs": obs,
        "next_obs": obs + 0.1 * torch.randn(rows, OBS, generator=g),
        "critic_obs": torch.randn(rows, COBS, generator=g),
        "next_critic_obs": torch.randn(rows, COBS, generator=g),
        "actions": torch.randn(rows, ACT, generator=g),
        "rewards": torch.randn(rows, generator=g),
        "dones": (torch.rand(rows, generator=g) < 0.2).long(),
        "truncations": (torch.rand(rows, generator=g) < 0.1).long(),
        "effective_n_steps": torch.ones(rows, dtype=torch.long),
    }


def _worker(rank: int, result_queue, seeds) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(seeds["port"]))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    import traceback

    # gloo has no ReduceOp.AVG — emulate it (NCCL supports it in production)
    _orig_all_reduce = dist.all_reduce

    def _avg_all_reduce(tensor, op=None):
        if op == dist.ReduceOp.AVG:
            _orig_all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(2)
            return tensor
        return _orig_all_reduce(tensor, op=op)

    dist.all_reduce = _avg_all_reduce

    try:
        agent = _make_agent(2, seeds["init"])
        half = _fixed_batch(64, seeds["batch"])
        # rank r sees half r of the SAME global batch
        half = {k: v[rank * 32 : (rank + 1) * 32] if v.shape[0] == 64 else v for k, v in half.items()}
        rows = 32
        for step in range(seeds["steps"]):
            # Align the RNG streams so the union of both ranks' draws equals
            # the single-process draw: same seed everywhere, rank 1 skips
            # rank 0's worth of normals first (CPU normal_ is a sequential
            # stream, so half+half == full).
            torch.manual_seed(10_000 * (step + 1))
            if rank == 1:
                torch.randn(rows, ACT)  # discard rank 0's draw
            agent._update_main(half)
            if step % 2 == 0:
                torch.manual_seed(10_000 * (step + 1) + 1)
                if rank == 1:
                    torch.randn(rows, ACT)
                agent._update_pol(half)
        import io

        buffer = io.BytesIO()
        torch.save({k: v for k, v in agent.state_dict().items() if "optimizer" not in k}, buffer)
        if rank == 0:
            result_queue.put(buffer.getvalue())
    except Exception:
        if rank == 0:
            result_queue.put({"__error__": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


def test_ddp_update_matches_single_process() -> None:
    """2-rank half-batch + grad AVG == 1-rank full batch, per update step."""
    seeds = {"init": 1234, "batch": 42, "steps": 3, "port": 29711}
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, result_queue, seeds)) for r in range(2)]
    for p in procs:
        p.start()
    import io

    payload = result_queue.get(timeout=120)
    if isinstance(payload, dict) and "__error__" in payload:
        pytest.fail("DDP worker crashed:\n" + payload["__error__"])
    ddp_state = torch.load(io.BytesIO(payload), weights_only=False)
    for p in procs:
        p.join(timeout=60)

    single = _make_agent(1, seeds["init"])
    full = _fixed_batch(64, seeds["batch"])
    for step in range(seeds["steps"]):
        torch.manual_seed(10_000 * (step + 1))
        single._update_main(full)
        if step % 2 == 0:
            torch.manual_seed(10_000 * (step + 1) + 1)
            single._update_pol(full)
    ref_state = {k: v for k, v in single.state_dict().items() if "optimizer" not in k}

    def _walk(prefix, a, b):
        if torch.is_tensor(a):
            if a.is_floating_point():
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4, msg=lambda m: f"{prefix}: {m}")
            else:
                assert torch.equal(a, b), prefix
        elif isinstance(a, dict):
            for k in a:
                _walk(f"{prefix}.{k}", a[k], b[k])
        else:
            assert a == b, prefix

    for key in ref_state:
        _walk(key, ddp_state[key], ref_state[key])
