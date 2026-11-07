# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reset-path microbenchmark: static full-batch masked vs gathered direct.

Four variants at k in {64, 512, 2048} selected lanes of a 2048 batch:
- static full-batch + where merge, compiled (dynamic=False)  [current design]
- static full-batch + where merge, eager
- gather/scatter on selected rows, compiled (dynamic=True)   [dynamic design]
- gather/scatter on selected rows, eager
"""

import sys
import time

import torch

N, W = 2048, 32


def reset_math(t):
    t.mul_(1.7).remainder_(97.0)
    t.add_(torch.rand_like(t) * 0.1)
    return t


def static_full_batch(state, clone, mask, ids):
    mask.fill_(False)
    mask[ids] = True
    clone.copy_(state)
    reset_math(clone)
    state.copy_(torch.where(mask, clone, state))


def gathered_direct(state, ids):
    rows = state.index_select(0, ids)
    reset_math(rows)
    state.index_copy_(0, ids, rows)


def bench(fn, args, iters=2000, warmup=300):
    for _ in range(warmup):
        fn(*args)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
    return best


def main():
    for threads in (8, 96):
        torch.set_num_threads(threads)
        print(f"== torch threads = {threads} ==")
        state = torch.randn(N, W)
        clone = torch.empty_like(state)
        mask = torch.zeros(N, 1, dtype=torch.bool)
        st_c = torch.compile(static_full_batch, dynamic=False, fullgraph=True)
        gd_c = torch.compile(gathered_direct, dynamic=True, fullgraph=True)
        for k in (64, 512, 2048):
            ids = torch.randperm(N)[:k]
            rows = {
                "static+where compiled": bench(st_c, (state, clone, mask, ids)),
                "static+where eager   ": bench(static_full_batch, (state, clone, mask, ids)),
                "gather    compiled   ": bench(gd_c, (state, ids)),
                "gather    eager      ": bench(gathered_direct, (state, ids)),
            }
            print(f"-- k = {k:4d}/{N} --")
            for name, us in rows.items():
                print(f"   {name} {us:9.1f} us")
        print(flush=True)


if __name__ == "__main__":
    main()
