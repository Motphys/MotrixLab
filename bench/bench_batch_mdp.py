# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""One batch MDP source executed by NumPy and Torch; no simulator required."""

import argparse
import json
from time import perf_counter

import numpy as np
import torch

from motrix_env_core.array import mx
from motrix_env_core.manager.dispatch import dispatch


@dispatch
def batch_rewards(position, default, weights, action, previous, velocity, command):
    delta = position - default
    pose = mx.sum(delta * delta * weights, axis=-1)
    change = action - previous
    rate = mx.sum(change * change, axis=-1)
    error = velocity - command
    tracking = mx.exp(-mx.sum(error * error, axis=-1))
    return pose, rate, tracking


def lane_rewards(position, default, weights, action, previous, velocity, command):
    """Independent scalar oracle and lane-fused performance reference."""
    count, joints = position.shape
    pose = np.empty(count, dtype=position.dtype)
    rate = np.empty(count, dtype=position.dtype)
    tracking = np.empty(count, dtype=position.dtype)
    for lane in range(count):
        p = np.float32(0)
        r = np.float32(0)
        e = np.float32(0)
        for joint in range(joints):
            delta = position[lane, joint] - default[joint]
            p += delta * delta * weights[joint]
            change = action[lane, joint] - previous[lane, joint]
            r += change * change
        for component in range(velocity.shape[1]):
            error = velocity[lane, component] - command[lane, component]
            e += error * error
        pose[lane] = p
        rate[lane] = r
        tracking[lane] = np.exp(-e)
    return pose, rate, tracking


def inputs(count, joints, seed=7):
    rng = np.random.default_rng(seed)
    shapes = ((count, joints), (joints,), (joints,), (count, joints), (count, joints), (count, 3), (count, 3))
    return tuple(rng.normal(size=shape).astype(np.float32) for shape in shapes)


def measure(function, args, repeats):
    start = perf_counter()
    result = function(*args)
    first_ms = (perf_counter() - start) * 1000
    for _ in range(10):
        function(*args)
    samples = []
    for _ in range(7):
        start = perf_counter()
        for _ in range(repeats):
            function(*args)
        samples.append((perf_counter() - start) * 1e6 / repeats)
    return result, {"first_call_ms": first_ms, "median_us": float(np.median(samples))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--joints", type=int, default=29)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=200)
    options = parser.parse_args()
    torch.set_num_threads(options.threads)
    arrays = inputs(options.num_envs, options.joints)
    tensors = tuple(torch.from_numpy(value) for value in arrays)
    expected = lane_rewards(*arrays)
    functions = {
        "numpy_batch": (batch_rewards, arrays),
        "numpy_lane": (lane_rewards, arrays),
        "torch_eager_cpu": (batch_rewards, tensors),
        "torch_inductor_cpu": (torch.compile(batch_rewards, fullgraph=True), tensors),
    }
    report = {"num_envs": options.num_envs, "joints": options.joints, "threads": options.threads, "results": {}}
    for name, (function, args) in functions.items():
        result, timing = measure(function, args, options.repeats)
        for actual, reference in zip(result, expected, strict=True):
            if isinstance(actual, torch.Tensor):
                actual = actual.detach().numpy()
            np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=3e-5)
            assert actual.dtype == reference.dtype
        report["results"][name] = timing
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
