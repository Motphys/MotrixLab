# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Collector worker: physics rollout process feeding the transition ring.

Spawn target. Builds its own environment (registry name + sim string are
spawn-safe) so collector and learner each hold a TaskMDP instance built
from the same config. Physics runs on CPU; the policy is refreshed from
the latest inference bundle before every block.

Reset-retry contract: on an engine-level numerical failure the whole block
is discarded (never mixed into the replay), the batch is reset, the history
is re-seeded from the post-reset observation, and collection restarts with a
fresh block. Old observations/goals are never paired with reset transitions.
"""

from __future__ import annotations

import time
import traceback

import torch

from motrix_env_core import registry as env_registry
from motrix_rl.transport.weight_channel import HostWeightReceiver
from motrix_rl_supermimic.mbrl_env import MbrlEnvBase
from motrix_rl_supermimic.networks import Policy
from motrix_rl_supermimic.task_mdp import optional_task_hook


def _numerical_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return "numerical" in message or "simulation step" in message


def collector_process(  # noqa: PLR0913
    env_name: str,
    sim: str | None,
    num_envs: int,
    seed: int | None,
    goal_frames: int,
    action_noise: float,
    action_limit: float,
    block_steps: int,
    hidden: int,
    hist: int,
    pi_ref_skip: bool,
    policy_device: str,
    ring_cursors,
    ring_slots,
    ring,
    weight_shared,
    weight_params,
    stop,
    errors,
    preregister: str | None = None,
) -> None:
    env: MbrlEnvBase | None = None
    try:
        # Keep the collector's TaskMDP/kinematics on the CPU regardless of
        # CUDA visibility; the learner-side copy lives on its device.
        import os

        os.environ["SUPERMIMIC_TASK_DEVICE"] = "cpu"
        if seed is not None:
            torch.manual_seed(seed)
        if preregister:
            import importlib

            module = importlib.import_module(preregister)
            register = getattr(module, "register", None)
            if register is not None:
                register()
        env = env_registry.resolve(env_name, sim=sim).make(num_envs=num_envs, mode="train", seed=seed)
        task, spec = env.task_mdp, env.spec
        from motrix_rl_supermimic.learner import build_policy

        class _PolicyCfg:
            agent = type("A", (), {"hidden": hidden, "hist": hist, "z_dim": 32, "pi_ref_skip": pi_ref_skip})()

        pi = build_policy(_PolicyCfg(), task, spec.command_dim * goal_frames, "cpu")
        inference_dev = torch.device(policy_device) if torch.cuda.is_available() else torch.device("cpu")
        if inference_dev.type == "cuda":
            inference_dev = torch.device("cuda", min(inference_dev.index or 0, torch.cuda.device_count() - 1))
        pi = pi.to(inference_dev)
        receiver = HostWeightReceiver(weight_shared, weight_params)
        from motrix_rl_supermimic.networks import NormStatsView

        norm_views = {"obs_norm": NormStatsView(pi.obs_norm), "goal_norm": NormStatsView(pi.goal_norm)}
        transport_hook = optional_task_hook(task, "transport_fields")
        transport_buffers = transport_hook() if transport_hook is not None else {}
        if ring_slots is not None:
            from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
            from motrix_rl_supermimic.async_impl.schema import transition_schema

            ring = GpuIpcTransitionRing(
                ring_cursors, ring_slots, transition_schema(spec, goal_frames),
                ring_slots.shape[0], num_envs,
            )
        history = torch.zeros(num_envs, hist, spec.observation_dim)
        total_steps = 0  # cumulative per-env control steps across blocks
        env.init_state()
        with torch.inference_mode():
            while not stop.is_set():
                receiver.maybe_load({"policy": pi}, norm_views)
                probe_step = getattr(env, "probe_step", None)
                if probe_step is not None:
                    # Curriculum guidance; failures here are real errors and
                    # surface like any other collector failure.
                    probe_step(pi, hist, goal_frames, total_steps, str(inference_dev))
                rows: list[dict] = []
                discarded = 0
                for _ in range(block_steps):
                    obs = task.add_observation_noise(
                        task.observation(env.state(), env.prev_action, env.context, env.frame)
                    )
                    shifted = torch.cat((history[:, 1:], obs[:, None]), 1)
                    history = torch.where(
                        (env.episode_steps == 0)[:, None, None], obs[:, None].expand_as(history), shifted
                    )
                    goal = task.command(env.frame, goal_frames).flatten(-2)
                    action = pi(history.to(inference_dev), goal.to(inference_dev)).cpu()
                    action = (action + action_noise * torch.randn_like(action)).clamp(-action_limit, action_limit)
                    try:
                        env.step(action)
                    except RuntimeError as exc:
                        if not _numerical_failure(exc):
                            raise
                        # Engine numerical failure: discard the whole block so
                        # stale observations/goals are never written next to
                        # fresh reset transitions, then restart cleanly.
                        discarded += 1
                        rows = []
                        env.init_state()
                        obs = task.add_observation_noise(
                            task.observation(env.state(), env.prev_action, env.context, env.frame)
                        )
                        history = obs[:, None, :].expand(num_envs, hist, spec.observation_dim).clone()
                        break
                    transition = env.last_transition
                    transition["obs"] = obs
                    transition["goal"] = goal
                    rows.append(transition)
                if rows:
                    for name, buffer in transport_buffers.items():
                        for row in rows:
                            row[name] = buffer.clone()
                    for transition in rows:
                        while not ring.push(transition):
                            if stop.is_set():
                                return
                            time.sleep(0.001)
                    total_steps += len(rows)
    except BaseException:
        try:
            errors.send(traceback.format_exc())
        except Exception:
            pass
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if close is not None:
                close()
