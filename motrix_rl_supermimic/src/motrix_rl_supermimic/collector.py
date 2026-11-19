# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Collector worker: physics rollout process feeding the transition ring.

Spawn target. Builds its own real environment and sampling binding, without
constructing an imagined TaskMDP. Physics runs on CPU; the policy is refreshed
from the latest inference bundle before every block."""

from __future__ import annotations

import time
import traceback
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event

import torch

from motrix_env_core import registry as env_registry
from motrix_rl.transport.common import bind_flat_params
from motrix_rl.transport.weight_channel import WeightChannelShared, weight_receiver_for
from motrix_rl_supermimic.config import AgentCfg
from motrix_rl_supermimic.mbrl_env import MbrlEnv, MbrlEnvCfg
from motrix_rl_supermimic.networks import NormStatsView, Policy, build_policy
from motrix_rl_supermimic.randomness import derive_seed, model_initialization, resolve_seed
from motrix_rl_supermimic.transport import connect_producer


def collector_process(  # noqa: PLR0913
    env_name: str,
    sim: str | None,
    num_envs: int,
    seed: int | None,
    action_noise: float,
    action_limit: float,
    block_steps: int,
    agent: AgentCfg,
    policy_device: str,
    weight_shared: WeightChannelShared,
    weight_params: list[torch.Tensor],
    stop: Event,
    errors: Connection,
    startup: Connection,
    preregister: str | None = None,
    amp_dtype: torch.dtype | None = None,
) -> None:
    env: MbrlEnv | None = None
    try:
        root_seed = resolve_seed(seed)
        env_seed = derive_seed(root_seed, "environment", worker_id=0)
        # Manager numerical terms share the process-default Torch stream.
        torch.manual_seed(env_seed)
        exploration = torch.Generator(device="cpu").manual_seed(derive_seed(root_seed, "exploration", worker_id=0))
        if preregister:
            import importlib

            module = importlib.import_module(preregister)
            module.register()
        build = env_registry.resolve(env_name, sim=sim)
        if not isinstance(build.env_cfg, MbrlEnvCfg):
            raise TypeError("SuperMimic requires a Manager task configuration inheriting MbrlEnvCfg")
        env = build.make(num_envs=num_envs, mode="train", seed=env_seed)
        if not isinstance(env, MbrlEnv):
            raise TypeError("SuperMimic requires an environment inheriting MbrlEnv")
        spec = env.state_spec
        observation_spec = env.observation_spec
        obs_dim = observation_spec.policy.frame_space.shape[0]
        action_dim = env.action_space.shape[0]
        startup.send((spec, observation_spec, env.action_space))
        ring = connect_producer(startup.recv())
        startup.close()

        with model_initialization(derive_seed(root_seed, "model/policy")):
            reference_action = env.cfg.reference_action() if agent.pi_ref_skip else None
            pi = build_policy(
                agent,
                obs_dim,
                observation_spec["goal"].space.shape[0],
                action_dim,
                "cpu",
                observation_spec=observation_spec.policy,
                reference_action=reference_action,
                amp_dtype=amp_dtype,
            )
        inference_dev = torch.device(policy_device)
        pi = pi.to(inference_dev)
        receiver = weight_receiver_for(weight_shared, weight_params)
        flat_params = {"policy": bind_flat_params(pi)} if inference_dev.type == "cuda" else None

        modules: dict[str, Policy] = {"policy": pi}
        norm_views = {"obs_norm": NormStatsView(pi.obs_norm), "goal_norm": NormStatsView(pi.goal_norm)}
        result = env.init_state()
        receiver.maybe_load(modules, norm_views, flat_params)
        with torch.inference_mode():
            while not stop.is_set():
                receiver.maybe_load(modules, norm_views, flat_params)
                for _ in range(block_steps):
                    obs = (
                        result.obs["policy"]
                        .to(inference_dev)
                        .reshape(num_envs, observation_spec.policy.history_length, obs_dim)
                    )
                    goal = result.obs["goal"]
                    action = pi(obs, goal.to(inference_dev)).cpu()
                    noise = torch.randn(action.shape, dtype=action.dtype, device="cpu", generator=exploration)
                    action = (action + action_noise * noise).clamp(-action_limit, action_limit)
                    result = env.step(action)
                    transition = env.snapshot_transition(result)
                    while not ring.push(transition):
                        if stop.is_set():
                            return
                        time.sleep(0.001)
                    if stop.is_set():
                        return
    except BaseException:
        try:
            errors.send(traceback.format_exc())
        except (BrokenPipeError, EOFError, OSError):
            # The original worker failure must still produce a nonzero exit.
            pass
        raise
    finally:
        startup.close()
