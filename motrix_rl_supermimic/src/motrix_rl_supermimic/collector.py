# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Collector worker: physics rollout process feeding the transition ring.

Spawned through :class:`motrix_rl.worker.Worker`. Builds its own real
environment and sampling runtime, without constructing an imagined TaskMDP.
Physics runs on CPU; the policy is refreshed from the latest inference
bundle before every block."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event

import torch

from motrix_env_core import registry as env_registry
from motrix_rl.transport.common import bind_flat_params
from motrix_rl.transport.weight_channel import WeightChannelShared, weight_receiver_for
from motrix_rl.worker import Worker
from motrix_rl_supermimic.config import AgentCfg
from motrix_rl_supermimic.env import SuperMimicEnv, SuperMimicEnvCfg
from motrix_rl_supermimic.networks import NormStatsView, Policy, build_policy
from motrix_rl_supermimic.randomness import derive_seed, model_initialization, resolve_seed
from motrix_rl_supermimic.replay import transition_schema
from motrix_rl_supermimic.transport import connect_producer


@dataclass(frozen=True)
class CollectorWorker(Worker):
    """Collector task: configuration and IPC objects only; runtime lives in ``run``."""

    env_name: str
    sim: str | None
    num_envs: int
    seed: int | None
    action_noise: float
    action_limit: float
    block_steps: int
    agent: AgentCfg
    policy_device: str
    weight_shared: WeightChannelShared
    weight_params: list[torch.Tensor]
    stop: Event
    errors: Connection
    startup: Connection
    preregister: str | None = None
    amp_dtype: torch.dtype | None = None

    def run(self) -> None:
        env: SuperMimicEnv | None = None
        try:
            root_seed = resolve_seed(self.seed)
            env_seed = derive_seed(root_seed, "environment", worker_id=0)
            # Manager numerical terms share the process-default Torch stream.
            torch.manual_seed(env_seed)
            if self.preregister:
                import importlib

                module = importlib.import_module(self.preregister)
                module.register()
            build = env_registry.resolve(self.env_name, sim=self.sim)
            if not isinstance(build.env_cfg, SuperMimicEnvCfg):
                raise TypeError("SuperMimic requires a Manager task configuration inheriting SuperMimicEnvCfg")
            env = build.make(num_envs=self.num_envs, mode="train", seed=env_seed)
            if not isinstance(env, SuperMimicEnv):
                raise TypeError("SuperMimic requires an environment inheriting SuperMimicEnv")
            observation_spec = env.observation_spec
            obs_dim = observation_spec.policy.frame_space.shape[0]
            action_dim = env.action_space.shape[0]
            schema = transition_schema(
                state_dim=env.world_state.size, context_dim=env.world_context.size, action_dim=action_dim
            )
            self.startup.send((schema, observation_spec, env.action_space))
            ring = connect_producer(self.startup.recv())
            self.startup.close()

            with model_initialization(derive_seed(root_seed, "model/policy")):
                reference_action = env.cfg.reference_action() if self.agent.pi_ref_skip else None
                pi = build_policy(
                    self.agent,
                    obs_dim,
                    observation_spec["goal"].space.shape[0],
                    action_dim,
                    "cpu",
                    observation_spec=observation_spec.policy,
                    reference_action=reference_action,
                    amp_dtype=self.amp_dtype,
                )
            inference_dev = torch.device(self.policy_device)
            pi = pi.to(inference_dev)
            receiver = weight_receiver_for(self.weight_shared, self.weight_params)
            flat_params = {"policy": bind_flat_params(pi)} if inference_dev.type == "cuda" else None

            modules: dict[str, Policy] = {"policy": pi}
            norm_views = {"obs_norm": NormStatsView(pi.obs_norm), "goal_norm": NormStatsView(pi.goal_norm)}
            env.initialize()
            env_device = env.device
            exploration = torch.Generator(device=env_device).manual_seed(
                derive_seed(root_seed, "exploration", worker_id=0)
            )
            result = env.state
            receiver.maybe_load(modules, norm_views, flat_params)
            with torch.inference_mode():
                while not self.stop.is_set():
                    receiver.maybe_load(modules, norm_views, flat_params)
                    for _ in range(self.block_steps):
                        obs = (
                            result.obs["policy"]
                            .to(inference_dev)
                            .reshape(self.num_envs, observation_spec.policy.history_length, obs_dim)
                        )
                        goal = result.obs["goal"]
                        action = pi(obs, goal.to(inference_dev)).to(env_device)
                        noise = torch.randn(action.shape, dtype=action.dtype, device=env_device, generator=exploration)
                        action = (action + self.action_noise * noise).clamp(-self.action_limit, self.action_limit)
                        result = env.step(action)
                        # Both ring producers currently accept host tensors. Keep
                        # this transfer at transport ingress, not in env computation.
                        transition = {name: value.cpu() for name, value in env.snapshot_transition(result).items()}
                        while not ring.push(transition):
                            if self.stop.is_set():
                                return
                            time.sleep(0.001)
                        if self.stop.is_set():
                            return
        except BaseException:
            try:
                self.errors.send(traceback.format_exc())
            except (BrokenPipeError, EOFError, OSError):
                # The original worker failure must still produce a nonzero exit.
                pass
            raise
        finally:
            self.startup.close()
