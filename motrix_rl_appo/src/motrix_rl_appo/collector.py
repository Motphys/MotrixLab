# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Spawned environment owner and chunk-aligned inference snapshots."""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event

import gymnasium as gym
import torch

from motrix_env_core import registry as env_registry
from motrix_env_core.array.env import ArrayEnv
from motrix_rl.transport.common import bind_flat_params
from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.ring import HostTransitionRing
from motrix_rl.transport.weight_channel import weight_receiver_for
from motrix_rl.worker import Worker
from motrix_rl_appo.config import AppoCfg
from motrix_rl_appo.networks import Actor, Critic, build_actor, build_critic
from motrix_rl_appo.transport import CollectorConnection, CollectorMetadata, GpuRingConnection


def publish_chunk(
    ring: HostTransitionRing | GpuIpcTransitionRing,
    chunk: dict[str, torch.Tensor],
    stop: Event,
    timeout: float,
) -> bool:
    """Retry backpressure and drive lazy IPC publication before collecting again."""
    started = time.monotonic()
    while not stop.is_set():
        if ring.push(chunk):
            break
        if time.monotonic() - started > timeout:
            raise TimeoutError("APPO rollout ring remained full")
        stop.wait(0.001)
    else:
        return False
    issued = ring.write_issued_count
    # A push only enqueues IPC H2D/D2D. Without this poll the final chunk can
    # remain unpublished for an entire rollout (or forever at shutdown).
    while ring.write_completed_count < issued:
        if stop.wait(0.001):
            return False
        if time.monotonic() - started > timeout:
            raise TimeoutError("APPO rollout ring write did not complete")
    return True


@dataclass(frozen=True)
class CollectorMetrics:
    returns: list[float]
    lengths: list[int]
    latency_s: float
    weight_version: int
    published_version: int
    env_metrics: dict[str, float]


class ChunkCollector:
    """Own rollout storage; copy borrowed observations before calling step."""

    def __init__(
        self,
        env: ArrayEnv,
        actor: Actor,
        critic: Critic,
        cfg: AppoCfg,
        device: torch.device,
        storage_device: torch.device | None = None,
    ) -> None:
        self.env = env
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.device = device
        storage_device = storage_device or torch.device("cpu")
        self.state = env.initialize()
        self.env_device = self.state.obs.policy.device
        self._warned_timeout_approximation = False
        self.returns = torch.zeros(env.num_envs, device=self.env_device)
        self.lengths = torch.zeros(env.num_envs, dtype=torch.long, device=self.env_device)
        obs_dim = self.state.obs.policy.shape[-1]
        critic_dim = self.state.obs.value_or_policy.shape[-1]
        action_dim = env.action_space.shape[0]
        self._storage: dict[str, torch.Tensor] = {
            "obs": torch.empty((env.num_envs, cfg.rollout_steps, obs_dim), device=storage_device),
            "critic_obs": torch.empty((env.num_envs, cfg.rollout_steps, critic_dim), device=storage_device),
            "actions": torch.empty((env.num_envs, cfg.rollout_steps, action_dim), device=storage_device),
            "rewards": torch.empty((env.num_envs, cfg.rollout_steps), device=storage_device),
            "dones": torch.empty((env.num_envs, cfg.rollout_steps), dtype=torch.bool, device=storage_device),
            "behaviour_logp": torch.empty((env.num_envs, cfg.rollout_steps), device=storage_device),
            "last_critic_obs": torch.empty((env.num_envs, critic_dim), device=storage_device),
        }

    @torch.no_grad()
    def collect(self) -> tuple[dict[str, torch.Tensor], list[float], list[int], dict[str, float]]:
        """Return borrowed storage, valid until the next collect call.

        The caller must finish publication (including device-copy completion)
        before collecting again and overwriting this carrier.
        """
        chunk = self._storage
        returns: list[float] = []
        lengths: list[int] = []
        for step in range(self.cfg.rollout_steps):
            obs = self.state.obs.policy.to(self.device)
            chunk["obs"][:, step].copy_(obs)
            chunk["critic_obs"][:, step].copy_(self.state.obs.value_or_policy)
            actions, logp = self.actor.sample(obs)
            chunk["actions"][:, step].copy_(actions)
            chunk["behaviour_logp"][:, step].copy_(logp)
            state = self.env.step(actions.to(self.env_device))
            reward = state.reward.to(self.device, copy=True)
            timeout = state.truncated & ~state.terminated
            if timeout.any():
                final = state.final_obs
                if final is None:
                    if not self._warned_timeout_approximation:
                        logging.getLogger(__name__).warning(
                            "Environment supplies no final_obs; timeout bootstrap uses the pre-action V(s_t) "
                            "approximation instead of the true final-state value"
                        )
                        self._warned_timeout_approximation = True
                    ids = timeout.nonzero(as_tuple=False).flatten().to(self.device)
                    # Step may overwrite borrowed environment observations. The
                    # carrier already owns the actual pre-action critic input.
                    values = self.critic(chunk["critic_obs"][:, step].to(self.device)[ids])
                else:
                    # Final observations are compact, not indexed by original env id.
                    selected = timeout[final.env_ids]
                    ids = final.env_ids[selected].to(self.device)
                    if ids.numel() != int(timeout.sum()):
                        raise RuntimeError("APPO final observations do not cover all timeout rows")
                    values = self.critic(final.obs.value_or_policy[selected].to(self.device))
                reward[ids] += self.cfg.algorithm.gamma * values
            self.returns += state.reward
            self.lengths += 1
            done = state.done
            returns.extend(self.returns[done].cpu().tolist())
            lengths.extend(self.lengths[done].cpu().tolist())
            self.returns[done] = 0
            self.lengths[done] = 0
            chunk["rewards"][:, step].copy_(reward)
            chunk["dones"][:, step].copy_(done)
            self.state = state
        chunk["last_critic_obs"].copy_(self.state.obs.value_or_policy)
        return chunk, returns, lengths, self.state.process_metrics()


@dataclass(frozen=True)
class CollectorWorker(Worker):
    env_name: str
    sim: str | None
    num_envs: int
    seed: int
    cfg: AppoCfg
    policy_device: str
    stop: Event
    errors: Connection
    startup: Connection
    metrics: Queue

    def run(self) -> None:
        try:
            env = env_registry.make(self.env_name, sim=self.sim, num_envs=self.num_envs, mode="train", seed=self.seed)
            if not isinstance(env, ArrayEnv):
                raise TypeError("APPO requires a Torch ArrayEnv frontend")
            if not isinstance(env.action_space, gym.spaces.Box) or len(env.action_space.shape) != 1:
                raise TypeError("APPO requires a vector Box action space")
            dims = CollectorMetadata(
                env.observation_spec.policy.space.shape[0],
                env.observation_spec.value_or_policy.space.shape[0],
                env.action_space.shape[0],
            )
            self.startup.send(dims)
            if not self.startup.poll(self.cfg.worker_timeout):
                raise TimeoutError("APPO learner did not complete collector handshake")
            connection: CollectorConnection = self.startup.recv()
            self.startup.close()
            device = torch.device(self.policy_device)
            if device.type == "cuda":
                torch.cuda.set_device(device)
            ring = connection.ring.endpoint() if isinstance(connection.ring, GpuRingConnection) else connection.ring
            actor = build_actor(self.cfg.actor, dims.obs_dim, dims.action_dim).to(device).eval()
            critic = build_critic(self.cfg.critic, dims.critic_dim).to(device).eval()
            modules = {"actor": actor, "critic": critic}
            norms = {"actor_norm": actor.obs_normalizer, "critic_norm": critic.obs_normalizer}
            flat = (
                {name: bind_flat_params(module) for name, module in modules.items()} if device.type == "cuda" else None
            )
            receiver = weight_receiver_for(connection.weights, connection.weight_params)
            collector = ChunkCollector(env, actor, critic, self.cfg, device, torch.device(connection.chunk_device))
            while not self.stop.is_set():
                receiver.maybe_load(modules, norms, flat)
                started = time.monotonic()
                chunk, returns, lengths, env_metrics = collector.collect()
                # A metrics record precedes its ring publication, so the parent
                # can always pair exactly one record with the next chunk.
                self.metrics.put(
                    CollectorMetrics(
                        returns,
                        lengths,
                        time.monotonic() - started,
                        receiver.loaded_version,
                        receiver.version,
                        env_metrics,
                    )
                )
                if not publish_chunk(ring, chunk, self.stop, self.cfg.worker_timeout):
                    return
        except BaseException:
            try:
                self.errors.send(traceback.format_exc())
            except (BrokenPipeError, EOFError, OSError):
                pass
            raise
        finally:
            self.startup.close()
            self.errors.close()
            # Do not wait for a queue feeder after the consumer has stopped.
            self.metrics.cancel_join_thread()
            self.metrics.close()
