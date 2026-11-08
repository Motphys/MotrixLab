# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import platform
from queue import Empty
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from motrix_rl_builtin.fastsac.async_impl.transport import Control, SharedTransitionRing, StartupHandshake
from motrix_rl_builtin.fastsac.async_impl.transport.weight_channel import HostWeightSender, WeightChannelShared
from motrix_rl_builtin.fastsac.async_impl.worker import actor_param_numel, run_collector_process
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap, FastSacStepResult

from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_env_core.base import EnvCfg
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.registry import EnvBuildSpec

_NUM_ENVS = 2

_OBS_DIM = 3

_ACT_DIM = 2


def _make_env():
    policy_space = gym.spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([-2.0, -1.0], dtype=np.float32),
        high=np.array([2.0, 3.0], dtype=np.float32),
    )
    env = Mock(spec=DirectEnv)
    state = ArrayEnvState(
        obs=EnvObs(torch.zeros((_NUM_ENVS, _OBS_DIM), dtype=torch.float32)),
        reward=torch.ones(_NUM_ENVS, dtype=torch.float32),
        terminated=torch.zeros(_NUM_ENVS, dtype=torch.bool),
        truncated=torch.zeros(_NUM_ENVS, dtype=torch.bool),
        episode_steps=torch.zeros(_NUM_ENVS, dtype=torch.int64),
    )
    env.cfg = SimpleNamespace(max_episode_steps=100)
    env.num_envs = _NUM_ENVS
    env.action_space = action_space
    env.policy_observation_space = policy_space
    env.value_observation_space = policy_space
    env.has_value_observation = False
    env.state = state
    env.init_state.return_value = state
    env.step.return_value = state
    return env


def _wrap_env(monkeypatch, env, renderer):
    import motrix_rl_builtin.fastsac.wrap as wrap_module

    monkeypatch.setattr(wrap_module, "create_renderer", Mock(return_value=renderer))
    return wrap_module.FastSacEnvWrap(env, torch.device("cpu"), render=object())


def test_rl_trainer_supports_sim_backend(monkeypatch) -> None:
    env = _make_env()
    renderer = Mock()
    wrapped = _wrap_env(monkeypatch, env, renderer)

    assert isinstance(wrapped, FastSacEnvWrap)
    initial = wrapped.reset()
    result = wrapped.step(torch.zeros((_NUM_ENVS, _ACT_DIM)))
    assert isinstance(result, FastSacStepResult)
    tensors = (*initial, result.obs, result.critic_obs, result.reward, result.terminated, result.truncated)
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert result.final_obs is None
    assert isinstance(result.info, dict)

    actions = env.step.call_args.args[0]
    assert isinstance(actions, torch.Tensor)
    assert actions.device.type == "cpu"

    wrapped.render("ignored", ignored=True)
    wrapped.close()
    renderer.render.assert_called_once_with()
    renderer.close.assert_called_once_with()


def test_fastsac_clips_actions_for_sim_backend() -> None:
    env = _make_env()
    wrapped = FastSacEnvWrap(env, torch.device("cpu"))
    actions = torch.tensor([[3.0, -2.0], [0.5, 4.0]])

    wrapped.step(actions)

    expected = actions.clamp(torch.from_numpy(env.action_space.low), torch.from_numpy(env.action_space.high))
    actual = env.step.call_args.args[0]
    np.testing.assert_allclose(actual, expected.numpy())


class _AsyncEnv:
    def __init__(self, cfg: EnvCfg, num_envs: int = 1, backend=None) -> None:
        self._cfg = cfg
        self._num_envs = num_envs
        self._device = torch.device("cpu")
        self._action_space = gym.spaces.Box(-1.0, 1.0, (_ACT_DIM,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def apply_action(self, actions, state):
        return state

    def update_state(self, state):
        return state

    def init_state(self):
        self._state = self._make_state(0.0)
        return self._state

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            assert actions.device.type == "cpu"
        self._state = self._make_state(1.0)
        return self._state

    def reset(self, data):
        return self._make_state(0.0).obs, {}


class _AsyncTestEnv(_AsyncEnv, DirectEnv):
    def compute_transition(self, state):
        return state

    def _make_state(self, value: float) -> ArrayEnvState:
        return ArrayEnvState(
            obs=EnvObs(np.full((self.num_envs, _OBS_DIM), value, dtype=np.float32)),
            reward=np.full(self.num_envs, value, dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
            episode_steps=np.zeros(self.num_envs, dtype=np.uint64),
            metrics={"progress": 0.5},
        )


def _async_cfg():
    return SimpleNamespace(
        agent=SimpleNamespace(
            actor_hidden_dim=16,
            log_std_max=2.0,
            log_std_min=-5.0,
            use_tanh=True,
            use_layer_norm=False,
            obs_normalization=False,
            learning_starts=10,
        ),
        trainer=SimpleNamespace(
            async_options=SimpleNamespace(
                idle_sleep_s=0.001,
                weight_poll_interval=1,
                collector_inference_device="cpu",
                collector_compile=False,
                collector_amp=False,
                collector_amp_dtype="fp16",
                learner_cpu_cores=None,
                collector_cpu_cores=None,
            )
        ),
    )


def _collect_in_spawn(tmp_path) -> tuple[torch.Tensor, ...]:
    cfg = _async_cfg()
    dims = (_OBS_DIM, _OBS_DIM, _ACT_DIM)
    action_scale = torch.ones(_ACT_DIM)
    action_bias = torch.zeros(_ACT_DIM)
    ring = SharedTransitionRing(2, _NUM_ENVS, *dims)
    weights = WeightChannelShared(_OBS_DIM)
    control = Control()
    ctx = mp.get_context("spawn")
    collector_data_queue = ctx.Queue(maxsize=2)
    error_queue = ctx.Queue(maxsize=2)
    weight_tx = HostWeightSender(weights, actor_param_numel(cfg, dims, action_scale, action_bias))
    # Ship the real handshake messages: weight slots (publisher learner) and
    # ring slots (draining learner; None for the host SharedTransitionRing).
    handshake = StartupHandshake(ctx, num_collectors=1, barrier=False)
    handshake.ship_weight_slots(0, weight_tx.params)
    handshake.ship_ring_slots(0, None)

    env_cls = _AsyncTestEnv
    env_spec = EnvBuildSpec(env_cls, EnvCfg(scene=SceneCfg()))
    process = ctx.Process(
        target=run_collector_process,
        kwargs={
            "env_spec": env_spec,
            "cfg": cfg,
            "num_envs": _NUM_ENVS,
            "dims": dims,
            "action_scale": action_scale,
            "action_bias": action_bias,
            "ring": ring,
            "weights": weights,
            "control": control,
            "collector_data_queue": collector_data_queue,
            "error_queue": error_queue,
            "num_iterations": 1,
            "logging_interval": 1,
            "is_resume": False,
            "seed": 7,
            "run_dir": str(tmp_path),
            "handshake": handshake,
        },
    )

    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("collector process did not exit")

    try:
        child_error = error_queue.get_nowait()
    except Empty:
        child_error = None
    finally:
        collector_data_queue.close()
        error_queue.close()

    assert child_error is None
    assert process.exitcode == 0
    assert control.collector_steps == 1
    count, views = ring.read_span()
    assert count >= 1
    return tuple(tensor.clone() for tensor in views)


@pytest.mark.integration
@pytest.mark.skipif(
    platform.machine().lower() not in {"amd64", "x86_64"},
    reason="FastSAC async shared memory currently supports x86-64 only",
)
def test_fastsac_async_supports_sim_backends(tmp_path) -> None:
    slot = _collect_in_spawn(tmp_path)

    obs, critic_obs, _actions, rewards, dones, truncations = slot
    assert all(
        (tensor.device.type, tensor.dtype, tensor.shape[:1]) == ("cpu", torch.float32, (_NUM_ENVS,)) for tensor in slot
    )
    # The test env steps every field from 0.0 to 1.0: the stored transition
    # batch must carry the post-step values through the shared ring.
    torch.testing.assert_close(obs, torch.full_like(obs, 1.0))
    torch.testing.assert_close(critic_obs, torch.full_like(critic_obs, 1.0))
    torch.testing.assert_close(rewards, torch.full_like(rewards, 1.0))
    assert not dones.any()
    assert not truncations.any()
