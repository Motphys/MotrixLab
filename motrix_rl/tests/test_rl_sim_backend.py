# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib
import platform
from queue import Empty
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.base import EnvCfg
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.registry import EnvBuildSpec
from motrix_env_motrixsim.torch_env import TorchEnv, TorchEnvState, TorchObs
from motrix_rl.fastsac.async_impl.transport import Control, SharedTransitionRing
from motrix_rl.fastsac.async_impl.transport.weight_channel import HostWeightSender, WeightChannelShared
from motrix_rl.fastsac.async_impl.worker import actor_param_numel, run_collector_process
from motrix_rl.fastsac.wrap import FastSacEnvWrap

_TRAINERS = ["skrl.torch", "skrl.jax", "rslrl.torch", "fastsac"]
_SIM_BACKENDS = ["np", "torch"]
_FASTSAC_WRAPPERS = {"np": "FastSacNpEnvWrap", "torch": "FastSacTorchEnvWrap"}

_NUM_ENVS = 2
_OBS_DIM = 3
_ACT_DIM = 2


def _make_env(sim_backend: str):
    policy_space = gym.spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([-2.0, -1.0], dtype=np.float32),
        high=np.array([2.0, 3.0], dtype=np.float32),
    )
    if sim_backend == "np":
        env = Mock(spec=DirectEnv)
        state = ArrayEnvState(
            obs=NpObs(np.zeros((_NUM_ENVS, _OBS_DIM), dtype=np.float32)),
            reward=np.ones(_NUM_ENVS, dtype=np.float32),
            terminated=np.zeros(_NUM_ENVS, dtype=bool),
            truncated=np.zeros(_NUM_ENVS, dtype=bool),
            episode_steps=np.zeros(_NUM_ENVS, dtype=np.uint64),
        )
    else:
        env = Mock(spec=TorchEnv)
        env.device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
        state = TorchEnvState(
            data=None,
            obs=TorchObs(torch.zeros((_NUM_ENVS, _OBS_DIM), device=env.device)),
            reward=torch.ones(_NUM_ENVS, device=env.device),
            terminated=torch.zeros(_NUM_ENVS, dtype=torch.bool, device=env.device),
            truncated=torch.zeros(_NUM_ENVS, dtype=torch.bool, device=env.device),
            episode_steps=torch.zeros(_NUM_ENVS, dtype=torch.int64, device=env.device),
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


def _require_trainer(trainer: str) -> None:
    if trainer.startswith("skrl"):
        pytest.importorskip("skrl")
        if trainer == "skrl.jax":
            pytest.importorskip("jax")
    elif trainer == "rslrl.torch":
        pytest.importorskip("rsl_rl")


def _wrap_env(monkeypatch, trainer: str, sim_backend: str, env, renderer):
    _require_trainer(trainer)
    wrapper_module = importlib.import_module(f"motrix_rl.{trainer}.wrap_{sim_backend}")
    monkeypatch.setattr(wrapper_module, "create_renderer", Mock(return_value=renderer))

    device = torch.device("cpu")
    if trainer == "fastsac":
        return getattr(wrapper_module, _FASTSAC_WRAPPERS[sim_backend])(env, device, render=object())

    trainer_module = importlib.import_module(f"motrix_rl.{trainer}")
    if trainer == "rslrl.torch":
        return trainer_module.wrap_env(env, device, render=object())
    return trainer_module.wrap_env(env, render=object())


@pytest.mark.parametrize("trainer", _TRAINERS)
@pytest.mark.parametrize("sim_backend", _SIM_BACKENDS)
def test_rl_trainer_supports_sim_backend(monkeypatch, trainer: str, sim_backend: str) -> None:
    env = _make_env(sim_backend)
    renderer = Mock()
    wrapped = _wrap_env(monkeypatch, trainer, sim_backend, env, renderer)

    if trainer == "skrl.torch":
        obs, _ = wrapped.reset()
        result = wrapped.step(torch.zeros((_NUM_ENVS, _ACT_DIM), device=wrapped.device))
        assert all(tensor.device == wrapped.device for tensor in (obs, *result[:4]))
    elif trainer == "skrl.jax":
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        obs, _ = wrapped.reset()
        result = wrapped.step(jnp.zeros((_NUM_ENVS, _ACT_DIM)))
        assert all(isinstance(array, jax.Array) for array in (obs, *result[:4]))
        assert all(array.device == wrapped.device for array in (obs, *result[:4]))
    elif trainer == "rslrl.torch":
        obs, _ = wrapped.reset()
        next_obs, rewards, dones, extras = wrapped.step(torch.zeros((_NUM_ENVS, _ACT_DIM)))
        assert all(tensor.device.type == "cpu" for tensor in (obs["policy"], next_obs["policy"], rewards, dones))
        assert extras["time_outs"].device.type == "cpu"
    else:
        assert isinstance(wrapped, FastSacEnvWrap)
        initial = wrapped.reset()
        result = wrapped.step(torch.zeros((_NUM_ENVS, _ACT_DIM)))
        assert all(tensor.device.type == "cpu" for tensor in (*initial, *result))

    actions = env.step.call_args.args[0]
    assert isinstance(actions, np.ndarray if sim_backend == "np" else torch.Tensor)
    if sim_backend == "torch":
        assert actions.device == env.device

    wrapped.render("ignored", ignored=True)
    wrapped.close()
    renderer.render.assert_called_once_with()
    renderer.close.assert_called_once_with()


@pytest.mark.parametrize("sim_backend", _SIM_BACKENDS)
def test_fastsac_clips_actions_for_sim_backend(sim_backend: str) -> None:
    env = _make_env(sim_backend)
    module = importlib.import_module(f"motrix_rl.fastsac.wrap_{sim_backend}")
    wrapped = getattr(module, _FASTSAC_WRAPPERS[sim_backend])(env, torch.device("cpu"))
    actions = torch.tensor([[3.0, -2.0], [0.5, 4.0]])

    wrapped.step(actions)

    expected = actions.clamp(torch.from_numpy(env.action_space.low), torch.from_numpy(env.action_space.high))
    actual = env.step.call_args.args[0]
    if sim_backend == "np":
        np.testing.assert_allclose(actual, expected.numpy())
    else:
        torch.testing.assert_close(actual, expected.to(env.device))


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


def _async_info(array):
    return {"Reward": {"alive": array([1.0, 1.0])}}


class _AsyncNpEnv(_AsyncEnv, DirectEnv):
    def compute_transition(self, state):
        return state

    def _make_state(self, value: float) -> ArrayEnvState:
        return ArrayEnvState(
            obs=NpObs(np.full((self.num_envs, _OBS_DIM), value, dtype=np.float32)),
            reward=np.full(self.num_envs, value, dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
            episode_steps=np.zeros(self.num_envs, dtype=np.uint64),
            metrics={"progress": 0.5},
        )


class _AsyncTorchEnv(_AsyncEnv, TorchEnv):
    def _make_state(self, value: float) -> TorchEnvState:
        return TorchEnvState(
            data=None,
            obs=TorchObs(torch.full((self.num_envs, _OBS_DIM), value)),
            reward=torch.full((self.num_envs,), value),
            terminated=torch.zeros(self.num_envs, dtype=torch.bool),
            truncated=torch.zeros(self.num_envs, dtype=torch.bool),
            episode_steps=torch.zeros(self.num_envs, dtype=torch.int64),
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


def _collect_in_spawn(sim_backend: str) -> tuple[torch.Tensor, ...]:
    cfg = _async_cfg()
    dims = (_OBS_DIM, _OBS_DIM, _ACT_DIM)
    action_scale = torch.ones(_ACT_DIM)
    action_bias = torch.zeros(_ACT_DIM)
    ring = SharedTransitionRing(2, _NUM_ENVS, *dims)
    weights = WeightChannelShared(_OBS_DIM)
    control = Control()
    ctx = mp.get_context("spawn")
    stats_queue = ctx.Queue(maxsize=2)
    error_queue = ctx.Queue(maxsize=2)
    slot_queue = ctx.Queue(maxsize=1)
    weight_tx = HostWeightSender(weights, actor_param_numel(cfg, dims, action_scale, action_bias))
    # Ship the real handshake message shape: (weight slots, ring slots). The
    # host SharedTransitionRing needs no ring slots, so the ring field is None.
    slot_queue.put((weight_tx.params, None))  # ship before the collector process starts
    env_cls = _AsyncNpEnv if sim_backend == "np" else _AsyncTorchEnv
    env_spec = EnvBuildSpec(env_cls, EnvCfg(scene=SceneCfg()))
    ipc_resources = (ring, weights, control, stats_queue, error_queue)
    process = ctx.Process(
        target=run_collector_process,
        args=(env_spec, cfg, _NUM_ENVS, dims, action_scale, action_bias, *ipc_resources, 1, 1, False, 7, slot_queue),
    )

    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail(f"{sim_backend} collector process did not exit")

    try:
        child_error = error_queue.get_nowait()
    except Empty:
        child_error = None
    finally:
        stats_queue.close()
        error_queue.close()

    assert child_error is None
    assert process.exitcode == 0
    assert control.collector_steps == 1
    count, views = ring.read_span()
    assert count >= 1
    return tuple(tensor.clone() for tensor in views)


@pytest.mark.skipif(
    platform.machine().lower() not in {"amd64", "x86_64"},
    reason="FastSAC async shared memory currently supports x86-64 only",
)
def test_fastsac_async_supports_sim_backends() -> None:
    np_slot = _collect_in_spawn("np")
    torch_slot = _collect_in_spawn("torch")

    assert len(np_slot) == len(torch_slot)
    for np_tensor, torch_tensor in zip(np_slot, torch_slot):
        assert (np_tensor.device.type, np_tensor.dtype, np_tensor.shape) == (
            torch_tensor.device.type,
            torch_tensor.dtype,
            torch_tensor.shape,
        )
        torch.testing.assert_close(np_tensor, torch_tensor)
