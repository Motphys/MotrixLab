# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Async topology smoke: spawned CPU collector feeding the learner ring."""

from pathlib import Path

import pytest
import torch
from _helpers import make_toy
from motrix_rl_supermimic.config import AgentCfg, CollectorCfg, SuperMimicCfg, TrainerCfg
from motrix_rl_supermimic.mbrl_env import assemble_task
from motrix_rl_supermimic.toy import ToyEnvCfg
from motrix_rl_supermimic.toy import register as register_toy

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("pi_ema", [0.0, 0.995])
def test_trainer_runs(tmp_path, monkeypatch, pi_ema):
    _run_trainer(tmp_path, monkeypatch, pi_ema)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="overlap stream contract requires CUDA")
def test_overlap_wm_orders_ingest_and_checkpoint_after_queued_updates(tmp_path, monkeypatch):
    _run_trainer(tmp_path, monkeypatch, 0.0, cuda_overlap=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC requires CUDA")
@pytest.mark.parametrize("learner_device", ["cuda:0", "cuda"])
def test_trainer_starts_cuda_ipc_transport_after_collector_metadata(tmp_path, monkeypatch, learner_device):
    _run_trainer(tmp_path, monkeypatch, 0.0, cuda_ipc=True, learner_device=learner_device)


def _run_trainer(tmp_path, monkeypatch, pi_ema, *, cuda_overlap=False, cuda_ipc=False, learner_device="cuda:0"):
    pytest.importorskip("motrix_rl")
    register_toy()
    env_name = "supermimic-toy"
    preregister = "motrix_rl_supermimic.toy"
    from motrix_rl_supermimic.provider import SuperMimicProvider

    from motrix_rl.contracts import TrainerContext
    from motrix_rl.metrics import MetricsLogger

    cfg = SuperMimicCfg(
        device=learner_device if cuda_overlap or cuda_ipc else "cpu",
        agent=AgentCfg(hidden=32, z_dim=8, pi_ema=pi_ema),
        trainer=TrainerCfg(
            iterations=20,
            buffer_steps=64,
            utd=0.5,
            horizon=4,
            wm_window=2,
            wm_batches=1,
            wm_batch=64,
            pi_batch=64,
            warmup_steps=4,
            wm_only_iters=2,
            critic_epochs=1,
            critic_minibatches=2,
            lr=3e-3,
            log_every=5,
            ckpt_every=20,
        ),
        collector=CollectorCfg(
            block_steps=2,
            queue_blocks=2,
            snapshot_every=5,
            worker_timeout=60.0,
            policy_device="cuda:0" if cuda_ipc else "cpu",
            preregister=preregister,
        ),
    )
    if cuda_overlap:
        cfg.trainer.iterations = 4
        cfg.trainer.wm_only_iters = 4
        cfg.trainer.wm_batch = 8
        cfg.trainer.overlap_wm = True
        cfg.trainer.utd = 1.0
        cfg.trainer.ckpt_every = 4
    context = TrainerContext(
        run=None,
        env_name=env_name,
        run_dir=Path(tmp_path),
        checkpoint_dir=Path(tmp_path) / "ckpt",
        sim=None,
        checkpoint_format="pt",
        num_envs=4,
        play_num_envs=1,
        seed=0,
        rl_cfg=cfg,
        logging=None,
        checkpoint=None,
        metrics=MetricsLogger.disabled(),
    )
    import motrix_rl_supermimic.trainer as trainer_module

    specs = []
    channel_type = trainer_module.WeightChannelShared

    def channel(spec):
        specs.append(spec)
        return channel_type(spec)

    monkeypatch.setattr(trainer_module, "WeightChannelShared", channel)
    senders = []
    sender_type = trainer_module.GpuIpcWeightSender if cuda_ipc else trainer_module.HostWeightSender

    def sender(*args):
        endpoint = sender_type(*args)
        senders.append(endpoint)
        return endpoint

    monkeypatch.setattr(trainer_module, "GpuIpcWeightSender" if cuda_ipc else "HostWeightSender", sender)
    epochs = {"queued": 0, "waited": 0, "ingest": 0, "save": 0}
    if cuda_overlap:
        # Scope the proxy to the trainer's imports: torch CUDA internals and
        # Learner receive real streams, including the side-stream context.
        class StreamWait:
            def __init__(self, stream):
                self.stream = stream

            def wait_stream(self, other):
                self.stream.wait_stream(other)
                epochs["waited"] = epochs["queued"]

            def record_event(self, event=None):
                return self.stream.record_event(event)

            def __cuda_stream__(self):
                return self.stream.__cuda_stream__()

            @property
            def cuda_stream(self):
                return self.stream.cuda_stream

            @property
            def device(self):
                return self.stream.device

        class CudaProxy:
            def __getattr__(self, name):
                return getattr(torch.cuda, name)

            def current_stream(self, device=None):
                return StreamWait(torch.cuda.current_stream(device))

        class TorchProxy:
            cuda = CudaProxy()

            def __getattr__(self, name):
                return getattr(torch, name)

        monkeypatch.setattr(trainer_module, "torch", TorchProxy())
        learner_type = trainer_module.Learner
        wm_update = learner_type.wm_update
        save = learner_type.save
        observe = learner_type.observe
        import motrix_rl_supermimic.learner as learner_module

        monkeypatch.setattr(learner_module, "torch", TorchProxy())

        def queued_update(learner):
            assert torch.cuda.current_stream(learner.dev) == learner.wm_side
            loss = wm_update(learner)
            epochs["queued"] += 1
            return loss

        def guarded_observe(learner, transitions):
            result = observe(learner, transitions)
            assert epochs["waited"] == epochs["queued"], "observe must wait for pending WM updates"
            epochs["ingest"] += 1
            return result

        def guarded_save(learner, wall_time):
            assert epochs["waited"] == epochs["queued"], "checkpoint can read pending WM state"
            epochs["save"] += 1
            return save(learner, wall_time)

        monkeypatch.setattr(learner_type, "wm_update", queued_update)
        monkeypatch.setattr(learner_type, "save", guarded_save)
        monkeypatch.setattr(learner_type, "observe", guarded_observe)
    trainer = SuperMimicProvider().create_trainer(context)
    trainer.train()
    assert len(senders) == 1
    assert senders[0].params[0].is_cuda == cuda_ipc
    assert senders[0].version == 1 + cfg.trainer.iterations // cfg.collector.snapshot_every
    assert tuple(section.name for section in specs[0].modules) == ("policy",)

    ckpt = Path(tmp_path) / "ckpt"
    assert (ckpt / "last.pt").exists()
    snapshot = torch.load(ckpt / "last.pt", map_location="cpu", weights_only=False)
    assert snapshot["iteration"] == cfg.trainer.iterations
    if cuda_overlap:
        assert epochs["queued"] == cfg.trainer.iterations
        assert epochs["ingest"] > 1
        assert epochs["save"] == 1


class _RandomReference(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("offset", torch.rand(1), persistent=False)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.offset.expand(goal.shape[0], 1)


@pytest.mark.parametrize("checkpoint_goal_frames", [2, 3])
def test_play_loads_policy_and_preserves_step_failure(tmp_path, monkeypatch, checkpoint_goal_frames):
    pytest.importorskip("motrix_rl")
    register_toy()
    from motrix_rl_supermimic.trainer import SuperMimicTrainer

    from motrix_rl.contracts import TrainerContext
    from motrix_rl.metrics import MetricsLogger

    cfg = SuperMimicCfg(device="cpu", agent=AgentCfg(hidden=16, pi_ref_skip=True))
    monkeypatch.setattr(ToyEnvCfg, "reference_action", lambda self: _RandomReference())
    play_env = make_toy(num_envs=1, cfg=ToyEnvCfg(goal_frames=2))
    play_env.init_state()
    cfg.trainer.iterations = 1
    from motrix_rl_supermimic.learner import Learner

    task, task_binding = assemble_task(play_env.cfg.for_imagined(), "cpu")
    learner = Learner(cfg, task, task_binding, play_env.num_envs, play_env.cfg.ctrl_dt, "cpu", tmp_path, seed=17)
    with torch.no_grad():
        for parameter in learner.pi.parameters():
            parameter.zero_()
    learner.save(0.0)
    checkpoint_path = tmp_path / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["goal_frames"] = checkpoint_goal_frames
    torch.save(checkpoint, checkpoint_path)

    class _Stop(RuntimeError):
        pass

    play_env = make_toy(num_envs=1, cfg=ToyEnvCfg(goal_frames=2))
    play_env.init_state()

    def stop_step(action):
        torch.testing.assert_close(action, learner.pi.reference_action.offset.expand_as(action))
        raise _Stop

    monkeypatch.setattr(play_env, "step", stop_step)
    from motrix_env_core.registry import EnvBuildSpec

    monkeypatch.setattr(EnvBuildSpec, "make", lambda self, **kwargs: play_env)
    context = TrainerContext(
        run=None,
        env_name="supermimic-toy",
        run_dir=tmp_path,
        checkpoint_dir=tmp_path,
        sim=None,
        checkpoint_format="pt",
        num_envs=1,
        play_num_envs=1,
        seed=0,
        rl_cfg=cfg,
        logging=None,
        checkpoint=None,
        metrics=MetricsLogger.disabled(),
    )
    trainer = SuperMimicTrainer(context=context)
    if checkpoint_goal_frames != play_env.cfg.goal_frames:
        with pytest.raises(ValueError, match="goal_frames"):
            trainer.play(str(checkpoint_path))
    else:
        with pytest.raises(_Stop):
            trainer.play(str(checkpoint_path))


def test_resume_uses_checkpoint_root_before_nonpersistent_reference_factory(tmp_path, monkeypatch):
    import motrix_rl_supermimic.trainer as trainer_module
    from motrix_rl_supermimic.learner import Learner
    from motrix_rl_supermimic.trainer import SuperMimicTrainer

    from motrix_rl.contracts import TrainerContext
    from motrix_rl.metrics import MetricsLogger

    register_toy()
    cfg = SuperMimicCfg(device="cpu", agent=AgentCfg(hidden=8, pi_ref_skip=True))
    monkeypatch.setattr(ToyEnvCfg, "reference_action", lambda self: _RandomReference())
    task, task_binding = assemble_task(ToyEnvCfg().for_imagined(), "cpu")
    original = Learner(cfg, task, task_binding, 1, 0.02, "cpu", tmp_path, seed=17)
    original.save(0)
    checkpoint_path = str(tmp_path / "last.pt")
    context = TrainerContext(
        run=None,
        env_name="supermimic-toy",
        run_dir=tmp_path,
        checkpoint_dir=tmp_path,
        sim=None,
        checkpoint_format="pt",
        num_envs=1,
        play_num_envs=1,
        seed=99,
        rl_cfg=cfg,
        logging=None,
        checkpoint=None,
        metrics=MetricsLogger.disabled(),
        resume_from=checkpoint_path,
    )
    monkeypatch.setattr(trainer_module, "assemble_task", lambda cfg, device: (task, task_binding))
    loads = []
    load = torch.load

    def read_checkpoint(*args, **kwargs):
        loads.append(args[0])
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", read_checkpoint)
    load_checkpoint = Learner.load_checkpoint

    class _Restored(RuntimeError):
        pass

    def restored(learner, checkpoint):
        # This value is absent from state_dict and cannot be repaired by load.
        torch.testing.assert_close(learner.pi.reference_action.offset, original.pi.reference_action.offset)
        assert learner.root_seed == original.root_seed
        load_checkpoint(learner, checkpoint)
        raise _Restored

    monkeypatch.setattr(Learner, "load_checkpoint", restored)
    with pytest.raises(_Restored):
        SuperMimicTrainer(context=context).train()
    assert loads == [checkpoint_path]
