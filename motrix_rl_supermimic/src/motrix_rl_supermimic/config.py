# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic provider configuration.

Field structure mirrors ``motrix_rl/configs/algo_base/supermimic.supermimic.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentCfg:
    """Network shapes and per-update batch sizes."""

    hidden: int = 0
    hist: int = 10
    z_dim: int = 32
    # Polyak average of the policy weights (decay per update, 0 = off);
    # checkpoints store the average as 'pi' and the live policy as 'pi_live'.
    pi_ema: float = 0.0
    # Semi-implicit world-model integration:
    # velocities are predicted directly, positions follow them plus a small
    # learned correction — rollout positions stop compounding increment error.
    wm_structured: bool = False
    # Per-contact-mode increment heads with self-organized gating (see WorldModel).
    wm_contact_modes: bool = False
    # Dedicated world-model MLP width; 0 falls back to ``hidden``.
    wm_hidden: int = 0
    # Reference-action residual policy: the task's ``reference_action`` module
    # supplies the base action from the goal and the policy outputs a residual.
    pi_ref_skip: bool = False
    # Fraction of WM replay windows replaced by their task-mirror image
    # (batch augmentation; 0 disables).
    wm_mirror: float = 0.0


@dataclass
class TrainerCfg:
    """Update governance: warmup, update credit, imagination horizon, checkpoints."""

    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr: float = 2e-4
    lr_decay_iters: int = 250_000
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    log_every: int = 50
    ckpt_every: int = 2000
    # Policy-lr cosine anneal (x1 -> x pi_anneal_min) starting pi_anneal_from
    # iterations after policy updates begin, over pi_anneal_len iterations;
    # guards the post-ignigation data-drift phase. 0 disables.
    pi_anneal_from: int = 0
    pi_anneal_len: int = 0
    pi_anneal_min: float = 0.03
    # Optimizer selection: "muon" puts Muon on hidden MLP matrices and fused
    # Adam on the rest (CUDA only; CPU falls back to plain Adam).
    pi_opt: str = "adam"
    wm_opt: str = "adam"
    # Policy-lr multiplier applied before the one-shot decay trigger.
    pi_lr_mult: float = 1.0
    # One-shot policy-lr decay trigger: normalized training return threshold
    # (0 disables) or hard iteration deadline (0 disables), whichever first.
    lr_decay_ret: float = 0.0
    lr_decay_by: int = 0
    lr_decay_len: int = 300
    amp: str = "none"
    compile: bool = False
    compile_wm: bool = False
    # torch.compile the tensor-only WM/actor loss kernels.
    compile_actor_kernel: bool = False
    compile_wm_kernel: bool = False
    # CUDA-graph capture of the WM/actor loss kernels (mutually exclusive with
    # compile/compile_wm; requires CUDA).
    cuda_graph_wm_loss: bool = False
    cuda_graph_actor_loss: bool = False
    # Cache normalized critic features across the critic Adam epochs.
    cache_critic_features: bool = False
    # Defer the WM-loss all-reduce style sync (single-process provider keeps
    # the flag for recipe parity only).
    defer_wm_loss_sync: bool = False
    # Encode all imagined-step goals in one batch before the rollout instead
    # of per step (same math).
    batch_goal_enc: bool = False
    # Run the world-model updates on a side CUDA stream concurrently with the
    # actor update; the actor imagines with the WM of the iteration start
    # Requires CUDA; the async loop drives it.
    overlap_wm: bool = False


@dataclass
class CollectorCfg:
    """Async collector topology (ignored when ``asynchronous`` is false)."""

    block_steps: int = 2
    queue_blocks: int = 2
    snapshot_every: int = 10
    worker_timeout: float = 180.0
    # Device for collector-side policy inference ("cpu" or "cuda[:i]").
    policy_device: str = "cuda:0"
    # Import path whose ``register()`` runs in the collector process before
    # env resolution; only needed for programmatically registered test envs.
    preregister: str | None = None


@dataclass
class SuperMimicCfg:
    device: str | None = None
    asynchronous: bool = False

    agent: AgentCfg = field(default_factory=AgentCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    collector: CollectorCfg = field(default_factory=CollectorCfg)
