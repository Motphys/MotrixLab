# Motrix RL Interface

`motrix-rl-interface` contains stable, implementation-neutral contracts that connect reinforcement-learning providers to
MotrixLab. Its public Python API is `motrix_rl_interface`, including `RlFramework`, `AgentProvider`, `TrainerBase`,
`TrainerContext`, `TrainerHandle`, `AgentRegistration` and `ExecutionMode`.

The package deliberately has no third-party dependencies: no Torch, JAX, SKRL, RSLRL, FastSAC, Hydra or MotrixSim. Runtime
context references are supplied by the caller; concrete providers and MotrixLab-specific control-plane services are not
implemented here.

Concrete integrations live in `motrix-rl-builtin`, `motrix-rl-skrl` and `motrix-rl-rslrl`. They implement these contracts
and register with the control plane through the `motrix_rl.frameworks` entry-point group. Hydra schema installation,
framework discovery, run metadata and checkpoint services remain in `motrix-rl`, not this interface package.
