# RL Plugin Packages 实施清单

## 摘要

本清单跟踪 RL 集成从单一实现树拆分为控制平面 `motrix_rl`、稳定接口 `motrix_rl_interface` 与 provider 插件包 `motrix_rl_builtin`（FastSAC）、`motrix_rl_skrl`（SKRL）和 `motrix_rl_rslrl`（RSLRL）的落地工作。边界、依赖和 entry-point discovery 设计见 [RL 多算法架构设计](../design/rl-multi-algorithm-architecture.md)。本文只覆盖 package 拆分、依赖声明、注册发现、相关调用方与 package README，不覆盖 `docs/source` 教程或额外算法开发。清单描述后续实现验收；本次文档编辑本身不修改 implementation/pyproject 文件。

## TODO

- [ ] 确认 `motrix_rl` 只保留控制平面：runner、run/checkpoint、backend selection、`motrix_rl.frameworks` 和 plugin loader；移除 provider 实现的直接所有权。
- [ ] 确认 `motrix_rl_interface` 只暴露 implementation-neutral provider/trainer contracts，并保持无 Torch/JAX/SKRL/RSLRL/FastSAC/Hydra/MotrixSim 依赖。
- [ ] 将 FastSAC 的配置、模型、memory、同步/异步 trainer、wrapper 和 export 实现归属 `motrix_rl_builtin`，注册 `motrix.fastsac`。
- [ ] 将 SKRL PPO 的 JAX/Torch provider、配置、wrapper、trainer 和 export 实现归属 `motrix_rl_skrl`，注册 `skrl.ppo`。
- [ ] 将 RSLRL PPO provider、wrapper、trainer 和 export 实现归属 `motrix_rl_rslrl`，注册 `rslrl.ppo`。
- [ ] 为三个插件保留 `motrix_rl.frameworks` entry-point group；entry point 指向无参数 `register()`，加载失败向调用方传播，未安装插件不阻塞控制平面。
- [ ] 审核依赖方向：插件可依赖 `motrix_rl_interface` 与控制平面，但接口包不得反向依赖 provider；provider-specific third-party dependencies 只存在于所属插件及其 extras。
- [ ] 审核 workspace、安装 extras、Hydra task 组合和 play/checkpoint discovery 在插件可选安装矩阵下的行为。
- [x] 更新各 package README，说明职责、支持的 provider、entry-point discovery 和安装边界；不修改 `docs/source` 教程。
- [ ] 以静态 import/entry-point 扫描、package import、provider discovery 和现有测试验证拆分完成。

## 验收边界

- `motrix_rl.frameworks` 是控制平面的稳定查询/注册入口；调用方不需要 import `motrix_rl_builtin`、`motrix_rl_skrl` 或 `motrix_rl_rslrl` 的内部模块。
- `(rllib, train_backend, algo)` 仍是 provider 查询键，run metadata 与 checkpoint manifest 语义不变。
- 未安装可选插件时，控制平面可以导入；选择缺失 provider 时给出明确错误。
- 本清单完成后应删除该 plan，并同步更新 `wiki/plan/index.md`。
