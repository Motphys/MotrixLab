# Go2 FlatWalk v2 落地计划

- [x] 从冻结 v1 提交创建 `feat/go2-flatwalk-v2-20261126` 分支。
- [x] 将 Go2 任务注册与配置重命名为 `go2-walk-flat-v2`，并删除旧 flat/rough 任务配置。
- [x] 应用 v2 训练配方：1024 env、2000 iterations、4 minibatches、seed 1。
- [x] 应用纯转向 hard cases、hard-case 课程与 yaw 跟踪权重调整。
- [x] 迁移仿真与真机部署 recipe，并更新相关测试。
- [x] 更新用户文档、package README 与 v2 设计说明。
- [x] 通过 Hydra dry-run 与受影响测试。
- [x] 重新执行 2000 iterations 训练并导出新 artifact。
- [x] 完成 TensorBoard 收敛分析、16 场景审计、220 矩阵审计和 60 秒站立审计。
- [x] 与已实机验证参考策略对比，确认配置迁移正确。
- [x] 完成仿真部署冒烟。
- [x] 修正 hard-case 采样顺序并完成严格复现训练：TensorBoard 学习曲线逐点一致、checkpoint sha256 一致、ONNX 推理输出逐位一致。
- [x] 完成复现版本 artifact 的真机验收。
