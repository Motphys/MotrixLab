# 训练总览

环境注册完成后，训练一个强化学习任务只需要三步：

```text
创建 Task 配置        指定环境、RL 框架、算法与超参数
      ↓
python scripts/train.py task=...   Hydra 组合配置并启动训练
      ↓
runs/{env}/...       保存 metadata、日志与 checkpoint
```

各页面按使用顺序展开：

- [Task 配置与命令行参数覆盖](task_config.md)：创建 Task 文件、调整运行与算法参数、
  使用 `key=value` 临时覆盖；
- [执行训练与分析结果](training_and_result.md)：启动训练、读取 TensorBoard 日志、
  用 play 回放策略；
- [训练产物：runs 目录与 checkpoint](runs_and_checkpoints.md)：`runs/` 的目录结构、
  best policy 的选择逻辑与续训方式。

```{toctree}
:hidden:

task_config
training_and_result
runs_and_checkpoints
```
