# SONIC G1 动作跟踪

SONIC 是面向 Unitree G1 的 Manager 环境，并通过 FastSAC 的 `sonic` PolicyVariant 构建专用
actor。策略将 10 帧本体感知历史、G1 未来参考和 SMPL 未来参考组合起来，通过两个值的
encoder mask 选择参考编码器。该时序深度是 `g1-sonic` 的公共契约；训练、LAFAN 数据和
小规模验证都使用同一个环境。

任务配置把 `algo.variant` 直接内联在 `configs/task/g1-sonic/motrix.fastsac.yaml` 中。
其中 `model.num_future_frames` 同时决定环境 policy observation 宽度与 actor 输入宽度；
`auxiliary` 是三个命名辅助损失的权重，由 FastSAC 统一加权求和并记录 `aux_*` 指标。
`g1_control_decoder_hidden_dims` 不出现在训练配置中：训练 actor 不构造该 decoder，
并用 FastSAC policy head 替代它。

## 观察与动作

`g1-sonic` 的 policy observation 宽度为 2412，privileged value observation 宽度为 1645。
动作是 29 维归一化关节位置目标，动作缩放和偏置在环境 action term 内完成；FastSAC actor
保持 identity action affine。

## 数据与运行

环境默认回退到仓库内置小型 smoke store，因此干净 checkout 可以直接完成 registry 检查和
短时集成运行。正式训练需要提供完整动作集，并通过环境变量指定原生 packed store：

```bash
source .venv/bin/activate
SONIC_PACKED_STORE=$PWD/data/sonic/lafan1-packed \
  python scripts/train.py task=g1-sonic/motrix.fastsac
```

也可以通过 `SONIC_DATA_ROOT` 指向包含单个 `sonic.npz` 的目录。两者都存在时，
`SONIC_PACKED_STORE` 选择只读 memory-mapped 动作集。

## 构建 packed store

打包器读取成对的 robot 与 SMPL NPZ 目录。输入四元数为 `wxyz`，输出转换为 `xyzw`，
并把关节与 body 列重排为任务约定顺序。

```bash
python scripts/motion/pack_sonic_data.py \
  /path/to/robot_filtered \
  /path/to/smpl_filtered \
  data/sonic/lafan1-packed
```

输出格式版本为 `motrixlab_sonic_packed_v1`。生成目录被 Git 忽略；仓库只包含小型 smoke store。

## 小规模验证

从仓库根目录执行时，内置 smoke store 不需要外部动作数据。小规模验证继续使用同一个
`g1-sonic` 配置，只覆盖并行数和训练长度：

```bash
python scripts/train.py task=g1-sonic/motrix.fastsac \
  num_envs=32 play_num_envs=4 checkpoint.interval=100 \
  algo.trainer.num_learning_iterations=1000
```

MotrixLab checkpoint 的回放方式见
[训练产物](../../tutorial/training/runs_and_checkpoints.md)。
