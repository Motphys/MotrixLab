# SONIC G1 动作跟踪

SONIC 是面向 Unitree G1 的 Manager 环境，并使用专用 FastSAC actor。策略将本体感知历史、
G1 未来参考和 SMPL 未来参考组合起来，通过两个值的 encoder mask 选择参考编码器。

## 环境

| 环境 | 时序配置 | 用途 |
| --- | --- | --- |
| `g1-sonic` | 10 帧 | 发布规模训练与官方 checkpoint 回放 |
| `g1-sonic-lafan` | 4 帧 | 中等规模 packed corpus 训练 |
| `g1-sonic-smoke` | 1 帧 | 使用仓库内置小型 store 的契约与集成测试 |

三个变体默认回退到仓库内置 smoke store，因此干净 checkout 可以直接完成 registry 检查和
短时集成运行。使用 release 或 LAFAN 配置进行有效训练仍需另行提供完整动作集，并通过环境变量
指定原生 packed store：

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

## Smoke 运行

从仓库根目录执行时，smoke 配置不需要外部动作数据：

```bash
python scripts/train.py task=g1-sonic-smoke/motrix.fastsac \
  num_envs=4 algo.trainer.num_learning_iterations=2
```

MotrixLab checkpoint 与官方 SONIC release checkpoint 的回放方式见
[训练产物](../../tutorial/runs_and_checkpoints.md)。
