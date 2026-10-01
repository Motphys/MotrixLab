# ONNX 模型导出设计

## 1. 目标

MotrixLab 通过统一入口将 metadata-backed training run 的最佳策略导出为 ONNX。导出结果满足以下约束：

- 输入为未归一化的原始 actor observation；
- observation normalizer 烘焙在 ONNX 内；
- 输出为 deterministic policy action，不包含探索噪声、critic 或 optimizer；
- ONNX 接口统一为动态 batch 的 `obs -> actions`；
- 导出后必须执行训练框架 reference 与 ONNX Runtime 的数值一致性验证；
- 框架发现复用 `RlFramework`，不维护独立 exporter registry。

当前首版支持：

| RL framework | Train backend | Algorithm | Checkpoint | 状态 |
|---|---|---|---|---|
| RSL-RL | Torch | PPO | `.pt` | 支持 |
| SKRL | Torch | PPO | `.pt` | 支持 |
| SKRL | JAX | PPO | `.pickle` | 后续 adapter |
| Motrix | Torch | FastSAC | `.pt` | 支持 |

FastSAC 的规范身份是 `(motrix, torch, fastsac)`。同步/异步是 algorithm config，不构成新的 exporter key。

## 2. 架构

Exporter 是现有 RL framework integration 的一项可选 capability：

```text
metadata.json + task_config.yaml + checkpoints/manifest.json
                            |
                            v
                    export_onnx(run_dir)
                            |
                            v
             get_framework(metadata.rllib)
                            |
                            v
              RlFramework.export_policy(...)
                            |
             get_agent_provider(algo, backend)
                            |
                            v
              provider OnnxPolicyExporter
                 restore / export / parity
                            |
                            v
                       policy.onnx
```

`RlFramework` 已经按 agent 和 train backend 索引 `AgentProvider`。Provider 本身唯一对应
`(rllib, train_backend, algo)`，因此 exporter capability 直接挂在 provider 上，不建立第二套 registry。

公共编排与具体实现按所有权拆分：

```text
motrix_rl/deploy/api.py             policy/deployment protocol and data
motrix_rl/deploy/onnx.py            metadata/checkpoint-driven ONNX entry point
motrix_rl/deploy/onnx_validation.py shared ONNX Runtime parity validation
motrix_rl/rslrl/export.py           RSL-RL policy restoration and export
motrix_rl/skrl/torch/export.py      SKRL/Torch policy restoration and export
```

具体 exporter 不放在公共 `motrix_rl.deploy` package 中，新增 rllib/backend 时由其自己的模块维护实现。

```python
class AgentProvider:
    def create_policy_exporter(self) -> OnnxPolicyExporter | None:
        return None


class RlFramework:
    def export_policy(self, request: OnnxExportRequest) -> OnnxModelArtifact: ...
```

不支持导出的 provider 返回 `None`；公共错误会同时报告请求组合和当前支持矩阵。

## 3. 公共接口

公共数据类型定义在 `motrix_rl.deploy.api`，只依赖 `motrix_rl` 的 run/config 数据和 `motrix_deploy` core contract，
不导入 Torch、JAX、SKRL、RSL-RL、MuJoCo 或厂商 SDK。

```python
@dataclass(frozen=True)
class OnnxExportRequest:
    run: RunContext
    checkpoint: Path
    task_config: TaskConfig
    opset: int
    parity: OnnxParityConfig


@dataclass(frozen=True)
class OnnxExportReport:
    input_spec: PolicyTensorSpec
    output_spec: PolicyTensorSpec
    parity: OnnxParityMetrics


@dataclass(frozen=True)
class OnnxModelArtifact:
    model_bytes: bytes
    report: OnnxExportReport


class OnnxPolicyExporter(ABC):
    @abstractmethod
    def export(self, request: OnnxExportRequest) -> OnnxModelArtifact: ...
```

Framework exporter 返回已验证的内存 artifact；公共 service 原子落盘后返回包含 `path` 和同一份
`OnnxExportReport` 的 `OnnxExportResult`。这两个阶段不通过可选的 `model/path` 字段混合在一个类型里。

公共 service 负责：

1. 打开 `RunContext`；
2. 校验 task snapshot 与 metadata 身份一致；
3. 通过 checkpoint manifest 定位 `best_policy`；
4. 调用 `RlFramework.export_policy()`；
5. parity 成功后原子写入 `.onnx`。

Exporter 不搜索文件名，也不创建训练环境。

## 4. ONNX 契约

| | 名称 | 形状 | 类型 |
|---|---|---|---|
| 输入 | `obs` | `(batch_size, obs_dim)` | `float32` |
| 输出 | `actions` | `(batch_size, action_dim)` | `float32` |

其中 `batch_size` 是动态维度。ONNX metadata properties 记录：

- `env_name`；
- `rllib`；
- `train_backend`；
- `algo`；
- `obs_dim`；
- `action_dim`；
- `source_checkpoint`。

公共 validator 从 ONNX Runtime session 读取实际 input/output 名称、dtype 和 shape，不能仅信任 exporter 声明。

## 5. RSL-RL / Torch / PPO

RSL-RL 4.0.1 自带两层 ONNX 能力：

- `MLPModel.as_onnx()` 返回适合导出的 wrapper；
- `OnPolicyRunner.export_policy_to_onnx()` 将 runner 中的 policy 写入文件。

Runner 入口要求先构造完整 runner 和环境。MotrixLab exporter 已经从 metadata、typed task config 和 checkpoint
获得所需信息，因此使用模型级 `as_onnx()`：

1. 从 `actor_state_dict` 的 MLP 权重得到 observation/action dimension；
2. 使用 `RslrlCfg.actor` 和 `obs_groups` 重建 `MLPModel`；
3. strict load `actor_state_dict`；
4. 调用 `actor.as_onnx()`；
5. 使用动态 batch axes 导出。

RSL-RL 的 `_OnnxMLPModel` 会复制 actor 的 `EmpiricalNormalization` 和 MLP。Actor 默认 forward 返回
deterministic mean，因此无需导出 stochastic sampling path。

首版只支持：

- `actor.class_name=MLPModel`；
- 单一 `obs_groups.actor=[policy]`；
- 非 state-dependent standard deviation。

其他结构 fail fast，不猜测 checkpoint 语义。

## 6. SKRL / Torch / PPO

SKRL 2.1.0 没有 ONNX exporter。它的完整 agent checkpoint 是 module state mapping，当前 PPO 训练产物至少包含：

- `policy`：policy model state dict；
- `observation_preprocessor`：`RunningStandardScaler` state dict；
- value、optimizer 和其他训练模块。

旧设计中的 `state_preprocessor` 名称不适用于当前 actor observation 路径；实际 checkpoint key 是
`observation_preprocessor`。

Adapter 只读取 deployment inference 所需状态：

```text
policy.net.*
policy.mean_layer.*
observation_preprocessor.running_mean
observation_preprocessor.running_variance
```

`log_std_parameter`、value network 和 optimizer 不进入 ONNX。

从 checkpoint 提取出的纯 NumPy 权重用于直接构造 ONNX graph：

```text
obs
 -> Sub(running_mean)
 -> Div(Sqrt(running_variance) + epsilon)
 -> Clip(-5, 5)
 -> MLP hidden layers
 -> mean layer
 -> actions
```

支持当前 trainer 的 ELU、ReLU、Tanh、Sigmoid、LeakyReLU 和 SELU。网络 shape 必须与 resolved `SkrlCfg`
一致，否则在生成 ONNX 前失败。

数值 reference 使用 SKRL 2.1 的 `RunningStandardScaler` 和 Torch layer/activation 实现，避免用同一份 ONNX
构图逻辑自证正确。

## 7. Motrix / Torch / FastSAC

FastSAC checkpoint 同时包含训练状态和部署所需的 actor 状态。Adapter 只恢复：

- `actor`：deterministic mean policy、action scale 和 action bias；
- `obs_normalizer`：actor observation 的运行均值、方差和标准差。

Critic、target critic、entropy temperature、optimizer 和 replay buffer 不进入 ONNX。FastSAC agent 始终保留
canonical actor/critic module，`torch.compile` 只创建训练期 runtime wrapper；checkpoint 只序列化 canonical
module，因此开启 compile 不会引入 `_orig_mod.` 前缀或改变 checkpoint contract。Adapter 直接 strict load
canonical actor state，不理解 PyTorch compile wrapper 的内部结构。

导出的 inference path 为：

```text
obs
 -> EmpiricalNormalization (when enabled)
 -> actor MLP
 -> deterministic mean
 -> optional tanh
 -> action scale + bias
 -> actions
```

同步和异步 trainer 产生相同 checkpoint contract，因此共享 `(motrix, torch, fastsac)` exporter。

## 8. Parity validation

每次导出都显式配置：

- random seed，默认 `1`；
- sample count，默认 `32`；
- `atol`，默认 `1e-4`；
- `rtol`，默认 `1e-5`。

样本包含一条全零 observation 和固定 seed 的标准正态随机输入。公共 validator 检查：

1. ONNX 恰好有一个输入和一个输出；
2. 名称、dynamic batch、feature dimension 和 `float32` dtype；
3. framework reference 与 ONNX 输出 shape；
4. 两侧输出全部 finite；
5. `numpy.allclose` 及最大绝对/相对误差。

验证失败时不会写入或覆盖目标 ONNX 文件。

## 9. 用户接口

ONNX 导出依赖已包含在运行环境中；FastSAC 与 SKRL Torch 必需安装，RSLRL 按需安装：

```bash
sh install.sh
sh install.sh --rslrl
source .venv/bin/activate
```

CLI 使用 run directory，不接受脱离 metadata 的裸 checkpoint：

```bash
python scripts/export_onnx.py \
  run_dir=runs/cartpole/skrl/torch/ppo/<run-id> \
  output=/tmp/cartpole.onnx
```

可显式覆盖验证参数：

```bash
python scripts/export_onnx.py \
  run_dir=<run-dir> \
  parity.seed=7 \
  parity.samples=64 \
  parity.atol=1e-5 \
  parity.rtol=1e-5
```

Python API：

```python
from motrix_rl.deploy import export_onnx

result = export_onnx("runs/cartpole/skrl/torch/ppo/<run-id>")
print(result.path, result.report.parity.max_abs_error)
```

未指定 output 时，默认写到 best checkpoint 同目录的 `policy.onnx`。

## 10. 扩展新 backend

新增 adapter 时：

1. 实现 `OnnxPolicyExporter`；
2. 在对应 `AgentProvider.create_policy_exporter()` 中延迟构造；
3. 从 typed task config 和 manifest checkpoint 严格恢复 deterministic policy；
4. 把 observation normalization 烘焙进 ONNX；
5. 使用真实 framework reference 通过公共 parity validator；
6. 增加小型 synthetic checkpoint 测试，不提交真实训练 checkpoint。

SKRL/JAX 可以复用 SKRL/Torch 的 NumPy weights 到 ONNX graph 路径，只需增加 Flax checkpoint extractor。
