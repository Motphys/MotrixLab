# 训练实时指标（Live Metrics）设计

## 摘要

本文描述 MotrixLab 训练进程把训练指标原生推送到外部 ingest 服务的设计。设置环境变量 `MOTRIX_METRICS_URL`
后，训练进程把 `motrix.metrics/v1` 记录通过 HTTP 批量推送给兼容 `motrix.metrics/v1` 的 ingest 服务；设置
`MOTRIX_METRICS_JSONL=1` 后，同一份记录还会写入 `run_dir/metrics.jsonl`。两者都未设置时功能完全不生效。两路输出都在
后台线程写出，训练线程只入队。训练器只通过控制平面的 `TrainingLogger` 一个接口记录指标，由它同时写 TensorBoard（tag
不变）与 live metrics。终端面板保持原样。

范围：公开协议 `motrix.metrics/v1`（环境变量、记录格式、请求体、状态机、幂等、重试规则）、规范 key 与各
provider 的映射、控制平面中的 `MetricsLogger` 与 `TrainingLogger`、FastSAC sync/async 的接入点、失败语义与测试
策略。非目标：SKRL / RSLRL 接入（见 §5.4）、wandb sink、media 上传、pod sidecar 采集、新增 `logging.backend` 取值、
服务端实现。

RL 控制平面与 provider 插件的分层见 [RL 多算法架构设计](./rl-multi-algorithm-architecture.md)；FastSAC async
的进程结构见 [FastSAC 异构训练器设计](./fastsac-async-heterogeneous-trainer.md)。

## 1. 目标与非目标

**目标**

- 训练中以秒级粒度在外部看板上看到进度、吞吐、loss、reward 曲线与关键事件（warmup 结束、checkpoint、崩溃）。
- 一份协议覆盖所有 RL 算法，跨算法对比统一按 `env_steps` 对齐。本期接入 FastSAC（off-policy）；SKRL / RSLRL 后续
  接入（§5.4）。
- 训练器只面对一个日志接口 `TrainingLogger`，不分别调用 TensorBoard 与 live metrics。
- 只用标准库，不新增依赖；服务不可达、慢或返回错误时，训练行为与耗时不受影响。
- 开关只认环境变量，`task_config` 结构不变，旧 run 的 `task_config.yaml` 与 `metadata.json` 读取不受影响。

**非目标**

- 不替换 TensorBoard：它是 `TrainingLogger` 的一路输出，tag 不变。终端面板消费结构化的 `TrainingPanelStats`，不经过
  `TrainingLogger`。
- 本期不接入 SKRL / RSLRL：它们的日志由框架内部的 writer 产生，接入只能拦截框架 writer。后续改为只使用它们的纯算法
  部分，日志统一走 `TrainingLogger`，届时再接入（§5.4）。
- 不新增 `logging.backend` 取值。FastSAC sync/async 只接受 `tensorboard`，live metrics 是附加输出。
- 不做 media 上传、eval 记录生产、wandb sink；协议为它们预留了 kind，但本仓库当前不产生。
- 不在库内安装信号处理器（见 §6）。

## 2. 协议 `motrix.metrics/v1`

本节是 MotrixLab 侧对协议的唯一定义。服务端是任意兼容 `motrix.metrics/v1` 的 ingest 服务；协议变化需要与服务端和
外部编排器同步，测试用 golden fixtures 锁定（§7）。

### 2.1 环境变量

| 变量                     | 说明                                                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `MOTRIX_METRICS_URL`     | ingest 服务基础地址。设置后启用 http 输出                                                                    |
| `MOTRIX_METRICS_JSONL`   | `1` / `true` / `yes` / `on` 时启用本地 `run_dir/metrics.jsonl` 输出，默认关闭。与 URL 都未设置时 logger 为 disabled，不建文件、不起线程 |
| `MOTRIX_METRICS_TOKEN`   | 写 token，只放在 `Authorization` 请求头，不写入 `task_config.yaml`、`metadata.json` 或 jsonl                 |
| `MOTRIX_METRICS_RUN_ID`  | 满足 `^[A-Za-z0-9._-]{1,200}$`。未设置时由 `run_dir` 相对路径派生：非法字符替换为 `_`，截断到 200 字符      |
| `MOTRIX_METRICS_PROJECT` | 看板上的项目名                                                                                               |
| `MOTRIX_METRICS_GROUP`   | 对比分组名                                                                                                   |
| `MOTRIX_METRICS_META`    | JSON 对象，原样放进 running 记录的 `meta`；解析失败时告警并忽略                                              |

HTTP 请求使用 `urllib` 默认的 proxy handler，代理与 `no_proxy` 由调用方环境决定。

### 2.2 记录格式

jsonl 每行一条记录，HTTP 每次发一批记录，两路都启用时是同一份记录：

```json
{
  "v": 1,
  "run_id": "...",
  "writer": "host:pid:role",
  "seq": 1791798324428939,
  "t": 1791798324.43,
  "kind": "run",
  "step": 12000,
  "axes": { "env_steps": 49152000 },
  "data": {}
}
```

- `writer` 为 `{hostname}:{pid}:{role}`，训练进程的 role 为 `trainer`。
- `seq` 在同一 writer 内单调递增，**初值取构造时的 `time.time_ns() // 1000`**，之后每条加 1（加锁）。同一容器重启后
  hostname 与 pid 可能完全相同，从 0 开始会让新记录被服务端按 `(writer, seq)` 当成重复丢弃。
- `t` 为 Unix 秒（float）。`step` 与 `axes` 只在 scalars / event 上出现。
- 数值统一转成 float；NaN 与 ±Inf 编码为 `null`。清洗对 `data` 递归生效（包括 running 的 `config`、event 的附加字段）；
  numpy 与 torch 标量先转 Python float。jsonl 序列化（在写出线程中）使用 `allow_nan=False` 作兜底，序列化异常吞掉并告警。

| kind        | data                                                                                                           |
| ----------- | -------------------------------------------------------------------------------------------------------------- |
| `run`       | `state` 与元信息，字段见 §2.4                                                                                  |
| `scalars`   | `{key: value}`，key 见 §3                                                                                      |
| `event`     | `name` 加附加字段，`name` 取值见下表                                                                           |
| `eval`      | 预留：`suite`、`task`、`ckpt_step`、`episodes`、`success_rate`、`metrics{}`，本仓库当前不产生                  |
| `media`     | 预留：run_dir 下的相对路径，本仓库当前不产生                                                                   |
| `heartbeat` | 空。只走 HTTP，不写 jsonl；空闲时补发，避免 `torch.compile` 等长时间无输出期间被服务端判为 lost               |

**事件名**

| name               | 时机                                                                                  | 附加字段                       |
| ------------------ | ------------------------------------------------------------------------------------- | ------------------------------ |
| `run_started`      | provider 完成启动、进入训练主循环（FastSAC async 为所有 worker ready 之后）           | provider 相关，例如 `workers` |
| `warmup_done`      | off-policy warmup 结束（`warming` 由 True 变 False）                                  | —                              |
| `checkpoint_saved` | 周期 checkpoint 与最终 checkpoint 写完                                                | `path`                         |
| `resumed`          | 从 checkpoint 恢复                                                                    | `path`、`global_step`         |
| `worker_crashed`   | FastSAC async 的 learner / collector 子进程异常退出                                   | `worker`、`exitcode`、`error_log` |
| `nonfinite`        | 某个 scalar key **第一次**出现 NaN/Inf；同一 logger 内每个 key 只发一次              | `key`                          |

`run_finished` / `run_failed` 由服务端在状态迁移时合成，客户端不发。

### 2.3 端点与请求

- `POST {URL}/api/v1/ingest`：请求体 `{"schema": "motrix.metrics/v1", "records": [...]}`，`Content-Type: application/json`，
  `Content-Encoding: gzip`；设置了 token 时带 `Authorization: Bearer <token>`。
- `GET {URL}/healthz`：服务存活探测，客户端不依赖。
- 每批最多 500 条。
- 2xx 表示已接收，响应体里单独列出的被拒记录不重试。
- **重试规则**：只在网络错误、408、429、5xx 时指数退避重试，503 遵守 `Retry-After`；其他 4xx（例如缺 token 的 401）
  丢弃该批并只告警一次，避免无限重试堵塞队列。

### 2.4 状态机与 run 记录字段

状态单调前进：`pending → running → finished | failed | killed`。训练进程只发 `running` 与三种终态；`pending` 与编排侧
`failed` 由外部编排器（`writer` 的 role 为 `pipeline`）发出。`lost` 由服务端推导：只有 `running` 超过 120 s 未收到任何
记录才判为 lost，`pending` 永不变 lost。训练进程的终态可以覆盖编排器写的终态，被覆盖的原因由服务端保存为
`infra_reason`。

| 时机     | `data` 字段                                                                                                                                                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 开始训练 | `state: "running"`、`project`、`group`、`meta`；`metadata.json` 的字段（env_name、rllib、train_backend、algo、seed、sim、checkpoint_format）；`family`；`job_type: "train"`；`primary_axis: "env_steps"`；`total_steps`；`num_envs`；`run_dir`；`config`；`motrixlab_version` |
| 正常结束 | `state: "finished"`                                                                                                                                                                                                                                                      |
| 异常     | `state: "failed"`、`reason`（`异常类型: 消息`，截断到 2 KB；FastSAC async 存在 `async_errors/*.log` 时附上路径）                                                                                                                                                         |
| Ctrl+C   | `state: "killed"`                                                                                                                                                                                                                                                        |

- `family`：FastSAC 为 `rl.offpolicy`，PPO 为 `rl.onpolicy`（PPO 目前只有 run 生命周期记录，见 §5.4）。
- `config`：`task_config` 的 resolved dict（与 `task_config.yaml` 同源）。
- `motrixlab_version`：`importlib.metadata.version("motrix-rl")`。

### 2.5 幂等

- scalars 由服务端按 `(run_id, key, step)` upsert，同一 step 后到的为准；event 按 `(run_id, writer, seq)` 去重。
- 两路都启用时，同一条记录（同一 seq）必须同时进入 jsonl 与 HTTP，保证事后用 jsonl 回填时可按 `(writer, seq)` 去重，两边
  记录逐条一致。jsonl 包含 run / scalars / event 全部 kind（heartbeat 除外）。需要回填保底的部署应同时设置
  `MOTRIX_METRICS_JSONL=1`。

## 3. 规范 key 与 provider 映射

规范 key 直接沿用 FastSAC 现有 TensorBoard tag，FastSAC 不需要映射。

| 规范 key                                | FastSAC                             |
| --------------------------------------- | ----------------------------------- |
| `rollout/mean_return`                   | 同名                                |
| `rollout/mean_ep_len`                   | 同名                                |
| `perf/env_steps_per_s`                  | 同名                                |
| `train/<loss>`                          | `qf_loss`、`actor_loss`、`alpha` 等 |
| `reward/<term>`、`metrics/<k>`          | 同名                                |
| `train/learning_rate`、`train/action_std` | 预留给后续接入的 on-policy 算法     |

- `reward/*` 的口径是"每步瞬时值在日志窗口内的均值"。
- `sys/*` 是固定清单，只发到 live，不写 TensorBoard：`sys/cpu_percent`、`sys/mem_used_gb`、`sys/gpu{i}/util_percent`、
  `sys/gpu{i}/mem_used_gb`（内存单位与终端面板一致，为 GiB）。不发每核 CPU 数据，避免大核数机器占用服务端 key 配额。

**step 与轴**

| provider      | `step`                                | `axes.env_steps`                                         | 频率                       |
| ------------- | ------------------------------------- | -------------------------------------------------------- | -------------------------- |
| FastSAC sync  | `global_step`                         | `(global_step + 1) * num_envs`（日志点上第 `global_step` 次迭代已采集完） | `logging.interval`         |
| FastSAC async | `collector_steps // num_collectors`   | `collector_steps * (num_envs // num_collectors)`，与 `performance.total_env_steps` 同口径 | 与 TensorBoard 写出点一致 |

`running` 记录里的 `total_steps` 由 `AgentProvider.total_env_steps(config, num_envs)` 给出（默认 `None`）：FastSAC 为
`num_learning_iterations * num_envs`，FastSAC sync 与 `env_steps` 同口径。
FastSAC async 只是近似：训练按 learner 的迭代数停止，而 `env_steps` 按 collector 计数，非严格 UTD 下两者会漂移，
`num_envs` 不能被 `num_collectors` 整除时也会少计，所以进度可能略超或略低于 100%。看板按进度展示时应截断到
`[0, 1]`，不能把它当作结束判据（结束以 `run` 状态为准）。

## 4. 组件：`motrix_rl/src/motrix_rl/metrics.py`

控制平面只新增这一个模块，只用标准库（json、gzip、urllib.request、threading、queue、socket）。

```python
class MetricsLogger:
    @classmethod
    def from_env(cls, run_dir: Path, *, role: str = "trainer") -> MetricsLogger: ...
    @classmethod
    def disabled(cls) -> MetricsLogger: ...
    enabled: bool
    run_id: str
    def run(self, state: str, **data) -> None: ...
    def scalars(self, values: Mapping[str, float], step: int, *, env_steps: int | None = None) -> None: ...
    def event(self, name: str, step: int | None = None, **data) -> None: ...
    def close(self) -> None: ...                 # 没有后续数据了；立即返回，幂等，异常吞掉
    def join(self, timeout: float) -> None: ...  # 可选：等 sink 收尾，最多 timeout
```

- **disabled**：所有方法是 no-op，不建文件、不起线程。
- **`JsonlSink`**：与 TensorBoard 的 `EventFileWriter` 相同，采用异步写出：`write` 只入队，daemon 线程负责 JSON 序列化、
  追加写 `run_dir/metrics.jsonl`（每条记录一行），每 2 s 批量写一次并 flush 一次，慢盘或网络文件系统不阻塞训练线程。
  与 TensorBoard 队列满时阻塞调用方不同，队列（10 万条）满时丢最旧的 scalars 并告警，run / event 不丢。`close` 之后写完队列即
  退出。序列化或 IO 异常吞掉并告警，不打断训练。
- **`HttpSink`**：daemon 线程 + 单一有界队列；入队一律 `put_nowait`，满了丢最旧的 scalars 并计数告警（run / event
  记录不丢）；每 2 s 或攒满 500 条发一批，请求超时 5 s，重试规则见 §2.3；空闲时补发 heartbeat。
- 两个 sink 都是具体类，不建基类或注册表。`MetricsLogger._emit` 在训练线程只做数据清洗、分配 seq 与入队；scalars
  在 `scalars()` 中已转成 float，不再二次清洗。
- **关闭是异步的**：`close()` 只表示没有后续数据，立即返回。每个 sink 按自己的节奏收尾：jsonl 写完队列后退出；http 发完
  队列后退出，失败按 §2.3 重试，最多重试 `flush_budget`（60 s）后丢弃剩余。训练结束不等待任何 sink；下一个训练任务用
  新的 logger，与这一个无关。`join(timeout)` 只供需要等待的调用方（测试、进程退出）使用，两个 sink 共用一个截止时间。
- **进程退出**：sink 是 daemon 线程，解释器退出时会被直接杀掉。`atexit` 钩子 `_join_at_exit` 在此之前关闭仍打开的
  logger 并 `join`，合计最多 `EXIT_TIMEOUT`（5 s），保证训练完立即退出的脚本也能送出最后几条记录（含终态）。健康时
  只需毫秒。
- logger 带线程与锁，不可 pickle；FastSAC async 只有父进程持有 logger，worker 子进程不接触。

### 4.1 统一日志接口：`motrix_rl/src/motrix_rl/training_logger.py`

```python
class TrainingLogger:
    def __init__(self, log_dir: Path, live: MetricsLogger) -> None: ...
    def scalars(self, values: Mapping[str, float], step: int, *, env_steps: int,
                live_extra: Callable[[], Mapping[str, float]] | None = None,
                placeholders: Collection[str] = ()) -> None: ...
    def event(self, name: str, step: int | None = None, **data) -> None: ...
    def close(self) -> None: ...
```

- 训练器每个日志窗口只调用一次 `scalars`：`values` 以 key 为 tag 写 TensorBoard，同一份值发 live；指标只组装、归约一次。
- `live_extra` 返回只发 live 的值（例如 `sys/*`，TensorBoard 原本没有），只在 live 启用时才调用，关闭 live 时不做任何
  额外计算；`placeholders` 列出尚未定义的值（例如第一个 episode
  结束前 NaN 的 rollout 均值）：TensorBoard 照写，live 不发，`nonfinite` 名额留给真正的发散。
- `event` 只发 live。`close` 只关 TensorBoard（异常吞掉，可放在 `finally`）；live logger 的终态与关闭归控制平面
  （`TrainerHandle`）。
- TensorBoard writer 在第一次 `scalars` 时才打开（`torch.utils.tensorboard.SummaryWriter`，打不开时告警，只发 live），
  训练在记录任何指标之前失败不会留下空 event 文件。

## 5. 接入点

### 5.1 控制平面

- **构造**：只在 `runner.create_training_handle` 中用 `MetricsLogger.from_env(run.run_dir)` 构造，经 `metrics=` 参数传给
  `create_trainer_context`。play 路径（`create_run_handle`）一律使用 disabled，避免设置了环境变量时 play 也起线程或
  改写原训练 run 的 `metadata.json`。
- **契约字段**：`TrainerContext.metrics` 与 `TrainerHandle.metrics`，均有 disabled 默认值，第三方 provider 不接入也能
  得到状态记录。`create_trainer` 从 typed context 填入 `TrainerHandle.metrics`。
- **生命周期**（`TrainerHandle.train()`）：发 `running` → `trainer.train()` → 发 `finished`；`KeyboardInterrupt` 发
  `killed`；`SystemExit` 非 0 与其他异常发 `failed`；最后在 `finally` 中 `close()`，不等待 sink。异常照常
  向上抛。
- **run 标注**：`annotate_run(run_dir, live={"run_id": ...})`，只写 run_id，不写 URL 与 token。旧版本读取
  `metadata.json` 时忽略未知顶层 key。

### 5.2 FastSAC sync（`fastsac/sync/train.py`）

- 日志点组装一份 `scalars` 字典（tag 集合逐字不变）交给 `TrainingLogger.scalars`，`sys/*`（复用面板已采好的样本）
  经 `live_extra` 只发 live。
- 还没有 episode 结束时，`rollout/mean_return`、`rollout/mean_ep_len` 是 NaN 占位，经 `placeholders` 传入：TensorBoard
  照写，live 不发。否则每次训练开局都会把这两个 key 的 `nonfinite` 名额用掉，之后真正的发散反而不报。
- 事件：`run_started`、`warmup_done`、`checkpoint_saved`（周期与最终）、`resumed`。
- `TrainingLogger.close()` 放在 `finally` 中，训练异常时也关闭 TensorBoard。

### 5.3 FastSAC async（父进程 `fastsac/async_impl/train.py`，learner 只多发 checkpoint 通知）

- 与 sync 相同，日志窗口经 `TrainingLogger.scalars` 写出；汇总的 episode 数为 0 时 rollout 均值作为 `placeholders`。
  TensorBoard 不可用时 live 照常输出。
- 事件：所有 worker ready 后发 `run_started`；resume 时发 `resumed`；`warming` 由 True 变 False 时发 `warmup_done`；
  primary learner 每写完一个周期 checkpoint，就经 panel 队列单独发一条通知（仅当 live 启用，由 learner 参数
  `notify_checkpoints` 控制；`put_nowait`，队列满时丢弃并告警，不阻塞 learner）（`{"rank", "checkpoint_saved": {"step",
  "path"}}`），父进程 drain 到即按其中的 step 发 `checkpoint_saved`。不能借用日志窗口 payload 的 `checkpoint_path`：
  learner 在同一轮里先发 payload 后存 checkpoint，路径要等下一个日志窗口才送达，最后一个窗口之后的 checkpoint 会丢。
  所有 worker 正常退出且最终 checkpoint 存在时，父进程再发一次；监控循环检测到子进程崩溃时发 `worker_crashed`。
  终态 `failed` 由父进程抛出的异常经 `TrainerHandle` 统一发出。

### 5.4 SKRL 与 RSLRL（后续）

本期不接入。它们的 TensorBoard 由框架内部的 writer 写出，接入只能替换或包裹框架 writer，日志逻辑仍分散在框架里。
后续计划只使用两者的纯算法部分（更新、网络与损失），rollout、日志窗口与指标由 MotrixLab 自己组织，统一经
`TrainingLogger` 写出，与 FastSAC 共用一个接口与一套规范 key（`train/learning_rate`、`train/action_std` 已预留）。

在此之前，SKRL / RSLRL 训练仍会得到控制平面发出的 run 生命周期记录（`running` / `finished` / `failed` / `killed`，
`TrainerHandle` 与 provider 无关），但没有 scalars 与事件；`total_steps` 为 `None`。

## 6. 失败语义

- **服务不可达或慢**：入队非阻塞，发送在 daemon 线程中进行，单次请求超时 5 s；训练结束不等待发送，进程退出最多多等
  `EXIT_TIMEOUT`。未送出的记录丢弃；启用了 jsonl 时本地记录仍保留，可事后回填。
- **服务拒绝**：非重试类 4xx 丢弃该批并告警一次，训练继续。
- **队列满**：丢最旧的 scalars，计数并告警；run / event 记录保留。
- **本地写失败或慢盘**：jsonl 在后台线程写出，序列化或 IO 异常吞掉并告警；慢盘只会让队列积压，不阻塞训练，训练结束
  也不等待；磁盘卡死时进程退出最多多等 `EXIT_TIMEOUT`。
- **SIGTERM（例如 pod 被删除）**：Python 默认处理直接退出，`finally` 不执行，训练进程不会发出终态，jsonl 也可能来不及
  被外部归档。库内不安装信号处理器（侵入性过强），由外部编排器写 `failed` 与服务端的 lost 推导兜底。这是接受的缺口。
- **分布式**：只有 rank0 持有 enabled logger。FastSAC async 只在父进程。

## 7. 测试策略

测试面向公开契约，不测私有实现，不新增 conftest 与 marker。

- `motrix_rl/tests/test_metrics.py`：未设置 URL 时不生效（不建文件、不起线程）；记录字段、writer 格式与 seq
  初值和单调性；NaN / Inf / numpy / torch 值与嵌套 `data` 的清洗；`nonfinite` 每 key 一次；run_id 清洗与截断；
  用标准库 `http.server` 校验 gzip、Bearer 与请求体；500 后补发成功且 seq 不重复；4xx 丢弃；服务不可达时 `close`
  按时返回；heartbeat；与 golden fixtures（`motrix_rl/tests/fixtures/metrics_v1/`，外部编排器的 pending / failed 请求
  体与 MotrixLab 产出的 running / scalars / event 样例）结构一致。
- `motrix_rl/tests/test_rl_contracts.py`：成功、异常、Ctrl+C 分别发出 finished、failed（带 reason）、killed，异常照常
  抛出，`close` 在终态之后执行；磁盘卡住时 `train()` 照常立即返回，记录随后写完。`test_metrics.py`：`close` 立即返回、
  `join` 共用一个截止时间；退出钩子写完从未关闭的 logger 的记录；服务不可达时 http 在 `flush_budget` 内放弃。`test_runner.py`：play 不构造 enabled logger；未设置 `MOTRIX_METRICS_JSONL` 时不产生 jsonl。
  `test_metrics.py` 另覆盖 URL 与 jsonl 开关的组合，以及磁盘卡住时 emit 不阻塞、记录不丢且按批写出。
- 单元测试只断言行为，不断言耗时：等待一律用 Event 或条件轮询（条件满足立即返回，超时只在失败时出现），重试测试把
  `HttpSink(backoff=...)` 调小，不依赖真实的 1 s 退避。
- 性能由 `bench/bench_live_metrics.py` 单独度量，不进 CI。同一个合成训练循环（`--workload numpy`：BLAS 释放 GIL，
  接近 torch 训练步；`--workload python`：训练线程全程持有 GIL，后台线程的最坏情况）每 `--log-every` 次迭代经
  `TrainingLogger.scalars` 记录一次，所有场景都开着 TensorBoard，与 off 的差值即 live 本身的代价。场景：off（与重复一次的
  `off, control` 噪声基线）、jsonl、jsonl 慢盘（每批卡 0.5 s）、jsonl 磁盘卡死（永不返回）、jsonl 写入报错、http、
  http 慢服务（0.5 s 才应答）、http 无响应（接受连接但永不应答）、http 拒绝连接、jsonl + http、磁盘卡死 + 无响应。
  只度量训练循环：报告训练线程上每次 emit 的耗时分位数、循环吞吐相对 off 的开销、落盘与送达的记录数；关闭不计时，
  故障场景只 `close()` 不 `join`，不必每轮等满超时。ingest 服务在独立进程中运行，不与训练进程争 GIL。`--max-overhead <百分比>` 超限时以非零退出码结束。
- FastSAC：锁定 sync / async 的 TensorBoard tag 集合；scalars key 为 TB tag 加 `sys/*`；`env_steps` 口径；事件；
  async worker 进程参数中不出现 logger；learner 的 `notify_checkpoints` 只在 live 启用时为真。
- `motrix_rl/tests/test_training_logger.py`：一次 `scalars` 同时写 TensorBoard 与 live；`placeholders` 只写
  TensorBoard、`live_extra` 只发 live；只发事件时不打开 TensorBoard；TensorBoard 不可用时 live 照常；live 关闭时只写
  TensorBoard、不产生文件；`close` 只关一次且不抛异常。
