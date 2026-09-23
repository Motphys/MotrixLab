# FastSAC 异构（Collector/Learner 分进程）训练器设计

## 摘要

异构 FastSAC 训练器把**仿真采样（collector）**与**网络训练（learner）**拆到两个进程，通过共享内存交换 transition 与权重，使 CPU 物理仿真与 GPU 梯度计算重叠，消除同步实现里「采样 → 训练 → 采样」串行循环中的 GPU 空转。多 NUMA node 服务器上可配置多个 collector 进程（每 node 一个），拓扑见 §8；本节先描述默认的 1 collector × 1 learner。

同步与异步执行共用 `motrix` framework 下唯一的 `fastsac` provider，对外方法名统一为 `motrix.fastsac`。`algo.asynchronous` 只选择执行拓扑，不改变算法、配置类型、run 身份或 checkpoint 格式。算法本身（`Actor`/`Critic`/`SimpleReplayBuffer`/`EmpiricalNormalization`/`FastSacAgent`）**原样复用、逐字节一致**。异构执行是默认模式，首要目标是让采样与训练各自满速。

算法原理见 [SAC 与 FastSAC 算法原理入门](../research/sac-and-fastsac-primer.md)；所处的多算法框架见 [RL 多算法架构设计](./rl-multi-algorithm-architecture.md)。

---

## 1. 设计前提

分进程之所以在**不改算法**的前提下成立，来自 FastSAC 的两个既有性质：

1. **off-policy**：learner 从 replay buffer 采样，不要求数据来自最新策略，因此 collector 用「稍旧几步」的策略采样是合法的（Ape-X / SEED 类分布式 off-policy 架构的共同前提）。策略滞后（staleness）是需要被监控和约束的量，而非正确性障碍。
2. **normalizer 是 learner 独占的写方**：observation normalizer 只在梯度更新中以 `update=True` 更新，采样路径（`FastSacAgent.act`）用 `update=False` 只读。因此 collector 侧**无需回写** normalizer 统计量，只需接收 learner 下发的快照——这消除了异构化最常见的一个双向同步难点，权重通道退化为单向 learner → collector。

---

## 2. 总体架构

```
                     进程 A: Collector                              进程 B: Learner (GPU)
    ┌──────────────────────────────────────────┐      ┌──────────────────────────────────────────┐
    │  DirectEnv (numpy 物理仿真, num_envs)        │      │  FastSacAgent（复用同步版，不驱动 env）      │
    │  Actor 副本 (CPU/CUDA 推理, eval) ◄ weights │      │   ├ actor / qnet / qnet_target             │
    │  obs_normalizer 副本 (同 actor, 只读)        │      │   ├ optimizers                             │
    │                                            │      │   ├ SimpleReplayBuffer (GPU) ◄── ingest    │
    │  loop step_once():                         │      │   └ EmpiricalNormalization (唯一写方)       │
    │    a = actor.explore(norm(obs))            │      │                                            │
    │    obs',r,term,trunc = env.step(a) ──┐     │      │  loop:                                     │
    │    ring.push(transition) ────────────┼─────┼──►   │    drain: ring → rb.extend (→GPU)          │
    │    episode 记账                       │     │      │    if ready: agent.update(n)               │
    │    poll weights  ◄───────────────────┼─────┼───   │    每 weight_publish_interval: publish 权重 ─┼─►
    │    满环则 backoff sleep               │     │      │    owns 日志 / checkpoint                    │
    └──────────────────────────────────────┘    │      └──────────────────────────────────────────┘
    ┌──────────── 共享内存 (torch.multiprocessing, spawn 继承的 CPU 共享 tensor) ─────────────────────┐
    │  ① SharedTransitionRing   collector→learner   原始 transition 的 SPSC 有界背压环                │
    │  ② WeightSnapshot         learner→collector   actor 权重 + obs_normalizer 统计量（seqlock 双缓冲）│
    │  ③ Control                共享标量：stop / global_step / collector_steps                        │
    └────────────────────────────────────────────────────────────────────────────────────────────┘
    ④ StatsQueue (mp.Queue, maxsize=8)  collector→learner  episode return/length、reward 分项、timing
```

**进程角色**

- **Collector（进程 A）**：拥有唯一的 CPU `DirectEnv` 与一个**推理专用**的 `Actor` + `obs_normalizer` 副本（CUDA 默认，也可显式选择 CPU；两者均 `eval`）。只做前向、`env.step`、把 transition 批推入共享环、读权重快照、维护 episode 记账。不持有 optimizer、qnet、replay buffer。
- **Learner（进程 B，GPU）**：就是现有的 `FastSacAgent`，但**不驱动 env**。它从共享环把 transition 灌进自己的 GPU replay buffer，照常调用 `agent.update(n)`；周期性把 actor 权重 + normalizer 快照发布到共享内存；并独占日志与 checkpoint。

**为什么 transition 走「共享环 + learner 端 ingest」，而不是两进程直接共享 GPU replay buffer？**

- `SimpleReplayBuffer.sample()` 含大量 gather / n-step 计算，是 learner 独占的读路径；让 collector 也触碰会引入跨进程锁与 GPU 上下文共享。
- 环与 buffer 解耦：环只做无锁搬运，buffer 的环形索引、n-step 语义完全归属 learner。

环本身有两种物理传输，由 `transition_ipc` 选择（见 §4.1）：

- **host 共享内存环**（默认回退路径）：CPU `share_memory_()` tensor，任何设备组合都可用。
- **CUDA-IPC 设备环**：collector 与 learner 推理/训练在同一 GPU 时，slot 直接放在显存里。env 的 CPU 输出在 collector 侧一次融合 H2D 直写 slot，learner 的 ingest 变为纯 D2D——env 产生的字节只过一次 PCIe，且被 collector 的 env step 时间掩盖；learner 关键路径上不再有任何 H2D/staging memcpy。

**collector 的环境固定在 CPU，Actor 的设备独立配置**：基于 2048/4096 环境的服务器端到端吞吐结果，默认把 actor 与只读 observation normalizer 放到 CUDA；CPU 保留为显式兼容配置。CUDA 路径不改变 env wrapper 的 device，也不移动 critic observation、reward/done、bookkeeping；transition 环的物理位置由 `transition_ipc` 决定，与推理设备解耦。learner 与 collector 同卡时仍需结合具体任务确认资源竞争边界。

CUDA 推理的数据边界为：CPU policy observation 先复制到预分配 pinned host buffer，再异步 H2D 到固定 shape device buffer；actor 输出立即异步 D2H 到预分配 pinned action buffer，并在返回 CPU env 前同步。若 `torch.compile(mode="reduce-overhead")` 复用 CUDA Graph output storage，D2H 已在下一次 replay 前完成，因此环境不会持有随后被覆盖的 device output 引用。权重由 learner 发布到权重通道（seqlock 双 buffer），物理位置由 `weight_ipc` 选择：`auto`（默认）在 learner 与 collector 推理同卡且 actor 参数达到 `weight_ipc_min_bytes` 时使用 CUDA-IPC 设备槽（publish 是一次 D2D 拷贝），否则使用 host 共享内存槽（一次融合 pinned D2H + shm 写入）。collector actor 参数绑定到一个 contiguous device flat buffer：host 路径每个新版本只做一次 pinned H2D，而不是逐参数传输；device 路径则完全不经 host。

---

## 3. 代码结构与接入

通用算法组件留在 `motrix_rl/fastsac/`，同步与异步各自只保留编排层。由于 `async` 是 Python 关键字，异步实现的内部包名用 `async_impl`；公共配置字段使用 `asynchronous`。

```
motrix_rl/src/motrix_rl/fastsac/
├── agent.py / buffer.py / networks.py / wrap_np.py
├── config.py                 # FastSacCfg 与共享配置
├── framework.py              # 注册 motrix.fastsac，根据配置选择 Trainer
├── sync/
│   └── train.py              # 同步 Trainer
└── async_impl/
    ├── shm.py                # 共享内存原语：SharedTransitionRing / WeightSnapshot / Control
    ├── numa.py               # 多 collector 的 NUMA/CPU 绑定（sched_setaffinity + libnuma set_membind）
    ├── collector.py          # Collector：CPU 采样进程逻辑
    ├── learner.py            # Learner：GPU 训练进程逻辑 + UTD 治理
    ├── worker.py             # module-level 进程入口（可被 spawn pickle）+ 共享 builder
    └── train.py              # Trainer(TrainerBase)：分配共享内存、spawn 两进程、编排生命周期
```

**复用（import，不复制）**：`fastsac.networks.{Actor, Critic}`、`fastsac.buffer.{SimpleReplayBuffer, EmpiricalNormalization}`、`fastsac.agent.FastSacAgent`、`fastsac.wrap_np.FastSacNpEnvWrap`、`fastsac.config.FastSacAgentCfg`。

**Framework 接入**：`FastSacProvider` 以 `agent_name="fastsac"` 注册在 `motrix` framework 下，`train_backend="torch"`、`checkpoint_format="pt"`。`create_trainer` 根据 `FastSacCfg.asynchronous` 返回同步或异步 Trainer，因此两种拓扑天然共用 checkpoint 与 play 路径。

**任务配置**：每个环境只保留一个 `configs/task/<env>/motrix.fastsac.yaml`。Task 固定记录 `rllib: motrix`、`algo: fastsac`；`algo.asynchronous` 默认为 `true`，需要同步 baseline 时覆盖为 `false`。

```bash
uv run scripts/train.py task=g1-walk-flat/motrix.fastsac                         # 异构（默认）
uv run scripts/train.py task=g1-walk-flat/motrix.fastsac algo.asynchronous=false # 同步 baseline
# 二者产出的 checkpoint 均为 .pt，同一 scripts/play.py 可回放
```

---

## 4. 共享内存原语（`shm.py`）

三个通道全部由父进程在 spawn 前分配、`.share_memory_()`，子进程通过 spawn reduction 继承句柄。所有跨进程标量都是**对齐 int64 共享 tensor**：每个游标 / 计数器**只有单一写方**，故无需原子 RMW（CAS / fetch_add）；「数据先于游标可见」这一顺序依赖**当前仅在强内存序 ISA（x86-64 / TSO）上成立**，代码不插入任何内存屏障。**弱内存序（ARM64）暂不支持**：Python 无可移植的独立 fence，`atomics` 包也只提供 `atomicview` 上的有序 load/store（且无 x86 wheel），要支持 ARM 需把每个游标 / `_seq` 的 store/load 改走 `atomicview` 的 `RELEASE`/`ACQUIRE`，属于对这些站点的重构。

### 4.1 SharedTransitionRing — SPSC 有界背压环

单生产者（collector）单消费者（learner）的环形缓冲。每个 slot 存**一个 env-step 的全 `num_envs` 批**，与同步版「一次 `extend` 写 `num_envs` 条」的粒度一致。八个字段各为一块 `(capacity, num_envs, dim)` 共享 tensor，与 `SimpleReplayBuffer.extend` 入参一一对应：

| 字段 | dtype | 形状 |
|---|---|---|
| `obs` / `next_obs` | f32 | `(C, N, obs_dim)` |
| `critic_obs` / `next_critic_obs` | f32 | `(C, N, critic_obs_dim)` |
| `actions` | f32 | `(C, N, act_dim)` |
| `rewards` | f32 | `(C, N)` |
| `dones` / `truncations` | i64 | `(C, N)` |

（`C = ring_capacity`，默认 64；`N = num_envs`。）

两个共享游标 `_write` / `_read` 实现无锁环：

- **生产**：`push()` 满环（`write - read >= C`）时返回 `False`，**不 step env、不丢数据**；写入 slot 各字段后再 `_write += 1`（x86/TSO 下字段写保证先于游标 bump 可见）。
- **消费**：`read_span()` 返回最长连续未读 run 的长度与六个字段的 `(count, num_envs, dim)` 视图但**不推进** `_read`；learner 拷入 replay buffer 后再 `commit_reads(count)`（`_read += count`，同様由 event 定序，见 IPC 环）。读游标只在拷贝完成后前进，故生产者永不覆盖仍在 ingest 的 slot。
- 游标（`RingCursors`）是独立于字段存储的 host 共享 tensor，由父进程创建——host 环与 IPC 环复用同一游标协议。

**背压方向是核心旋钮**：环满（collector 快）→ collector 阻塞采样，天然把采样速率压到 learner 消费速率，防止无界内存增长、防止 replay buffer 被过新数据刷爆而 off-policy 失真；环空（learner 快）→ learner 无新数据可 ingest，由 §5 的 UTD 治理决定「等数据」还是「在已有 buffer 上继续更新」。`C` 需足够吸收两进程抖动（一次 GC、一次 CUDA sync），但不宜过大以免抬高在途 staleness。

#### CUDA-IPC 设备环（`transition_ipc`）

host 环下 learner 的 drain 承担全部搬运：ring→pinned 的 CPU memcpy 与 H2D 都在 learner 关键路径上。设备环把这段搬运整体移到 collector 侧并隐藏：

- **融合 slot 布局**：全部字段放进一块 `(C, num_envs, obs_dim + critic_obs_dim + act_dim + 3)` 的 f32 设备 tensor（reward/done/truncation 以 0/1 float 存储，learner 拷入 i64 buffer 时由 `copy_` 顺手完成数值恒等的类型转换）。collector 每步把六个 CPU 字段拼进一块 pinned staging，**一次 H2D** 写入整个 slot；learner 侧按最后一维 offset 切出六个 strided 视图直接喂 `extend_batch`（纯 D2D）。
- **归属与交接**：设备字段由 learner 进程分配（IPC handle 导出方需要 CUDA context 且必须保活），经既有的 `slot_queue` 握手随权重 slot 一起 ship 给 collector（`torch.multiprocessing` 已注册 CUDA tensor 的跨进程 reducer）；collector 侧从到达的 tensor 建立接收端。游标沿用父进程创建的 host `RingCursors`，两进程可见。
- **发布定序（正确性核心）**：GPU 写入不受 x86/TSO 保障，游标 bump 前必须让设备写入落定——生产者在 H2D 入队后对该 slot record 一个 CUDA event，`_write` 只推进到「event 已完成的最旧 slot + 1」（惰性 flush：`is_full`/`push` 前查询队首 event，完成即推进）。消费侧对称：`commit_reads` 在 D2D 读取入队后 record event，`_read` 只推进到「event 已完成」的边界。staging 复用同样由 event 守护：覆写前确认上一次 H2D 已完成（正常节奏下 env.step 的毫秒级间隔使其成为 no-op）。
- **采样无竞争**：learner 的 D2D ingest 与 `sample()` 在同一条默认流上，流内天然有序，不需要额外 event；event 只服务于跨进程游标推进。
- **传输选择**：`transition_ipc: auto/on/off`。`auto` 要求 learner 设备与 `collector_inference_device` 解析后同为 CUDA（未显式给 index 时视为同卡；显式 index 不同则回退）；`on` 在不满足时告警回退 host 环。host 环是永久保留的回退路径，覆盖 CPU collector、异卡与 IPC 建立失败的边界。
- **代价**：显存增加约 `C × num_envs × feat × 4` 字节（microduck-walk-flat 量级约 130MB），高维 critic_obs 任务需把 `ring_capacity` 纳入显存预算。



**语义一致性**：collector 只是把同步版 collect 相位原样搬到另一进程，transition 的构造代码相同——调用顺序、dtype（dones/truncations 用 long）、auto-reset 后的 next_obs 语义与同步版逐字节一致。这是「算法未变、只变执行拓扑」的基础。

### 4.2 WeightSnapshot — seqlock 双缓冲（learner → collector）

collector 前向只需两样东西，打包成一个快照：**actor 权重**（展平成单个 float 向量）与 **obs_normalizer 统计量**（`_mean` / `_std` / `_var` / `count`）。不含 qnet、critic_obs_normalizer、optimizer，故体积小。

传输用 **seqlock 保护的双缓冲**。朴素双缓冲（「写方发布到另一槽、读方读当前槽」）本身**并非无竞争**：若 learner 在 collector 一次 `maybe_load` 期间连续发布两次，第二次会复用 collector 仍在拷贝的槽，产生**撕裂快照**并随后驱动策略数千步——难复现、难调试。seqlock 用一个共享计数器 `_seq` 消除它：

- 写方 `publish()`：`_seq` 先自增到**奇数**（标记写入中）→ 写入与当前活跃槽相对的另一槽 → `_seq` 回到**偶数**（x86/TSO 下数据写保证先于这次 seq 偶数写可见）。
- 读方 `maybe_load()`：读 `_seq`，若为奇数则重试；记 `version = _seq // 2`，仅当比本地新才拷出整槽到本地临时张量，再复检 `_seq`；若期间发生过发布（`s1 != s2`）则丢弃重试，否则将 params 载入 actor、stats 载入 normalizer。

因唯一写方是 learner，公共路径下单次发布落在**另一槽**、`_seq` 甚至不与读方相撞，读几乎永不重试；罕见的「一次读期间两次发布」由 `s1 != s2` 检出并重试，**永不把撕裂快照暴露给 actor**。全程无锁、无 CAS。公开 `version = _seq // 2` 从 0 起，与 collector 初始 `_local_version = 0` 对齐，保证首次发布被看到。

`weight_publish_interval` 越小 staleness 越低、GPU→CPU 拷贝越频，是新鲜度/吞吐的权衡旋钮。

#### GPU collector 权重同步剖析与传输边界

RTX 5090 上以 K1 WBT、2048 environments 和默认 `weight_publish_interval=4` 对完整 `train.py` 做 Nsight Systems
剖析后，actor flat snapshot 为 932,528 bytes，单次 H2D 的 GPU device time 平均约 35.8 us；128 次权重 H2D
合计 4.58 ms，而同一训练窗口累计的 collector `sync` wall time 约 176.6 ms。原始 PCIe H2D 只占 `sync`
约 2.6%，不是主瓶颈。细分 wall time 还包括 collector 等待 learner 完成发布、稳定 CPU snapshot 拷贝，以及同卡
learner 竞争下的 CUDA stream completion；不能把最后一项全部归因于 memcpy。

因此当前传输继续使用 CPU shared-memory snapshot，不引入 CUDA IPC。先执行两个更小且直接针对剖析结果的改动：

- learner 在把 seqlock 置为 odd 之前完成 actor/normalizer 的 D2H materialization，使 collector 能继续读取上一个完整
  version，不为 learner 的 CUDA completion 自旋；actor 参数先在 device 上 flatten，再做一次 D2H。
- CPU/CUDA collector 都复用持久 local param/normalizer staging，CUDA staging 使用 pinned memory enqueue H2D，不在 `maybe_load` 末尾单独同步。下一次 inference
  与权重加载位于同一 CUDA stream，stream ordering 保证 policy forward 看到完整新权重，而 inference 原有的 action D2H
  barrier 保证 CPU environment 收到动作前全部完成。每次成功 env step 最多 poll 一次权重，下一次 staging reuse 之前必定
  先执行这次 inference，因此 staging 生命周期覆盖异步拷贝。

`perf/collector_sync_{wait_writer,host_snapshot,actor_load}_ms` 分别记录 writer 等待、stable host snapshot 和 actor-load
enqueue 的平均 wall time。CUDA completion 若被流水到下一次 inference，会体现在 `collector_sample_actions_ms`；判断优化
必须同时看 `collect_ms_per_batch` 与 `env_steps_per_s`，不能只看 `collector_sync_ms`。

只有后续完整训练剖析证明剩余 GPU transfer 在端到端 collector critical path 中占主导，并且 CPU snapshot 方案无法通过
流水隐藏时，才升级到 persistent CUDA IPC snapshot。届时必须同时解决跨进程 CUDA event、slot reuse acknowledgement、
producer lifetime 和 compiled collector 固定参数地址，不能只把 H2D 替换成无同步的 D2D。

### 4.3 Control — 共享标量

一小组共享 int64：`stop`（停止标志）、`global_step`（learner 迭代计数）、`collector_steps`（已产出的 env-step 批数，即训练进度基准；多 collector 下为 per-collector 计数器数组，每个 collector 单写自己的计数，聚合属性求和，见 §8）。seed 不放这里，作为进程入口参数直接传入（collector 的 seed 为 `seed + collector_id`）。

---

## 5. Learner 与 update-to-data（UTD）比例治理

同步版是**确定性比例**：每 `num_envs` 条新 transition 恰好做 `num_updates` 次更新。异构下两进程自由奔跑，比例会漂移，直接影响样本效率、稳定性与「与同步版可比性」，因此需显式治理。

定义 **UTD = 梯度更新次数 / 已产出环境步批数**。learner 主循环每轮：`drain()` 灌入至多 `max_ingest_per_iter` 个 slot → `maybe_train(ingested)` 依 `utd_mode` 决定本轮更新数 `n` → `agent.update(n)`。单次 `update(n)` 内部完成 n 步更新，并用 agent 上持久的 `update_idx` 计数器做 `policy_frequency` 门控（actor/Q 更新比例跨调用精确为 `1/policy_frequency`），`update_idx` 是 sync/async 共享的唯一真源。

**三种模式（`utd_mode`）**：

1. `learner_bound`（**吞吐优先**）：只要 buffer 就绪，每轮固定做 `num_updates` 次更新、GPU 打满；collector 满速采样，UTD 随两进程相对速度浮动 → 最高 wall-clock 吞吐。代价是 UTD 偏离同步版，样本效率曲线会偏移，故实际 UTD 作为 `async/utd` 一等指标输出。
2. `strict`（对比验证用）：`n = ingested * num_updates`，严格维持 `UTD == num_updates`，只买「相位重叠」的加速、不改样本效率 → 与同步版最可比；无新数据时 `n=0`，learner 让出时间片等数据。


**权重发布与 staleness**：learner 每 `weight_publish_interval` 次更新调用一次 `publish`。collector 用的是「上次拉取的权重版本」，滞后 ≈ 发布间隔 × learner 速度 + 环内在途 slots。`learner_bound` 下 staleness 比 `strict` 大，故 `async/policy_lag`（= `weights.version − collector 本地版本`）作为一等诊断指标输出，必要时缩短 `weight_publish_interval` 主动压低。

---

## 6. 进程生命周期、日志与 checkpoint

### 6.1 spawn 与自建资源

- 用 `torch.multiprocessing.get_context("spawn")`——**必须 spawn 而非 fork**（CUDA + fork 不安全）。
- 共享内存原语在父进程（`Trainer`）分配并 `share_memory_()`，作为参数传给两个子进程，spawn 下靠 reduction 传句柄。
- `DirectEnv` 持有 motrixsim 原生句柄、**不可 pickle**：子进程**不接收 env 对象**，而是接收 `env_name + num_envs + seed`，在进程内用 `env_registry.make(...)` 自建（与同步版构造一致）。learner 同理自建 agent。父进程只用一次「1 env」的探针构建读出 obs/critic/act 维度，随即丢弃。
- 当前仅支持 Motrix `np` 仿真后端（collector 环境天然是 CPU 负载）；learner device 由 `FastSacCfg.device` 或 CUDA 可用性决定，collector inference device 由 `trainer.async_options.collector_inference_device` 独立决定且默认为 CUDA。CUDA 不可用时直接失败，不回退到 CPU；CPU-only 运行需显式配置 `collector_inference_device=cpu`。

### 6.2 日志归属

episode return / length、reward 分项、env metrics、collector timing 都发生在 collector（它才有 reward/done）。collector 按根级 `logging.interval` 把一份紧凑 `snapshot_stats()` 放进 `StatsQueue`（先清掉旧快照，保证 learner 总见最新）；learner 在日志相位 drain 出来，喂给与同步版**完全复用**的 rich 训练面板。

TensorBoard scalar 与同步版同名（`rollout/mean_return`、`rollout/mean_ep_len`、`perf/env_steps_per_s` 等），并新增异构专属：`async/policy_lag`、`async/ring_fill`、`async/weight_version`、`async/utd`，以及 collector 细分 timing `perf/collector_{sample_actions,env_step,env_step.*,push,bookkeep,sync,sync.*}_ms`（嵌套阶段为点分路径，如 `collector_env_step.physics.read_ms`）、整体 `perf/collect_ms_per_batch`，learner 侧 `perf/learn_ms_per_update`、`perf/learn_pct`、`perf/updates_per_s`。

> 面板中 `collect_ms` / `learn_ms` / `learn_pct` 因两进程并发，**不像同步版那样相加为 100%**：`learn_pct` 表示 learner wall-clock 中真正用于更新（vs 空转/欠数据）的比例，≈100% 表示 GPU-bound，偏低表示 collector 喂不满 buffer。

### 6.3 checkpoint 兼容

learner 持有完整 `FastSacAgent`，直接复用其 `state_dict()` 与通用 checkpoint 记录逻辑。产出的 `.pt` 与同步版**结构相同**，因此：同一 `scripts/play.py` 可回放异构版 checkpoint；异构版可 resume 同步版 checkpoint，反之亦然，支持「同步预训练 + 异构继续」这类混合实验。

### 6.4 关闭与容错

- 正常结束：learner 达到 `num_iterations`（以 `collector_steps` 为基准）→ 主循环退出 → 父进程置 `Control.stop`。
- 异常：父进程监控循环发现 learner 退出即结束；collector 崩溃（非零 exit code）则置 stop、终止另一进程。任一子进程 `finally` 都会 `set_stop()`，使对端及时退出。
- 父进程 `join`（带 timeout）后对仍存活的进程 `terminate`，并 drain / close `StatsQueue`；learner 非零退出码会被重新抛出为错误。共享内存段随进程退出由 `torch.multiprocessing` 回收。

### 6.5 warmup 对齐

learner 启动即 `publish_weights()`，让 collector 在正式采样前拿到初版策略。`learning_starts` 前 collector 用**随机动作**填充（与同步版 warmup 一致）、learner 不训练、不发布有效更新。resume 场景（`global_step > 0`）下 collector 直接用已加载策略采样，不走随机 warmup。

---

## 7. 配置（`FastSacCfg`）

同步与异步拓扑共用一个 `FastSacCfg`。`asynchronous` 选择执行拓扑；通用 Trainer 配置与异步专属配置分别由 `FastSacTrainerCfg` 和 `FastSacAsyncOptionsCfg` 表达，避免在根配置中混入只对异步拓扑有效的字段：

```python
@dataclass
class FastSacAsyncOptionsCfg:
    ring_capacity: int = 64  # SharedTransitionRing slot 数（每 collector 一条独立 ring）
    utd_mode: str = "strict"  # strict=精确比例；learner_bound=吞吐优先
    weight_publish_interval: int = 4  # learner 每 N 次更新发布一次权重（逐 collector 广播）
    weight_poll_interval: int = 1  # collector 每 N 个 env-step 检查一次新权重
    max_ingest_per_iter: int = 8  # learner 每轮对每条 ring 最多 drain 多少 slot
    idle_sleep_s: float = 0.0005  # 满环/欠数据时的退避睡眠
    collector_inference_device: str = "cuda"  # cpu / cuda / cuda:N；只控制 actor + policy normalizer
    collector_compile: bool = True  # CUDA 固定 batch 推理使用 reduce-overhead
    collector_amp: bool = True  # 默认使用实测吞吐最优的 FP16 collector autocast
    collector_amp_dtype: str = "fp16"  # fp16 / bf16
    transition_ipc: str = "auto"  # 设备 transition 环：auto/on/off（见 §4.1）
    weight_ipc: str = "auto"  # 权重快照 CUDA-IPC 传输：auto/on/off + 大小门控
    num_collectors: int = 1  # collector 进程数；num_envs 均分（须整除）
    cpus_per_collector: int | None = None  # 每 collector 从其（自动分配的）node 取的 CPU 数；None 用全部


@dataclass
class FastSacTrainerCfg:
    num_learning_iterations: int = 10000
    async_options: FastSacAsyncOptionsCfg = field(default_factory=FastSacAsyncOptionsCfg)


@dataclass
class FastSacCfg:
    asynchronous: bool = True
    trainer: FastSacTrainerCfg = field(default_factory=FastSacTrainerCfg)
```

基础配置默认使用 `strict` 保持同步/异步 UTD 可比；吞吐优先的任务可通过 `algo.trainer.async_options.utd_mode=learner_bound` 让两进程各自满速。较短的 `weight_publish_interval` 用于限制策略 staleness。

---

## 8. 多 collector 与 NUMA 绑定（多 NUMA node 服务器）

默认拓扑仍是 1 collector × 1 learner。配置 `num_collectors > 1` 后，`num_envs` 均分给 N 个 collector 进程（要求整除），每个 collector 绑定一个 NUMA node 满速采样。核心原则：**learner 的 GPU 永不因数据断粮空转，collector 全速自由跑，一切同步点允许松弛**。

**拓扑扩展**：

- **每 collector 一条独立 SPSC ring**（不做共享 MPSC 环）：ring 的 slot 形状为 `(capacity, num_envs/N, dim)`，所有无锁原语（单写方游标、seqlock）原样保留。满环背压按 ring 独立——快的 collector 阻塞在自己的满环上，不拖住别人。
- **每 collector 一份独立 `WeightSnapshot`**：learner 逐份非阻塞 `publish`，某份正在被读（seqlock odd）不影响其他份；staleness 以 `async/policy_lag`（聚合 max）与 `async/policy_lag_collector{i}`（多 collector 时分列）监控，不做版本对齐屏障。
- **learner 轮询多路 ring 并按「代」合并**：`drain()` 对每条 ring 至多消费 `max_ingest_per_iter` 个 slot；第 k 代（各 collector 的第 k 个 slot）拼成完整 `num_envs` 批（env 按 collector 连续分块映射，保证 n-step 的时间相邻性）。一个循环内所有完整批次先各自 `.to(device)`，再在 GPU 侧沿时间轴 stack，最后用 `SimpleReplayBuffer.extend_batch` 每字段一次连续列写入（H2D 次数与 GPU kernel 数不随批量增长——RTX 3090 + 双 CUDA collector 实测，逐批 `extend` 的阻塞式 CPU 源拷贝在多 CUDA context 争用下显著变慢）。未集齐的「代」以 CPU slot 视图挂起——慢 collector 自己的 ring 会先填满并背压它自己，不拖住别人；所有 ring 的读游标只在合并批到达 GPU 后推进，保持 ring 的 no-clobber 保证。`strict` 模式的 UTD 记账因此无需按 collector 数缩放（合并批就是完整 `num_envs` 批），任意 collector 数下长期 UTD 精确等于 `num_updates`；`learner_bound` 下 ring 全空时 learner 不等待，在已有 buffer 上继续 `num_updates`。
- **数据顺序无关**：off-policy + i.i.d. 采样，多 collector 的 transition 入 buffer 顺序无关，不引入全局序号。
- **seed 按 collector 划分**（`seed + collector_id`），保证可复现且样本不重复。
- **进度与日志**：`Control.collector_steps` 是 per-collector 计数器数组（单写方不变），聚合求和后除以 collector 数得到「完整 num_envs 批等价步」，作为训练进度 / 日志 / checkpoint / resume 的统一基准——任意 collector 数下 `num_learning_iterations`、`learning_starts`、TensorBoard x 轴与同步版语义一致；每 collector 有独立 `StatsQueue`（快照携带 `collector_id`）、每个 learner rank 每 log 窗口发一份轻量 payload，全部汇到**父进程**——worker 完全对等，父进程聚合（return/ep_len/timing 取均值、episodes 求和、policy_lag 取 max）后渲染面板并写 TensorBoard。面板/系统采样由未绑定的父进程执行，CPU 视图是全机（所有 core）而非某个 node；多 collector 时额外输出 `async/policy_lag_collector{i}`、`async/ring_fill_collector{i}`，单 collector 的标量键保持不变。
- **resume 语义**：多 collector 下 resume 与单 collector 相同——learner 从 checkpoint 恢复网络与优化器，所有 collector 重新 reset env 后从 checkpointed iteration 继续采样；ring / 在途 transition 不跨进程恢复。

**NUMA 绑定**（拓扑决策在 `async_impl/topology.py`，绑定原语在 `async_impl/numa.py`，等价 `numactl --cpunodebind= --membind=`，best-effort，**自动分配**）：

- **分配策略**（`resolve_worker_topology`）：learner 绑到其 GPU 的 PCIe 本地 node（经 NVML/pynvml 查 PCI bus id + sysfs `numa_node`，与 system_metrics 同款进程内会话，不创建 CUDA context）；每个 collector 跟随其归属 learner 的 node——collector/learner 对绝不跨 NUMA 边界，transition ring、pinned staging 与权重快照全部 node 本地。单 node 宿主机 / CPU learner / GPU 拓扑未知时不绑定（OS 默认放置）。实测依据：双路双卡机器上整树 node 绑定较不绑 +58%（first-touch 内存页跨 node 是主要开销），collector 与 learner 同 node 的局部性收益远大于带宽减半的损失。
- CPU affinity：`os.sched_setaffinity` 绑到 node 的 CPU 列表（sysfs `node{X}/cpulist`）；配置 `cpus_per_collector` 时进一步切成不重叠的连续分片，避免 collector 之间抢核。
- 内存策略：libnuma 的 `set_membind`（ctypes 加载，numactl 同款调用）把该进程**后续**分配绑到本地 node——因此绑定发生在 worker 进程的第一行（父进程在 spawn 前预绑，保证 import 期分配也 node 本地），先于 env / staging / pinned buffer 的任何分配。libnuma 不可用或 node 未知时告警并沿用 OS 默认放置，单 NUMA 机器与容器行为不变。

**collector 推理设备**：多 collector 改变 CPU/CUDA 推理的权衡（N 个 CUDA collector 与 learner 产生 N 倍 H2D/D2H burst 争用）。首版保持默认 `collector_inference_device="cuda"`（与单 collector 一致），以实测吞吐决定多 collector 场景默认值是否调整。

**明确不做**：共享 MPSC 环 / 原子游标、collector 间同步、per-step 权重同步、多机。

---

## 9. 单机多卡（多 learner × DDP 数据并行）

`num_learners > 1` 时每个 GPU 一个 learner 进程，`torch.distributed`（NCCL，file rendezvous 由 parent 注入）做梯度平均；collector 进一步按 `num_collectors % num_learners == 0` 均分给各 learner。所有共享内存原语零改动——每条 ring / 每条 weight channel / Control 仍是严格单写者单读者。

```text
parent (mp spawn, 创建全部 shm 原语)
│
├── learner rank 0 @ cuda:0 ───────── DDP/NCCL 梯度同步（每 rank 本地 batch）──────────┐
│    ├─ drain ring 0..k-1（本 rank 分片，按「代」合并成完整 num_envs 批）                 │
│    ├─ 本地 sharded replay buffer（env 分片 → 每 env 完整轨迹在本 rank，n-step 邻接保持） │
│    ├─ 唯一 weight 发布者：向【全部】collector publish                                  │
│    │   （CUDA-IPC 仅限与本 rank 同卡的 collector，其余走 host shm）                     │
│    └─ 唯一 checkpoint / TensorBoard / 面板 writer（drain 全部 StatsQueue 聚合）         │
│                                                                                        ▼
├── learner rank 1 @ cuda:1 ── drain ring k..2k-1 ── 本地 rb ── 梯度 allreduce ── 与 rank 0 同步
│
├── collector 0 (env CPU, 推理 cuda:0) ──ring 0 (SPSC)──► learner 0
├── collector 1 (env CPU, 推理 cuda:1) ──ring 1 (SPSC)──► learner 1
└── ...                                    ▲
                                           └── weight channel（rank 0 → 每个 collector 一份）

Control（全局共享标量，所有进程可见）：collector_steps[] 聚合 // num_collectors
  = 全局进度基准 → 迭代 / 日志 / checkpoint / update 次数全部由它推导（跨 rank 锁步）
```

**数据连接与 pipeline（一次循环）**：collector 本地采样（推理在归属 learner 的卡上）→ push 进自己的 SPSC ring，满环背压只作用于自己 → 归属 learner `drain`：把 k 条分片 ring 按代合并成完整 shard 批，一次性写入本地 replay buffer（按 shard env 数定尺寸）→ update：各 rank 以 `batch_size / num_learners` 采样，backward 后手动 all-reduce 梯度取均值（actor/qnet 每次反向后合并为一次扁平 all-reduce；标量 `log_alpha` 单独 all-reduce——不走 DDP 包装，因为更新边界调用的是 `get_actions_and_log_probs`/`projection`/`get_value` 等自定义方法而非 `module.forward`，DDP 既不代理也不同步它们）→ rank 0 按 `weight_publish_interval` 向全部 collector 发布权重 + normalizer 统计。

**关键规则**：

- **update 次数锁步**：梯度同步要求各 rank backward 次数严格一致，因此多 learner 下 update 次数不从本地 `ingested` 推导，而从全局进度基准（`Control.collector_steps` 聚合）的增量推导——所有 rank 看到同一个值，天然锁步；分片慢的 rank 只是让全体多等，不会死锁。
- **算法量全局、资源量本地**：`batch_size` 全局（内部按 rank 均分，DDP 平均后等价同步版全局 batch）；`num_envs` / `num_collectors` / NUMA 均为每节点量。本地 rb 容量 = 全局 `buffer_size`（总内存 × num_learners，是 env 分片保 n-step 的直接代价）。
- **配置只给数量**：`num_learners` / `learner_devices`（null → 复制 `device`）；NUMA node 全部自动分配（collector round-robin、learner GPU 本地），无手工列表。`world_size == 1` 时跳过 `init_process_group` 与 DDP 包装，单卡行为逐字节不变；多 learner 下暂禁 learner 侧 `torch.compile`（reduce-overhead CUDA graph 与 DDP hook 的组合首版不碰）。

**已知取舍**：obs-normalizer 各 rank 只见本地分片，running stats 有轻微偏差，首版直接发布 rank 0 版本（严格一致可后续加发布前 all-reduce count/mean/var，量极小）；learner 侧 perf 指标只反映 rank 0。多机（torchrun / 网络 rendezvous / 跨机传输）本期不做——逻辑拓扑与部署机制分离，rendezvous 换来源即可平移。

---

## 10. 不变量与关键取舍

- **算法不变**：transition 构造、`rb.extend` 调用与 `sample` 语义、更新数学与同步版逐字节一致；异构只改「谁在哪个进程执行」。
- **normalizer 单写方**：只有 learner 以 `update=True` 更新 normalizer；collector 只读快照。权重通道因此是单向的。
- **单生产者单消费者**：环与权重快照都建立在「每个共享量只有一个写方」之上，这是无锁 / 无 CAS 的前提；多 collector 通过「每 collector 一条独立 ring + 一份独立 WeightSnapshot」保持该前提，而不是引入 MPSC / 原子游标。游标与数据之间的顺序则依赖 x86/TSO（不插屏障，故当前 **x86-only**，弱内存序 ISA 暂不支持）。当前支持 1..N collector × 1..M learner（单机）、不做多机分布式。
- **背压优先于放开比例**：collector 快时阻塞采样而非无界缓冲，用 `ring_capacity` 吸收抖动，避免 off-policy 失真。
- **checkpoint 与同步版字节兼容**：保证 play / resume 互通与 A/B 对比有效。
- **非确定性**：两进程相对速度随机，逐步复现不可能；正确性以「固定 seed 下 `strict` 模式收敛曲线落在同步版 run-to-run 方差带内」在统计层面成立。seed 同时播撒 collector（env + 采样噪声）与 learner（网络初始化 + 采样噪声）。
- **torch.compile 与多进程**：learner 沿用同步版的 in-process 编译约定；CUDA collector 默认启用 `mode="reduce-overhead"`，在首次 env step 前按固定 `num_envs` shape warmup；连续 stochastic 调用必须保留独立随机 sample。显式 CPU collector 不执行 compile。
- **精度边界**：collector 默认使用服务器实测吞吐最优的 FP16 compiled CUDA 路径；FP32 仍是 correctness baseline，可通过 `collector_amp=false` 显式选择。低精度的完整训练质量仍需与 FP32 单独比较，不能由吞吐结果代替。

---

## 11. 一句话总结

FastSAC 的 off-policy 属性 + normalizer 为 learner 独占，使「collector/learner 分进程 + 共享内存」在不改算法、不改同步版的前提下成立；唯一的 `motrix.fastsac` provider 通过 `asynchronous` 字段选择 Trainer，并共用 env/config/checkpoint。默认异步执行；基础配置用 `strict` 验证算法等价，吞吐优先的任务用 `learner_bound` 让采样与训练各自满速。三个必须做对的点是：**SPSC 有界背压环**（防内存失控 / off-policy 失真）、**UTD 比例治理**（吞吐模式下监控并标注实际 UTD）、**seqlock 双缓冲权重快照**（无锁读、杜绝撕裂、靠短发布间隔压低 staleness）。
