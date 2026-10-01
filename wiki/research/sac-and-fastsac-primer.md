# SAC 与 FastSAC 算法原理入门

## 摘要

本文面向强化学习初学者，解释 Soft Actor-Critic（SAC）的训练目标、critic target、actor loss、温度系数、经验回放与 n-step return，并把这些概念对应到当前仓库的 FastSAC 实现。本文属于研究说明，不定义稳定接口；当前代码实现入口主要位于 `motrix_rl/src/motrix_rl/fastsac/`。

源码入口：

- [agent.py](../../motrix_rl/src/motrix_rl/fastsac/agent.py)：训练循环、critic update、actor update、alpha update。
- [buffer.py](../../motrix_rl/src/motrix_rl/fastsac/buffer.py)：经验回放与 n-step reward 计算。
- [networks.py](../../motrix_rl/src/motrix_rl/fastsac/networks.py)：actor、distributional critic 与 Bellman projection。
- [config.py](../../motrix_rl/src/motrix_rl/fastsac/config.py)：默认超参数。

## 先建立几个概念

强化学习训练可以理解成：智能体在状态 `s` 下选择动作 `a`，环境返回奖励 `r` 和下一个状态 `s'`。训练的目标不是只让当前一步奖励最大，而是让从当前时刻开始的长期累计奖励最大。

普通 return 通常写作：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

其中 $\gamma$ 是折扣因子。$\gamma$ 越接近 1，算法越重视长期收益；$\gamma$ 越接近 0，算法越短视。

SAC 在普通 return 上增加了一个熵项：

$$
G_t^{\mathrm{soft}}
= \sum_{k=0}^{\infty} \gamma^k
\left(
    r_{t+k} + \alpha \mathcal{H}\left(\pi(\cdot \mid s_{t+k})\right)
\right)
$$

这里：

- $\pi(a \mid s)$ 是策略，也就是 actor 给出的动作分布。
- $\mathcal{H}(\pi(\cdot \mid s))$ 是策略熵，表示策略有多随机。
- $\alpha$ 是温度系数，控制“更看重奖励”还是“更鼓励探索”。

直观理解：SAC 不只奖励“拿高分”，也奖励“不要过早变得过于确定”。这对连续控制任务很重要，因为早期如果策略太快收敛到某个动作附近，后续可能很难探索到更好的动作。

## SAC 由哪些部分组成

SAC 通常包含三类模型：

1. actor，也就是策略网络，输入状态，输出动作分布。
2. critic，也就是 Q 网络，估计 `Q(s, a)`。
3. target critic，用较慢变化的参数产生稳定的训练目标。

当前 FastSAC 实现中，actor 和 critic 定义在 `motrix_rl/src/motrix_rl/fastsac/networks.py`。

actor 是 tanh-squashed Gaussian policy。它先输出高斯分布参数 `mean` 和 `log_std`，再采样 raw action，最后经过 `tanh` 映射到动作范围：

```python
_, mean, log_std = self(obs)
std = log_std.exp()
dist = torch.distributions.Normal(mean, std)
raw_action = dist.rsample()
tanh_action = torch.tanh(raw_action)
action = tanh_action * self.action_scale + self.action_bias
```

代码位置：`Actor.get_actions_and_log_probs()`。

在继续解释 `tanh` 前，先说明这里提到的 actor loss 是什么。训练 actor 时，我们不是直接告诉网络“正确动作是多少”，而是让 actor 采样一个动作，再交给 critic 评价这个动作的价值。actor loss 就是用来更新 actor 参数的损失函数：它希望 actor 更常采样 critic 认为价值高的动作，同时保留一定随机性用于探索。

SAC 中 actor 希望最大化：

$$
Q(s, a) - \alpha \log \pi(a \mid s)
$$

训练代码通常最小化 loss，所以会写成相反方向：

$$
\alpha \log \pi(a \mid s) - Q(s, a)
$$

这个 loss 不是随意拼出来的，它来自 SAC 的 soft policy improvement。语义上，actor 在每个状态下都在做一个权衡：一方面希望动作的 Q 值高，另一方面不希望策略过早变成只输出一个确定动作。

其中 $-Q(s, a)$ 是“利用”项。最小化 loss 时，$Q(s, a)$ 越大，loss 越小，所以 actor 会被推向 critic 认为长期收益更高的动作。

$\alpha \log \pi(a \mid s)$ 是“熵”或“探索”项。因为概率密度越集中，策略熵越低；保留这个项等价于惩罚过于确定的策略。$\alpha$ 控制这个惩罚的强度：$\alpha$ 大时，actor 更愿意保持随机性；$\alpha$ 小时，actor 更接近贪心地追求高 Q 动作。

从分布匹配角度看，固定当前 critic 后，SAC 希望 actor 接近下面这个“软最优”动作分布：

$$
\pi^\star(a \mid s)
\propto
\exp\left(\frac{Q(s, a)}{\alpha}\right)
$$

也就是说，Q 值高的动作应该有更大概率，但 Q 值低的动作不一定被完全排除。actor loss 可以理解成让当前策略 $\pi(\cdot \mid s)$ 靠近这个按 Q 值加权的 softmax 分布：

$$
J_\pi(s)
=
\mathbb{E}_{a \sim \pi(\cdot \mid s)}
\left[
    \alpha \log \pi(a \mid s) - Q(s, a)
\right]
$$

因此它的直观语义是：让 actor 学会“更常选择高价值动作，同时保持可控随机性”。

本文后面的“Actor loss 如何对应 SAC 目标”小节会结合当前实现展开这条公式。这里先记住一点：actor loss 是 actor 的训练信号，它依赖 actor 采样出的动作、动作的概率，以及 critic 对该动作的 Q 值评价。

这里的 `tanh` 有两个作用。第一个作用是把无界的高斯采样动作压到有限范围内。高斯分布采样得到的 `raw_action` 理论上可以是任意实数，但机器人控制里的动作通常有上下界，例如关节目标、速度命令或力矩命令不能无限大。`torch.tanh(raw_action)` 会把动作压到 $(-1, 1)$，再通过 `action_scale` 和 `action_bias` 映射到环境真实动作空间：

$$
u = \tanh(x)
$$

$$
a = u \cdot \mathrm{action\_scale} + \mathrm{action\_bias}
$$

其中 $x$ 是 `raw_action`，$u$ 是 squashed action，$a$ 是最终送给环境的动作。

第二个作用是保持 actor 可训练。代码使用 `dist.rsample()`，也就是 reparameterization trick。可以把采样理解成：

$$
x = \mu(s) + \sigma(s)\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

然后再做：

$$
a = \tanh(x) \cdot \mathrm{action\_scale} + \mathrm{action\_bias}
$$

这样 actor loss 对 $\mu(s)$ 和 $\sigma(s)$ 仍然可求导，actor 可以通过 critic 给出的梯度学习“哪些动作更好”。

需要注意，经过 `tanh` 后，动作分布已经不再是原始高斯分布。SAC 的 actor loss 和 critic target 都需要 $\log \pi(a \mid s)$，所以代码必须对 `log_prob` 做变量变换修正：

```python
log_prob = dist.log_prob(raw_action)
log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
log_prob -= torch.log(self.action_scale + 1e-6)
```

其中 `torch.log(1 - tanh_action.pow(2) + 1e-6)` 来自 `tanh` 的导数：

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)
$$

如果不做这一步修正，算法会把 squashed action 的概率算错，进而影响熵项 $\alpha \log \pi(a \mid s)$，actor 更新和温度系数更新都会偏离真实目标。

critic 是 distributional Q network，不是直接输出一个标量 Q 值，而是输出 Q 值在一组离散支撑点 `q_support` 上的分布。当前实现默认使用多个 Q 网络：

```python
return torch.stack([qnet(obs, actions) for qnet in self.qnets], dim=0)
```

代码位置：`Critic.forward()`。

## SAC 的 critic target

普通 Q-learning 的 Bellman target 可以写成：

$$
y = r + \gamma Q_{\mathrm{target}}(s', a')
$$

SAC 的 soft Bellman target 多了熵项：

$$
y_{\mathrm{soft}}
= r + \gamma
\left(
    Q_{\mathrm{target}}(s', a') - \alpha \log \pi(a' \mid s')
\right)
$$

这个式子里的 $-\log \pi(a' \mid s')$ 就是熵的采样形式。因为熵可以写作：

$$
\mathcal{H}\left(\pi(\cdot \mid s)\right)
= \mathbb{E}_{a \sim \pi(\cdot \mid s)}
\left[-\log \pi(a \mid s)\right]
$$

所以在采样动作 $a'$ 后，熵奖励就体现在 $-\alpha \log \pi(a' \mid s')$ 上。

当前 FastSAC 的 target 构造在 `FastSacAgent._update_main()` 中：

```python
next_actions, next_logp = self.actor.get_actions_and_log_probs(b["next_obs"])
discount = cfg.gamma ** b["effective_n_steps"]
target_distributions = self.qnet_target.projection(
    b["next_critic_obs"],
    next_actions,
    rewards - discount * bootstrap * self.log_alpha.exp() * next_logp,
    bootstrap,
    discount,
)
```

这段代码可以拆开看：

- `next_actions` 是从当前 actor 在下一状态 `s'` 采样出来的动作。
- `next_logp` 是 $\log \pi(a' \mid s')$。
- `self.log_alpha.exp()` 是 `alpha`。
- `discount` 是 $\gamma^n$，因为实现支持 n-step return。
- `bootstrap` 控制是否允许从下一状态继续 bootstrap。

传给 projection 的 reward 参数是：

$$
r^{(n)} - \gamma^n b \alpha \log \pi(a' \mid s')
$$

其中 $r^{(n)}$ 对应代码中的 `rewards`，$b$ 对应代码中的 `bootstrap`。

随后 distributional critic 的 projection 里再加上未来 Q 支撑：

```python
target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
```

合起来就是：

$$
z_{\mathrm{target}}
= r^{(n)}
- \gamma^n b \alpha \log \pi(a' \mid s')
+ \gamma^n b z
$$

这里 $z$ 是 `q_support` 上的一个价值支撑点。

也就是 SAC soft target 的 distributional 版本：

$$
y_{\mathrm{soft}}^{(n)}
= r^{(n)}
+ \gamma^n b
\left(
    Q_{\mathrm{target}}(s', a') - \alpha \log \pi(a' \mid s')
\right)
$$

## FastSAC 中的 return 在哪里算

当前 FastSAC 有两个容易混淆的 return。

第一个是日志用的 episode return。它只用于展示训练过程，不直接参与 loss：

```python
ep_return = torch.zeros(self.num_envs, device=device)
...
ep_return += rewards
...
recent_returns.append(float(ep_return[j]))
```

代码位置：`FastSacAgent.learn()`。这个 return 是一个 episode 内环境 reward 的简单累计，没有乘 `gamma`，用于打印 `rollout/mean_return`。

第二个是训练 critic 用的 n-step reward。它在 replay buffer 采样时计算：

```python
discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device))
n_step_rew = (all_rew * done_mask * discounts.view(1, 1, -1)).sum(dim=2)
```

代码位置：`SimpleReplayBuffer.sample()`。如果配置里的 `num_steps == 1`，则直接使用单步 reward：

```python
"rewards": torch.gather(self.rewards, 1, idx).reshape(flat)
```

当前默认配置为：

```python
gamma: float = 0.97
num_steps: int = 1
```

所以默认训练时 critic target 的 reward 部分就是单步 reward；如果之后把 `num_steps` 调大，就会变成折扣 n-step reward。

## Actor loss 如何对应 SAC 目标

critic 学的是“某个状态动作对有多好”。actor 则要学会选择更好的动作。SAC 的 actor 优化目标可以写成：

$$
\max_{\pi}\;
\mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi(\cdot \mid s)}
\left[
    Q(s, a) - \alpha \log \pi(a \mid s)
\right]
$$

因为训练框架通常最小化 loss，所以代码写成：

$$
J_{\pi}
= \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi(\cdot \mid s)}
\left[
    \alpha \log \pi(a \mid s) - Q(s, a)
\right]
$$

当前实现对应 `FastSacAgent._update_pol()`：

```python
actions, log_probs = self.actor.get_actions_and_log_probs(b["obs"])
q_outputs = self.qnet(b["critic_obs"], actions)
q_values = self.qnet.get_value(F.softmax(q_outputs, dim=-1))
qf_value = q_values.mean(dim=0)
actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()
```

这段代码说明：

- actor 根据当前观察 `b["obs"]` 采样动作。
- critic 估计这些动作的 Q 值。
- actor loss 鼓励 `qf_value` 变大。
- actor loss 同时包含 `alpha * log_probs`，也就是 $\alpha \log \pi(a \mid s)$，对应 SAC 的熵正则。

需要注意：`log_probs` 通常是负数。最小化 $\alpha \log \pi(a \mid s) - Q(s, a)$ 会鼓励高 Q 动作，同时通过熵项避免策略过早塌缩。

## Alpha 温度系数

$\alpha$ 控制探索强度。$\alpha$ 越大，actor 越在意熵；$\alpha$ 越小，actor 越在意 Q 值。

当前实现将 `alpha` 存为 `log_alpha`：

```python
self.log_alpha = torch.tensor([math.log(cfg.alpha_init)], requires_grad=True, device=device)
```

这样做的好处是：训练的是无约束的 `log_alpha`，真正使用时取 `exp()`，保证 `alpha` 永远为正数。

自动调节在 `_update_main()` 中：

```python
alpha_loss = (-self.log_alpha.exp() * (next_logp.detach() + self.target_entropy)).mean()
```

目标是让当前策略熵接近 `target_entropy`。如果策略太确定，熵偏低，`alpha` 会被调大；如果策略太随机，`alpha` 会被调小。

## 为什么当前实现叫 FastSAC

当前 `motrix_rl_builtin.fastsac` 不是最朴素的 SAC，而是一个偏工程化的高吞吐实现。它在 SAC 基础上加入了几个特征。

第一，使用 replay buffer。采样训练批次来自 `SimpleReplayBuffer.sample()`，所以 FastSAC 是 off-policy 算法。环境交互得到的数据可以被多次复用，样本效率比纯 on-policy 方法更高。

第二，支持 n-step return。虽然默认 `num_steps = 1`，但 buffer 已经支持把连续多步 reward 折扣求和，再用第 n 步后的状态 bootstrap。

第三，使用 distributional critic。critic 输出的是 Q 分布，`projection()` 把 soft Bellman target 投影回固定的 Q 支撑点。相比直接回归标量 Q，这类 C51 风格的 critic 可以保留价值分布信息。

第四，支持 asymmetric observation。actor 使用 policy observation，critic 可以使用 privileged critic observation。相关字段在 replay buffer 中分别存储为：

```python
self.observations
self.critic_observations
self.next_observations
self.next_critic_observations
```

第五，面向大规模并行环境。FastSAC 的训练循环一次处理 `num_envs` 个环境，把数据按 per-environment circular buffer 存储，并用较大的 batch 做多次更新。

第六，包含性能优化选项。配置中有 `compile`、`amp`、`amp_dtype`，CUDA 上可启用 `torch.compile` 和自动混合精度。

## 一次训练迭代的代码流程

当前训练主循环在 `FastSacAgent.learn()` 中。一个迭代可以按下面顺序理解：

```text
1. 根据当前 obs 选择 actions
2. env.step(actions) 得到 next_obs, rewards, terminated, truncated
3. 把 transition 写入 replay buffer
4. 更新日志用的 episode return
5. 如果 warmup 结束，从 replay buffer 采样 batch
6. 更新 critic
7. 按 policy_frequency 更新 actor
8. soft update target critic
```

对应代码片段：

```python
actions = self.act(obs, deterministic=False)
next_obs, next_critic_obs, rewards, terminated, truncated = env.step(actions)

self.rb.extend(obs, critic_obs, actions, rewards, terminated.long(), truncated.long(), next_obs, next_critic_obs)

if not warming:
    metrics = self._train_step()
```

`_train_step()` 中会重复执行若干次 gradient update：

```python
for i in range(cfg.num_updates):
    b = self.rb.sample(batch_per_env)
    qf_loss, alpha_loss, qf_max, qf_min = self._update_main(b)
    if i % cfg.policy_frequency == 0:
        actor_loss, entropy = self._update_pol(b)
    self._soft_update()
```

## Target network 与 soft update

critic target 不直接使用当前 `qnet`，而是使用较慢更新的 `qnet_target`。这样可以减少训练目标快速变化带来的不稳定。

当前实现的 soft update：

```python
tau = self.cfg.tau
torch._foreach_mul_(tgt, 1.0 - tau)
torch._foreach_add_(tgt, src, alpha=tau)
```

数学上就是：

$$
\theta_{\mathrm{target}}
\leftarrow
(1 - \tau)\theta_{\mathrm{target}} + \tau \theta
$$

当前默认 `tau = 0.125`。

## done、truncated 与 bootstrap

训练 target 时，是否允许继续加未来价值由 `bootstrap` 控制：

```python
dones = b["dones"].bool()
truncations = b["truncations"].bool()
bootstrap = (truncations | ~dones).float()
```

含义是：

- 真正 terminated 的 episode 不继续 bootstrap。
- 因时间限制等原因 truncated 的 episode 可以 bootstrap。
- 没结束的 transition 也可以 bootstrap。

这会影响 target：

$$
y = r^{(n)} + b \gamma^n V_{\mathrm{future}}
$$

如果 `bootstrap = 0`，target 就只剩已经拿到的 reward。

## 初学者如何读这份实现

建议按下面顺序读代码：

1. `config.py`：先看默认超参数，例如 `gamma`、`tau`、`alpha_init`、`num_steps`、`batch_size`、`num_updates`。
2. `networks.py`：看 actor 如何采样动作和计算 log probability，再看 critic 如何输出 Q 分布。
3. `buffer.py`：看 transition 如何存储，n-step reward 如何计算。
4. `agent.py`：看 `_update_main()` 的 critic target、`_update_pol()` 的 actor loss、`learn()` 的训练循环。
5. `train.py`：看 trainer 如何创建环境、agent、checkpoint 和日志。

如果只想抓住核心，可以先记住三条主线：

- critic 学习：$r^{(n)} + \gamma^n (Q_{\mathrm{target}} - \alpha \log \pi)$。
- actor 学习：最大化 $Q - \alpha \log \pi$。
- alpha 学习：让策略熵接近目标熵。

## 与架构设计文档的关系

本页解释当前 FastSAC 代码和 SAC 理论。多算法接入、训练入口和 checkpoint 布局等工程架构，见 [RL 多算法架构设计](../design/rl-multi-algorithm-architecture.md)。如果后续要把这里的研究说明沉淀成稳定训练接口或新算法接入方案，应迁移或同步到 `design/`。
