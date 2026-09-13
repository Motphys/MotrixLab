# 编写 DirectEnv 环境

`DirectEnv` 是 MotrixLab 的直接工作流前端：环境实现直接持有仿真后端接口 `SimBackend`（通过 `self.sim` 访问），
在固定的生命周期钩子中实现动作施加、奖励、终止与观测。它适合任务逻辑高度定制、
不需要 reward/observation/termination 逐项声明式组合的环境。

本页先给出一个完整的最小示例建立整体印象，再逐层展开背后的概念；何时选择 DirectEnv
而不是 Manager 工作流，见[环境构建总览](index.md)。
## 最小示例

以内置 `cartpole` 环境为例（完整源码见 `motrix_envs/src/motrix_envs/basic/cartpole/`）。
一个 DirectEnv 环境由**配置类**和**环境类**两部分组成，分别注册后即可训练和预览：

```python
import os

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnv, DirectEnvCfg
from motrix_env_core.sim import DofPositionQuery, DofVelocityQuery
from motrix_env_core.sim.write import CtrlTargetsWrite, JointPositionWrite, JointVelocityWrite

model_file = os.path.dirname(__file__) + "/cartpole.xml"

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
}


@registry.envcfg("cartpole")
@configclass
class CartPoleEnvCfg(DirectEnvCfg):
    """Move a cart to keep an inverted pendulum upright.

    zh_CN: 移动小车以保持倒立摆直立。
    """

    scene: SceneCfg = SceneCfg(file=model_file)
    max_episode_seconds: float = 10
    reset_noise_scale: float = 0.01


@registry.env("cartpole")
class CartPoleEnv(DirectEnv):
    def __init__(self, cfg: CartPoleEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.compile_writes({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.compile_writes(
            {
                "position": JointPositionWrite(("slider", "hinge")),
                "velocity": JointVelocityWrite(("slider", "hinge")),
            },
            reset=True,
        )
        self._action_space = gym.spaces.Box(-3.0, 3.0, (1,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, env_ids: np.ndarray):
        rows = len(env_ids)
        scale = self._cfg.reset_noise_scale
        self._reset_program.buffer("position")[env_ids] = np.random.uniform(
            -scale, scale, (rows, 2)
        ).astype(np.float32)
        self._reset_program.buffer("velocity")[env_ids] = np.random.uniform(
            -scale, scale, (rows, 2)
        ).astype(np.float32)
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
        return {}

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState):
        self._ctrl_writes.buffer("ctrl")[:] = actions.astype(np.float32, copy=False)
        self._ctrl_writes.execute()
        return state

    def compute_transition(self, state: ArrayEnvState):
        self.sim_data.execute()
        dof_pos = self.sim_data["dof_pos"]
        cart_pos, angle = dof_pos[:, 0], dof_pos[:, 1]
        state.reward = np.ones((self.num_envs,), dtype=np.float32)
        state.terminated = (
            np.isnan(angle)
            | (np.abs(angle) > 0.2)
            | (cart_pos < -0.8)
            | (cart_pos > 0.8)
        )
        return state

    def compute_observation(self, state: ArrayEnvState):
        obs = np.concatenate([self.sim_data["dof_pos"], self.sim_data["dof_vel"]], axis=-1)
        return state.replace(obs=obs)
```

注册完成后，`python scripts/view.py env=cartpole` 可以预览，按
[Task 配置](../training/task_config.md)创建训练 Task 即可开始训练。

从示例可以读出 DirectEnv 的三个核心要素：

1. **配置**继承 `DirectEnvCfg`，声明场景与任务参数；
2. **环境类**在构造函数中对 `self.sim` 编译查询与读写程序——这层边界见
   [SimBackend：与仿真器解耦](sim_backend.md)；
3. **任务逻辑**写在 `reset` / `apply_action` / `compute_transition` / `compute_observation`
   四个生命周期钩子里。

下一节先用生命周期图建立整体认识，再逐节展开配置与 SimBackend 程序的细节。

## 生命周期

`DirectEnv` 继承 `ArrayEnv`，step / auto-reset / truncation 生命周期由基类管理，
子类不要重复实现。`step(actions)` 的固定时序为，橙色标注的四个阶段
就是 Direct 工作流需要填充的位置：

```{image} /_static/images/tutorial/direct-env-lifecycle-light.svg
:alt: DirectEnv 单个控制步流水线：apply_action（hook）→ physics_step → compute_transition（hook）→ 截断判定 → auto-reset（reset(env_ids)）→ compute_observation（hook）
:class: only-light
```

```{image} /_static/images/tutorial/direct-env-lifecycle-dark.svg
:alt: DirectEnv 单个控制步流水线：apply_action（hook）→ physics_step → compute_transition（hook）→ 截断判定 → auto-reset（reset(env_ids)）→ compute_observation（hook）
:class: only-dark
```

子类需要实现的钩子：

| 钩子                                 | 职责                                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------------------------- |
| `reset(env_ids)`                     | 为选中的环境写入重置状态（随机化初始姿态等），返回 info 字典；观测交给后续的 `compute_observation` |
| `apply_action(actions, state)`       | 将动作写入仿真（通常是 ctrl 目标）                                                       |
| `compute_transition(state)`          | 执行读取程序并派生 `state.reward`、`state.terminated` 等；这是每步唯一的全量数据刷新点，**不得**写 `state.obs` |
| `compute_observation(state)`         | 纯粹用已刷新的仿真数据拼装 `state.obs`，不再执行读取                                     |
| `observation_space` / `action_space` | 作为 property 定义，通常在 `__init__` 中预构造                                          |

语义约定：

- `terminated` 表示任务失败等回合终止条件；`truncated` 表示达到 `max_episode_steps`
  的时间截断。两者由 `ArrayEnv` 合成 `done` 并触发 auto-reset；
  `info["time_outs"]` 标记"截断但未失败"的行。
- 环境维度必须使用 NumPy 向量化操作；只有遍历固定数量的关节、脚或 term 时才允许普通循环。
- 常量（初始姿态、空间定义、query 名称）在 `__init__` 或配置构造阶段预计算，
  不在 step 循环中重复创建。


## 配置

`DirectEnvCfg` 继承 `EnvCfg`，配置写成 `class MyEnvCfg(..., DirectEnvCfg)`。
`EnvCfg` 提供以下公共字段：

| 字段                  | 含义                                                                 |
| --------------------- | -------------------------------------------------------------------- |
| `scene`               | `SceneCfg`，场景与模型来源；必须配置，否则 `validate()` 失败          |
| `sim`                 | `SimCfg`，仿真参数（`dt`、`solver_iterations` 等）                   |
| `ctrl_dt`             | 控制步长（秒），默认 `0.01`                                          |
| `max_episode_seconds` | 回合最长时长（秒），换算为 `max_episode_steps = max_episode_seconds / ctrl_dt`；`None` 表示不限制 |
| `render_spacing`      | 多环境网格渲染时的间距                                               |

每个控制步内仿真推进 `sim_substeps = round(ctrl_dt / sim.dt)` 个物理子步，
因此 `sim.dt` 必须小于等于 `ctrl_dt`。场景文件与仿真参数的详细说明见
[搭建物理场景](scene.md)。



## 注册约定

- **配置先于环境类注册**：`@registry.env("name")` 要求同名配置已通过
  `@registry.envcfg("name")` 注册，否则报错。
- 配置注册对象可以是配置类，也可以是带返回类型标注的零参数 factory
  （适合返回经过定制的配置实例）。
- registry 根据环境类的继承关系推断前端类型（`DirectEnv` 属于 `"np"` 数据后端），
  不要手动指定。
- 配置 docstring 的首行与 `zh_CN:` 行分别作为环境的英文、中文描述，
  展示在环境概览中。
