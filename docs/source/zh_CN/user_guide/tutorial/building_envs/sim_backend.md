# SimBackend：与仿真器解耦

`DirectEnv` 与 `ManagerEnv` 的所有仿真交互都通过 backend 中立的 `SimBackend` 接口完成
（即环境持有的 `self.sim`）。具体仿真器类型不会跨越 core 边界——同一份环境代码可以运行在
任何实现了该接口的后端上。

## 编译三类程序

环境在构造阶段对 `self.sim` 编译三类程序，之后每步只做轻量的 buffer 写入和 `execute()`：

- `compile_model(queries)`：模型查询，即不随仿真变化的结构信息（执行器列表、关节限位等）。
- `compile_reads(queries)`：仿真数据读取程序（关节位置、速度、link 位姿等），
  每次 `execute(env_ids)` 批量刷新，读取结果按声明时的 key 索引。
- `compile_writes({name: Write})`：写入程序（由 `write_compiler` 编译器承载）。控制目标用 `CtrlTargetsWrite`；
  重置写入（初始位姿等）传入 `reset=True`。写入时先填 `buffer(name)`，再 `execute(env_ids)`。

两类工作流的分工：`DirectEnv` 在自己的构造函数中编译这些程序
（见[编写 DirectEnv 环境](direct_env.md)的最小示例）；`ManagerEnv` 则由配置的
`queries` 组与各项自动生成，用户通常无需直接操作
（见[编写 ManagerEnv 环境](manager_env.md)的查询一节）。

## backend 的选择与注册

- backend 是构造级别的字符串选择：创建环境时 `backend=None` 使用注册的默认后端
  （当前为 `motrixsim`），也可显式传入后端名，或通过
  `registry.make(..., sim="motrixsim")` 指定。
- backend 通过 `motrix_env.sim_backends` entry-point 组懒发现：
  第三方仿真器实现一个 `SimBackend` 工厂并注册到该组即可接入，导入开销为零。
- `motrix_env_mujoco` 仅用于 MuJoCo scene 编译，不作为训练后端。
