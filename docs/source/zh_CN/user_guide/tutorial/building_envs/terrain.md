# 程序化地形生成

MotrixLab 支持在场景配置中直接声明程序化高度场地形：不依赖外部高度图文件，
在模型 build 阶段由**地形生成器**（terrain generator）确定性地产出高度数据。粗糙地面、台阶、
金字塔坡、离散障碍物等地形只需要几行配置，通过 `seed` 即可精确复现。

## 最小示例

在 `SceneCfg.assets` 中声明一个程序化高度场资产（`ProceduralHFieldAssetCfg`），
在 `SceneCfg.objs` 中用 `HFieldTerrainCfg` 把它挂为地面：

```python
from motrix_env_core.base import EnvCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    HFieldTerrainCfg,
    MaterialCfg,
    NoiseTerrainGeneratorCfg,
    ProceduralHFieldAssetCfg,
    SceneAssetsCfg,
    SceneCfg,
    SceneObjsCfg,
)


@configclass
class TerrainAssetsCfg(SceneAssetsCfg):
    mat_ground: MaterialCfg = MaterialCfg()
    terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
        generator=NoiseTerrainGeneratorCfg(seed=0, height_scale=0.1),
        size=(64.0, 64.0),   # 世界空间 X/Y 全宽，单位 m
        shape=(320, 320),     # 高度场分辨率（行、列）
    )


@configclass
class TerrainObjsCfg(SceneObjsCfg):
    floor: HFieldTerrainCfg = HFieldTerrainCfg(hfield="terrain", material="mat_ground")


@configclass
class MyTaskEnvCfg(EnvCfg):
    scene: SceneCfg = SceneCfg(assets=TerrainAssetsCfg(), objs=TerrainObjsCfg())
```

内置环境中，`go1-walk-rough`、`go2-walk-rough`、`anymalc-walk-rough` 等粗糙地形任务
都通过同一个 `NoiseTerrainGeneratorCfg(seed=0, height_scale=0.1, flip_y=True)` 生成地形，
可以直接预览效果：

```bash
python scripts/view.py env=go1-walk-rough
```

## 生成器契约

所有生成器继承 `TerrainGeneratorCfg`，遵循同一条契约：

- `generate(size, shape)` 返回 **归一化到 [0, 1]** 的高度数组；物理高度 = 归一化值 ×
  `height_scale`（米）。例如 `height_scale=0.1` 时，整张地形的高差最大为 0.1 m。
- 数组为 MuJoCo 行主序：第一维（行，`shape[0]`）对应高度场 x 轴，第二维（列，`shape[1]`）
  对应 y 轴。
- `seed` 驱动所有随机性，同一种子总是产出同一张地形。
- 地形在**模型 build 时一次性生成**，训练过程中不再变化；编译器会校验返回数组的形状与数值有限性。
- 各生成器的 `validate()` 会拒绝超出 `height_scale` 的配置（例如台阶总爬升超过
  `height_scale`），错误信息给出具体上限。

## 内置生成器

每个生成器都是 `@configclass`，字段即全部可调参数。以下截图均由
`examples/terrain_generate.py` 的展示配置渲染。

### FlatTerrainGeneratorCfg —— 平面

常量归一化高度 `height`（默认 0.0），多用作组合地形的基底或重置平台。

### NoiseTerrainGeneratorCfg —— 噪声

逐单元均匀采样 `[0, 1)` 的随机高度。`downsampled_scale`（米）设置后，先在以该间距
采样的粗网格上生成噪声，再双线性插值到目标分辨率，得到起伏平缓的丘陵噪声；
不设置则是逐单元的粗糙颗粒噪声。

```{figure} /_static/images/tutorial/terrain/noise_rough.jpg
:alt: 逐单元噪声地形截图：粗糙颗粒状起伏

逐单元噪声（`height_scale=0.08`），颗粒状起伏。
```

```{figure} /_static/images/tutorial/terrain/noise_smooth.jpg
:alt: 下采样噪声地形截图：平缓丘陵起伏

`downsampled_scale=0.4` 的同一颗噪声，双线性插值后成为平缓丘陵。
```

### QuantizedTerrainGeneratorCfg —— 台地化

把 `source` 生成器的归一化输出量化为 `levels` 个离散高度级（含 0 和 1），形成台地。
被包裹生成器自己的 `height_scale` 被忽略，物理尺度由本配置的 `height_scale` 决定。

```{figure} /_static/images/tutorial/terrain/noise_terraced.jpg
:alt: 噪声量化台地地形截图：离散阶梯状平台

噪声量化为 `levels=5` 的台地。
```

### StairsTerrainGeneratorCfg —— 台阶

沿一个轴或从中心径向铺规则台阶，主要字段：

| 字段             | 含义                                                                                      |
| ---------------- | ----------------------------------------------------------------------------------------- |
| `axis`           | `"x"` / `"y"` 沿轴线性；`"radial"` 围绕场地中心铺同心方环                                  |
| `profile`        | `ascending`、`descending`、`pyramid`、`inverted_pyramid` 四种台阶走向                       |
| `step_count`     | 台阶数；`step_width` 未设置时台阶均分整个轴向跨度                                            |
| `step_height`    | 单级台阶的物理高度（m）                                                                     |
| `step_width`     | 可选的固定踏面宽度（m）；设置后 `step_count` 退化为最大级数上限                              |
| `platform_width` | 中心平台宽度（m），保持在轮廓的自然中心高度；0 关闭                                           |
| `base_level`     | 归一化基础抬升，供坑洞留出向下空间（高度场高度不能为负）                                       |

```{figure} /_static/images/tutorial/terrain/stairs_ascending.jpg
:alt: 沿 x 轴上升的线性台阶地形截图

`axis="x", profile="ascending"`：沿 x 轴从低到高的 8 级台阶。
```

```{figure} /_static/images/tutorial/terrain/stairs_pyramid.jpg
:alt: 径向下降台阶地形截图：中心高台向四周逐级下降

`axis="radial", profile="descending"`：中心高台向四周逐级下降。
```

```{figure} /_static/images/tutorial/terrain/stairs_inverted_pyramid.jpg
:alt: 径向反转金字塔台阶地形截图：中心坑洞四周环形上升

`axis="radial", profile="inverted_pyramid"`：中心坑洞，四周环形上升。
```

### DiscreteObstaclesTerrainGeneratorCfg —— 离散障碍物

在平坦基底（归一化高度 0.5）上随机散布矩形凸起或凹坑。`height_mode="fixed"` 时
每个障碍物都是 `height`（可正可负）；`height_mode="choice"` 时从
{+height, +height/2, −height/2, −height} 中随机抽取，得到凸坑混合的障碍场。
障碍物边长按 `size_min`–`size_max`（场地边长比例）逐轴均匀采样。注意校验要求
`|height| <= 0.5 * height_scale`。

```{figure} /_static/images/tutorial/terrain/obstacles_choice.jpg
:alt: 离散障碍物地形截图：随机散布的凸起与凹坑

`height_mode="choice"`：随机散布的凸起与凹坑混合。
```

### PyramidSlopeTerrainGeneratorCfg —— 金字塔坡

高度随到场地边界的切比雪夫距离线性上升（`slope` 为每米爬升），中心形成方形尖峰；
`inverted=True` 反转为中心凹坑。峰值取决于场地尺寸，生成时若超过 `height_scale`
会报错，需要增大 `height_scale` 或减小 `slope`。

```{figure} /_static/images/tutorial/terrain/slope_compare.jpg
:alt: 金字塔坡对比图：左侧为中心土丘，右侧为反转的中心凹坑

左：`slope=0.1`、`platform_width=0.8` 的中心土丘；右：`inverted=True` 时反转为中心凹坑。
坡度平缓时（每米仅爬升 0.1 m），俯视视角下起伏不明显，建议用低角度相机检查坡面。
```

## 组合地形

`CompositeTerrainGeneratorCfg` 把多个生成器拼进一张场地：先用 `base` 铺满全场，
再按 `regions` 中的 `TerrainRegionCfg` 逐个贴上矩形区域块。要点：

- `center` 与 `size` 都是**全场比例**（0–1），不是米。
- 区域按声明顺序粘贴，重叠时后声明的区域覆盖先声明的。
- 子生成器按 `子 height_scale / 组合 height_scale` 重缩放后粘贴，**物理高度守恒**——
  台阶的 `step_height`、坡的 `slope` 在拼接后保持原来的物理尺寸。
- 组合 `height_scale` 必须覆盖每个子生成器的物理峰值，否则生成时报错。
- `blend`（0–0.5）是区域边缘的过渡带宽度（占区域边长比例）；与场地边界齐平的边不做过渡。

```{figure} /_static/images/tutorial/terrain/composite.jpg
:alt: 组合地形截图：台地化噪声基底上拼贴台阶、坑洞、障碍物与坡

组合地形：噪声台地基底上依次贴上径向台阶、反转金字塔坑、离散障碍物与金字塔坡。
```

`grid_terrain()` 是组合地形的网格快捷方式——把生成器按行列摆成难度网格，每个格子
保留自己的生成器，`height_scale` 默认取所有子生成器中的最大值：

```python
from motrix_env_core.config.scene import (
    FlatTerrainGeneratorCfg,
    StairsTerrainGeneratorCfg,
    grid_terrain,
)

terrain = grid_terrain(
    [
        [FlatTerrainGeneratorCfg(), StairsTerrainGeneratorCfg(step_count=3, step_height=0.05)],
        [StairsTerrainGeneratorCfg(step_count=6, step_height=0.08), FlatTerrainGeneratorCfg()],
    ],
    blend=0.1,  # 所有格子边界过渡带占格子边长的比例
)
```

```{figure} /_static/images/tutorial/terrain/grid.jpg
:alt: 网格地形截图：2x3 台阶格子按难度排列

`grid_terrain` 把不同 `step_count`、`step_height` 的台阶摆成难度网格。
```

## 在环境中使用地形

与平地任务共用同一套环境逻辑，只替换场景，是内置四足/双足任务的固定做法：
平坦配置用 `FlatTerrainCfg` 做地面，粗糙变体继承平坦配置、只覆写 `scene`——
`assets` 换成带地形资产的配置组，`objs.floor` 换成 `HFieldTerrainCfg(hfield="terrain")`：

```python
@registry.envcfg("my-robot-walk-rough")
@configclass
class MyRobotRoughEnvCfg(MyRobotFlatEnvCfg):
    scene: MySceneCfg = MySceneCfg(
        assets=TerrainAssetsCfg(),
        objs=StandardSceneObjsCfg(floor=HFieldTerrainCfg(hfield="terrain", material="mat_ground")),
    )
```

在地形上训练时，重置与奖励通常要以**地形相对高度**为基准。运行时通过 backend 中立的
SimBackend 接口采样任意 (x, y) 处的地面高度：

```python
ground_height = env.sim.sample_terrain_height(
    env.cfg.ground_geom_name, env_ids, base_pos[:, None, :2]
)[:, 0]
```

`go1-walk-rough` 在重置时先采样出生点附近的地面高度，再把基座抬到最高点之上，
避免机器人出生在地里；body-height 奖励同样以局部地形高度为基准。

## 编写自定义生成器

继承 `TerrainGeneratorCfg` 并实现 `generate(size, shape)`，返回归一化 `[0, 1]` 高度数组
即可接入所有组合机制：

```python
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import TerrainGeneratorCfg

import numpy as np


@configclass
class SinusoidalTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Sine waves along the x axis."""

    waves: int = 4

    def validate(self) -> None:
        super().validate()
        if self.waves < 1:
            raise ValueError(f"waves must be at least 1, got {self.waves}")

    def generate(self, size: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
        phase = np.linspace(0.0, self.waves * 2.0 * np.pi, shape[0])[:, None]
        heights = 0.5 + 0.5 * np.sin(phase)  # 归一化到 [0, 1]
        return np.broadcast_to(heights, shape).astype(np.float32)
```

## 预览与调试

`examples/terrain_generate.py` 为每种内置生成器与组合布局各生成一张展示地形，
打印高度统计，并把归一化高度图保存为灰度 PNG；`--render` 用 MotrixSim 渲染组合地形：

```bash
python examples/terrain_generate.py                       # 全部生成器，输出到 terrain_previews/
python examples/terrain_generate.py --resolution 256      # 指定高度场分辨率
python examples/terrain_generate.py --render              # 打开查看器查看组合地形
```

本文截图即由该脚本的展示配置离屏渲染得到。
