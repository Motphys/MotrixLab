# Scene / Robot 配置设计

## 摘要

本文定义 MotrixLab 的场景配置模型。`SceneCfg` 统一保存基础模型文件、asset registry、scene-level visual、system camera 和 object registry：asset 通过 `SceneAssetsCfg` 的字段名获得稳定名称，scene object 通过 `SceneObjsCfg` 的字段名获得稳定名称与组装顺序，全局视觉环境通过 `SceneVisualCfg` 配置，interactive viewer 与视频录制的默认视角通过 `SystemCameraCfg` 配置。`StandardSceneCfg` 在通用 `SceneCfg` 之上提供标准渐变天空、地面、材质、贴图、haze 与方向光，并允许用户通过 configclass 继承覆写。场景最终通过 MotrixSim MSD API 直接组装为 `World`，不通过字符串或临时 MJCF 文件拼接。

## 感知相机与挂载

`SceneCfg.sensors` 使用 `SceneSensorsCfg` 字段名作为传感器名。`RaycastCameraSensorCfg` 描述物理深度相机，不是 `SystemCameraCfg` 渲染视角；配置定义在 core 的 `config.scene.sensor`，不暴露原生 simulator 类型。

- `site` 引用完整组装后场景中的 optical site。统一的 `SiteCfg` 是 scene object，`parent_link_name` 使用导入模型中的逻辑 link 名；编译器根据所属 `BodyCfg` 的 prefix/suffix 解析最终 link，并在所有 body 附加后挂载，因此不依赖字段声明顺序。URDF 文件内的 `sites` 也使用同一配置类型并显式提供 `name`。
- `PinholeCameraCfg` 使用像素单位的 `(fx,fy,cx,cy)`，`from_fov` 接受角度制 FOV；`OrthographicCameraCfg` 使用两个方向的像素间距。光学轴为 +X 右、+Y 下、+Z 前，像素中心为 `(u+.5,v+.5)`。
- 原始测量与图像预处理分离。`outputs` 启用光轴深度、射线距离、世界坐标命中点/法线；至少启用一个距离字段。near/far 限制光轴深度，近处遮挡物不会被穿透跳过。
- 自身排除与几何白名单独立于接触响应；具体后端不支持配置时显式报错。MuJoCo compiler 支持 link site，但不支持该原生相机。

`RaycastCameraImageQuery(sensor=...,field=...)` 从一次原生像素交错读取建立只读视图：距离为 `(N,H,W)`，点/法线为 `(N,H,W,3)`。同一相机多个字段共享原生读取；既有 `SensorValuesQuery` 继续提供扁平布局。距离单位米，miss 为 -1，miss 点/法线为零；validity 可由距离非负得到，但不能据此判定是否已经采样。相机的宽高和投影参数直接由场景配置提供，不额外建立未被读取路径使用的 spec 对象。

### 当前原生生命周期边界

MotrixSim `0.10.2.dev132022` 在 physics step 的积分前状态采样，`update_period=0` 每物理步更新，其余周期由原生调度。读取不推进物理、不强制采样。`SceneData.reset(..., forward_kinematic=True)` 会在 reset 状态刷新 raycast camera 图像；写入关节/浮动基状态并执行带 `forward_kinematic=True` 的 write program 也会刷新图像。普通 `reset`（不请求 FK）仍保留缓存图像，直接修改 DoF 后单独调用 FK 也不会使已有 camera 帧失效或重新采样。

当前版本仍不提供 camera ready、帧序号、采样时间、采样位姿，也没有独立的无时间推进 refresh/invalidate API。因而 query 仍是原生缓存读取接口，不能从 `-1` 区分未就绪与全 miss，也不能把当前 site pose 当作缓存图像的采样外参。不通过图像差分或周期推算伪造帧元数据；完整的有效性和采样元数据契约仍需上游接口支持。

## 目标

- `SceneCfg` 能完整声明程序化场景所需的 asset 与 object。
- texture、material、hfield 等 MSD asset 是显式配置，不由 terrain 隐式创建。
- skybox 作为显式 asset 配置，并由该配置绑定为 world 的活动天空。
- asset 可以通过具名字段继承、覆写和引用，并具有稳定的 Hydra 配置路径。
- scene object 可以按字段名继承和覆写，并由字段声明顺序提供稳定组装顺序。
- haze、ambient light 和 head light 等 scene-level visual 参数不伪装成 asset 或 object。
- `StandardSceneCfg` 提供可复用的标准场景，但 `SceneCfg()` 本身保持空白和通用。
- 完整 MJCF 场景与机器人模型保持不同语义，不因为输入文件格式相同而合并抽象。
- 场景组装直接操作 `motrixsim.msd.World`。

## 配置层级

```text
EnvCfg
└── scene: SceneCfg | None

SceneCfg
├── file: str | Path | None
├── assets: SceneAssetsCfg
├── visual: SceneVisualCfg
├── system_camera: SystemCameraCfg
└── objs: SceneObjsCfg

SceneAssetsCfg
└── <field name>: SceneAssetCfg | None
    ├── TextureCfg
    ├── SkyboxCfg
    ├── MaterialCfg
    └── HFieldAssetCfg

SceneObjsCfg
└── <field name>: SceneObjCfg | None
    ├── RobotCfg
    │   └── QuadrupedRobotCfg
    ├── LightCfg
    └── GeomCfg
        ├── FlatTerrainCfg
        └── HFieldTerrainCfg

ModelFileCfg
├── MjcfFileCfg
└── UrdfFileCfg
```

两个 registry 都通过 configclass 字段提供名称、继承、覆写和 Hydra 路径。asset 在全部注册完成后由 object 引用；scene object 则按照 `dataclasses.fields()` 返回的字段顺序依次附加到 MSD World。

## SceneAssetsCfg 契约

`SceneAssetsCfg` 是字段式、dict-like 的 asset registry。所有 configclass 字段都表示一个 asset，字段名同时作为配置名称和最终 MSD asset name：

```python
@configclass
class SceneAssetsCfg:
    def items(self) -> Iterator[tuple[str, SceneAssetCfg]]: ...


@configclass
class SceneCfg:
    file: str | Path | None = None
    assets: SceneAssetsCfg = SceneAssetsCfg()
    visual: SceneVisualCfg = SceneVisualCfg()
    system_camera: SystemCameraCfg = SystemCameraCfg()
    objs: SceneObjsCfg = SceneObjsCfg()
```

用户通过继承 `SceneAssetsCfg` 声明或覆写 asset：

```python
@configclass
class StandardSceneAssetsCfg(SceneAssetsCfg):
    skybox: SkyboxCfg = SkyboxCfg(
        color_top=(0.4, 0.4, 0.4),
        color_bottom=(0.0, 0.0, 0.0),
    )
    tex_ground: TextureCfg = TextureCfg(
        file=MOTPHYS_GROUND_TEXTURE,
    )
    mat_ground: MaterialCfg = MaterialCfg(
        texture="tex_ground",
        texture_repeat=(0.4, 0.4),
    )
```

该设计具有以下约束：

- `SceneAssetCfg` 不包含 `name` 字段。
- registry 字段名必须是合法 Python 标识符，并直接作为 MSD asset name。
- 同一个 registry 中字段名全局唯一；子类重新声明同名字段表示覆写。
- 值为 `None` 的可选字段不参与最终 asset registry，可用于禁用继承的 asset。
- asset 之间使用字段名字符串引用，例如 material 的 `texture="tex_ground"`。
- registry 提供 `items()` 和按名称读取能力，但不提供绕过 configclass schema 的动态字段写入。

与普通 `dict[str, SceneAssetCfg]` 相比，字段式 registry 保留具体字段类型、IDE 补全、configclass 继承以及 Hydra structured config 路径。

## SceneObjsCfg 契约

`SceneObjsCfg` 是与 asset registry 对称的字段式、dict-like object registry。字段名同时作为配置名称和最终 MSD object name：

```python
@configclass
class SceneObjsCfg:
    def items(self) -> Iterator[tuple[str, SceneObjCfg]]: ...
```

object registry 保留 dataclass 字段顺序：

- 基类字段按照声明顺序排列。
- 子类覆写已有字段时保留原位置。
- 子类新增字段追加到继承字段之后。
- 值为 `None` 的可选字段不参与最终 object 序列。

首版不提供显式重排能力。如果场景要求不同于继承顺序的排列，应定义一个直接继承 `SceneObjsCfg` 的新 registry，并按目标顺序重新声明字段。

`SceneObjsCfg` 使用单一 object 名称空间，因此同一个 registry 中的字段名天然唯一。asset 与 object 属于不同 registry，允许分别存在同名字段。

## SceneAssetCfg 契约

`SceneAssetCfg` 表示能够注册到 `world.assets` 的配置。asset name 由外部 registry 传入：

```python
@configclass
class SceneAssetCfg:
    def add_to_world(
        self,
        world: motrixsim.msd.World,
        name: str,
    ) -> None: ...
```

首批 asset 配置覆盖当前程序化 terrain 已经使用的 MSD 能力：

- `TextureCfg`：file-backed 2D texture、颜色空间和 mipmap 配置。
- `SkyboxCfg`：渐变 skybox texture，并将自身字段名绑定到 `world.assets.skybox`。
- `MaterialCfg`：基础颜色、主 texture 引用、UV repeat、metallic 与 roughness。
- `HFieldAssetCfg`：height-field 文件、XY 尺寸和高度缩放。

一个 scene 最多启用一个 `SkyboxCfg`。该限制避免多个 asset 按注册顺序反复覆盖活动 skybox，使最终视觉环境保持显式且确定。

外部 MJCF 或机器人文件携带的内部 asset 属于该文件自身，不进入 `SceneAssetsCfg` 的显式引用空间。

## StandardSceneCfg

`SceneCfg()` 表示不带预设 asset/object、并保留 MSD 默认视觉参数的通用场景。`StandardSceneCfg` 是独立的可继承 preset，提供 MotrixLab 标准场景：

```python
@configclass
class StandardSceneObjsCfg(SceneObjsCfg):
    floor: FlatTerrainCfg = FlatTerrainCfg(
        material="mat_ground",
    )
    sun: LightCfg = LightCfg(
        color=(0.7, 0.7, 0.7),
        illuminance=10000.0,
    )
    robot: RobotCfg = MISSING


@configclass
class StandardSceneCfg(SceneCfg):
    assets: StandardSceneAssetsCfg = StandardSceneAssetsCfg()
    visual: SceneVisualCfg = SceneVisualCfg(
        ambient_light_color=(0.3, 0.3, 0.3),
        ambient_light_brightness=1000.0,
        head_light_color=(0.6, 0.6, 0.6),
        head_light_luminous_power=1000.0,
        haze=(0.1, 0.1, 0.1, 1.0),
        tone_mapping="none",
    )
    objs: StandardSceneObjsCfg = StandardSceneObjsCfg()
```

`floor`、`sun` 和 `robot` 是普通 registry 字段；`robot` 是标准场景必须提供的插槽。大多数场景直接在构造时传入具体机器人，无需定义新的 registry 类：

```python
scene = StandardSceneCfg(
    objs=StandardSceneObjsCfg(
        floor=FlatTerrainCfg(
            material="mat_ground",
            height=0.1,
        ),
        robot=UnitreeG129Dof(),
    ),
)
```

只有需要新增 object 字段或改变 registry schema 时，才定义新的 `SceneObjsCfg` 子类。无机器人的场景使用通用 `SceneCfg` 或自定义 `SceneObjsCfg`，而不是 `StandardSceneCfg`。

`StandardSceneCfg` 不在初始化后复制或缓存另一份 asset/object 列表；构建和校验始终读取 registry 当前字段，避免出现两份配置事实源。

默认 skybox 使用从灰色到黑色的渐变。环境光颜色 `(0.3, 0.3, 0.3)`、头灯颜色
`(0.6, 0.6, 0.6)`、对应的 MSD 强度 `1000`，以及方向光颜色 `(0.7, 0.7, 0.7)` 和照度
`10000 lux` 与原有 G1 MJCF scene 的加载结果一致；方向光保留标准场景的斜向照射。方向光负责照亮场景物体，
skybox 负责背景，两者职责相互独立。

## SceneVisualCfg

`SceneVisualCfg` 表达 `world.visual` 上的全局视觉环境参数，而不是可命名、可引用的 asset 或 hierarchy object。首版覆盖当前标准场景需要的字段：

```python
@configclass
class SceneVisualCfg:
    ambient_light_color: Vec3 | None = None
    ambient_light_brightness: float | None = None
    head_light_color: Vec3 | None = None
    head_light_luminous_power: float | None = None
    haze: Vec4 | None = None
    tone_mapping: str | None = None
```

字段值为 `None` 时保留 MotrixSim MSD 的默认值；非 `None` 时在 `World.build()` 前写入对应的 `world.visual` 字段。`tone_mapping` 接受 `"none"` 或 `"aces"`；标准场景使用 `"none"`，使渐变 skybox 的颜色变换与原 MJCF scene 一致。该语义让空白 `SceneCfg()` 不需要复制 MotrixSim 的全部默认配置，同时允许 `StandardSceneCfg` 只固定自身确实关心的视觉参数。

## SceneObjCfg 契约

`SceneObjCfg` 是所有附加到场景 hierarchy 的对象配置基类：

```python
@configclass(kw_only=True)
class SceneObjCfg:
    def append_to_world(
        self,
        world: motrixsim.msd.World,
        name: str,
    ) -> None: ...
```

`SceneObjCfg` 不包含 `name` 字段。object name 由 `SceneObjsCfg` 字段名传入，并用于 geom、light 和其他程序化对象的 MSD name。scene object 配置保持可修改，使 Hydra 能覆写 `scene.objs.floor.height` 等嵌套字段。

## GeomCfg 与 terrain

`GeomCfg` 承载程序化 geometry 共有的视觉引用与碰撞配置：

```python
@configclass(kw_only=True)
class GeomCfg(SceneObjCfg):
    material: str | None = None
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    condim: int = 3
    contype: int = 1
    conaffinity: int = 1
    priority: int = 1
```

`FlatTerrainCfg` 只负责 InfinitePlane geometry，不创建 texture 或 material：

```python
@configclass
class FlatTerrainCfg(GeomCfg):
    height: float = 0.0
```

`HFieldTerrainCfg` 只负责引用具名 hfield asset 并创建对应 geometry：

```python
@configclass
class HFieldTerrainCfg(GeomCfg):
    hfield: str
```

texture 文件、UV repeat、hfield 文件、尺寸与高度缩放分别属于 `TextureCfg`、`MaterialCfg` 和 `HFieldAssetCfg`，不属于 terrain object。

## LightCfg

`LightCfg` 表达 world-level directional light，并通过所在 `SceneObjsCfg` 的字段名获得 MSD light name。它包含 position、direction、color、illuminance 和 cast-shadows 配置。其他灯型应使用独立的窄配置类型，不向方向光配置混入无关字段。

## 模型文件与 robot

`SceneCfg.file` 表示完整 MJCF scene，加载后可继续叠加 asset、visual、object 和 sensor 配置。

`ModelFileCfg` 封装模型文件的路径、校验和 MSD World 加载。`MjcfFileCfg` 直接加载 MJCF；`UrdfFileCfg` 在加载 URDF 后可补充 simulation-only geom、site、joint 参数和 actuator。

`RobotCfg` 通过 `model: ModelFileCfg` 组合模型来源，并以必填的 `base_link_name` 选择要挂载的机器人子树。组装后 Body name 与经 prefix/suffix 处理的 base link name 相同，因此运行时也用该字段定位机器人 Body。`RobotCfg` 还包含 translation、rotation、prefix 和 suffix。同一机器人语义配置可以组合 MJCF 或 URDF 来源，而无需为每种格式建立机器人类型的交叉继承。

`QuadrupedRobotCfg` 在 `RobotCfg` 上增加 source-independent 的四足语义：四条具名 leg、可选的 foot contact geom、可选的 foot position sensor，以及按 joint name 索引的默认关节位置。`QuadrupedSceneCfg` 根据传入的 quadruped robot 自动生成 foot contact sensors；步态配对、初始 base 位置和奖励目标高度仍属于 locomotion task 配置。

完整 scene 与 robot 的区别由模型语义决定，而不是由文件格式决定。

## 校验规则

SceneCfg 在修改 MSD World 前完成以下校验：

- 所有启用的 registry 字段值必须是 `SceneAssetCfg`。
- 每个 asset 配置校验自身文件和字段形状。
- scene 中最多启用一个 `SkyboxCfg`。
- `MaterialCfg.texture` 必须引用 `TextureCfg` 字段。
- `GeomCfg.material` 必须引用 `MaterialCfg` 字段。
- `HFieldTerrainCfg.hfield` 必须引用 `HFieldAssetCfg` 字段。
- 所有启用的 object registry 字段值必须是 `SceneObjCfg`。
- object 字段名作为最终对象名称，并由 registry 保证唯一。
- 被设置为 `None` 的默认组件不参与引用集合或场景组装。
- visual 颜色与 haze 必须具有正确维度，亮度和 luminous power 必须非负。

asset registry 采用单一名称空间，因此 texture、material 与 hfield 不能使用相同字段名。该约束让所有 asset 引用都能通过一个稳定名称解析。

## 场景组装

场景组装先加载可选的基础文件，再注册全部 asset，并按顺序附加全部 scene object：

```python
def build_scene_world(scene: SceneCfg) -> motrixsim.msd.World:
    validate_scene_cfg(scene)
    world = motrixsim.msd.World() if scene.file is None else motrixsim.msd.from_file(resolve_path(scene.file))

    for name, asset in scene.iter_assets():
        asset.add_to_world(world, name)

    scene.visual.apply_to_world(world)

    for name, obj in scene.objs.items():
        obj.append_to_world(world, name)

    return world
```

构建过程直接操作 MSD object。`scene.file` 加载完整基础模型；外部 MJCF / URDF 文件也仍可作为 scene object 或机器人输入。SceneCfg 本身不生成 XML。仿真 timestep、solver 和 gravity 仍由 `SimCfg` 在 `World.build()` 前统一配置。

## 完整示例

```python
@configclass
class G1WalkCfg(EnvCfg):
    scene: SceneCfg = StandardSceneCfg(
        assets=StandardSceneAssetsCfg(
            mat_ground=MaterialCfg(
                texture="tex_ground",
                texture_repeat=(0.2, 0.2),
                roughness=0.8,
            ),
        ),
        objs=StandardSceneObjsCfg(robot=UnitreeG129Dof()),
    )
```
