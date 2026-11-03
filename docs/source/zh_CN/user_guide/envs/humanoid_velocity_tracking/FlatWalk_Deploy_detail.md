# G1 平地行走：完整部署指南

本指南以官方Walk示例artifact为主线，介绍从仿真验证到真机部署的完整流程。

## 部署演示视频

以下视频展示完整的G1机器人部署过程：

```{video} /_static/videos/Deploy_details_video.mp4
:width: 100%
```

---

## 前提条件

- 一台G1机器人
- 官方示例artifact：`artifacts/g1-flat.deploy`（仓库自带）
- 手柄用于控制机器人移动

**使用个人训练策略：**
如果使用个人训练的策略，请先注册环境并修改相应的sim2sim配置，完成sim2sim验证后再按照本文示例进行sim2real部署。

---

## 第一部分：仿真验证（Sim2Sim）

### 步骤1：验证官方Artifact

在训练服务器或本地机器上执行：

```bash
cd /path/to/MotrixLab
source .venv/bin/activate

# 检查官方示例artifact
mxlab deploy inspect artifact=artifacts/g1-flat.deploy
```

**预期输出：**
```
Artifact: artifacts/g1-flat.deploy
  valid: true
  task: g1_walk/v1
  robot: g1-29dof (29 joints)
  policy: actor.onnx
```

确认 `valid: true`。

### 步骤2：MuJoCo仿真测试

```bash
# 在MuJoCo仿真中测试官方策略
mxlab deploy task=g1-walk-flat/sim artifact=artifacts/g1-flat.deploy
```

**操作说明：**
- **P键**：启动行走策略
- **W/A/S/D**：前进/左移/后退/右移
- **Q/E**：左转/右转
- **Esc/Ctrl+C**：退出

**验证要点：**
- 机器人能否稳定完成启动过渡
- 行走策略是否正确响应速度指令
- 关节位置和速度是否在合理范围内
- 没有异常抖动或不稳定现象
- 机器人能够平滑地前进、后退、平移和转向

✅ **只有仿真验证成功后才能进行真机部署！**

### 步骤3：MotrixSim仿真测试（可选）

```bash
# 使用MotrixSim后端验证
mxlab deploy task=g1-walk-flat/sim \
  artifact=artifacts/g1-flat.deploy \
  runtime.backend=motrixsim
```

---

## 第二部分：真机部署（Sim2Real）

### 步骤4：连接到机器人

### 有线连接（推荐）

```bash
# 通过以太网直连机器人
# 配置PC网络：192.168.123.X/24（机器人默认：192.168.123.10）

# SSH登录到机器人板载PC2
ssh unitree@192.168.123.10
# 默认密码：123
```

### 无线连接

```bash
# 确保PC和机器人在同一WiFi网络
# 查找机器人IP地址后登录
ssh unitree@<机器人IP>
```

---

## 步骤2：克隆仓库到机器人

在机器人PC2上执行：

```bash
cd ~
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
```

---

## 步骤3：安装Unitree SDK2

在机器人PC2上执行：

```bash
# 克隆Unitree SDK2
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2

# 安装到系统目录（推荐）
sudo mkdir -p /opt/unitree_robotics
sudo cp -r . /opt/unitree_robotics/unitree_sdk2

# 或设置环境变量指向SDK2位置
export UNITREE_SDK2_ROOT=~/unitree_sdk2
```

**注意：**
- Unitree SDK2包含CycloneDDS库，是与机器人硬件通信的基础
- `motrix_unitree_native` 在编译时需要链接SDK2
- SDK2已包含所需的DDS库，无需单独安装CycloneDDS

---

## 步骤4：构建最小部署环境

### 安装uv包管理器

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 创建虚拟环境并安装依赖

```bash
# 返回MotrixLab目录
cd ~/MotrixLab

# 创建Python 3.10虚拟环境
uv venv --python 3.10

# 激活环境
source .venv/bin/activate

# 设置SDK2路径（如果未安装到/opt/unitree_robotics）
export UNITREE_SDK2_ROOT=/opt/unitree_robotics/unitree_sdk2

# 安装最小部署依赖（仅sim2real相关）
uv pip install -e motrix_deploy
uv pip install -e motrix_deploy_tasks
uv pip install -e motrix_deploy_unitree[native]
uv pip install -e motrix_robots
uv pip install -e motrixlab
```

**注意：**
- 不需要安装训练相关包（motrixlab、motrix-rl-*）
- 不需要安装仿真后端（mujoco、motrixsim）
- `motrix_unitree_native` 会在安装时编译并链接Unitree SDK2

---

## 步骤5：验证安装和通信

### 验证部署环境

```bash
# 验证Native SDK安装
python -c "from motrix_deploy_unitree.g1 import native; print('部署环境就绪')"
```

### 验证DDS通信

```bash
# 测试与机器人硬件的DDS通信
motrix-deploy-unitree read-lowstate eth0 --robot g1-29dof

# 如果正常，会持续输出机器人状态数据
# 按Ctrl+C停止
```

**如果无法读取状态：**
- 检查网卡名称：`ip -br addr`（常见：`eth0`, `enp*`, `wlan*`）
- 确认机器人上电且系统运行正常

---

## 步骤5：传输Artifact到机器人

**从训练服务器传输：**

```bash
# 在训练服务器上执行
scp -r artifacts/g1-flat.deploy unitree@192.168.123.10:~/MotrixLab/artifacts/
```

**或使用官方示例：**

```bash
# 如果仓库包含官方示例artifact
ls ~/MotrixLab/artifacts/g1-flat.deploy
```

---

## 步骤6：部署前安全准备

### 硬件安全检查

**必须完成的安全措施：**
- [ ] 机器人固定在支撑架或吊带上
- [ ] 准备物理急停按钮，确保可触及
- [ ] 至少两名操作人员在场
- [ ] 清理机器人周围障碍物，确保足够行走空间
- [ ] 关闭其他可能控制机器人的程序

**警告：**
- ⚠️ 行走策略会移动机器人
- ⚠️ 软件急停不能替代物理急停
- ⚠️ 首次部署必须使用支撑

### 手柄准备

**连接手柄到机器人PC2：**
- 同时长按L1+R1进入机器人的调试模式


---

## 步骤7：启动部署

### 部署命令

在机器人PC2上执行：

```bash
cd ~/MotrixLab
source .venv/bin/activate

# 使用eth0网卡（机器人板载部署）
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/g1-flat.deploy \
  runtime.transport.network_interface=eth0
```

**网卡说明：**
- `eth0`：板载以太网（常用）
- `lo`：本地回环（仅本地测试）
- 使用 `ip -br addr` 查看可用网卡

### 启动流程

1. **程序启动**
   - 连接机器人硬件
   - 显示提示：`G1 remote: LT+Up stand  RB+X enable policy  LT+B passive  select emergency stop`

2. **确认站立（手柄操作）**
   - 按住 **LT** + 按 **↑（方向键上）** → 机器人开始站立过渡
   - 默认过渡时间：3秒

3. **启用策略（手柄操作）**
   - 站立完成后，按住 **RB** + 按 **X** → 行走策略启用
   - 策略现在准备接收速度指令

4. **控制移动（手柄操作）**
   - **左摇杆**：前进/后退和左/右平移
   - **右摇杆**：yaw旋转
   - **从小幅度运动开始测试！**

### 手柄控制说明

#### 启动阶段

| 操作 | 手柄按键 | 说明 |
|------|---------|------|
| 确认站立 | LT + ↑ | 按住LT，再按方向键上，开始站立 |
| 启用策略 | RB + X | 按住RB，再按X，启动行走策略 |
| 被动模式 | LT + B | 按住LT，再按B，进入阻尼模式 |
| 紧急停止 | Select | 触发软件急停阻尼 |
| 物理急停 | 急停按钮 | 立即切断电机电源（推荐） |

#### 行走控制

| 操作 | 手柄操作 | 说明 |
|------|---------|------|
| 前进 | 左摇杆下推 | 线速度向前 |
| 后退 | 左摇杆上推 | 线速度向后 |
| 左平移 | 左摇杆右推 | 向左侧移动 |
| 右平移 | 左摇杆左推 | 向右侧移动 |
| 左转 | 右摇杆左推 | 逆时针旋转 |
| 右转 | 右摇杆右推 | 顺时针旋转 |

**注意：**
- 摇杆方向可能反向，具体以实际控制为准
- 默认速度范围：[0.25, 0.25, 0.25]（linear_x, linear_y, yaw）
- 摇杆死区：0.12（小幅度移动会被忽略）

---

## 步骤8：停止和退出

### 正常停止

- 在控制台按 **Ctrl+C** 退出程序

### 紧急停止

1. **软件急停**：按手柄 **Select按钮**
2. **物理急停**：按物理急停按钮（推荐）

**注意：停止后机器人会进入阻尼模式，无法维持站立，需要支撑！**

---

## 进阶：Hot-Swapping多策略模式

如果你已经准备了多个artifact（例如walk和wbt），可以使用hot-swapping模式在不重启程序的情况下切换策略。

### 前提条件

- 已准备walk artifact：`artifacts/g1-flat.deploy`
- 已准备wbt artifact：`artifacts/g1-wbt-dance.deploy`
- 两个artifact都已完成sim2sim验证

### Hot-Swapping部署命令

在机器人PC2上执行：

```bash
cd ~/MotrixLab
source .venv/bin/activate

# 启动多策略热切换模式
mxlab deploy \
  program=g1-hot-swapping \
  runtime=g1-unitree-state-machine \
  program.walk_artifact=artifacts/g1-flat.deploy \
  program.wbt_artifact=artifacts/g1-wbt-dance.deploy \
  runtime.transport.network_interface=eth0
```

### 策略切换操作（手柄）

| 操作 | 手柄组合键 | 说明 |
|------|-----------|------|
| 切换到Fixstand | LT + ↑ | 固定站立模式 |
| 切换到Walk | RB + X | 行走策略 |
| 切换到WBT | RB + Y | 全身跟踪策略 |
| 进入被动模式 | LT + B | 阻尼模式 |

### 使用流程

1. **启动程序**
   - 程序以fixstand模式启动
   - 按 **Start** 确认站立

2. **切换到Walk**
   - 按住 **RB** + 按 **X** → 切换到行走模式
   - 使用左右摇杆控制移动

3. **切换到WBT**
   - 按住 **RB** + 按 **Y** → 切换到WBT模式
   - 机器人执行舞蹈动作

4. **返回Fixstand**
   - 按住 **LT** + 按 **↑** → 返回固定站立
   - 机器人保持站立姿态

### 注意事项

- ⚠️ 切换策略时确保机器人处于稳定状态
- ⚠️ 从WBT切换回其他模式前等待动作完成
- ⚠️ 首次使用建议在支撑架上测试切换
- ⚠️ 策略切换有短暂过渡，属正常现象

### 优势

- ✅ 无需重启程序即可切换策略
- ✅ 快速测试多种运动模式
- ✅ 适合演示和研究对比
- ✅ 减少启动停止次数

---

## 配置调整（可选）

### 调整过渡时间

```bash
# 延长站立过渡时间（更安全）
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/g1-flat.deploy \
  runtime.transport.network_interface=eth0 \
  task_settings.preparation.default_transition_duration_s=5.0
```

### 降低移动速度

```bash
# 降低速度范围（更慢、更安全）
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/g1-flat.deploy \
  runtime.transport.network_interface=eth0 \
  runtime.operator.planar_velocity.range_scale=[0.15,0.15,0.15]
```

### 启用Deadman按钮

```bash
# 需要按住LB才能移动（安全功能）
runtime.operator.planar_velocity.deadman_button=LB
```

### 调整摇杆死区

```bash
# 减小死区（更灵敏）
runtime.operator.planar_velocity.deadzone=0.05
```

### 跳过手动确认

```bash
# 自动启动（充分测试后使用）
task_settings.preparation.wait_for_remote_buttons=false
```

---

## 故障排查

### 无法SSH连接机器人

**检查：**
```bash
# 1. Ping机器人
ping 192.168.123.10

# 2. 检查PC网络配置
ip addr show

# 3. 确认机器人上电
```

### 无法读取机器人状态

**检查：**
```bash
# 1. 确认网卡名称
ip -br addr

# 2. 尝试不同网卡
motrix-deploy-unitree read-lowstate lo --robot g1-29dof
motrix-deploy-unitree read-lowstate eth0 --robot g1-29dof

# 3. 检查防火墙
sudo ufw status
```

### 手柄无响应

**检查：**
```bash
# 1. 确认手柄连接
ls /dev/input/js*

# 2. 测试手柄输入
jstest /dev/input/js0

# 3. 增加速度范围
runtime.operator.planar_velocity.range_scale=[0.5,0.5,0.5]

# 4. 减小死区
runtime.operator.planar_velocity.deadzone=0.05
```

### 部署环境安装失败

**检查Python版本：**
```bash
python --version
# 必须是Python 3.10
```

**重新安装：**
```bash
rm -rf .venv
uv venv --python 3.10
source .venv/bin/activate
# 重新安装依赖
```

### 机器人启动时抖动

**尝试：**
```bash
# 降低启动增益
task_settings.preparation.kp=60.0
task_settings.preparation.kd=2.5

# 延长过渡时间
task_settings.preparation.default_transition_duration_s=5.0
```

### 机器人行走不稳定

**检查：**
1. 先使用小速度测试
```bash
runtime.operator.planar_velocity.range_scale=[0.1,0.1,0.1]
```

2. PD参数是否与训练配置匹配
```bash
cat artifacts/g1-flat.deploy/manifest.json | grep -A 30 '"kp"'
```

3. 地面条件是否与训练环境相似

4. 机器人硬件状态是否正常

---

## 常用命令速查

```bash
# 验证部署环境
python -c "from motrix_deploy_unitree.g1 import native; print('OK')"

# 测试DDS通信
motrix-deploy-unitree read-lowstate eth0 --robot g1-29dof

# 检查artifact
mxlab deploy inspect artifact=artifacts/g1-flat.deploy

# 测试手柄
jstest /dev/input/js0

# 基本部署
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/g1-flat.deploy \
  runtime.transport.network_interface=eth0

# 慢速部署（更安全）
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/g1-flat.deploy \
  runtime.transport.network_interface=eth0 \
  runtime.operator.planar_velocity.range_scale=[0.15,0.15,0.15]
```

---

## 使用官方示例Artifact

如果想快速测试，使用仓库自带的示例：

```bash
# Walk行走示例
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/g1-flat.deploy \
  runtime.transport.network_interface=eth0

# WBT舞蹈示例
mxlab deploy task=g1-wbt-dance/hardware \
  artifact=artifacts/g1-wbt-dance.deploy \
  runtime.transport.network_interface=eth0
```

---

## 安全提醒

- ✅ 首次部署必须使用支撑架或吊带
- ✅ 准备独立的物理急停按钮
- ✅ 至少两名操作人员在场
- ✅ 先使用小速度指令测试
- ✅ 确保行走区域无障碍物
- ✅ 软件急停不能替代物理急停
- ✅ 停止后机器人无法维持站立，需要支撑

---

## 参考文档

- [MotrixLab训练教程](../../tutorial/basic_frame.md)
- [Motrix Deploy高级指南](../../tutorial/advanced/motrix_deploy.md)
- [Unitree Deploy README](../../../../../../motrix_deploy_unitree/README.md)
- [WBT部署指南](../whole_body_tracking/WBT_Deploy_detail.md)
