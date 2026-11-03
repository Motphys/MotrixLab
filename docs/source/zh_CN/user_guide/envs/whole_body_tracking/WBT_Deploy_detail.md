# G1 全身跟踪（WBT）：完整部署指南

本指南以官方WBT示例artifact为主线，介绍从仿真验证到真机部署的完整流程。

---

## 前提条件

- 一台G1机器人
- 官方示例artifact：`artifacts/g1-wbt-dance.deploy`（仓库自带）

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
mxlab deploy inspect artifact=artifacts/g1-wbt-dance.deploy
```

**预期输出：**
```
Artifact: artifacts/g1-wbt-dance.deploy
  valid: true
  task: g1_wbt/v1
  robot: g1-29dof (29 joints)
  policy: actor.onnx
```

确认 `valid: true`。

### 步骤2：MuJoCo仿真测试

```bash
# 在MuJoCo仿真中测试官方策略
mxlab deploy task=g1-wbt-dance/sim artifact=artifacts/g1-wbt-dance.deploy
```

**操作说明：**
- **P键**：启动WBT策略执行
- **O键**：取消启动
- **Esc/Ctrl+C**：退出

**验证要点：**
- 机器人能否稳定完成启动过渡
- WBT策略是否正确跟踪运动轨迹
- 关节位置和速度是否在合理范围内
- 没有异常抖动或不稳定现象

✅ **只有仿真验证成功后才能进行真机部署！**

### 步骤3：MotrixSim仿真测试（可选）

```bash
# 使用MotrixSim后端验证
mxlab deploy task=g1-wbt-dance/sim \
  artifact=artifacts/g1-wbt-dance.deploy \
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
- 不需要安装训练相关包（motrix-rl-*）
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
# 将训练并已完成sim2sim的文件，传输到机器人PC2上
scp -r artifacts/g1-wbt-dance.deploy unitree@192.168.123.10:~/MotrixLab/artifacts/
```

**或使用官方示例：**

```bash
# 如果仓库包含官方示例artifact
ls ~/MotrixLab/artifacts/g1-wbt-dance.deploy
```

---

## 步骤6：部署前安全准备

### 硬件安全检查

**必须完成的安全措施：**
- [ ] 机器人固定在支撑架或吊带上
- [ ] 准备物理急停按钮，确保可触及
- [ ] 至少两名操作人员在场
- [ ] 清理机器人周围障碍物，确保足够活动空间
- [ ] 关闭其他可能控制机器人的程序

**警告：**
- ⚠️ WBT策略可能产生大幅度动作
- ⚠️ 软件急停不能替代物理急停
- ⚠️ 首次部署必须使用支撑

### 手柄准备

**连接手柄到机器人PC2：**
- 通过同时长按L1+R1进入机器人的调试模式

---

## 步骤7：启动部署

### 部署命令

在机器人PC2上打开终端，并执行：

```bash
cd ~/MotrixLab
source .venv/bin/activate

# 使用eth0网卡（机器人板载部署）
mxlab deploy task=g1-wbt-dance/hardware \
  artifact=artifacts/g1-wbt-dance.deploy \
  runtime.transport.network_interface=eth0
```

**网卡说明：**
- `eth0`：板载以太网（常用）
- `lo`：本地回环（仅本地测试）
- 使用 `ip -br addr` 查看可用网卡

### 启动流程

1. **程序启动**
   - 连接机器人硬件
   - 等待手柄输入

2. **确认站立（手柄操作）**
   - 按 **Start按钮** → 机器人开始站立过渡
   - 默认过渡时间：3秒

3. **启动策略（手柄操作）**
   - 站立完成后，按 **A按钮** → 启动WBT策略
   - 机器人自动执行运动序列（约20秒）

4. **策略完成**
   - 自动完成或按 **Select按钮** 紧急停止

### 启动过程中的按键

| 操作 | 手柄按键 | 说明 |
|------|---------|------|
| 确认站立 | Start | 开始从当前姿态过渡到站立 |
| 启动策略 | A | 站立完成后启动WBT运动 |
| 紧急停止 | Select | 触发软件急停阻尼 |
| 物理急停 | 急停按钮 | 立即切断电机电源（推荐） |

**注意：**
- WBT硬件部署需要手柄
- P/M/O键用于仿真模式，硬件模式使用Start/A按钮

---

## 步骤8：停止和退出

### 正常停止

- WBT策略播放完成后自动停止
- 或在控制台按 **Ctrl+C** 退出程序

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

### 策略执行不稳定

**检查：**
1. PD参数是否与训练配置匹配
```bash
cat artifacts/g1-wbt-dance.deploy/manifest.json | grep -A 30 '"kp"'
```

2. 地面条件是否与训练环境相似

3. 机器人硬件状态是否正常

---

## 常用命令速查

```bash
# 验证部署环境
python -c "from motrix_deploy_unitree.g1 import native; print('OK')"

# 测试DDS通信
motrix-deploy-unitree read-lowstate eth0 --robot g1-29dof

# 检查artifact
mxlab deploy inspect artifact=artifacts/g1-wbt-dance.deploy

# 基本部署
mxlab deploy task=g1-wbt-dance/hardware \
  artifact=artifacts/g1-wbt-dance.deploy \
  runtime.transport.network_interface=eth0

# 慢速过渡部署（更安全）
mxlab deploy task=g1-wbt-dance/hardware \
  artifact=artifacts/g1-wbt-dance.deploy \
  runtime.transport.network_interface=eth0 \
  task_settings.preparation.default_transition_duration_s=5.0
```

---

## 使用官方示例Artifact

如果想快速测试，使用仓库自带的示例：

```bash
# WBT舞蹈示例
mxlab deploy task=g1-wbt-dance/hardware \
  artifact=artifacts/g1-wbt-dance.deploy \
  runtime.transport.network_interface=eth0

# Walk行走示例
mxlab deploy task=g1-walk-flat/hardware \
  artifact=artifacts/g1-flat.deploy \
  runtime.transport.network_interface=eth0
```

---

## 安全提醒

- ✅ 首次部署必须使用支撑架或吊带
- ✅ 准备独立的物理急停按钮
- ✅ 至少两名操作人员在场
- ✅ WBT可能产生大幅度动作，确保安全空间
- ✅ 软件急停不能替代物理急停
- ✅ 停止后机器人无法维持站立，需要支撑

---

## 参考文档

- [MotrixLab训练教程](../../tutorial/basic_frame.md)
- [Motrix Deploy高级指南](../../tutorial/advanced/motrix_deploy.md)
- [WBT任务添加指南](./adding_wbt_task.md)
- [Unitree Deploy README](../../../../../../motrix_deploy_unitree/README.md)
