# TankTrouble - 深度强化学习坦克对战游戏

## 1. 项目简介

TankTrouble 是一个用 C++17 打造的坦克对战游戏（Linux版），支持单机和联网模式。本项目基于原版 TankTrouble 游戏，集成了**深度强化学习（DRL）**训练功能，可以训练智能体（Agent）学习如何玩坦克对战游戏。

### 主要特性

- **单机模式**：与高智能的 AI 对手 Agent Smith 对战（会躲子弹、追踪敌人、智能攻击）
- **强化学习训练**：提供完整的 RL 训练环境，支持 DQN、PPO 等算法
- **GUI 训练界面**：可视化训练过程，实时观察智能体的学习效果
- **随机地图生成**：每局游戏都有不同的迷宫地图
- **物理引擎**：包含碰撞检测、子弹反弹等真实物理效果

**相关仓库**：[TankTroubleServer](https://github.com/JustDoIt0910/TankTroubleServer)

---

## 2. 快速开始

### 2.1 环境要求

- **操作系统**：Linux (建议 Ubuntu 18.04+)
- **编译器**：支持 C++17 的 GCC/Clang
- **Python**：Python 3.8+ (用于强化学习训练)
- **Conda**：Miniconda 或 Anaconda（用于管理 Python 环境）

### 2.2 安装依赖

#### 安装系统依赖

```bash
# 安装 gtkmm-3.0 图形界面库
sudo apt-get update
sudo apt-get install libgtkmm-3.0-dev

# 安装 CMake（如果没有）
sudo apt-get install cmake
```

#### 创建 Conda 环境

```bash
# 创建名为 RL 的 conda 环境
conda create -n RL python=3.10

# 激活环境
conda activate RL

# 安装 Python 依赖
pip install torch numpy pybind11 gymnasium
```

### 2.3 编译项目

使用提供的 `build.sh` 脚本进行一键编译：

```bash
# 克隆仓库
git clone https://github.com/your-repo/DRL_for_TankTrouble.git
cd DRL_for_TankTrouble

# 初始化子模块（如果有）
git submodule update --init --recursive

# 运行编译脚本
bash build.sh
```

编译脚本会自动：
1. 激活 conda RL 环境
2. 检查并安装必要的 Python 依赖（pybind11、PyTorch）
3. 配置 CMake（指定 Python 路径和 RPATH）
4. 编译项目生成可执行文件和 Python 模块

### 2.4 运行游戏

```bash
cd build
./TankTrouble
```

启动后可以选择：
- **单机模式**：与 Agent Smith AI 对战
- **在线模式**：连接服务器进行多人对战
- **Agent训练**：开始强化学习训练（需要先编译成功）

### 2.5 开始 RL 训练

#### 方式 1：使用 GUI 训练界面

```bash
cd build
./TankTrouble
# 点击 "Agent训练" 按钮
```

---

## 3. 关键文件解释

### 3.1 核心游戏逻辑

#### `main.cc`
- **作用**：程序入口，初始化 GTK 应用并启动主窗口
- **内容**：创建 GTK Application 实例，运行游戏主窗口

#### `Window.cc / Window.h`
- **作用**：游戏主窗口管理器
- **内容**：管理所有视图（入口视图、游戏视图、大厅视图），处理视图切换，嵌入 Python 解释器用于 RL 训练回调

#### `Controller.cc / Controller.h`
- **作用**：游戏控制器的抽象基类
- **内容**：定义了 LocalController 和 OnlineController 的统一接口，包括游戏对象管理、移动控制等

### 3.2 游戏对象

#### `Tank.cc / Tank.h`
- **作用**：坦克对象，继承自 Object
- **内容**：
  - 坦克的绘制、移动、旋转逻辑
  - 前进、后退、顺时针/逆时针旋转控制
  - 子弹发射管理（剩余弹药数量）
  - 坦克尺寸定义：宽20、高28像素

#### `Shell.cc / Shell.h`
- **作用**：炮弹对象，继承自 Object
- **内容**：
  - 炮弹的绘制和移动逻辑
  - 子弹反弹计算（与墙壁碰撞后的反射）
  - 子弹生命周期管理

#### `Block.cc / Block.h`
- **作用**：墙壁对象
- **内容**：
  - 地图中的墙壁方块
  - 不继承 Object，根据地图生成算法创建
  - 用于碰撞检测的矩形区域

#### `Object.cc / Object.h`
- **作用**：游戏对象的多态基类
- **内容**：
  - 定义了所有游戏对象的通用接口：`draw()`、`getCurrentPosition()`、`getNextPosition()`
  - 包含位置信息结构 `PosInfo`（坐标 + 角度）
  - 移动状态枚举

### 3.3 地图生成

#### `Maze.cc / Maze.h`
- **作用**：随机迷宫地图生成算法
- **内容**：
  - 使用类似 Prim 最小生成树的算法
  - 从左上角开始，随机打通相邻格子之间的墙
  - 保证每次生成的地图都不同且连通

### 3.4 游戏控制器

#### `controller/LocalController.cc / LocalController.h`
- **作用**：单机模式的游戏控制器
- **内容**：
  - 管理单机游戏的所有逻辑：对象移动、碰撞检测、得分计算
  - 管理 Agent Smith AI 的行为
  - 维护全局步数 `globalSteps`（用于 AI 决策）
  - 处理碰撞检测优化表（按网格划分可能碰撞的墙壁）

#### `controller/OnlineController.cc / OnlineController.h`
- **作用**：联网模式的游戏控制器
- **内容**：
  - 负责与服务器通信，同步游戏状态
  - 从服务器获取游戏对象数据并更新本地视图
  - 发送玩家操作到服务器

#### `controller/RLController.cc / RLController.h`
- **作用**：强化学习模式的游戏控制器
- **内容**：
  - 专门为 RL 训练设计的控制器
  - 提供状态观测、动作执行、奖励计算接口
  - 与 Python 端的 RL 算法交互

### 3.5 AI 人机（Agent Smith）

#### `smithAI/AgentSmith.cc / AgentSmith.h`
- **作用**：AI 对手的决策大脑
- **内容**：
  - 危险检测：预测炮弹弹道，判断是否处于威胁中
  - 躲避决策：计算最优躲避路径（旋转、移动、边转边移）
  - 攻击决策：瞄准并射击敌人
  - 路径规划：调用 A* 算法接近敌人

#### `smithAI/DodgeStrategy.cc / DodgeStrategy.h`
- **作用**：躲避策略执行者
- **内容**：
  - 存储并执行 AgentSmith 生成的躲避命令队列
  - 命令格式：`{ROTATE_CW, 3}, {MOVE_FORWARD, 15}`
  - 根据 globalSteps 判断每个命令是否执行完毕

#### `smithAI/ContactStrategy.cc / ContactStrategy.h`
- **作用**：接近策略执行者
- **内容**：
  - 存储 A* 算法生成的路径点
  - 控制坦克沿路径点移动，接近敌人

#### `smithAI/AttackStrategy.cc / AttackStrategy.h`
- **作用**：攻击策略执行者
- **内容**：
  - 存储瞄准角度
  - 通过弹道模拟寻找最佳射击角度

#### `smithAI/AStar.cc / AStar.h`
- **作用**：A* 路径规划算法
- **内容**：
  - 经典 A* 算法实现
  - 在迷宫地图中寻找从起点到终点的最短路径
  - 返回路径点列表供 ContactStrategy 使用

### 3.6 强化学习环境

#### `rl/TankEnv.cc / rl/TankEnv.h`
- **作用**：RL 训练环境（Gym-like 接口）
- **内容**：
  - 提供标准的 RL 接口：`reset()`、`step(action)`
  - 动作空间：DO_NOTHING、MOVE_FORWARD、MOVE_BACKWARD、ROTATE_CW、ROTATE_CCW、SHOOT
  - 状态空间：坦克位置、敌人位置、炮弹信息、墙壁信息等
  - 奖励函数：击中敌人奖励、被击中惩罚、存活奖励等
  - 管理智能体坦克（agent_tank_id_）和敌人坦克（enemy_tank_id_）

### 3.7 Python 绑定

#### `bindings/bindings.cc`
- **作用**：TankEnv 的 Python 绑定
- **内容**：
  - 使用 pybind11 将 C++ 的 TankEnv 暴露给 Python
  - 生成 `tank_trouble_env` Python 模块
  - 允许在 Python 中调用 `env.reset()` 和 `env.step(action)`

#### `bindings/rl_bindings.cc`
- **作用**：RLController 的 Python 绑定
- **内容**：
  - 使用 pybind11 将 C++ 的 RLController 暴露给 Python
  - 生成 `rl_controller` Python 模块
  - 用于 GUI 训练模式下的回调

### 3.8 工具类

#### `util/Vec.cc / util/Vec.h`
- **作用**：二维向量类
- **内容**：
  - 向量的加减乘除、点乘、叉乘
  - 向量长度、单位向量、旋转等运算
  - 游戏中所有位置和方向计算的基础

#### `util/Math.cc / util/Math.h`
- **作用**：几何数学工具库
- **内容**：
  - 圆形与矩形的碰撞检测
  - 矩形与矩形的碰撞检测（投影法）
  - 线段与线段的交点计算
  - 子弹反弹方向计算
  - 包围盒（Bounding Box）算法

#### `util/Id.cc / util/Id.h`
- **作用**：唯一 ID 生成器
- **内容**：
  - 为游戏对象（坦克、炮弹）生成唯一标识符
  - 使用原子操作保证线程安全

### 3.9 视图层（GUI）

#### `view/EntryView.cc / EntryView.h`
- **作用**：游戏入口界面
- **内容**：
  - 显示游戏 Logo
  - 提供"单机模式"、"在线模式"、"Agent训练"按钮

#### `view/GameView.cc / GameView.h`
- **作用**：游戏主界面
- **内容**：
  - 绘制游戏场景（坦克、炮弹、墙壁）
  - 显示玩家信息（血量、得分）
  - 处理键盘输入（WASD 移动，空格射击）

#### `view/GameLobby.cc / GameLobby.h`
- **作用**：在线模式游戏大厅
- **内容**：
  - 显示可用房间列表
  - 创建新房间、加入房间功能
  - 房间信息同步

#### `view/component/GameArea.cc / GameArea.h`
- **作用**：游戏绘制区域组件
- **内容**：
  - 使用 Cairo 绘制游戏场景
  - 调用每个对象的 `draw()` 方法进行渲染
  - 处理画面刷新和动画

#### `view/component/PlayerInfoItem.cc / PlayerInfoItem.h`
- **作用**：玩家信息显示组件
- **内容**：显示玩家名称、血量、得分等信息

#### `view/component/RoomItem.cc / RoomItem.h`
- **作用**：房间列表项组件
- **内容**：在游戏大厅中显示单个房间的信息

### 3.10 事件系统

#### `event/ControlEvent.cc / ControlEvent.h`
- **作用**：游戏控制事件封装
- **内容**：
  - 将键盘操作封装成事件对象
  - 方便集成到事件驱动模型中
  - 包含移动、旋转、射击等事件类型


### 3.12 配置文件

#### `defs.h`
- **作用**：游戏全局宏定义
- **内容**：
  - 游戏区域尺寸、网格大小
  - 坦克和炮弹的物理参数
  - 移动步长、旋转步长等常量

#### `CMakeLists.txt`
- **作用**：CMake 构建配置文件
- **内容**：
  - 配置编译选项（C++17）
  - 查找依赖库（gtkmm、Python、pybind11）
  - 定义编译目标：
    - `TankTrouble`：主可执行文件
    - `tank_trouble_env.so`：Python 环境模块
    - `rl_controller.so`：RL 控制器模块
  - 设置 RPATH 解决 libstdc++.so.6 版本冲突

#### `build.sh`
- **作用**：一键编译脚本
- **内容**：
  1. 激活 conda RL 环境
  2. 检查 Python、pybind11、PyTorch 依赖
  3. 清理并创建 build 目录
  4. 配置 CMake（指定 Python 路径、RPATH）
  5. 使用多核编译 `make -j$(nproc)`
  6. 提示编译成功和运行方式

#### `build_python_module.sh`
- **作用**：单独编译 Python 模块的脚本
- **内容**：仅编译 `tank_trouble_env.so` 和 `rl_controller.so`，不编译 GUI 部分

---

## 4. 项目架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                         主程序入口                           │
│                        main.cc                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │   Window.cc/h       │  ← 主窗口管理器
        │  (GTK Application)  │
        └─────────┬───────────┘
                  │
         ┌────────┼────────┐
         │        │        │
         ▼        ▼        ▼
    ┌────────┐ ┌────────┐ ┌──────────┐
    │ Entry  │ │ Game   │ │  Lobby   │  ← 视图层
    │ View   │ │ View   │ │  View    │
    └────────┘ └───┬────┘ └──────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │   Controller        │  ← 控制器层
         │  (基类)             │
         └──────┬──────────────┘
                │
      ┌─────────┼─────────┐
      │         │         │
      ▼         ▼         ▼
┌──────────┐ ┌────────┐ ┌──────────┐
│ Local    │ │ Online │ │ RL       │
│Controller│ │Ctrl    │ │Controller│
└────┬─────┘ └────────┘ └─────┬────┘
     │                         │
     ▼                         ▼
┌──────────┐            ┌─────────────┐
│ Smith AI │            │ TankEnv     │  ← RL 环境
│ (策略)   │            │ (Gym-like)  │
└──────────┘            └─────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │ Python 训练   │
                      │ (DQN/PPO)     │
                      └───────────────┘
```

### 线程模型

- **GUI 线程**：处理界面渲染和用户输入
- **Controller 线程**：运行游戏逻辑（由自定义事件驱动库驱动）
- **网络线程**（联网模式）：处理与服务器的通信

### 数据流

1. **单机模式**：LocalController → AgentSmith AI → GameView
2. **RL 训练模式**：Python (DQN/PPO) → TankEnv → RLController → GameView

---


## 5. 技术亮点

1. **智能 AI 对手**：Agent Smith 使用弹道预测、A* 路径规划、躲避策略，难度极高
2. **碰撞检测优化**：通过网格划分减少不必要的碰撞检测，提升性能
3. **投影法碰撞检测**：支持任意角度旋转的矩形碰撞
4. **包围盒算法**：精确计算子弹反弹方向
5. **事件驱动架构**：游戏逻辑与 GUI 完全解耦，符合单一职责原则
6. **自定义网络协议**：高效的消息序列化，无需第三方库
7. **RL 环境集成**：完整的 Gym-like 接口，支持多种 RL 算法
8. **跨语言调用**：C++ 和 Python 通过 pybind11 无缝集成

---

## 7. 开发者信息

本项目基于原版 TankTrouble 游戏，增加了深度强化学习训练功能。

### 贡献指南

欢迎提交 Issue 和 Pull Request！

### 许可证

详见 [LICENSE](LICENSE) 文件。

---

## 8. 常见问题

### Q1: 编译时提示找不到 pybind11？
**A**: 确保已激活 RL 环境并安装 pybind11：
```bash
conda activate RL
pip install pybind11
```

### Q2: 运行时报错 `libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`？
**A**: 这是库版本冲突，使用 `build.sh` 脚本编译可以自动解决（会设置正确的 RPATH）。

### Q3: 训练时智能体一直原地不动？
**A**: 检查奖励函数设计，确保鼓励探索行为。可以参考 `REWARD_IMPROVEMENTS_V2.md` 调整奖励。

### Q4: GUI 训练界面卡顿？
**A**: 降低渲染频率或使用无 GUI 的训练脚本（`train_dqn.py`）。

### Q5: 如何加载训练好的模型？
**A**: 参考 `python/MODEL_USAGE.md` 文档，使用 `torch.load()` 加载模型权重。

---

