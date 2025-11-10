# Tank Trouble RL Training Guide

## 项目概述

本项目成功集成了深度强化学习（Deep Reinforcement Learning）训练框架到 TankTrouble 游戏中，实现了 RL Agent vs SmithAI 的对战训练模式。

## 架构设计

### 核心组件

1. **TankEnv (C++)**: `rl/TankEnv.{h,cc}`
   - 标准 Gym-style 强化学习环境
   - 提供 `reset()` 和 `step(action)` 接口
   - 状态空间：57维向量（位置、角度、距离、16个射线检测）
   - 动作空间：6个离散动作（静止、前进、后退、左转、右转、射击）
   - 奖励函数：距离奖励、对准奖励、转圈惩罚、步骤成本、终局奖励

2. **RLController (C++)**: `controller/RLController.{h,cc}`
   - 继承自 `LocalController`
   - 管理 agent 决策线程和 episode 监控线程
   - 支持 Python 回调注入（DQN 策略）
   - 提供 C++ fallback 启发式策略
   - 实时计算奖励并输出调试信息

3. **DQN Agent (Python)**: `python/train_with_gui.py`
   - Double DQN 算法实现
   - Experience Replay Buffer
   - 提供 `get_action_from_state()` 和 `on_episode_end()` 回调
   - 自动保存模型到 `saved_models/`

4. **Python-C++ 桥接**: `bindings/bindings.cc`
   - 使用 pybind11 将 `TankEnv` 暴露给 Python
   - 编译为 `tank_trouble_env.so` Python 模块

### 关键技术要点

#### GIL (Global Interpreter Lock) 管理
- 主线程在初始化 Python 解释器后释放 GIL (`PyEval_SaveThread()`)
- Agent 线程通过 lambda wrapper 在调用 Python 回调时获取 GIL (`py::gil_scoped_acquire`)
- 所有 Python 对象（模块、函数）存储为 static 并增加引用计数，避免 GIL 释放后被销毁

#### 多线程架构
- **主线程**: GTK GUI 事件循环
- **物理线程**: `LocalController::run()` - 游戏逻辑和 SmithAI 控制
- **Agent 线程**: `RLController::agentLoop()` - 每 50ms 决策一次
- **Episode 线程**: `RLController::episodeLoop()` - 监控对局结束，触发模型保存

#### 奖励塑形（Reward Shaping）
```cpp
// 距离奖励：鼓励接近敌人
reward += 0.5 * (prev_dist_norm - current_dist_norm);

// 对准奖励：鼓励瞄准敌人
reward += 0.1 * (prev_align_norm - current_align_norm);

// 转圈惩罚：避免原地打转
if (displacement < 1.0 && angle_change > 5.0) {
    reward -= 0.02;
}

// 步骤成本：鼓励快速结束
reward -= 0.001;

// 终局奖励
if (win) reward = 100.0;
if (lose) reward = -100.0;
```

## 使用指南

### 环境准备

1. **创建 Conda 环境**:
```bash
conda create -n RL python=3.10
conda activate RL
pip install torch numpy pybind11
```

2. **编译项目**:
```bash
cd /home/rachel/CLionProjects/DRL_for_TankTrouble
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 运行模式

#### 1. GUI 训练模式（推荐）
```bash
cd build
./TankTrouble
```
- 点击界面上的 **"Agent训练"** 按钮
- 观察终端输出的训练日志
- Agent 自动与 SmithAI 对战
- 每局结束自动保存模型

#### 2. 非 GUI 训练模式
```bash
cd build
conda activate RL
python ../python/test_env_standalone.py
```
- 纯命令行训练
- 更快的训练速度（无渲染开销）

#### 3. Python 环境直接训练
```bash
cd python
conda activate RL
python train_dqn.py
```
- 使用 `tank_trouble_env` 模块
- 可调整超参数

## 调试技巧

### 查看 Agent 行为
终端会实时打印：
```
[AGENT] step=100 action=3 (python) r=0.005 dp=0.002 ap=0.004 sp=0 sc=-0.001
```
- `step`: 当前步数
- `action`: 选择的动作（0-5）
- `(python)` or `(fallback)`: 使用 Python 策略还是 C++ 启发式
- `r`: 总奖励
- `dp`: 距离进步奖励
- `ap`: 对准进步奖励
- `sp`: 转圈惩罚
- `sc`: 步骤成本

### 常见问题

#### Q: Agent 一直原地打转？
A: 检查奖励函数，可能需要增加转圈惩罚权重或添加前进奖励。

#### Q: Python 回调未注入（日志显示 fallback）？
A: 检查：
1. 是否激活了 RL conda 环境编译
2. `train_with_gui.py` 是否在正确路径
3. CMake 是否找到了正确的 Python 解释器

#### Q: 程序崩溃，提示 GIL 错误？
A: 确保所有 Python 对象声明为 static，并在释放 GIL 前增加引用计数。

#### Q: 退出时程序卡住？
A: 线程清理逻辑已优化，正常情况下 2 秒内可退出。

## 性能优化建议

1. **训练速度**: 
   - 使用非 GUI 模式训练更快
   - 调整 `agentLoop` 的决策频率（默认 50ms）

2. **奖励调优**:
   - 根据训练日志调整各项奖励权重
   - 监控 `dp`, `ap`, `sp` 的分布

3. **网络结构**:
   - 修改 `train_with_gui.py` 中的 `QNetwork` 层数和神经元数
   - 调整学习率和 batch size

## 文件结构
```
DRL_for_TankTrouble/
├── rl/
│   ├── TankEnv.h           # 环境接口
│   └── TankEnv.cc          # 环境实现
├── controller/
│   ├── RLController.h      # RL 控制器接口
│   └── RLController.cc     # RL 控制器实现
├── bindings/
│   └── bindings.cc         # pybind11 绑定
├── python/
│   ├── train_with_gui.py   # DQN agent（GUI 使用）
│   ├── train_dqn.py        # DQN agent（独立训练）
│   └── test_env_standalone.py  # 测试脚本
├── saved_models/           # 模型保存目录
└── Window.cc              # GUI 集成
```

## 下一步改进

- [ ] 添加 Tensorboard 日志
- [ ] 实现 Prioritized Experience Replay
- [ ] 尝试其他算法（PPO, A3C）
- [ ] 添加课程学习（从简单地图到复杂地图）
- [ ] 多 agent 训练（自我对弈）

## 致谢

本项目基于 TankTrouble 游戏，集成了现代深度强化学习技术。感谢 pybind11 和 PyTorch 社区的支持。

