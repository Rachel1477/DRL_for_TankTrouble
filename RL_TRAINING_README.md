# Agent vs SmithAI 训练模式使用说明

## 功能概述

已实现 Agent vs SmithAI 的训练模式，包括：

1. ✅ **新的目标识别规则**：SmithAI 以 Agent 为目标，Agent 以 SmithAI 为目标
2. ✅ **新的游戏模式**：在入口界面添加了 "Agent训练" 按钮
3. ✅ **自动保存**：每次对局结束后自动保存模型
4. ✅ **GUI 显示**：训练过程在游戏界面中实时显示

## 文件结构

### 新增文件

- `controller/RLController.h` / `controller/RLController.cc`: RL 训练控制器
- `python/train_with_gui.py`: GUI 训练集成脚本
- `python/start_rl_training.py`: 训练启动脚本
- `bindings/rl_bindings.cc`: RLController 的 pybind11 绑定

### 修改文件

- `view/EntryView.h` / `view/EntryView.cc`: 添加 "Agent训练" 按钮
- `Window.h` / `Window.cc`: 支持 RL 训练模式
- `CMakeLists.txt`: 添加 RLController 到构建

## 编译和运行

### 1. 编译项目

```bash
cd build
cmake ..
make -j$(nproc)
```

### 2. 运行游戏

```bash
./TankTrouble
```

### 3. 开始训练

1. 在游戏入口界面点击 **"Agent训练"** 按钮
2. 训练将自动开始，Agent 与 SmithAI 对战
3. 每次对局结束后，模型会自动保存到 `checkpoint_dqn_gui.pth`

## 技术实现

### 目标识别

- **Agent**: 使用 `PLAYER_TANK_ID` (ID=1)
- **SmithAI**: 使用 `AI_TANK_ID` (ID=2)
- SmithAI 通过 `getMyPosition()` 获取 Agent 位置，自动以 Agent 为目标
- Agent 通过 DQN 网络选择动作，以 SmithAI 为目标

### 训练循环

RLController 的 `run()` 方法中集成了训练循环：
- 每 50ms 获取一次 Agent 状态
- 调用 Python 回调函数获取 Agent 动作
- 应用动作并更新游戏状态
- 检测对局结束并触发保存

### 模型保存

每次对局结束后：
- 自动保存模型到 `checkpoint_dqn_gui.pth`
- 包含网络权重、episode 计数和总奖励

## Python 集成

### 设置回调（待实现）

由于从 C++ 直接调用 Python 需要嵌入 Python 解释器，当前实现提供了回调接口。完整的集成需要：

1. 在 C++ 中嵌入 Python 解释器
2. 加载 `train_with_gui.py` 模块
3. 设置回调函数到 RLController

### 简化方案

当前可以使用以下简化方案：

1. **方案 A**: 使用 TankEnv 进行训练（无 GUI），然后加载模型到 GUI 模式
2. **方案 B**: 修改代码以支持从 Python 启动 GUI 并设置回调

## 注意事项

1. **SmithAI 目标**: 已通过 `getMyPosition()` 实现，SmithAI 自动以 Agent 为目标
2. **状态表示**: 使用与 TankEnv 相同的 57 维状态向量
3. **动作空间**: 6 个动作（DO_NOTHING, MOVE_FORWARD, MOVE_BACKWARD, ROTATE_CW, ROTATE_CCW, SHOOT）
4. **奖励函数**: 
   - Agent 获胜: +100
   - Agent 失败: -100
   - 平局: 0
   - 存活奖励: +0.01

## 下一步工作

要完成完整的 Python 集成，需要：

1. 在 `Window::onUserChooseRLTraining()` 中嵌入 Python 解释器
2. 加载 `train_with_gui.py` 并设置回调
3. 启动训练循环

或者使用更简单的方案：创建一个独立的 Python 训练脚本，使用 TankEnv 进行训练，然后定期将状态同步到 GUI。

