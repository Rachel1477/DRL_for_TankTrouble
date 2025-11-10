# 模型使用说明

## 概述

`train_dqn.py` 训练出来的模型可以无缝用于 `train_with_gui.py`，因为它们使用相同的网络结构（`QNetwork` 和 `DQNAgent`）。

## 模型文件

### train_dqn.py 保存的模型

1. **checkpoint_dqn.pth** - 训练过程中最佳模型（平均分提升时保存）
2. **final_dqn.pth** - 训练结束时的最终模型

### train_with_gui.py 保存的模型

1. **checkpoint_dqn_gui.pth** - GUI训练过程中保存的模型（每10个episode保存一次）

## 使用方法

### 方法1: 自动加载（推荐）

`train_with_gui.py` 会自动按以下顺序尝试加载模型：

1. `checkpoint_dqn.pth` (train_dqn 的最佳模型)
2. `final_dqn.pth` (train_dqn 的最终模型)
3. `checkpoint_dqn_gui.pth` (GUI训练模型)

只需确保模型文件在当前工作目录，然后运行GUI训练即可：

```python
from train_with_gui import initialize_agent

# 自动加载模型（如果存在）
agent = initialize_agent()
```

### 方法2: 指定模型路径

```python
from train_with_gui import initialize_agent

# 指定要加载的模型
agent = initialize_agent(model_path='checkpoint_dqn.pth')
# 或
agent = initialize_agent(model_path='final_dqn.pth')
```

### 方法3: 手动指定状态大小

如果模型文件不在默认路径，可以手动指定状态大小：

```python
from train_with_gui import initialize_agent

# 自动检测状态大小（从模型文件或使用默认值129）
agent = initialize_agent(state_size=None, action_size=6, model_path='path/to/model.pth')
```

## 状态空间兼容性

- **train_dqn.py**: 使用 129 维状态空间
  - 9 (基础信息)
  - 4 (网格位置)
  - 3 (路径信息)
  - 64 (地图网格)
  - 1 (直线视线)
  - 48 (射线特征)

- **train_with_gui.py**: 自动适配状态空间
  - 如果模型文件包含 `state_size`，会自动使用
  - 如果未找到模型，默认使用 129

## 模型信息

加载模型时会显示以下信息：

```
Found model at checkpoint_dqn.pth
  Using state_size from checkpoint: 129
  Using action_size from checkpoint: 6
✓ Successfully loaded model from checkpoint_dqn.pth
  Episode: 500, Avg Score: 15.23, Epsilon: 0.01
```

## 注意事项

1. **状态空间匹配**: 模型必须是用相同状态空间训练的。当前版本使用129维状态空间。

2. **模型格式**: 模型文件必须包含以下键：
   - `state_dict`: 网络权重
   - `state_size`: 状态空间大小（可选，但推荐）
   - `action_size`: 动作空间大小（可选，但推荐）

3. **设备兼容性**: 模型会自动加载到CPU，确保兼容性。

4. **继续训练**: 使用 `train_with_gui.py` 加载 `train_dqn.py` 的模型后，可以继续训练，模型会保存到 `checkpoint_dqn_gui.pth`。

## 示例

### 使用 train_dqn 训练的模型进行GUI训练

```bash
# 1. 训练模型
python python/train_dqn.py

# 2. 模型会自动保存到 checkpoint_dqn.pth 和 final_dqn.pth

# 3. 运行GUI训练（会自动加载模型）
python python/start_rl_training.py
# 或在GUI中点击"Agent训练"按钮
```

### 在代码中使用

```python
from train_with_gui import initialize_agent, get_action_from_state

# 初始化agent（会自动加载 train_dqn 训练的模型）
agent = initialize_agent()

# 获取动作
state = [0.5] * 129  # 129维状态向量
action = get_action_from_state(state)
print(f"Selected action: {action}")
```

## 故障排除

### 问题1: 状态空间不匹配

**错误**: `RuntimeError: Error(s) in loading state_dict`

**解决**: 确保模型是用相同状态空间训练的。检查模型文件的 `state_size` 是否与当前环境匹配。

### 问题2: 找不到模型文件

**错误**: `No model found, creating new agent`

**解决**: 
- 确保模型文件在当前工作目录
- 或使用 `model_path` 参数指定完整路径
- 检查文件权限

### 问题3: 模型加载失败

**错误**: `Failed to load model weights`

**解决**:
- 检查模型文件是否损坏
- 确保 PyTorch 版本兼容
- 检查模型文件格式是否正确

## 更新日志

- **2024-12-XX**: 添加了自动模型加载功能，支持 train_dqn 和 train_with_gui 的模型格式
- **2024-12-XX**: 更新状态空间从122维到129维，包含路径信息

