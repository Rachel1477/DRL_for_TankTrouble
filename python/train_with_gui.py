"""
Training script that integrates with GUI for Agent vs SmithAI training
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 全局唯一环境，由C++传入或Python端创建
global_env = None


try:
    from python.train_dqn import QNetwork, DQNAgent, ReplayBuffer
except ImportError:
    # Fallback if not in path
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_dqn", "python/train_dqn.py")
    train_dqn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_dqn)
    QNetwork = train_dqn.QNetwork
    DQNAgent = train_dqn.DQNAgent
    ReplayBuffer = train_dqn.ReplayBuffer

# Try to import the rl_controller binding so we can call the live controller's smith action
try:
    import rl_controller
    _rl_controller_module = rl_controller
except Exception:
    _rl_controller_module = None

# TankEnv 绑定
try:
    import tank_trouble_env as tte
    print("✓ 成功导入 tank_trouble_env")
except ImportError as e:
    print(f"✗ 无法导入 tank_trouble_env: {e}")
    print("请先编译 C++ 环境并设置 PYTHONPATH")
    exit(1)




# 允许C++侧传入TankEnv实例
def set_global_env(env):
    global global_env
    global_env = env

# Global agent for GUI integration
_global_agent = None
_global_episode_count = 0
_global_model_path = "checkpoint_dqn_gui.pth"
_prev_state = None
_prev_action = None
_episode_rewards = []
_episode_step_count = 0

def on_step(prev_state, prev_action, reward, next_state, done):
    """Callback for each step - store experience in replay buffer"""
    print(f"[PY] on_step called: prev_action={prev_action}, reward={reward}, done={done}")
    global _global_agent, _episode_rewards
    if _global_agent is None:
        print("[PY] on_step: _global_agent is None")
        return
    # Add experience to replay buffer
    _global_agent.step(prev_state, prev_action, reward, next_state, done)
    _episode_rewards.append(reward)

def get_action_from_state(state):
    # 始终用唯一的 global_env 环境
    if _global_agent is not None:
        try:
            action = int(_global_agent.act(state))
            print(f"[PY] DQN agent act() -> {action}")
            # 推理时强制 eval，避免 BatchNorm 报错
            _global_agent.qnetwork_local.eval()
            return action
        except Exception as e:
            print(f"[PY] DQN agent act() failed: {e}")
    
    print("[PY] get_action_from_state: fallback to 0")
    return 0

def on_episode_end(episode, total_reward, agent_won):
    """Callback function when episode ends
    
    This is where we trigger learning from the replay buffer.
    """
    global _global_agent, _global_episode_count, _global_model_path
    global _prev_state, _prev_action, _episode_rewards, _episode_step_count
    
    _global_episode_count = episode
    
    # Perform multiple learning steps at episode end
    # This ensures the agent learns from accumulated experience
    if _global_agent is not None and len(_global_agent.memory) > _global_agent.batch_size:
        # Learn from experience multiple times
        num_learning_steps = min(10, _episode_step_count // 10)  # Adaptive learning
        for _ in range(num_learning_steps):
            experiences = _global_agent.memory.sample()
            _global_agent.learn(experiences, _global_agent.gamma)
        
        # Update target network more frequently for faster convergence on random maps
        _global_agent.soft_update(_global_agent.qnetwork_local, 
                                   _global_agent.qnetwork_target, 
                                   _global_agent.tau)
    
    # Calculate epsilon for display
    if episode < 500:
        eps = 1.0 - 0.95 * (episode / 500.0)
    elif episode < 1000:
        eps = 0.05 - 0.04 * ((episode - 500) / 500.0)
    else:
        eps = 0.01
    
    result = "WON" if agent_won else "LOST"
    print(f"\n[Episode {episode}] {result} | Steps: {_episode_step_count} | "
          f"Epsilon: {eps:.3f} | Buffer: {len(_global_agent.memory)}")
    
    # Save model periodically (every 10 episodes) to avoid I/O overhead
    # Do this before resetting episode tracking
    if _global_agent is not None and episode % 10 == 0:
        avg_reward = np.mean(_episode_rewards) if _episode_rewards else total_reward
        torch.save({
            'state_dict': _global_agent.qnetwork_local.state_dict(),
            'state_size': _global_agent.state_size,
            'action_size': _global_agent.action_size,
            'episode': episode,
            'agent_won': agent_won,
            'avg_score': avg_reward,
            'total_reward': total_reward,
            'epsilon': eps,
        }, _global_model_path)
        print(f"[Model saved to {_global_model_path}]")
    
    # Reset episode tracking
    _prev_state = None
    _prev_action = None
    _episode_rewards = []
    _episode_step_count = 0

def initialize_agent(state_size=None, action_size=6, model_path=None):
    global _global_agent, _global_model_path, global_env
    print(f"[PY] initialize_agent called: state_size={state_size}, action_size={action_size}, model_path={model_path}, global_env={global_env}")
    """Initialize the global agent
    
    Params
    ======
        state_size: State size (if None, will try to detect from model or use default 129)
        action_size: Action size (default 6)
        model_path: Path to model checkpoint (supports both train_dqn and train_with_gui formats)
                   If None, will try to load from default paths:
                   - checkpoint_dqn.pth (best model from train_dqn)
                   - final_dqn.pth (final model from train_dqn)
                   - checkpoint_dqn_gui.pth (GUI training model)
    """
    
    # Force CPU for compatibility
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Default state size (updated to match C++: 90 dimensions, 5x5 map grid)
    default_state_size = 82
    
    # Try to load model from various possible paths
    # IMPORTANT: Load checkpoint FIRST to get correct state_size before creating agent
    model_paths_to_try = []
    if model_path:
        model_paths_to_try.append(model_path)
    else:
        # Try common model paths in order of preference
        model_paths_to_try = [
            'checkpoint_dqn.pth',      # Best model from train_dqn
            'final_dqn.pth',           # Final model from train_dqn
            'checkpoint_dqn_gui.pth',  # GUI training model
        ]
    
    loaded_model_path = None
    checkpoint = None
    
    for path in model_paths_to_try:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                loaded_model_path = path
                print(f"Found model at {path}")
                break
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    # Determine state_size and action_size from checkpoint or use defaults
    # Priority: checkpoint > provided parameter > default
    if checkpoint:
        # Try to get state_size and action_size from checkpoint
        checkpoint_state_size = checkpoint.get('state_size', None)
        checkpoint_action_size = checkpoint.get('action_size', None)
        
        if checkpoint_state_size:
            # Always use checkpoint state_size if available (most reliable)
            state_size = checkpoint_state_size
            print(f"  Using state_size from checkpoint: {state_size}")
        elif state_size is None:
            state_size = default_state_size
            print(f"  Using default state_size: {state_size} (checkpoint didn't have state_size)")
        else:
            print(f"  Using provided state_size: {state_size} (checkpoint didn't have state_size)")
        
        if checkpoint_action_size:
            action_size = checkpoint_action_size
            print(f"  Using action_size from checkpoint: {action_size}")
    else:
        # No checkpoint found, use provided or default values
        state_size = state_size if state_size is not None else default_state_size
        print(f"No model found, creating new agent with state_size={state_size}, action_size={action_size}")
    
    # 如果global_env还没被C++传入，则在Python端创建
    if global_env is None:
        try:
            import tank_trouble_env as tte
            global_env = tte.TankEnv()
            print("[Python] 未检测到C++传入环境，已在Python端新建TankEnv")
        except Exception as e:
            print(f"[Python] 创建TankEnv失败: {e}")
            raise
    _global_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=5e-4,
        gamma=0.99,
        tau=1e-3,
        buffer_size=100000,
        batch_size=64,
        update_every=4,
        env=global_env
    )
    
    # Load model weights if checkpoint was found
    if checkpoint and 'state_dict' in checkpoint:
        try:
            _global_agent.qnetwork_local.load_state_dict(checkpoint['state_dict'])
            _global_agent.qnetwork_target.load_state_dict(checkpoint['state_dict'])
            
            # Display model info
            episode = checkpoint.get('episode', 'unknown')
            avg_score = checkpoint.get('avg_score', 'unknown')
            epsilon = checkpoint.get('epsilon', 'unknown')
            
            print(f"✓ Successfully loaded model from {loaded_model_path}")
            print(f"  Episode: {episode}, Avg Score: {avg_score}, Epsilon: {epsilon}")
            
            # Update global model path to the loaded one
            if model_path:
                _global_model_path = model_path
            elif loaded_model_path:
                _global_model_path = loaded_model_path
        except Exception as e:
            print(f"✗ Failed to load model weights: {e}")
            print(f"  Model state_size mismatch? Expected {state_size}")
    elif checkpoint:
        print(f"⚠ Warning: Checkpoint found but no 'state_dict' key. Model not loaded.")
    
    return _global_agent

# Export functions for C++ binding
__all__ = ['get_action_from_state', 'on_episode_end', 'on_step', 'initialize_agent', 'set_global_env']

