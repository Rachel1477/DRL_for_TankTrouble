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

# Global agent for GUI integration
_global_agent = None
_global_episode_count = 0
_global_model_path = "checkpoint_dqn_gui.pth"

def get_action_from_state(state):
    """Callback function for C++ to get action from agent"""
    global _global_agent
    if _global_agent is None:
        return 0  # Default: do nothing
    
    # Convert state to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    # Get action (with epsilon for exploration)
    eps = max(0.01, 0.995 ** _global_episode_count)
    action = _global_agent.act(state, eps)
    
    return action

def on_episode_end(episode, total_reward, agent_won):
    """Callback function when episode ends"""
    global _global_agent, _global_episode_count, _global_model_path
    _global_episode_count = episode
    
    print(f"Episode {episode} ended: reward={total_reward:.2f}, agent_won={agent_won}")
    
    # Save model after each episode
    if _global_agent is not None:
        torch.save({
            'state_dict': _global_agent.qnetwork_local.state_dict(),
            'episode': episode,
            'total_reward': total_reward,
        }, _global_model_path)
        print(f"Model saved to {_global_model_path}")

def initialize_agent(state_size=57, action_size=6, model_path=None):
    """Initialize the global agent"""
    global _global_agent, _global_model_path
    
    if model_path:
        _global_model_path = model_path
    
    # Force CPU for compatibility
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    _global_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=5e-4,
        gamma=0.99,
        tau=1e-3,
        buffer_size=100000,
        batch_size=64,
        update_every=4
    )
    
    # Load existing model if available
    if os.path.exists(_global_model_path):
        try:
            checkpoint = torch.load(_global_model_path, map_location='cpu')
            _global_agent.qnetwork_local.load_state_dict(checkpoint['state_dict'])
            _global_agent.qnetwork_target.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded model from {_global_model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    return _global_agent

# Export functions for C++ binding
__all__ = ['get_action_from_state', 'on_episode_end', 'initialize_agent']

