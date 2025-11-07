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
_prev_state = None
_prev_action = None
_episode_rewards = []
_episode_step_count = 0

def on_step(prev_state, prev_action, reward, next_state, done):
    """Callback for each step - store experience in replay buffer"""
    global _global_agent, _episode_rewards
    if _global_agent is None:
        return
    
    # Add experience to replay buffer
    _global_agent.step(prev_state, prev_action, reward, next_state, done)
    _episode_rewards.append(reward)

def get_action_from_state(state):
    """Callback function for C++ to get action from agent
    
    This function is called by C++ agentLoop every 50ms.
    """
    global _global_agent, _global_episode_count, _prev_state, _prev_action, _episode_step_count
    if _global_agent is None:
        return 0  # Default: do nothing
    
    # Improved epsilon decay: slower decay for better exploration on random maps
    # Start at 1.0, decay to 0.05 over 500 episodes, then to 0.01 over 1000 episodes
    if _global_episode_count < 500:
        eps = 1.0 - 0.95 * (_global_episode_count / 500.0)  # 1.0 -> 0.05
    elif _global_episode_count < 1000:
        eps = 0.05 - 0.04 * ((_global_episode_count - 500) / 500.0)  # 0.05 -> 0.01
    else:
        eps = 0.01  # minimum exploration
    
    action = _global_agent.act(state, eps)
    
    # Store current state and action for next step
    _prev_state = state
    _prev_action = action
    _episode_step_count += 1
    
    return action

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
    
    # Reset episode tracking
    _prev_state = None
    _prev_action = None
    _episode_rewards = []
    _episode_step_count = 0
    
    # Save model periodically (every 10 episodes) to avoid I/O overhead
    if _global_agent is not None and episode % 10 == 0:
        torch.save({
            'state_dict': _global_agent.qnetwork_local.state_dict(),
            'episode': episode,
            'agent_won': agent_won,
        }, _global_model_path)
        print(f"[Model saved to {_global_model_path}]")

def initialize_agent(state_size=122, action_size=6, model_path=None):  # Updated default state size
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
__all__ = ['get_action_from_state', 'on_episode_end', 'on_step', 'initialize_agent']

