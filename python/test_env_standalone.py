#!/usr/bin/env python3
"""
Standalone test for tank_trouble_env module (no GUI).
Tests agent vs SmithAI training loop.
"""

import sys
import os
import time

# Ensure we can find the compiled module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cmake-build-debug'))

print("[Test] Importing tank_trouble_env module...")
try:
    import tank_trouble_env
    print("[Test] Successfully imported tank_trouble_env")
except ImportError as e:
    print(f"[Test] Failed to import tank_trouble_env: {e}")
    print("[Test] Make sure the module is compiled and in the build directory")
    sys.exit(1)

print("[Test] Importing train_with_gui for agent...")
try:
    from train_with_gui import DQNAgent
    print("[Test] Successfully imported DQNAgent")
except ImportError as e:
    print(f"[Test] Failed to import DQNAgent: {e}")
    print("[Test] Falling back to random actions")
    DQNAgent = None

def test_basic_env():
    """Test basic environment functionality"""
    print("\n" + "="*60)
    print("[Test] Creating TankEnv...")
    env = tank_trouble_env.TankEnv()
    
    print("[Test] Resetting environment...")
    state = env.reset()
    print(f"[Test] Initial state shape: {len(state)}, first 10 values: {state[:10]}")
    
    print("[Test] Running 20 random steps...")
    for step in range(20):
        action = step % 6  # Cycle through all actions
        next_state, reward, done = env.step(action)
        print(f"[Test] Step {step+1}: action={action}, reward={reward:.4f}, done={done}, state_size={len(next_state)}")
        
        if done:
            print(f"[Test] Episode ended at step {step+1}")
            state = env.reset()
            print(f"[Test] Reset environment, new state size: {len(state)}")
        else:
            state = next_state
    
    print("[Test] Basic environment test completed successfully!")
    return True

def test_agent_training():
    """Test agent training loop (no GUI)"""
    print("\n" + "="*60)
    print("[Test] Starting agent training test...")
    
    env = tank_trouble_env.TankEnv()
    state = env.reset()
    state_size = len(state)
    action_size = 6
    
    print(f"[Test] State size: {state_size}, Action size: {action_size}")
    
    if DQNAgent is not None:
        print("[Test] Creating DQN agent...")
        agent = DQNAgent(state_size, action_size)
        use_agent = True
    else:
        print("[Test] Using random policy (DQNAgent not available)")
        use_agent = False
    
    num_episodes = 3
    max_steps_per_episode = 100
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n[Test] Episode {episode}/{num_episodes} starting...")
        
        for step in range(max_steps_per_episode):
            if use_agent:
                action = agent.act(state, eps=0.1)  # 10% exploration
            else:
                import random
                action = random.randint(0, 5)
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            if use_agent:
                agent.step(state, action, reward, next_state, done)
            
            if step % 10 == 0:
                print(f"[Test]   Step {step}: action={action}, reward={reward:.4f}, total_reward={total_reward:.4f}")
            
            state = next_state
            
            if done:
                print(f"[Test] Episode {episode} finished after {steps} steps, total_reward: {total_reward:.4f}")
                break
        
        if not done:
            print(f"[Test] Episode {episode} reached max steps ({max_steps_per_episode}), total_reward: {total_reward:.4f}")
    
    print("\n[Test] Agent training test completed successfully!")
    if use_agent:
        model_path = "/tmp/test_agent_model.pth"
        import torch
        torch.save(agent.qnetwork_local.state_dict(), model_path)
        print(f"[Test] Saved test model to {model_path}")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("Tank Trouble RL Environment Standalone Test")
    print("="*60)
    
    try:
        # Test 1: Basic environment functionality
        if not test_basic_env():
            print("[Test] FAILED: Basic environment test")
            sys.exit(1)
        
        # Test 2: Agent training loop
        if not test_agent_training():
            print("[Test] FAILED: Agent training test")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("[Test] ALL TESTS PASSED!")
        print("="*60)
    except Exception as e:
        print(f"\n[Test] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

