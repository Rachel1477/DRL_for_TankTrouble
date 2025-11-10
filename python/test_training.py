"""
Quick test to verify training script can run (short training run)
"""
import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from train_dqn import train, DQNAgent, QNetwork, ReplayBuffer
    import tank_trouble_env as tte
    print("✓ Successfully imported all training modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    exit(1)


def test_agent_creation():
    """Test that agent can be created"""
    print("\n=== Testing Agent Creation ===")
    
    try:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
        env = tte.TankEnv()
        state_size = len(env.reset())
        action_size = 6
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            lr=5e-4,
            gamma=0.99
        )
        print(f"✓ Agent created successfully")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        print(f"  Device: {agent.device}")
        
        return True
    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_act():
    """Test agent action selection"""
    print("\n=== Testing Agent Action Selection ===")
    
    try:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
        env = tte.TankEnv()
        state_size = len(env.reset())
        action_size = 6
        
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        
        state = env.reset()
        action = agent.act(state, eps=0.0)  # No exploration
        print(f"✓ Agent selected action: {action}")
        assert 0 <= action < action_size, f"Action {action} out of range [0, {action_size})"
        
        # Test with exploration
        actions = [agent.act(state, eps=1.0) for _ in range(10)]
        print(f"✓ Agent actions with exploration: {actions}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing agent actions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_replay_buffer():
    """Test replay buffer"""
    print("\n=== Testing Replay Buffer ===")
    
    try:
        buffer = ReplayBuffer(action_size=6, buffer_size=100, batch_size=32)
        
        # Add some experiences
        for i in range(50):
            state = [0.1] * 56  # Example state size
            action = i % 6
            reward = 0.1
            next_state = [0.2] * 56
            done = False
            buffer.add(state, action, reward, next_state, done)
        
        print(f"✓ Added 50 experiences to buffer")
        print(f"  Buffer size: {len(buffer)}")
        
        # Sample a batch
        if len(buffer) >= buffer.batch_size:
            batch = buffer.sample()
            print(f"✓ Sampled batch: {len(batch[0])} experiences")
        
        return True
    except Exception as e:
        print(f"✗ Error testing replay buffer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_short_training():
    """Test a very short training run"""
    print("\n=== Testing Short Training Run ===")
    
    try:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
        print("Running 5 episodes...")
        scores = train(
            n_episodes=5,
            max_t=50,
            eps_start=1.0,
            eps_end=0.9,
            eps_decay=0.95,
            print_every=5
        )
        print(f"✓ Training completed successfully")
        print(f"  Scores: {scores}")
        return True
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 50)
    print("Training Components Test Suite")
    print("=" * 50)
    
    results = []
    results.append(("Agent Creation", test_agent_creation()))
    results.append(("Agent Action Selection", test_agent_act()))
    results.append(("Replay Buffer", test_replay_buffer()))
    results.append(("Short Training Run", test_short_training()))
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All training component tests passed!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        exit(1)

