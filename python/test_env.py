"""
Test script to verify TankEnv bindings and basic functionality
"""
import numpy as np

try:
    import tank_trouble_env as tte
    print("✓ Successfully imported tank_trouble_env module")
except ImportError as e:
    print(f"✗ Failed to import tank_trouble_env: {e}")
    print("Make sure to:")
    print("  1. Build the module: cd build && cmake .. && make")
    print("  2. Set PYTHONPATH: export PYTHONPATH=$PWD/build:$PYTHONPATH")
    exit(1)


def test_env_basic():
    """Test basic environment functionality"""
    print("\n=== Testing Basic Environment ===")
    
    try:
        # Create environment
        env = tte.TankEnv()
        print("✓ Environment created successfully")
        
        # Test reset
        state = env.reset()
        print(f"✓ Reset successful, state shape: {len(state)}")
        print(f"  State sample (first 10 values): {state[:10]}")
        
        # Test step
        action = tte.Action.MOVE_FORWARD
        next_state, reward, done = env.step(action)
        print(f"✓ Step successful")
        print(f"  Next state shape: {len(next_state)}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        
        # Test all actions
        print("\n=== Testing All Actions ===")
        actions = [
            tte.Action.DO_NOTHING,
            tte.Action.MOVE_FORWARD,
            tte.Action.MOVE_BACKWARD,
            tte.Action.ROTATE_CW,
            tte.Action.ROTATE_CCW,
            tte.Action.SHOOT
        ]
        
        for action in actions:
            state = env.reset()
            next_state, reward, done = env.step(action)
            print(f"✓ Action {action}: reward={reward:.3f}, done={done}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_episodes():
    """Test running multiple episodes"""
    print("\n=== Testing Multiple Episodes ===")
    
    try:
        env = tte.TankEnv()
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(5):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 100:
                # Random action
                action = np.random.randint(0, 6)
                next_state, reward, done = env.step(action)
                total_reward += reward
                steps += 1
                state = next_state
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: reward={total_reward:.2f}, steps={steps}, done={done}")
        
        print(f"\nAverage reward: {np.mean(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.2f}")
        print("✓ Multiple episodes test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during multiple episodes test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_consistency():
    """Test state consistency"""
    print("\n=== Testing State Consistency ===")
    
    try:
        env = tte.TankEnv()
        state = env.reset()
        
        # State should be a vector of floats
        assert isinstance(state, list) or isinstance(state, tuple), "State should be list or tuple"
        assert len(state) > 0, "State should not be empty"
        assert all(isinstance(x, (int, float)) for x in state), "State should contain numbers"
        
        print(f"✓ State is valid: length={len(state)}, type={type(state)}")
        print(f"  State range: [{min(state):.3f}, {max(state):.3f}]")
        
        # Test that state changes after action
        next_state, _, _ = env.step(tte.Action.MOVE_FORWARD)
        # Handle case where state lengths might differ initially
        if len(state) == len(next_state):
            state_changed = not np.allclose(state, next_state, atol=1e-6)
        else:
            state_changed = True  # Different lengths means state changed
        print(f"✓ State changes after action: {state_changed} (state len: {len(state)} -> {len(next_state)})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during state consistency test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 50)
    print("TankTrouble Environment Test Suite")
    print("=" * 50)
    
    results = []
    results.append(("Basic Environment", test_env_basic()))
    results.append(("State Consistency", test_state_consistency()))
    results.append(("Multiple Episodes", test_multiple_episodes()))
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed! Environment is ready for training.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        exit(1)

