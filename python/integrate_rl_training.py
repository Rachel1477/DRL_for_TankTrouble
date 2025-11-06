"""
Python script to integrate RL training with GUI
This script should be imported and called from C++ to set up callbacks
"""
import sys
import os

# Add paths
build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.dirname(__file__))

try:
    import rl_controller
    from train_with_gui import initialize_agent, get_action_from_state, on_episode_end
except ImportError as e:
    print(f"Warning: Could not import RL modules: {e}")
    print("Training will use default random actions")

def setup_rl_controller(rl_ctl):
    """
    Set up callbacks for RLController
    
    Args:
        rl_ctl: RLController instance from C++
    """
    try:
        # Initialize agent
        agent = initialize_agent(state_size=57, action_size=6)
        print("RL Agent initialized for GUI training")
        
        # Set callbacks
        rl_ctl.setGetActionCallback(get_action_from_state)
        rl_ctl.setEpisodeEndCallback(on_episode_end)
        
        print("Callbacks set successfully")
        return True
    except Exception as e:
        print(f"Error setting up RL controller: {e}")
        import traceback
        traceback.print_exc()
        return False

# Export for C++ use
__all__ = ['setup_rl_controller']

