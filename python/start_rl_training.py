"""
Start RL training with GUI
This script should be run to start the training mode
"""
import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
sys.path.insert(0, build_dir)

try:
    import rl_controller
    from train_with_gui import initialize_agent, get_action_from_state, on_episode_end
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure to:")
    print("1. Build the project: cd build && cmake .. && make")
    print("2. Set PYTHONPATH to include build directory")
    sys.exit(1)

def main():
    """Main function to start RL training"""
    print("Initializing RL Agent...")
    
    # Initialize agent
    agent = initialize_agent(state_size=57, action_size=6)
    
    print("RL Agent initialized. Training will start when GUI is opened.")
    print("Click 'Agent训练' button in the game to start training.")
    
    # Note: The actual integration with GUI will be done through
    # the C++ code calling Python functions via callbacks
    # This is a placeholder for the integration

if __name__ == '__main__':
    main()

