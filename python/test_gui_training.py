# Test training using RLController (no GUI)
import os
import sys
import time

# Ensure build dir on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'build')
sys.path.insert(0, BUILD)
sys.path.insert(0, os.path.join(ROOT, 'python'))

import rl_controller  # type: ignore
from train_with_gui import initialize_agent, get_action_from_state, on_episode_end


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    initialize_agent(state_size=57, action_size=6, model_path=os.path.join(ROOT, 'checkpoint_dqn_gui.pth'))

    ctl = rl_controller.RLController()
    ctl.setGetActionCallback(get_action_from_state)
    ctl.setEpisodeEndCallback(on_episode_end)

    ctl.start()
    print('RLController started. Running for ~10 seconds...')
    time.sleep(10)
    print('Quitting...')
    ctl.quitGame()
    print('Done.')


if __name__ == '__main__':
    main()
