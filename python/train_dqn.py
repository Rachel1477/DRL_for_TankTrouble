"""
Double DQN Training Script for TankTrouble
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple
import os

try:
    import tank_trouble_env as tte
    print(f"[DEBUG] tank_trouble_env path: {tte.__file__}")
    # 检查 TankEnv().reset() 的 state 长度
    env_debug = tte.TankEnv()
    state_debug = env_debug.reset()
    print(f"[DEBUG] env.reset() state size: {len(state_debug)}")
except ImportError:
    raise RuntimeError(
        "tank_trouble_env not found. Build with pybind11 and install via pip or set PYTHONPATH to build dir."
    )

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    """Enhanced Q-Network for DQN with larger capacity for complex state space"""
    
    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
    # Deeper network to handle complex state (90 dim: 9 base + 4 grid pos + 3 path + 25 map grid + 1 LOS + 48 ray)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        
    def forward(self, state):
        # Handle both single state and batch
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Dual replay buffer: mix recent and long-term experiences for sampling

    - Maintains two buffers:
      * recent_memory: a small buffer emphasizing the most recent transitions
      * main_memory: a large buffer storing long-term transitions
    - Sampling draws a fraction from recent_memory and the rest from main_memory.
    """

    def __init__(
        self,
        action_size: int,
        buffer_size: int = 100000,
        batch_size: int = 64,
        seed: int = 0,
        recent_buffer_size: int = 10000,
        recent_fraction: float = 0.5,
    ):
        self.action_size = action_size
        self.main_memory = deque(maxlen=buffer_size)
        self.recent_memory = deque(maxlen=recent_buffer_size)
        self.batch_size = batch_size
        # Clamp recent_fraction to [0.0, 1.0]
        self.recent_fraction = max(0.0, min(1.0, recent_fraction))
        self.experience = Experience
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to both recent and main buffers"""
        e = self.experience(state, action, reward, next_state, done)
        self.main_memory.append(e)
        self.recent_memory.append(e)

    def sample(self):
        """Sample a mixed batch from recent and main buffers"""
        # Determine how many to draw from recent buffer
        want_recent = int(self.batch_size * self.recent_fraction)
        have_recent = len(self.recent_memory)
        k_recent = min(want_recent, have_recent)

        # Remaining from main buffer
        k_main = self.batch_size - k_recent
        # Ensure main buffer has enough samples (caller checks len(main) > batch_size)
        have_main = len(self.main_memory)
        if have_main < k_main:
            # Fallback: reduce recent portion and adjust main portion
            # This should rarely happen because we gate on __len__ > batch_size
            k_main = have_main
            k_recent = self.batch_size - k_main
            if k_recent > have_recent:
                # As a last resort, cap to available total
                k_recent = have_recent

        recent_samples = random.sample(self.recent_memory, k=k_recent) if k_recent > 0 else []
        main_samples = random.sample(self.main_memory, k=k_main) if k_main > 0 else []
        experiences = recent_samples + main_samples

        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of the main memory"""
        return len(self.main_memory)


class DQNAgent:
    """Double DQN Agent"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 0,
        lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 1e-3,
        buffer_size: int = 100000,
        batch_size: int = 64,
        update_every: int = 4
        , env=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        # Optional reference to the environment (used to query SmithAI when choosing random actions)
        self.env = env
        
        # Device - force CPU for compatibility
        # Use CUDA only if explicitly available and not busy
        if torch.cuda.is_available():
            try:
                # Test if CUDA is actually usable
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                self.device = torch.device("cuda")
            except:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Q-Network (local and target)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn"""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
    
    def act(self, state, eps: float = 0.01):
        """Returns actions for given state as per current policy with 1% shooting probability during random action selection
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon for epsilon-greedy action selection
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        # if random.random() > eps:  # follow the policy
        if False:  # follow the policy
            # Use the current Q-values to select the best action
            values_list = action_values.cpu().data.tolist()[0]  # 修正：取第一个 batch
            selected_action = values_list.index(max(values_list))
            print(f"[PY-DEBUG] Chose action {selected_action} via policy")
        else:  # 1% chance select random action
            # 优先用agent视角SmithAI，否则fallback到随机动作
            try:
                if self.env is not None and hasattr(self.env, 'get_agent_smith_action'):
                    smith_choice = int(self.env.get_agent_smith_action())
                    print(f"[PY-DEBUG] get_agent_smith_action returned {smith_choice}")
                    if 0 <= smith_choice < self.action_size:
                        selected_action = smith_choice
                    else:
                        selected_action = random.choice(np.arange(self.action_size))
                        print("[PY-DEBUG] AgentSmithAI returned invalid action, chose random action.")
                else:
                    selected_action = random.choice(np.arange(self.action_size))
                    print("[PY-DEBUG] Chose random action (no AgentSmithAI available).")
            except Exception as e:
                selected_action = random.choice(np.arange(self.action_size))
                print(f"[PY-DEBUG] Failed to query AgentSmithAI, selected random action. Exception: {e}")

        print(f"[PY-DEBUG] DQNAgent.act returning action={selected_action}")
        return selected_action

    
    def learn(self, experiences: Tuple, gamma: float):
        """Update value parameters using given batch of experience tuples
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get max predicted Q values (for next states) from target model
        # Double DQN: use local network to select actions, target network to evaluate
        Q_targets_next = self.qnetwork_target(next_states).detach()
        _, best_actions = self.qnetwork_local(next_states).detach().max(1, keepdim=True)
        Q_targets_next = Q_targets_next.gather(1, best_actions).squeeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        
        # Update target network (soft update)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def load_model(model_path: str, device=None):
    """Load a trained DQN model from file
    
    Params
    ======
        model_path (str): path to the saved model file
        device: torch device to load model on (default: auto-detect)
    
    Returns
    =======
        agent (DQNAgent): loaded agent
        info (dict): model information (episode, avg_score, etc.)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    state_size = checkpoint['state_size']
    action_size = checkpoint['action_size']
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=5e-4,
        gamma=0.99,
        tau=1e-3,
        buffer_size=100000,
        batch_size=64,
        update_every=4
    )
    
    agent.qnetwork_local.load_state_dict(checkpoint['state_dict'])
    agent.qnetwork_target.load_state_dict(checkpoint['state_dict'])
    
    info = {
        'episode': checkpoint.get('episode', 'unknown'),
        'avg_score': checkpoint.get('avg_score', 'unknown'),
        'epsilon': checkpoint.get('epsilon', 'unknown'),
        'scores': checkpoint.get('scores', None),
    }
    
    print(f"Model loaded from {model_path}")
    print(f"  Episode: {info['episode']}, Avg Score: {info['avg_score']}, Epsilon: {info['epsilon']}")
    
    return agent, info


def train(
    n_episodes: int = 2000,
    max_t: int = 1000,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.9999,
    print_every: int = 10,
    load_checkpoint: str = None  # Path to checkpoint to resume training
):
    """Train the DQN agent
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        print_every (int): how often to print training progress
    """
    # Create environment
    env = tte.TankEnv()
    state_size = len(env.reset())
    action_size = 6  # DO_NOTHING, MOVE_FORWARD, MOVE_BACKWARD, ROTATE_CW, ROTATE_CCW, SHOOT
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create agent (force CPU if CUDA issues)
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU for testing
    
    # Load checkpoint if provided
    start_episode = 1
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"Loading checkpoint from {load_checkpoint}...")
        agent, checkpoint_info = load_model(load_checkpoint)
        start_episode = checkpoint_info.get('episode', 1) + 1
        print(f"Resuming training from episode {start_episode}")
        # Attach environment reference so agent can query SmithAI for random actions
        agent.env = env
    else:
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            lr=1e-3,  # 提高学习率，加快学习速度
            gamma=0.99,
            tau=1e-3,
            buffer_size=100000,
            batch_size=64,
            update_every=4
        )
        agent.env = env
    
    # Training statistics
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    best_avg_score = -np.inf  # Track best average score
    model_save_path = 'checkpoint_dqn.pth'
    final_model_path = 'final_dqn.pth'
    
    for i_episode in range(start_episode, n_episodes + 1):
        state = env.reset()
        score = 0
        # Debug: 检查reset输出长度
        if len(state) != state_size:
            print(f"[ERROR] Episode {i_episode} reset state size mismatch: {len(state)} vs expected {state_size}")
        for t in range(max_t):
            # print(f"[PY-DEBUG] Episode {i_episode} Step {t} - state[:5]={state[:5] if hasattr(state, '__getitem__') else state}")
            # Select action
            action = agent.act(state, eps)
            # print(f"[PY-DEBUG] Episode {i_episode} Step {t} - action={action}")
            # Take step in environment
            try:
                next_state, reward, done = env.step(action)
                # print(f"[PY-DEBUG] Episode {i_episode} Step {t} - next_state[:5]={next_state[:5] if hasattr(next_state, '__getitem__') else next_state}, reward={reward}, done={done}")
            except Exception as e:
                print(f"[PY-DEBUG] Exception during env.step: {e}")
                raise
            # Debug: 检查step输出长度
            if len(next_state) != state_size:
                print(f"[ERROR] Episode {i_episode} step {t} next_state size mismatch: {len(next_state)} vs expected {state_size}")
            assert len(next_state) == state_size, f"next_state size {len(next_state)} != expected {state_size} at episode {i_episode} step {t}"
            # Agent step (store experience and learn)
            agent.step(state, action, reward, next_state, done)
            # print(f"[PY-DEBUG] Episode {i_episode} Step {t} - agent.step done")
            state = next_state
            score += reward
            if done:
                print(f"[PY-DEBUG] Episode {i_episode} finished after {t+1} steps, score={score}")
                break
        
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        
        # Print progress and save model if improved
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_window)
            print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {eps:.3f}')
            
            # Save model if average score improved (need at least 100 episodes for meaningful average)
            if len(scores_window) >= 10 and avg_score > best_avg_score:
                best_avg_score = avg_score
                torch.save({
                    'state_dict': agent.qnetwork_local.state_dict(),
                    'state_size': state_size,
                    'action_size': action_size,
                    'episode': i_episode,
                    'avg_score': avg_score,
                    'epsilon': eps,
                }, model_save_path)
                print(f'  -> Model saved to {model_save_path} (avg_score: {avg_score:.2f})')
    
    # Save final model at the end of training
    print(f'\nTraining completed! Saving final model...')
    final_avg_score = np.mean(scores_window) if len(scores_window) > 0 else 0.0
    torch.save({
        'state_dict': agent.qnetwork_local.state_dict(),
        'state_size': state_size,
        'action_size': action_size,
        'episode': n_episodes,
        'avg_score': final_avg_score,
        'epsilon': eps,
        'scores': scores,  # Save all scores for analysis
    }, final_model_path)
    print(f'Final model saved to {final_model_path}')
    print(f'Best average score: {best_avg_score:.2f}')
    print(f'Final average score: {final_avg_score:.2f}')
    
    return scores


if __name__ == '__main__':
    print("Starting Double DQN Training...")
    scores = train(
        n_episodes=500,  # 增加训练episode数
        max_t=1000,
        eps_start=1.0,
        eps_end=0.05,  # 提高最终epsilon，保持更多探索
        eps_decay=0.9999,  # 更慢的衰减，保持探索更长时间
        print_every=10
    )
    print("Training completed!")

