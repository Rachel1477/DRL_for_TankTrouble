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
except ImportError:
    raise RuntimeError(
        "tank_trouble_env not found. Build with pybind11 and install via pip or set PYTHONPATH to build dir."
    )

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    """Q-Network for DQN"""
    
    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples"""
    
    def __init__(self, action_size: int, buffer_size: int = 100000, batch_size: int = 64, seed: int = 0):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


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
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        
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
    
    def act(self, state, eps: float = 0.0):
        """Returns actions for given state as per current policy
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            # Use .item() or convert to list to avoid NumPy compatibility issues
            action_values_cpu = action_values.cpu().data
            # Convert to Python list and find max
            values_list = action_values_cpu.tolist() if hasattr(action_values_cpu, 'tolist') else [v.item() for v in action_values_cpu]
            return values_list.index(max(values_list))
        else:
            return random.choice(np.arange(self.action_size))
    
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


def train(
    n_episodes: int = 2000,
    max_t: int = 1000,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
    print_every: int = 100
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
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU for testing
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
    
    # Training statistics
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        
        for t in range(max_t):
            # Select action
            action = agent.act(state, eps)
            
            # Take step in environment
            next_state, reward, done = env.step(action)
            
            # Agent step (store experience and learn)
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        
        # Print progress
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_window)
            print(f'\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {eps:.3f}')
        
        # Save model if average score improved
        if i_episode % print_every == 0 and len(scores_window) >= 100:
            avg_score = np.mean(scores_window)
            if avg_score >= np.max(scores_window) if len(scores_window) == 100 else False:
                model_path = 'checkpoint_dqn.pth'
                torch.save({
                    'state_dict': agent.qnetwork_local.state_dict(),
                    'state_size': state_size,
                    'action_size': action_size,
                }, model_path)
                print(f'Model saved to {model_path}')
    
    return scores


if __name__ == '__main__':
    print("Starting Double DQN Training...")
    scores = train(
        n_episodes=2000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        print_every=100
    )
    print("Training completed!")

