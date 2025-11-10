import math
import json
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import tank_trouble_env as tte
except ImportError:
    raise RuntimeError("tank_trouble_env not found. Build with pybind11 and install via pip or set PYTHONPATH to build dir.")


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    batch_size: int = 1024
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_steps: int = 200000
    rollout_steps: int = 2048
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden = 128
        self.pi = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


def compute_gae(rewards, values, dones, gamma, lam):
    adv = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[t + 1] if t + 1 < len(rewards) else 1.0
        nextvalue = values[t + 1] if t + 1 < len(values) else values[t]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def rollout(env, policy, cfg: PPOConfig, device):
    s = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    states, actions, logps, rewards, dones, values = [], [], [], [], [], []
    for _ in range(cfg.rollout_steps):
        logits, v = policy(s.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()[0]
        logp = dist.log_prob(a)
        ns, r, done = env.step(int(a.item()))
        states.append(s)
        actions.append(a.detach())
        logps.append(logp.detach())
        rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
        dones.append(torch.tensor(float(done), dtype=torch.float32, device=device))
        values.append(v.squeeze(0).detach())
        if done:
            ns = env.reset()
        s = torch.tensor(ns, dtype=torch.float32, device=device)
    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(logps),
        torch.stack(rewards),
        torch.stack(dones),
        torch.stack(values),
    )


def train(cfg: PPOConfig, reward_json: str | None = None):
    env = tte.TankEnv()
    if reward_json:
        with open(reward_json, "r", encoding="utf-8") as f:
            env.set_reward_config(json.load(f))
    state_dim = len(env.reset())
    action_dim = 6
    device = torch.device(cfg.device)

    policy = ActorCritic(state_dim, action_dim).to(device)
    optim_ = optim.Adam(policy.parameters(), lr=cfg.lr)

    total_steps = 0
    while total_steps < cfg.max_steps:
        states, actions, logps_old, rewards, dones, values = rollout(env, policy, cfg, device)
        with torch.no_grad():
            adv, returns = compute_gae(rewards, values, dones, cfg.gamma, cfg.lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dataset_size = states.size(0)
        idxs = torch.randperm(dataset_size)

        for _ in range(cfg.epochs):
            for start in range(0, dataset_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = idxs[start:end]
                mb_s = states[mb_idx]
                mb_a = actions[mb_idx]
                mb_logp_old = logps_old[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = returns[mb_idx]

                logits, v = policy(mb_s)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_a)
                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (mb_ret - v).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                optim_.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim_.step()

        total_steps += dataset_size
        print(f"steps={total_steps} avgR={rewards.mean().item():.3f} V={values.mean().item():.3f}")


if __name__ == "__main__":
    cfg = PPOConfig()
    train(cfg, reward_json=None)


