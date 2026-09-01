"""
ddpg_agent.py
-------------
From-scratch implementation of Deep Deterministic Policy Gradient (DDPG),
used as the AGENT that solves the portfolio allocation MDP defined in
env/portfolio_env.py.

ATTRIBUTION: this implementation follows the algorithm and default
hyperparameter choices described in:
    Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T.,
    Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control
    with deep reinforcement learning. ICLR.
    https://arxiv.org/abs/1509.02971
No third-party RL library (example Stable-Baselines3) is used; the actor,
critic, replay buffer, target-network updates, and exploration process
below are implemented directly from that paper's algorithm description
(Algorithm 1) and Section 7's hyperparameter table, adapted for this
project's continuous portfolio-weight action space.

Components:
    - Actor network        : deterministic policy  mu(s) -> a
    - Critic network        : action-value function Q(s, a)
    - Target networks       : slow-moving copies of actor & critic for
                               stable TD targets (soft update, tau)
    - Replay buffer          : off-policy experience replay
    - Ornstein-Uhlenbeck noise: temporally-correlated exploration noise
                               added to the deterministic policy's actions
"""
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], action_dim), nn.Tanh(),
        )
        self.action_scale = 5.0  # matches env action_space bound

    def forward(self, state):
        return self.net(state) * self.action_scale


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


# ---------------------------------------------------------------------
class OUNoise:
    """Ornstein-Uhlenbeck process for temporally-correlated exploration noise.

    Parameterization (theta, sigma, dt defaults) follows Lillicrap et al.
    (2016), Section 7, "Ornstein-Uhlenbeck process with theta = 0.15 and
    sigma = 0.2" (sigma is set per-project below via noise_sigma and
    annealed over training rather than held fixed; see DDPGAgent).
    """

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.3, dt=1e-2, seed=0):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(self.action_dim)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    def __init__(self, capacity=100_000, seed=0):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------
class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=100_000,
        batch_size=128,
        noise_sigma=0.3,
        seed=0,
    ):
        torch.manual_seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_size, seed=seed)
        self.noise = OUNoise(action_dim, sigma=noise_sigma, seed=seed)

    def act(self, state, explore=True):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        if explore:
            action = action + self.noise.sample()
        return action

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def _soft_update(self, source, target):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None, None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=DEVICE)
        a = torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
        r = torch.as_tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=DEVICE)
        d = torch.as_tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # --- critic update ---
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target_next = self.critic_target(s2, a2)
            q_target = r + self.gamma * (1 - d) * q_target_next
        q_pred = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # --- actor update (deterministic policy gradient) ---
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # --- soft update targets ---
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

        return critic_loss.item(), actor_loss.item()
