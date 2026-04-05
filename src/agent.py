import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

# ── Neural Network ──────────────────────────────────────────
class DQNNetwork(nn.Module):
    """
    The brain of our agent.
    Takes the game screen as input and outputs Q-values for each action.
    """
    def __init__(self, input_shape, n_actions):
        super(DQNNetwork, self).__init__()

        # CNN layers — reads the game screen like eyes
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate size after CNN
        conv_out = self._get_conv_out(input_shape)

        # Fully connected layers — makes decisions
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0   # normalize pixel values
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


# ── Replay Buffer ────────────────────────────────────────────
class ReplayBuffer:
    """
    Memory of the agent.
    Stores past experiences so the agent can learn from them later.
    Like a diary of everything that happened in the game.
    """
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ── DDQN Agent ───────────────────────────────────────────────
class DDQNAgent:
    """
    The full agent that:
    - Looks at the screen
    - Decides what to do
    - Learns from mistakes
    - Gets better over time
    """
    def __init__(self, input_shape, n_actions, lr=0.0001):
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Two networks — online learns, target stays stable
        self.online_net = DQNNetwork(input_shape, n_actions).to(self.device)
        self.target_net = DQNNetwork(input_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        # Exploration — starts random, gets smarter over time
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.target_update = 1000
        self.steps = 0

    def select_action(self, state):
        """Explore randomly OR exploit what we know"""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).argmax().item()

    def learn(self):
        """Learn from a batch of past experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1))

        # DDQN target Q values
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1))
            target_q = rewards_t.unsqueeze(1) + self.gamma * next_q * (1 - dones_t.unsqueeze(1))

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")