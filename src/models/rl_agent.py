"""Deep Q-Network (DQN) reinforcement learning agent for trading decisions.

Uses PyTorch if available, falls back to a numpy-based neural network.

The agent learns to make Buy / Hold / Sell decisions based on:
- Current model predictions and confidence
- Portfolio state (position, unrealized P&L)
- Market features (volatility, volume ratio, etc.)

It self-corrects through experience replay: bad trades produce negative
rewards, and the network learns to avoid similar situations.
"""

import random
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Actions
ACTION_SELL = 0
ACTION_HOLD = 1
ACTION_BUY = 2
ACTION_NAMES = {ACTION_SELL: "SELL", ACTION_HOLD: "HOLD", ACTION_BUY: "BUY"}


# ── Numpy-based DQN (fallback when torch unavailable) ────────────────

class _NumpyDQN:
    """Minimal 3-layer neural network using only numpy."""

    def __init__(self, state_size, action_size=3, lr=1e-3):
        self.lr = lr
        scale1 = np.sqrt(2.0 / state_size)
        scale2 = np.sqrt(2.0 / 128)
        scale3 = np.sqrt(2.0 / 64)
        self.w1 = np.random.randn(state_size, 128) * scale1
        self.b1 = np.zeros(128)
        self.w2 = np.random.randn(128, 64) * scale2
        self.b2 = np.zeros(64)
        self.w3 = np.random.randn(64, action_size) * scale3
        self.b3 = np.zeros(action_size)

    def forward(self, x):
        self._z1 = x @ self.w1 + self.b1
        self._a1 = np.maximum(0, self._z1)
        self._z2 = self._a1 @ self.w2 + self.b2
        self._a2 = np.maximum(0, self._z2)
        self._out = self._a2 @ self.w3 + self.b3
        self._input = x
        return self._out

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    def train_step(self, states, targets, actions):
        """One gradient step. Returns loss."""
        q_all = self.forward(states)
        q_pred = q_all[np.arange(len(actions)), actions]
        loss = float(np.mean((q_pred - targets) ** 2))

        # Backward pass
        dout = np.zeros_like(q_all)
        dout[np.arange(len(actions)), actions] = 2 * (q_pred - targets) / len(actions)

        # Layer 3
        dw3 = self._a2.T @ dout
        db3 = dout.sum(axis=0)
        da2 = dout @ self.w3.T

        # Layer 2
        da2 = da2 * (self._z2 > 0)
        dw2 = self._a1.T @ da2
        db2 = da2.sum(axis=0)
        da1 = da2 @ self.w2.T

        # Layer 1
        da1 = da1 * (self._z1 > 0)
        dw1 = self._input.T @ da1
        db1 = da1.sum(axis=0)

        # Clip gradients
        for g in [dw1, dw2, dw3, db1, db2, db3]:
            np.clip(g, -1, 1, out=g)

        # Update
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3

        return loss

    def copy_from(self, other):
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()
        self.w3 = other.w3.copy()
        self.b3 = other.b3.copy()


# ── PyTorch DQN ───────────────────────────────────────────────────────

if HAS_TORCH:
    class _TorchDQN(nn.Module):
        def __init__(self, state_size, action_size=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, action_size),
            )

        def forward(self, x):
            return self.net(x)


# ── Trading Environment ──────────────────────────────────────────────

class TradingEnvironment:
    """Simulated intraday trading environment."""

    def __init__(self, prices, predictions, confidences, features,
                 initial_capital=100_000, transaction_cost=0.001):
        self.prices = prices
        self.predictions = predictions
        self.confidences = confidences
        self.features = features
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        self.position = 0
        self.capital = initial_capital
        self.entry_price = 0.0
        self.step_idx = 0
        self.trades: list[dict] = []

    @property
    def state_size(self):
        return 4 + self.features.shape[1]

    def reset(self):
        self.position = 0
        self.capital = self.initial_capital
        self.entry_price = 0.0
        self.step_idx = 0
        self.trades = []
        return self._get_state()

    def step(self, action):
        price = self.prices[self.step_idx]
        reward = 0.0

        if action == ACTION_BUY and self.position <= 0:
            if self.position == -1:
                pnl = (self.entry_price - price) / self.entry_price - self.transaction_cost
                reward = pnl * 100
                self.capital *= 1 + pnl
                self.trades.append({"type": "close_short", "price": price, "pnl": pnl})
            self.position = 1
            self.entry_price = price
            self.trades.append({"type": "buy", "price": price, "pnl": 0})

        elif action == ACTION_SELL and self.position >= 0:
            if self.position == 1:
                pnl = (price - self.entry_price) / self.entry_price - self.transaction_cost
                reward = pnl * 100
                self.capital *= 1 + pnl
                self.trades.append({"type": "close_long", "price": price, "pnl": pnl})
            self.position = -1
            self.entry_price = price
            self.trades.append({"type": "sell", "price": price, "pnl": 0})

        elif action == ACTION_HOLD:
            reward = -0.01

        self.step_idx += 1
        done = self.step_idx >= len(self.prices) - 1

        if done and self.position != 0:
            price = self.prices[self.step_idx]
            if self.position == 1:
                pnl = (price - self.entry_price) / self.entry_price - self.transaction_cost
            else:
                pnl = (self.entry_price - price) / self.entry_price - self.transaction_cost
            reward += pnl * 100
            self.capital *= 1 + pnl
            self.position = 0

        return self._get_state(), reward, done

    def _get_state(self):
        idx = min(self.step_idx, len(self.prices) - 1)
        price = self.prices[idx]
        unrealized_pnl = 0.0
        if self.position == 1:
            unrealized_pnl = (price - self.entry_price) / self.entry_price
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - price) / self.entry_price

        feat_idx = min(idx, len(self.features) - 1)
        state = np.concatenate([
            [self.predictions[min(idx, len(self.predictions) - 1)]],
            [self.confidences[min(idx, len(self.confidences) - 1)]],
            [float(self.position)],
            [unrealized_pnl],
            self.features[feat_idx],
        ])
        return state.astype(np.float32)


# ── DQN Agent ─────────────────────────────────────────────────────────

class DQNAgent:
    """DQN agent that learns trading actions through experience replay.

    Self-correction happens via:
    1. Experience replay: replays past mistakes and learns to avoid them.
    2. Target network: stabilizes learning.
    3. Epsilon decay: shifts from exploration to exploitation.
    """

    def __init__(self, state_size, lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=10_000,
                 batch_size=64, target_update=10):
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size)
        self.train_metrics: dict = {}
        self._use_torch = HAS_TORCH

        if self._use_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = _TorchDQN(state_size).to(self.device)
            self.target_net = _TorchDQN(state_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        else:
            self.policy_net = _NumpyDQN(state_size, lr=lr)
            self.target_net = _NumpyDQN(state_size, lr=lr)
            self.target_net.copy_from(self.policy_net)

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, 2)
        if self._use_torch:
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(t).argmax(dim=1).item()
        else:
            q = self.policy_net.predict(state)
            return int(np.argmax(q))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(list(self.memory), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones)

        if self._use_torch:
            return self._replay_torch(states, actions, rewards, next_states, dones)
        return self._replay_numpy(states, actions, rewards, next_states, dones)

    def _replay_numpy(self, states, actions, rewards, next_states, dones):
        # Double DQN target
        next_q_policy = self.policy_net.predict(next_states)
        next_actions = np.argmax(next_q_policy, axis=1)
        next_q_target = self.target_net.predict(next_states)
        next_q = next_q_target[np.arange(len(next_actions)), next_actions]
        next_q[dones] = 0.0
        targets = rewards + self.gamma * next_q

        return self.policy_net.train_step(states, targets, actions)

    def _replay_torch(self, states, actions, rewards, next_states, dones):
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze()

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze()
            next_q[dones_t] = 0.0
            target_q = rewards_t + self.gamma * next_q

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train_on_env(self, env, episodes=100):
        episode_rewards = []
        episode_capitals = []
        losses = []

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                loss = self.replay()
                if loss > 0:
                    losses.append(loss)

            episode_rewards.append(total_reward)
            episode_capitals.append(env.capital)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (ep + 1) % self.target_update == 0:
                if self._use_torch:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                else:
                    self.target_net.copy_from(self.policy_net)

        self.train_metrics = {
            "episode_rewards": episode_rewards,
            "episode_capitals": episode_capitals,
            "avg_reward": float(np.mean(episode_rewards[-20:])),
            "avg_capital": float(np.mean(episode_capitals[-20:])),
            "final_epsilon": self.epsilon,
            "avg_loss": float(np.mean(losses[-100:])) if losses else 0.0,
            "total_episodes": episodes,
        }
        return self.train_metrics

    def get_action_name(self, action):
        return ACTION_NAMES.get(action, "UNKNOWN")

    def get_q_values(self, state):
        if self._use_torch:
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(t).cpu().numpy().flatten()
        return self.policy_net.predict(state).flatten()
