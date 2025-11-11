import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.base_agent import BaseAgent


# Policy Network
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Policy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, state):
        return self.net(state)


# Value Network (baseline)
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(ValueNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class ReinforceAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-3,
        gamma=0.99,
        entropy_coef=0.01,
        batch_episodes=5,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.batch_episodes = batch_episodes

        self.policy = Policy(state_dim, action_dim)
        self.value = ValueNet(state_dim)

        # Single optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )

        # Episode buffer for current episode
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_values = []

        # Batch buffer for multiple episodes
        self.batch_buffer = []

        # Running statistics for return normalization
        self.global_return_mean = 0.0
        self.global_return_std = 1.0
        self.return_alpha = 0.01  # Exponential moving average weight

    def select_action(self, state, greedy=False) -> int:
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        # Get policy logits and value estimate
        logits = self.policy(state_t)
        value = self.value(state_t)

        dist = Categorical(logits=logits)

        if greedy:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        # Store trajectory information
        self.episode_states.append(state)
        self.episode_actions.append(action.item())
        self.episode_log_probs.append(log_prob)
        self.episode_values.append(value.detach())

        return int(action.item())

    def store_reward(self, reward):
        self.episode_rewards.append(float(reward))

    def _compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.append(R)
        returns.reverse()
        return returns

    def _update_running_stats(self, returns):
        """Update running mean and std of returns."""
        batch_mean = float(np.mean(returns))
        batch_std = float(np.std(returns) + 1e-8)

        self.global_return_mean = (
            1 - self.return_alpha
        ) * self.global_return_mean + self.return_alpha * batch_mean
        self.global_return_std = (
            1 - self.return_alpha
        ) * self.global_return_std + self.return_alpha * batch_std

    def end_episode(self):
        """Called at the end of each episode to store it in batch buffer."""
        if self.episode_rewards:
            self.batch_buffer.append(
                {
                    "states": self.episode_states.copy(),
                    "actions": self.episode_actions.copy(),
                    "log_probs": self.episode_log_probs.copy(),
                    "rewards": self.episode_rewards.copy(),
                    "values": self.episode_values.copy(),
                }
            )

        # Clear episode buffers
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_log_probs.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()

        # Learn if we have enough episodes in batch
        if len(self.batch_buffer) >= self.batch_episodes:
            self.learn()

    def learn(self):
        if not self.batch_buffer:
            return

        all_log_probs = []
        all_advantages = []
        all_values = []
        all_returns = []
        all_states = []

        # Process each episode in the batch
        for episode in self.batch_buffer:
            returns = self._compute_returns(episode["rewards"])
            self._update_running_stats(returns)

            # Normalize returns using running statistics
            norm_returns = [
                (g - self.global_return_mean) / math.sqrt(self.global_return_std)
                for g in returns
            ]

            # Compute advantages
            for i in range(len(episode["rewards"])):
                advantage = norm_returns[i] - episode["values"][i].item()
                all_advantages.append(advantage)
                all_log_probs.append(episode["log_probs"][i])
                all_values.append(episode["values"][i])
                all_returns.append(norm_returns[i])
                all_states.append(episode["states"][i])

        # Convert to tensors
        advantages_t = torch.tensor(all_advantages, dtype=torch.float32)
        log_probs_t = torch.stack(all_log_probs)
        values_t = torch.stack(all_values)
        returns_t = torch.tensor(all_returns, dtype=torch.float32)

        # Compute entropy for exploration
        states_t = torch.tensor(np.array(all_states), dtype=torch.float32)
        logits = self.policy(states_t)
        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()

        # Policy loss (REINFORCE with advantage)
        policy_loss = -(log_probs_t * advantages_t).mean()

        # Value loss (MSE between value predictions and returns)
        # Squeeze values to match returns shape
        values_squeezed = values_t.squeeze()
        value_loss = nn.functional.mse_loss(values_squeezed, returns_t)

        # Total loss with entropy regularization
        loss = policy_loss + value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        # Clear batch buffer
        self.batch_buffer.clear()

    def save(self, path) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "return_mean": self.global_return_mean,
                "return_std": self.global_return_std,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])
        self.global_return_mean = checkpoint.get("return_mean", 0.0)
        self.global_return_std = checkpoint.get("return_std", 1.0)


def train(
    env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
):
    agent = ReinforceAgent(
        state_dim, action_dim, lr=3e-3, gamma=0.99, entropy_coef=0.01, batch_episodes=5
    )

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting REINFORCE training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(float(reward))

            state = next_state
            episode_reward += float(reward)

            if done:
                break

        # End episode and potentially learn
        agent.end_episode()

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")

        if np.mean(scores_deque) >= target_score:
            avg_score = np.mean(scores_deque)
            print(
                f"\nEnvironment solved in {episode} episodes! "
                f"Average Score: {avg_score:.2f}"
            )
            break

    # Final batch update if needed
    if agent.batch_buffer:
        agent.learn()

    print("\nTraining complete.")

    return agent, scores
