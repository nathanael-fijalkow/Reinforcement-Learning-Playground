from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.base_agent import BaseAgent


# Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, state):
        return self.net(state)


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


# Prioritized Replay Buffer
# Prioritized replay buffers determine how likely transitions are to be
# sampled based on their TD error (importance)
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.alpha = alpha  # normalization exponent for priorities
        self.beta = 0.4  # importance-sampling exponent
        # the weights of stored transitions
        self.priorities = np.zeros(capacity)
        self.pos = 0  # position to insert the next transition

    def push(self, *args):
        """Save a transition."""
        # New transitions get high priority but capped at 95th percentile
        # to avoid outliers dominating sampling
        current_len = len(self.memory)
        if current_len > 10:
            # Use 95th percentile to avoid outlier dominance
            max_p = float(np.percentile(self.priorities[:current_len], 95))
            max_p = max(max_p, 1.0)  # Ensure minimum priority of 1.0
        elif current_len > 0 and self.priorities[:current_len].max() > 0:
            max_p = float(self.priorities[:current_len].max())
        else:
            max_p = 1.0

        if current_len < self.capacity:
            self.memory.append(None)

        self.memory[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_p
        # Update position
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions based on their priorities.

        Returns (samples, indices, is_weights)
        """
        # If memory is empty, return empty lists
        if len(self.memory) == 0:
            return [], [], []

        # Get priorities, apply alpha and small epsilon to avoid zero-sum
        priorities = self.priorities[: len(self.memory)].astype(float)
        eps = 1e-6
        probs = (priorities + eps) ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / (probs_sum + 1e-8)  # Add epsilon for stability

        # Allow sampling with replacement if batch_size > current memory
        replace = batch_size > len(self.memory)
        indices = np.random.choice(
            len(self.memory), batch_size, p=probs, replace=replace
        )
        samples = [self.memory[i] for i in indices]

        # Importance-sampling weights
        N = len(self.memory)
        beta = self.beta
        sampling_probs = probs[indices]
        is_weights = np.power(N * sampling_probs, -beta)
        is_weights = is_weights / (is_weights.max() + 1e-8)
        
        return samples, indices, is_weights

    def update_priorities(self, indices, errors, epsilon=1e-5):
        """Update priorities of sampled transitions."""
        for i, error in zip(indices, errors):
            val = float(error)
            self.priorities[int(i)] = abs(val) + epsilon

    def __len__(self):
        return len(self.memory)


class DQNPrioritisedExpReplayAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=64,
        target_update=10,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        # copy weights from policy_net to target_net
        self.update_target_net()

        # freeze target_net parameters
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.steps_done = 0
        self.episodes_done = 0  # Track episodes for beta annealing

    def select_action(self, state, greedy=False):
        self.steps_done += 1
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            # Convert state (numpy array) to tensor
            # float() ensures it is a float tensor
            # unsqueeze(0) adds batch dimension
            state = torch.from_numpy(state).float().unsqueeze(0)

            # max(1) returns (value, index). take [1] for the index
            # item() returns the value as a Python number
            return self.policy_net(state).max(1)[1].item()

    def anneal_beta(self, max_episodes=1000):
        """Gradually increase beta from 0.4 to 1.0 over training."""
        beta_start = 0.4
        beta_end = 1.0
        progress = min(self.episodes_done / max_episodes, 1.0)
        self.memory.beta = beta_start + (beta_end - beta_start) * progress

    def learn(self):
        # Do not learn if not enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        
        # Anneal beta (importance sampling correction)
        self.anneal_beta()
        
        batch_transitions, batch_indices, is_weights = self.memory.sample(
            self.batch_size
        )
        batch = Transition(*zip(*batch_transitions))

        # Convert batch-array of Transitions to tensors
        state_batch = torch.cat(
            [torch.from_numpy(s).float().unsqueeze(0) for s in batch.state]
        )
        action_batch = torch.tensor(batch.action).long().unsqueeze(1)
        reward_batch = torch.tensor(batch.reward).float()

        # Compute state-action values using policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next state values using target_net
        next_state_values = torch.zeros(self.batch_size)
        non_final_mask = torch.tensor(
            [next_state is not None for next_state in batch.next_state],
            dtype=torch.bool,
        )

        # Collect all non-terminal next_states and convert to tensor
        non_final_next_states = torch.cat(
            [
                torch.from_numpy(next_state).float().unsqueeze(0)
                for next_state in batch.next_state
                if next_state is not None
            ]
        )

        # Compute the target Q-values for non-terminal next_states
        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = (
                    self.target_net(non_final_next_states).max(1)[0]
                )

        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch

        # Compute element-wise MSE loss with importance-sampling weights
        elementwise_loss = (
            state_action_values - expected_state_action_values.unsqueeze(1)
        ).pow(2).squeeze()

        # Convert is_weights to tensor and apply
        is_w = torch.tensor(is_weights, dtype=torch.float32)
        is_w = is_w.to(elementwise_loss.device)
        loss = (is_w * elementwise_loss).mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute TD errors AFTER network update
        with torch.no_grad():
            # Re-compute Q-values with updated network
            updated_q_values = self.policy_net(state_batch).gather(
                1, action_batch
            )
            td_errors = torch.abs(
                updated_q_values - expected_state_action_values.unsqueeze(1)
            )
            td_errors_np = td_errors.cpu().numpy().flatten()

        # Update priorities with post-update TD errors
        self.memory.update_priorities(batch_indices, td_errors_np.tolist())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        # copy weights from policy_net to target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))


# Training loop
def train(
    env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
):
    agent = DQNPrioritisedExpReplayAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting DQN training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                next_state_for_buffer = None
            else:
                next_state_for_buffer = next_state

            agent.memory.push(state, action, reward, next_state_for_buffer, done)
            agent.learn()

            state = next_state
            episode_reward += reward

            if done:
                break

        # Increment episodes for beta annealing
        agent.episodes_done += 1

        if episode % agent.target_update == 0:
            agent.update_target_net()

        # Decay exploration
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % (num_episodes / 10) == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")

        if np.mean(scores_deque) >= target_score:
            print(
                f"\nEnvironment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}"
            )
            break

    print("\nTraining complete.")

    return agent, scores
