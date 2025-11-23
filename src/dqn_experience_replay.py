import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.base_agent import BaseAgent


# Q-Network
class DQN(nn.Module):
    """Deep Q-Network: neural network for approximating Q-values."""
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


# Replay buffer
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    
    Stores transitions and samples random batches to break correlation
    between consecutive samples, improving training stability.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNExpReplayAgent(BaseAgent):
    """
    Deep Q-Network (DQN) with experience replay and target network.
    
    DQN combines three key innovations:
    1. Q-learning with neural network function approximation
    2. Experience replay buffer to break sample correlation
    3. Separate target network for stable Q-value targets
    
    The target network is a copy of the policy network that is updated
    periodically (not every step). This prevents the "moving target" problem
    where both the predicted Q-values and the target Q-values change
    simultaneously, causing training instability.
    
    Update rule:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q_target(s',a') - Q(s,a)]
    
    Reference: Mnih et al. (2015) "Human-level control through deep RL"
    
    Parameters:
        state_dim: Dimension of state space
        action_dim: Number of discrete actions
        lr: Learning rate for optimizer
        gamma: Discount factor for future rewards
        epsilon: Initial exploration rate
        epsilon_decay: Multiplicative decay for epsilon
        epsilon_min: Minimum epsilon value
        buffer_size: Maximum replay buffer capacity
        batch_size: Number of transitions per training batch
        target_update_freq: Steps between target network updates
    """
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
        target_update_freq=1000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Policy network (updated every step)
        self.policy_net = DQN(state_dim, action_dim)
        
        # Target network (updated periodically for stability)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set to evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def select_action(self, state, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            # Convert state (numpy array) to tensor
            # float() ensures it is a float tensor
            # unsqueeze(0) adds batch dimension
            state = torch.from_numpy(state).float().unsqueeze(0)

            # max(1) returns (value, index)
            # we want the index of the max log-probability so we take [1] from the result
            # item() returns the value as a Python number
            return self.policy_net(state).max(1)[1].item()

    def learn(self):
        """Update policy network using batch from replay buffer."""
        # Do not learn if not enough samples in memory
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # We need to convert batch-array of Transitions to tensors
        # Each state in batch.state is a numpy array of shape (state_dim,)
        # So we convert each to tensor and then concatenate along batch dimension
        state_batch = torch.cat(
            [torch.from_numpy(s).float().unsqueeze(0) for s in batch.state]
        )
        # unsqueeze(1) adds action dimension
        action_batch = torch.tensor(batch.action).long().unsqueeze(1)
        reward_batch = torch.tensor(batch.reward).float()

        # Compute state-action values using policy_net
        # self.policy_net(state_batch) has shape (batch_size, action_dim)
        # gather(1, action_batch) selects the Q-value for the taken action
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next state values using target_net (KEY: use target network!)
        # initialize to zeros (to account for terminal states)
        next_state_values = torch.zeros(self.batch_size)
        # Identify which next_states are not terminal (i.e., not None)
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
        # Use target network for stability (prevents moving target problem)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1)[0]
            )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.mse_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "steps_done" in checkpoint:
            self.steps_done = checkpoint["steps_done"]


# Training loop
def train(
    env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
):
    agent = DQNExpReplayAgent(state_dim, action_dim)

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
            
            # Learn periodically (every 4 steps) instead of every step
            # This improves stability and reduces computational cost
            if step % 4 == 0:
                agent.learn()

            state = next_state
            episode_reward += reward

            if done:
                # Learn one final time at episode end
                agent.learn()
                # Decay epsilon after each episode
                agent.epsilon = max(
                    agent.epsilon_min, agent.epsilon * agent.epsilon_decay
                )
                break

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
