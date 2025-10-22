import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
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
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, state):
        return self.net(state)

# Replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Basic replay buffer without prioritization using deque for storage
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class VanillaDQNAgent(BaseAgent):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 lr=1e-4, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01, 
                 buffer_size=10000, 
                 batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.policy_net = DQN(state_dim, action_dim)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def select_action(self, state, greedy=False):
        self.steps_done += 1
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            # Convert state (numpy array) to tensor
            # float() ensures it is a float tensor
            # unsqueeze(0) adds batch dimension
            state = torch.from_numpy(state).float().unsqueeze(0)

            # max(1) returns (value, index)
            # we want the index of the max log-probability so we take [1] from the result
            # view(1, 1) adds batch dimension
            # item() returns the value as a Python number
            return self.policy_net(state).max(1)[1].view(1, 1).item()

    def learn(self):
        # Do not learn if not enough samples in memory
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # We need to convert batch-array of Transitions to tensors
        # Each state in batch.state is a numpy array of shape (state_dim,)
        # So we convert each to tensor and then concatenate along batch dimension
        state_batch = torch.cat([torch.from_numpy(s).float().unsqueeze(0) for s in batch.state])
        # unsqueeze(1) adds action dimension 
        action_batch = torch.tensor(batch.action).long().unsqueeze(1)
        reward_batch = torch.tensor(batch.reward).float()

        # Compute state-action values using policy_net
        # self.policy_net(state_batch) has shape (batch_size, action_dim)
        # gather(1, action_batch) selects the Q-value for the taken action
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next state values using target_net
        # initialize to zeros (to account for terminal states)
        next_state_values = torch.zeros(self.batch_size)
        # Identify which next_states are not terminal (i.e., not None)
        non_final_mask = torch.tensor(
            [next_state is not None for next_state in batch.next_state], dtype=torch.bool
        )

        # Collect all non-terminal next_states and convert to tensor
        non_final_next_states = torch.cat([
            torch.from_numpy(next_state).float().unsqueeze(0)
            for next_state in batch.next_state if next_state is not None
        ])

        # Compute the target Q-values for non-terminal next_states
        next_state_values[non_final_mask] = (
            self.policy_net(non_final_next_states).max(1)[0].detach()
        )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))

# Training loop
def train(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score):
    agent = VanillaDQNAgent(state_dim, action_dim)

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

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % (num_episodes / 10) == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")

        if np.mean(scores_deque) >= target_score:
            print(f"\nEnvironment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break
            
    print("\nTraining complete.")

    return agent, scores