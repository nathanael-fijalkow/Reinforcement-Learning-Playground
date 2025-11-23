import random
from collections import deque, namedtuple

import numpy as np

from src.base_agent import BaseAgent

# Replay buffer
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Experience replay breaks the correlation between consecutive samples,
    improving learning stability and sample efficiency by allowing the
    agent to learn from past experiences multiple times.
    
    Parameters:
        capacity: Maximum number of transitions to store
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)


class QLearningExpReplayAgent(BaseAgent):
    """
    Q-Learning agent with experience replay for tabular environments.
    
    Combines Q-learning updates with experience replay to improve sample
    efficiency and break correlation between consecutive experiences.
    
    Key features:
    - Stores transitions in a replay buffer
    - Samples random batches for learning (decorrelates samples)
    - Can learn from each experience multiple times
    
    Parameters:
        state_dim: Number of discrete states in the environment
        action_dim: Number of discrete actions available
        learning_rate: Step size for Q-value updates (α)
        gamma: Discount factor for future rewards (γ)
        epsilon: Initial exploration rate for ε-greedy policy
        epsilon_decay: Multiplicative decay factor for epsilon
        epsilon_min: Minimum epsilon value
        buffer_size: Maximum number of transitions to store
        batch_size: Number of transitions to sample per update
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=32,
    ):
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self):
        """
        Sample a batch from replay buffer and update Q-values.
        
        Uses Q-learning update rule on each sampled transition:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Only learns if buffer has enough samples to fill a batch.
        """
        # Don't learn until we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        
        # Process each transition in the batch
        for transition in transitions:
            state, action, reward, next_state, done = transition

            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state, :])
            
            # Q-learning update (off-policy: uses max)
            # Match vanilla Q-learning formula
            new_value = old_value + self.lr * (
                reward + self.gamma * next_max - old_value
            )
            self.q_table[state, action] = new_value

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)


def train(
    env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
):
    agent = QLearningExpReplayAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting Q-Learning training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store the transition in memory
            agent.memory.push(state, action, reward, next_state, done)
            
            # Learn periodically (every 4 steps) instead of every step
            # This reduces computational cost and improves stability
            if step % 4 == 0:
                agent.learn()

            state = next_state
            episode_reward += reward

            if done:
                # Learn one final time at episode end
                agent.learn()
                # Decay epsilon at end of episode
                agent.epsilon = max(
                    agent.epsilon_min, agent.epsilon * agent.epsilon_decay
                )
                break

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % (num_episodes / 10) == 0:
            print(f"Episode {episode}	Average Score: {np.mean(scores_deque):.2f}")

        if np.mean(scores_deque) >= target_score:
            print(
                f"Environment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}"
            )
            break

    print("\nTraining complete.")

    return agent, scores
