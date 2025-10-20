import numpy as np
from collections import deque, namedtuple
from src.base_agent import BaseAgent

# Replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

import numpy as np
import random
from collections import namedtuple, deque

# Transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)

    def push(self, transition, bias):
        self.memory.append(transition)  # Eliminated the oldest transitions
        self.priorities.append(bias + 1e-6)

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return []

        probs = np.array(self.priorities) / np.array(self.priorities).sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices, biases):
        """Update priorities after learning"""
        for i, bias in zip(indices, biases):
            self.priorities[i] = bias + 1e-6

    def __len__(self):
        return len(self.memory)

class QLearningPrioExpReplayAgent(BaseAgent):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 learning_rate=0.1, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01,
                 buffer_size=10000):
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim
        self.memory = PrioritizedReplayBuffer(buffer_size)

    def select_action(self, state, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, batch_size):
        transitions, indices = self.memory.sample(batch_size)
        biases = []
        for transition in transitions:
            state, action, reward, next_state, done = transition

            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state, :])
            target = reward if done else reward + self.gamma * next_max
            bias = abs(target - old_value)

            new_value = (1 - self.lr) * old_value + self.lr * target
            self.q_table[state, action] = new_value

            biases.append(bias)

        self.memory.update_priorities(indices, biases)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

def train(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score):
    agent = QLearningPrioExpReplayAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting Q-Learning training with prioritized experience replay...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward if done else reward + agent.gamma * np.max(agent.q_table[next_state, :])
            bias = abs(target - agent.q_table[state, action])
            agent.memory.push(Transition(state, action, reward, next_state, done), bias)

            state = next_state
            episode_reward += reward

            if done:
                break
        
        batch_size = 32
        agent.learn(batch_size)

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % (num_episodes / 10) == 0:
            print(f"Episode {episode}	Average Score: {np.mean(scores_deque):.2f}")

        if np.mean(scores_deque) >= target_score:
            print(f"Environment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break

    
    print("\nTraining complete.")
    
    return agent, scores
