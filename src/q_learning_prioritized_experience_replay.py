import numpy as np
import random
from collections import deque, namedtuple
from src.base_agent import BaseAgent

# Replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.biases = deque([], maxlen=capacity)
        self.is_cache_updated = False

    def push(self, transition, bias):
        """Save a transition"""
        self.memory.append(transition)
        self.biases.append(bias)
        self.is_cache_updated = False

    def sample(self):
        if not self.is_cache_updated:
            probs_unnormalized = np.exp(self.biases)
            self.distribution = probs_unnormalized / np.sum(probs_unnormalized)
            self.is_cache_updated = True
        return random.choices(self.memory, weights=self.distribution, k=1)[0]

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

    def learn(self):
        transition = self.memory.sample()
        state, action, reward, next_state, done = transition

        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

def train(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score):
    agent = QLearningPrioExpReplayAgent(state_dim, action_dim)

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

            bias = reward + agent.gamma * np.max(agent.q_table[next_state, :]) - agent.q_table[state, action]
            agent.memory.push(Transition(state, action, reward, next_state, done), bias)

            state = next_state
            episode_reward += reward

            if done:
                break

        for _ in range(len(agent.memory)):
            agent.learn()

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % (num_episodes / 10) == 0:
            print(f"Episode {episode}	Average Score: {np.mean(scores_deque):.2f}")
        
        if np.mean(scores_deque) >= target_score:
            print(f"Environment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break

    
    print("\nTraining complete.")
    
    return agent, scores
