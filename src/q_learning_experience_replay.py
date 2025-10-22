import numpy as np
import random
from collections import deque, namedtuple
from src.base_agent import BaseAgent

# Replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# This is a basic replay buffer without prioritization and using deque for storage
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        """Sample a single transition"""
        return random.sample(self.memory, 1)[0]

    def __len__(self):
        return len(self.memory)

class QLearningExpReplayAgent(BaseAgent):
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
        self.memory = ReplayBuffer(buffer_size)

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
            # Performs one step of the learning process
            agent.learn() 

            state = next_state
            episode_reward += reward

            if done:
                break

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % (num_episodes / 10) == 0:
            print(f"Episode {episode}	Average Score: {np.mean(scores_deque):.2f}")
        
        if np.mean(scores_deque) >= target_score:
            print(f"Environment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break

    
    print("\nTraining complete.")
    
    return agent, scores
