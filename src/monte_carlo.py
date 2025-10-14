import numpy as np
from collections import deque
from src.base_agent import BaseAgent

class MonteCarlo(BaseAgent):
    def __init__(self,
                state_dim,
                action_dim,
                learning_rate=0.1,
                epsilon=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                gamma=0.99):
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = learning_rate
        self.lr_decay = 0.995
        self.lr_min = 0.001
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim
        self.gamma = gamma

    def select_action(self, state, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, trajectory):
        G = 0
        visited_couples = set()
        for step in range(len(trajectory)):
            state, action, reward = trajectory[len(trajectory) - step - 1]
            G = reward + self.gamma * G
            if (state, action) not in visited_couples:
                visited_couples.add((state, action))
                old_value = self.q_table[state, action]
                self.q_table[state, action] = old_value + self.lr * (G - old_value)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.lr = max(self.lr_min, self.lr * self.lr_decay)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

def train(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score):
    agent = MonteCarlo(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting Monte Carlo training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        trajectory = []
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, action, reward))
            state = next_state
            episode_reward += reward

            if done:
                break

        agent.learn(trajectory)
        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % (num_episodes / 10) == 0:
            print(f"Episode {episode}	Average Score: {np.mean(scores_deque):.2f}")
        
        if np.mean(scores_deque) >= target_score:
            print(f"Environment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break

    
    print("\nTraining complete.")
    
    return agent,scores