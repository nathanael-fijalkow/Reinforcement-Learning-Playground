import numpy as np
from collections import deque
from src.base_agent import BaseAgent

class MonteCarlo(BaseAgent):
    def __init__(self, 
                 state_dim, 
                 action_dim):
        self.q_table = np.zeros((state_dim, action_dim))
        self.gamma = 0.99
        self.Learning_rate = 0.1
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.episode = []
    def select_action(self, state, greedy=False):
        if greedy or np.random.rand() > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return np.random.choice(len(self.q_table[state]))            

    def learn(self, state, action, reward, next_state, done):
        self.episode.append((state, action, reward))
        if done:
            G = 0.0
            for s, a, r in reversed(self.episode):
                G = r + self.gamma * G
                self.q_table[s, a] += self.Learning_rate * (G - self.q_table[s, a])
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode = []


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
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.learn(state, action, reward, next_state, done)

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
