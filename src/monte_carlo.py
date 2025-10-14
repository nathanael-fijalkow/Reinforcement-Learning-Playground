import numpy as np
from collections import deque
from src.base_agent import BaseAgent

class MonteCarlo(BaseAgent):
    def __init__(self, 
                 state_dim, 
                 action_dim,eps=0.3,gamma=0.99,lr=0.1,min_lr=0.01):
        self.q_table = np.zeros((state_dim, action_dim))
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.min_lr = min_lr
        self.trajectory = []

    def select_action(self, state, greedy=False):
        if greedy or np.random.rand() > self.eps:
            return np.argmax(self.q_table[state])
        else:
            return np.random.randint(self.q_table.shape[1])   

    def learn(self, state, action, reward, next_state, done):
        self.trajectory.append((state, action, reward))
        
        if done:
            G = 0
            for s, a, r in reversed(self.trajectory):
                G = r + self.gamma * G
                self.q_table[s, a] += self.lr * (G - self.q_table[s, a])
            self.trajectory = []

            self.lr = max(self.min_lr, self.lr * 0.99) 
            self.eps = max(0.1, self.eps * 0.9995)
        
        

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
