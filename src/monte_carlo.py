import numpy as np
from collections import deque
from src.base_agent import BaseAgent

class MonteCarlo(BaseAgent):
    def __init__(self, 
                 state_dim, 
                 action_dim, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.1, gamma=0.9):
        self.q_table = np.zeros((state_dim, action_dim))
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.gamma = gamma

    def select_action(self, state, greedy=False):
        if not greedy:
           eps = np.random.rand()
           if eps < self.epsilon:
               return np.random.choice(self.action_dim)
        return np.argmax(self.q_table[state])     

    def learn(self, trajectory):
        G = 0
        for state,action,reward in reversed(trajectory):
            G = reward + self.gamma * G
            self.q_table[state][action] += self.learning_rate*(G - self.q_table[state][action])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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

            trajectory.append([state,action,reward])

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
    
    return agent, scores