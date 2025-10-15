import numpy as np
from collections import deque
from src.base_agent import BaseAgent

class MonteCarlo(BaseAgent):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 learning_rate=0.1,
                 lr_decay=0.995,
                 lr_min=0.001, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01):
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim

    def select_action(self, state, greedy=False):
        if greedy or np.random.rand() >= self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return np.random.randint(self.q_table.shape[1])           

    def learn(self, trajectory):
        G = 0
        for state, action, reward in reversed(trajectory):
            G = reward + self.gamma * G 
            old_value = self.q_table[state, action]
            new_value = old_value + self.lr * (G - old_value)
            self.q_table[state, action] = new_value
        
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
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
    
    return agent, scores
