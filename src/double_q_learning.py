import numpy as np
from collections import deque
from src.base_agent import BaseAgent

class DoubleQLearningAgent(BaseAgent):
    def __init__(self, 
                 state_dim, 
                 action_dim, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.1, gamma=0.9):
        self.q_table1 = np.zeros((state_dim, action_dim))
        self.q_table2 = np.zeros((state_dim, action_dim))
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.table = 0

    def select_action(self, state, greedy=False):
        if np.random.rand() < 0.5:
            self.table = 0
        else :
            self.table = 1

        if not greedy:
           eps = np.random.rand()
           if eps < self.epsilon:
               return np.random.choice(self.action_dim)
        if self.table == 0:
            return np.argmax(self.q_table1[state])
        else:
            return np.argmax(self.q_table2[state])

    def learn(self, state, action, reward, next_state, done):
        if self.table == 0:
            next_action = np.argmax(self.q_table2[next_state])
            self.q_table1[state][action] += self.learning_rate*(reward + self.gamma*self.q_table1[next_state][next_action] - self.q_table1[state][action])
        else:
            next_action = np.argmax(self.q_table1[next_state])
            self.q_table2[state][action] += self.learning_rate*(reward + self.gamma*self.q_table2[next_state][next_action] - self.q_table2[state][action])
            
        if done :
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

def train(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score):
    agent = DoubleQLearningAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting DoubleQLearning training...")

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

    print(agent.q_table1, agent.q_table2)
    print("\nTraining complete.")
    
    return agent, scores
