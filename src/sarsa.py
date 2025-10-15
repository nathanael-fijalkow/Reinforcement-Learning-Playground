import numpy as np
from collections import deque
from src.base_agent import BaseAgent

class SARSA(BaseAgent):
    def __init__(self,
                 state_dim,
                 action_dim,
                 epsilon=1.0,
                 learning_rate=0.1,
                 gamma=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.q_table = np.zeros((state_dim, action_dim))
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim

    def select_action(self, state, greedy=False):
        if greedy or np.random.rand() > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return np.random.randint(self.action_dim)

    def learn(self, state, action, reward, next_state, next_action, done):
        old_value = self.q_table[state, action]
        
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state, next_action]
        
        td_error = td_target - old_value
        self.q_table[state, action] += self.learning_rate * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)


def train(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score):
    agent = SARSA(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting SARSA training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        action = agent.select_action(state)
        
        for step in range(max_steps_per_episode):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_action = agent.select_action(next_state)
            
            agent.learn(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            episode_reward += reward

            if done:
                break
        
        agent.decay_epsilon()

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % (num_episodes // 10) == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}\tEpsilon: {agent.epsilon:.3f}")
        
        if np.mean(scores_deque) >= target_score:
            print(f"Environment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break

    print("\nTraining complete.")
    
    return agent, scores