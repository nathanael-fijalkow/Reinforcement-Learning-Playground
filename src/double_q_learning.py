import numpy as np
from collections import deque
from src.base_agent import BaseAgent

class DoubleQLearningAgent(BaseAgent):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 learning_rate=0.1, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01):
        self.q1 = np.zeros((state_dim, action_dim))
        self.q2 = np.zeros((state_dim, action_dim))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim

    def mean(self):
        return (self.q1 + self.q2) / 2.0
    
    def select_action(self, state, greedy=False):
        state=int(state)
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        q = self.mean()
        return np.argmax(q[state, :])

    def learn(self, state, action, reward, next_state, done):
        state=int(state)
        next_state=int(next_state)
        # flip a coin pour choisir quelle table mettre à jour
        update_q1 = (np.random.rand() < 0.5)
        if update_q1:
            # sélection avec q1 et évaluation avec q2

            if done: 
                new_value = reward 
            else:
                next_action = np.argmax(self.q1[next_state, :])
                new_value = reward + self.gamma * self.q2[next_state, next_action]
            self.q1[state, action] += self.lr * (new_value - self.q1[state, action])
        else:
            # sélection avec q2 et évaluation avec q1
            if done:
                new_value = reward
            else:
                next_action = np.argmax(self.q2[next_state, :])
                new_value = reward + self.gamma * self.q1[next_state, next_action]
            self.q2[state, action] += self.lr * (new_value - self.q2[state, action])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.savez(path, q1=self.q1, q2=self.q2)

    def load(self, path):
        data = np.load(path)
        self.q1 = data["q1"]
        self.q2 = data["q2"]


def train(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score):
    agent = DoubleQLearningAgent(state_dim, action_dim)
    print_every = max(1, num_episodes // 10)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting Double Q-Learning training...")

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

        if episode % print_every == 0:
            print(f"Episode {episode}	Average Score: {np.mean(scores_deque):.2f}")
        
        if np.mean(scores_deque) >= target_score:
            print(f"Environment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break

    
    print("\nTraining complete.")
    
    return agent, scores
