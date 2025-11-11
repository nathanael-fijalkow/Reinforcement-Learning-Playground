import random
from collections import deque

import numpy as np

from src.base_agent import BaseAgent


class DynaQAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        planning_steps=5,
    ):
        self.q_table = np.zeros((state_dim, action_dim))
        self.model = {}  # model[(state, action)] = (reward, next_state)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.planning_steps = planning_steps
        self.action_dim = action_dim
        self.observed_states_actions = set()

    def select_action(self, state, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        new_value = (1 - self.lr) * old_value + self.lr * (
            reward + self.gamma * next_max
        )
        self.q_table[state, action] = new_value

    def update_model(self, state, action, reward, next_state):
        if (state, action) not in self.model:
            self.observed_states_actions.add((state, action))
        self.model[(state, action)] = (reward, next_state)

    def planning(self):
        for _ in range(self.planning_steps):
            if not self.observed_states_actions:
                continue

            state, action = random.choice(list(self.observed_states_actions))
            reward, next_state = self.model[(state, action)]
            self.learn(state, action, reward, next_state)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)


def train(
    env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
):
    agent = DynaQAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting Dyna-Q training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.learn(state, action, reward, next_state)
            agent.update_model(state, action, reward, next_state)
            agent.planning()

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.decay_epsilon()
        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")

        if np.mean(scores_deque) >= target_score:
            print(
                f"\nEnvironment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}"
            )
            break

    print("\nTraining complete.")

    return agent, scores
