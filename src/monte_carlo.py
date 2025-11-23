from collections import deque

import numpy as np

from src.base_agent import BaseAgent


class MonteCarlo(BaseAgent):
    """
    Monte Carlo control agent using every-visit MC updates with tabular representation.
    
    Monte Carlo methods learn from complete episodes rather than bootstrapping.
    Q-values are updated toward the actual observed returns:
    Q(s,a) ← Q(s,a) + α[G_t - Q(s,a)]
    
    where G_t is the discounted return from time t to the end of the episode.
    
    This is an every-visit implementation, meaning all occurrences of each
    state-action pair in an episode are updated (as opposed to first-visit
    which only updates the first occurrence).
    
    Parameters:
        state_dim: Number of discrete states in the environment
        action_dim: Number of discrete actions available
        learning_rate: Step size for Q-value updates (α)
        lr_decay: Multiplicative decay factor for learning rate
        lr_min: Minimum learning rate value
        gamma: Discount factor for future rewards (γ)
        epsilon: Initial exploration rate for ε-greedy policy
        epsilon_decay: Multiplicative decay factor for epsilon
        epsilon_min: Minimum epsilon value
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.1,
        lr_decay=0.999,  # Slower decay for high-variance environments
        lr_min=0.01,  # Higher minimum to keep learning
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
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
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, trajectory):
        """
        Update Q-values using every-visit Monte Carlo.
        
        Processes the trajectory in reverse to efficiently compute discounted returns.
        Each state-action pair encountered in the trajectory is updated toward
        the actual return observed from that point onward.
        
        Args:
            trajectory: List of (state, action, reward) tuples from one complete episode
        """
        G = 0
        for state, action, reward in reversed(trajectory):
            G = reward + self.gamma * G
            self.q_table[state, action] += self.lr * (G - self.q_table[state, action])

        # Decay learning rate and epsilon after each episode
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)


def train(
    env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
):
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
            print(
                f"Environment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}"
            )
            break

    print("\nTraining complete.")

    return agent, scores
