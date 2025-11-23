from collections import deque

import numpy as np

from src.base_agent import BaseAgent


class SarsaAgent(BaseAgent):
    """
    SARSA (State-Action-Reward-State-Action) agent using tabular representation.
    
    SARSA is an on-policy TD control algorithm that learns the action-value
    function for the policy being followed (including exploration):
    Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    
    Key difference from Q-learning: SARSA uses the actual next action a'
    taken by the ε-greedy policy, not max_a' Q(s',a'). This makes it on-policy.
    
    Parameters:
        state_dim: Number of discrete states in the environment
        action_dim: Number of discrete actions available
        learning_rate: Step size for Q-value updates (α)
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
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = learning_rate
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

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-value using the SARSA (on-policy) update rule.
        
        SARSA uses the actual next action taken by the policy:
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        
        This differs from Q-learning which uses max_a' Q(s',a'), making
        SARSA sensitive to the exploration strategy being used.
        
        Parameters:
            state: Current state
            action: Action taken in current state
            reward: Reward received
            next_state: Next state reached
            next_action: Actual next action selected by the policy (key for SARSA!)
            done: Whether episode terminated
        """
        old_value = self.q_table[state, action]
        
        # Use the actual next action (on-policy)
        self.q_table[state, action] += self.lr * (
            reward + self.gamma * self.q_table[next_state, next_action] - old_value
        )

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)


def train(
    env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
):
    agent = SarsaAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting SARSA training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        # SARSA: Select initial action
        action = agent.select_action(state)

        for step in range(max_steps_per_episode):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # SARSA: Select next action BEFORE learning
            # This is the actual action that will be taken in next_state
            if not done:
                next_action = agent.select_action(next_state)
            else:
                next_action = 0  # Arbitrary, won't be used since episode ends
            
            # Update using the actual next action (on-policy)
            agent.learn(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action  # Use the action we already selected
            episode_reward += reward

            if done:
                break

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
