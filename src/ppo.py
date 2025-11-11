from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.base_agent import BaseAgent


# --- 1. Define the Actor Network (Policy) ---
# The Actor decides which action to take given a state.
# For a discrete action space, it outputs probabilities over actions.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # Output log probabilities of actions
        return F.log_softmax(self.fc2(x), dim=-1)


# --- 2. Define the Critic Network (Value Function) ---
# The Critic estimates the value (expected return) of being in a given state.
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Output a single value (state-value)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)


# --- 3. Define the PPO Agent ---
# This class combines the Actor and Critic, handles training, and action selection.
class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon_clip=0.2,
        K_epochs=4,
        gae_lambda=0.95,
    ):
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        # Define optimizers for both networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon_clip = epsilon_clip  # Clipping parameter for PPO
        self.K_epochs = (
            K_epochs  # Number of epochs to optimize the policy and value function
        )
        self.gae_lambda = (
            gae_lambda  # Lambda for Generalized Advantage Estimation (GAE)
        )

        # Buffers to store experience for a single trajectory/batch
        self.states = []
        self.actions = []
        self.log_probs = []  # Log probabilities of actions taken with the OLD policy
        self.rewards = []
        self.values = (
            []
        )  # State values predicted by the critic for the states in trajectory
        self.dones = []

    def select_action(self, state, greedy=False):
        # Convert state (numpy array) to PyTorch tensor
        state = torch.from_numpy(state).float().unsqueeze(0)

        # Get log probabilities from the actor (current policy)
        log_probs = self.actor(state)
        dist = torch.distributions.Categorical(logits=log_probs)

        if greedy:
            action = torch.argmax(log_probs, dim=-1)
        else:
            action = dist.sample()

        # Get the value prediction from the critic
        value = self.critic(state)

        # Store the experience
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)

        return action.item()

    def store_transition(self, reward, done):
        # Store the reward and done flag for the current transition
        self.rewards.append(reward)
        self.dones.append(done)

    def calculate_gae(self, next_value):
        # Calculate Generalized Advantage Estimation (GAE)
        # This is a more stable way to estimate advantages compared to simple TD error
        advantages = []
        returns = []
        last_advantage = 0

        # Convert lists to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.cat(self.values).squeeze()
        dones = torch.tensor(self.dones, dtype=torch.float32)

        # Add the next_value to the end of the values for the last step's calculation
        # If the episode is done, next_value is 0
        ext_values = torch.cat((values, next_value.squeeze().unsqueeze(0)))

        for t in reversed(range(len(rewards))):
            # TD error for current step
            delta = (
                rewards[t] + self.gamma * ext_values[t + 1] * (1 - dones[t]) - values[t]
            )
            # GAE formula: detach delta to prevent gradients from flowing back through the value function
            last_advantage = (
                delta.detach()
                + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            )
            advantages.insert(0, last_advantage)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        # Calculate returns (Value + Advantage)
        returns = advantages + values
        return advantages, returns

    def learn(self, next_state_value):
        # Calculate advantages and returns using GAE
        advantages, returns = self.calculate_gae(next_state_value)

        # Convert stored lists to tensors for batch processing
        old_states = torch.cat(self.states).squeeze(
            1
        )  # Remove the unsqueeze(0) from select_action
        old_actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)

        # Normalize advantages (important for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Optimization Loop (K_epochs)
        for _ in range(self.K_epochs):
            # Get new log probabilities and values from current policy/value function
            new_log_probs_dist = torch.distributions.Categorical(
                logits=self.actor(old_states)
            )
            new_log_probs = new_log_probs_dist.log_prob(old_actions)
            new_values = self.critic(old_states).squeeze()

            # --- Critic Loss ---
            # Mean Squared Error between predicted values and actual returns
            critic_loss = F.mse_loss(
                new_values, returns.detach()
            )  # Detach returns for critic update

            # --- Actor Loss (Clipped Surrogate Objective) ---
            # Ratio of new policy probability to old policy probability
            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
                * advantages
            )
            actor_loss = -torch.min(
                surr1, surr2
            ).mean()  # Minimize negative PPO objective

            # --- Update Networks ---
            # Zero gradients, perform backward pass, and update weights for critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Zero gradients, perform backward pass, and update weights for actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Clear stored experience for the next batch/trajectory collection
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def save(self, path):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])


# --- 4. Training Loop ---
def train(
    env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
):
    # Initialize the PPO agent
    agent = PPOAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)  # To store recent scores for average calculation
    scores = []  # To store all scores

    print("Starting PPO training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()  # Reset environment for a new episode
        episode_reward = 0
        done = False

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)  # Agent selects an action
            next_state, reward, terminated, truncated, _ = env.step(
                action
            )  # Environment takes a step
            done = terminated or truncated  # Check if episode is done

            agent.store_transition(reward, done)  # Store the transition

            state = next_state  # Update current state
            episode_reward += reward  # Accumulate reward

            if done:  # If episode finished
                break

        # Get the value of the last state (if not done, or 0 if done)
        next_state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        next_state_value = (
            agent.critic(next_state_tensor) if not done else torch.tensor([[0.0]])
        )

        # After each episode, the agent learns from the collected experience
        agent.learn(next_state_value)

        scores_deque.append(episode_reward)  # Add current episode reward to deque
        scores.append(episode_reward)  # Add to all scores

        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")

        # Check if the environment is solved (average score over 100 episodes is 475 or more)
        if np.mean(scores_deque) >= target_score:
            print(
                f"\nEnvironment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}"
            )
            break

    print("\nTraining complete.")

    return agent, scores
