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
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, state):
        # Output log probabilities of actions
        return F.log_softmax(self.network(state), dim=-1)


# --- 2. Define the Critic Network (Value Function) ---
# The Critic estimates the value (expected return) of being in a given state.
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single value (state-value)
        )

    def forward(self, state):
        return self.network(state)


# --- 3. Define the Actor-Critic Agent ---
# This class combines the Actor and Critic, handles training, and action selection.
class ActorCriticAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
        value_clip=0.2,
        max_grad_norm=0.5,
        batch_size=5,
    ):
        """
        Actor-Critic agent with additional improvements.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            lr_actor: Learning rate for policy network
            lr_critic: Learning rate for value network
            gamma: Discount factor for future rewards
            entropy_coef: Coefficient for entropy bonus (exploration)
            value_clip: Clipping parameter for value function loss
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Number of episodes to collect before updating
        """
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        # Define optimizers for both networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        # Entropy coefficient for exploration bonus
        # Prevents premature convergence to deterministic policies
        self.entropy_coef = entropy_coef
        # Value clipping to prevent large value function updates
        self.value_clip = value_clip
        # Gradient clipping to prevent exploding gradients
        self.max_grad_norm = max_grad_norm
        # Batch size - collect multiple episodes before updating
        self.batch_size = batch_size

        # Buffers for current episode
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_dones = []
        self.episode_states = []

        # Batch buffers for collecting multiple episodes before update
        # Previous implementation updated after every single episode
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_values = []
        self.batch_dones = []

        self.episodes_collected = 0
        self.training_mode = True

    def set_training(self, training: bool):
        """Set training mode."""
        self.training_mode = training
        self.actor.train(training)
        self.critic.train(training)

    def select_action(self, state, greedy=False):
        # Convert state (numpy array) to PyTorch tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        with torch.set_grad_enabled(self.training_mode):
            # Get log probabilities from the actor
            log_probs = self.actor(state_tensor)
            # Create a categorical distribution from log probabilities
            dist = torch.distributions.Categorical(logits=log_probs)

            if greedy:
                action = torch.argmax(log_probs, dim=-1)
            else:
                # Sample an action from the distribution
                action = dist.sample()

            # Store experience during training
            if self.training_mode:
                self.episode_log_probs.append(dist.log_prob(action))
                self.episode_values.append(self.critic(state_tensor))
                self.episode_states.append(state_tensor)

        return int(action.item())

    def store_reward(self, reward):
        """Store reward for current step."""
        if self.training_mode:
            self.episode_rewards.append(reward)

    def end_episode(self, next_state=None, done=True):
        """
        New method for proper episode management and batch collection.
        Called at end of episode to finalize episode data and add to batch.
        When batch is full, triggers learning. This enables collecting multiple
        episodes before updating, improving sample efficiency.
        
        Args:
            next_state: Final state (unused for Actor-Critic)
            done: Whether episode terminated naturally
        """
        if not self.training_mode or len(self.episode_log_probs) == 0:
            return

        # Add episode to batch
        self.batch_log_probs.extend(self.episode_log_probs)
        self.batch_rewards.extend(self.episode_rewards)
        self.batch_values.extend(self.episode_values)
        self.batch_dones.extend(self.episode_dones)

        # Clear episode buffers
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_dones = []
        self.episode_states = []

        self.episodes_collected += 1

        # Learn only when batch is full (not after every episode)
        # This improves sample efficiency and training stability
        if self.episodes_collected >= self.batch_size:
            self.learn()
            self.episodes_collected = 0

    def store_transition(self, reward, done):
        """Legacy method for compatibility. Use store_reward() instead."""
        self.store_reward(reward)
        if self.training_mode:
            self.episode_dones.append(done)

    def learn(self):
        """Update policy and value networks using collected batch."""
        if len(self.batch_log_probs) == 0:
            return

        # Calculate discounted rewards (returns)
        returns = []
        R = 0
        # Iterate backwards to calculate discounted returns
        for i in reversed(range(len(self.batch_rewards))):
            if self.batch_dones[i]:  # If episode ended, reset R
                R = 0
            R = self.batch_rewards[i] + self.gamma * R
            returns.insert(0, R)  # Insert at the beginning to maintain order

        # Convert lists to PyTorch tensors
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.cat(self.batch_log_probs)
        old_values = torch.cat(self.batch_values).squeeze()

        # Normalize returns (helps stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate Advantage: TD Error = G_t - V(S_t)
        advantage = returns - old_values

        # Normalize advantages (important for stability)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Get current log probs and values for entropy calculation
        new_log_probs_dist = torch.distributions.Categorical(logits=log_probs)
        # Entropy bonus encourages exploration and prevents
        # premature convergence to deterministic policies
        entropy = new_log_probs_dist.entropy().mean()

        # --- Critic Loss with Value Clipping ---
        # Value function clipping (similar to PPO)
        # Prevents destructively large updates to the value function
        new_values = old_values  # In A2C we use the same values
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values,
            -self.value_clip,
            self.value_clip
        )
        value_loss_unclipped = F.mse_loss(old_values, returns.detach())
        value_loss_clipped = F.mse_loss(value_pred_clipped, returns.detach())
        critic_loss = torch.max(value_loss_unclipped, value_loss_clipped)

        # --- Actor Loss ---
        # Actor aims to maximize expected return, so we minimize -log_prob * advantage
        # Add entropy bonus to actor loss
        # Encourages exploration by rewarding policy diversity
        actor_loss = (
            -(log_probs * advantage.detach()).mean() - self.entropy_coef * entropy
        )

        # --- Update Networks with Gradient Clipping ---
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Clear batch buffers
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_values = []
        self.batch_dones = []

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
    # Initialize the Actor-Critic agent
    agent = ActorCriticAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)  # To store recent scores for average calculation
    scores = []  # To store all scores

    print("Starting Actor-Critic training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)

            state = next_state
            episode_reward += reward

            if done:
                break

        # End episode with state for potential bootstrapping
        agent.end_episode(next_state=state if not done else None, done=done)

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")

        # Check if solved (average over 100 episodes >= target)
        if np.mean(scores_deque) >= target_score:
            avg_score = np.mean(scores_deque)
            print(f"\nSolved in {episode} episodes! Average Score: {avg_score:.2f}")
            break

    print("\nTraining complete.")

    return agent, scores
