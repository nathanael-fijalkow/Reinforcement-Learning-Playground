import numpy as np
from collections import deque, namedtuple
from src.base_agent import BaseAgent

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Prioritized Replay Buffer
# Prioritized replay buffers determine how likely transitions are to be sampled based on their TD error (importance)
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.alpha = alpha # normalization exponent for priorities
        self.priorities = np.zeros(capacity) # the weights of stored transitions
        self.pos = 0 # position to insert the next transition

    def push(self, *args):
        """Save a transition."""
        # New transitions get maximum priority
        max_p = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_p
        # Update position
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=1):
        """Sample a batch of transitions based on their priorities."""
        # If memory is empty, return empty lists
        if len(self.memory) == 0:
            return [], []
            
        # Get priorities, apply alpha
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = np.array([self.memory[i] for i in indices])
        
        return samples, indices

    def update_priorities(self, indices, errors, epsilon=1e-5):
        """Update priorities of sampled transitions."""
        for i, error in zip(indices, errors):
            self.priorities[i] = abs(error) + epsilon

    def __len__(self):
        return len(self.memory)

class QLearningExpReplayAgent(BaseAgent):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 learning_rate=0.1, 
                 lr_decay=0.995, 
                 lr_min=0.001,
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01,
                 batch_size=32,
                 buffer_size=10000):
        self.q_table = np.zeros((state_dim, action_dim))
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory = PrioritizedReplayBuffer(buffer_size)

    def select_action(self, state, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self):
        if len(self.memory) == 0:
            return

        batch_transitions, batch_indices = self.memory.sample(self.batch_size)
        batch = Transition(*map(np.array, zip(*batch_transitions)))
        old_values = self.q_table[batch.state, batch.action]
        next_max = np.max(self.q_table[batch.next_state, :], axis=1)

        # Calculate TD error
        td_errors = batch.reward + self.gamma * next_max * (1 - batch.done) - old_values
        
        # Update Q-table
        new_value = old_values + self.lr * td_errors
        self.q_table[batch.state, batch.action] = new_value

        # Update priority in buffer
        self.memory.update_priorities(batch_indices, td_errors)
            
    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

def train(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score):
    agent = QLearningExpReplayAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting Q-Learning training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store the transition in memory
            agent.memory.push(state, action, reward, next_state, done)
            # Performs one step of the learning process
            agent.learn()
            
            state = next_state
            episode_reward += reward

            if done:
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
                agent.lr = max(agent.lr_min, agent.lr * agent.lr_decay)
                break

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        if episode % (num_episodes / 10) == 0:
            print(f"Episode {episode}	Average Score: {np.mean(scores_deque):.2f}")
        
        if np.mean(scores_deque) >= target_score:
            print(f"Environment solved in {episode} episodes! Average Score: {np.mean(scores_deque):.2f}")
            break

    
    print("\nTraining complete.")
    
    return agent, scores
