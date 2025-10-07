import numpy as np
from src.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    A simple random agent that selects actions uniformly at random.
    Useful for testing environments and as a baseline for comparison.
    """
    
    def __init__(self, action_dim):
        """
        Initialize the random agent.
        
        :param action_dim: The number of possible actions in the environment.
        """
        self.action_dim = action_dim
    
    def select_action(self, state, greedy=False):
        """
        Selects a random action regardless of the state or greedy parameter.
        
        :param state: The current state of the environment (ignored).
        :param greedy: Whether to act greedily (ignored for random agent).
        :return: A randomly selected action.
        """
        return np.random.choice(self.action_dim)
    
    def save(self, path):
        """
        Save method for compatibility with BaseAgent interface.
        Since this is a random agent, there are no parameters to save.
        
        :param path: The file path where the model should be saved (unused).
        """
        print(f"RandomAgent: No parameters to save. Skipping save to {path}")
    
    def load(self, path):
        """
        Load method for compatibility with BaseAgent interface.
        Since this is a random agent, there are no parameters to load.
        
        :param path: The file path from which to load the model (unused).
        """
        print(f"RandomAgent: No parameters to load. Skipping load from {path}")