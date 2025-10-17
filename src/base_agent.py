from abc import ABC, abstractmethod
from numpy import intp

class BaseAgent(ABC):
    """
    An abstract base class for all Reinforcement Learning agents.
    This class defines the common interface that every agent must implement.
    """
    
    @abstractmethod
    def select_action(self, state, greedy=False) -> int | intp :
        """
        Selects an action based on the current state.

        :param state: The current state of the environment.
        :param greedy: If True, select the action with the highest value (for evaluation).
                       If False, allow for exploration (for training).
        :return: The selected action.
        """
        pass

    @abstractmethod
    def save(self, path) -> None:
        """
        Saves the agent's model/parameters to a file.

        :param path: The file path where the model should be saved.
        """
        pass

    @abstractmethod
    def load(self, path) -> None:
        """
        Loads the agent's model/parameters from a file.

        :param path: The file path from which to load the model.
        """
        pass
