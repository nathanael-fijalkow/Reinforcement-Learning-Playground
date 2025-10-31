import gymnasium as gym
from gymnasium import spaces


def create_env(env_name, render_mode=None):
    """
    Creates a gymnasium environment.
    """
    env = gym.make(env_name, render_mode=render_mode)
    return env


def get_env_dimensions(env):
    """
    Gets the state and action dimensions of a gymnasium environment.
    """
    if isinstance(env.observation_space, spaces.Box):
        state_dim = env.observation_space.shape[0]
    else:
        state_dim = env.observation_space.n

    action_dim = env.action_space.n
    return state_dim, action_dim
