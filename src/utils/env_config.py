"""Utility for loading environment configuration defaults."""
import json
import os
from typing import Dict, Any


def load_env_config(env_name: str) -> Dict[str, Any]:
    """Load configuration for a specific environment.
    
    Args:
        env_name: Name of the Gymnasium environment
        
    Returns:
        Dictionary with num_episodes, max_steps, and target_score
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "env_configs.json"
    )
    
    with open(config_path, "r") as f:
        configs = json.load(f)
    
    # Return specific config or default
    return configs.get(env_name, configs["default"])


def get_env_params(
    env_name: str,
    num_episodes: int | None = None,
    max_steps: int | None = None,
    target_score: float | None = None,
) -> tuple:
    """Get environment parameters with CLI overrides.
    
    Args:
        env_name: Name of the environment
        num_episodes: Override for number of episodes (None = use default)
        max_steps: Override for max steps (None = use default)
        target_score: Override for target score (None = use default)
        
    Returns:
        Tuple of (num_episodes, max_steps, target_score)
    """
    config = load_env_config(env_name)
    
    final_episodes = (
        num_episodes if num_episodes is not None else config["num_episodes"]
    )
    final_steps = max_steps if max_steps is not None else config["max_steps"]
    final_score = (
        target_score if target_score is not None else config["target_score"]
    )
    
    return final_episodes, final_steps, final_score
