#!/usr/bin/env python3
"""
Hyperparameter Tuning Script using Optuna

This script performs automated hyperparameter optimization for reinforcement learning agents. 
It uses Optuna, a hyperparameter optimization
framework, to search for the best hyperparameters.

Key Concepts:
- Objective Function: Defines what to optimize (average reward)
- Search Space: Defines which hyperparameters to tune and their ranges
- Optimization: Optuna tries different combinations and learns from results
- Pruning: Early stopping of unpromising trials to save computation

Usage:
    python tune_hyperparameters.py reinforce CartPole-v1 --n-trials 50
    python tune_hyperparameters.py dqn CartPole-v1 --n-trials 30 --episodes 500
"""

import argparse
import sys
import numpy as np

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.utils.environments import create_env, get_env_dimensions
from src.utils.seeding import seed_env, set_global_seeds


def create_agent(algorithm, state_dim, action_dim, trial=None, **fixed_params):
    """
    Create an agent with hyperparameters suggested by Optuna.
    
    Args:
        algorithm: Name of the algorithm (e.g., 'reinforce', 'dqn')
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        trial: Optuna trial object for suggesting hyperparameters
        **fixed_params: Fixed hyperparameters that won't be tuned
    
    Returns:
        Agent instance with suggested hyperparameters
    """
    if algorithm == 'reinforce':
        from src.reinforce import ReinforceAgent
        
        if trial is not None:
            # Define search space for REINFORCE hyperparameters
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            gamma = trial.suggest_float('gamma', 0.95, 0.999)
            entropy_coef = trial.suggest_float('entropy_coef', 0.0, 0.1)
            batch_episodes = trial.suggest_int('batch_episodes', 1, 10)
        else:
            # Use default values if no trial
            lr = fixed_params.get('lr', 3e-3)
            gamma = fixed_params.get('gamma', 0.99)
            entropy_coef = fixed_params.get('entropy_coef', 0.01)
            batch_episodes = fixed_params.get('batch_episodes', 5)
        
        return ReinforceAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            entropy_coef=entropy_coef,
            batch_episodes=batch_episodes
        )
    
    elif algorithm == 'dqn_experience_replay':
        from src.dqn_experience_replay import DQNExpReplayAgent
        
        if trial is not None:
            # Define search space for DQN hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
            gamma = trial.suggest_float('gamma', 0.95, 0.999)
            epsilon = trial.suggest_float('epsilon', 0.8, 1.0)
            epsilon_decay = trial.suggest_float('epsilon_decay', 0.990, 0.999)
            epsilon_min = trial.suggest_float('epsilon_min', 0.01, 0.1)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            buffer_size = trial.suggest_categorical('buffer_size', [5000, 10000, 20000])
        else:
            lr = fixed_params.get('lr', 1e-4)
            gamma = fixed_params.get('gamma', 0.99)
            epsilon = fixed_params.get('epsilon', 1.0)
            epsilon_decay = fixed_params.get('epsilon_decay', 0.995)
            epsilon_min = fixed_params.get('epsilon_min', 0.01)
            batch_size = fixed_params.get('batch_size', 64)
            buffer_size = fixed_params.get('buffer_size', 10000)
        
        return DQNExpReplayAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            batch_size=batch_size,
            buffer_size=buffer_size
        )
    
    elif algorithm == 'dqn_prioritised_experience_replay':
        from src.dqn_prioritised_experience_replay import DQNPrioritisedExpReplayAgent
        
        if trial is not None:
            # Define search space for DQN with prioritised replay
            lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
            gamma = trial.suggest_float('gamma', 0.95, 0.999)
            epsilon = trial.suggest_float('epsilon', 0.8, 1.0)
            epsilon_decay = trial.suggest_float('epsilon_decay', 0.990, 0.999)
            epsilon_min = trial.suggest_float('epsilon_min', 0.01, 0.1)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            buffer_size = trial.suggest_categorical('buffer_size', [5000, 10000, 20000])
            target_update = trial.suggest_int('target_update', 5, 20)
        else:
            lr = fixed_params.get('lr', 1e-4)
            gamma = fixed_params.get('gamma', 0.99)
            epsilon = fixed_params.get('epsilon', 1.0)
            epsilon_decay = fixed_params.get('epsilon_decay', 0.995)
            epsilon_min = fixed_params.get('epsilon_min', 0.01)
            batch_size = fixed_params.get('batch_size', 64)
            buffer_size = fixed_params.get('buffer_size', 10000)
            target_update = fixed_params.get('target_update', 10)
        
        return DQNPrioritisedExpReplayAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            batch_size=batch_size,
            buffer_size=buffer_size,
            target_update=target_update
        )
    
    elif algorithm == 'actor_critic':
        from src.actor_critic import ActorCriticAgent
        
        if trial is not None:
            lr_actor = trial.suggest_float('lr_actor', 1e-4, 1e-2, log=True)
            lr_critic = trial.suggest_float('lr_critic', 1e-4, 1e-2, log=True)
            gamma = trial.suggest_float('gamma', 0.95, 0.999)
        else:
            lr_actor = fixed_params.get('lr_actor', 1e-3)
            lr_critic = fixed_params.get('lr_critic', 1e-3)
            gamma = fixed_params.get('gamma', 0.99)
        
        return ActorCriticAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma
        )
    
    elif algorithm == 'ppo':
        from src.ppo import PPOAgent
        
        if trial is not None:
            lr_actor = trial.suggest_float('lr_actor', 1e-5, 1e-3, log=True)
            lr_critic = trial.suggest_float('lr_critic', 1e-4, 1e-2, log=True)
            gamma = trial.suggest_float('gamma', 0.95, 0.999)
            epsilon_clip = trial.suggest_float('epsilon_clip', 0.1, 0.3)
            K_epochs = trial.suggest_int('K_epochs', 3, 10)
            gae_lambda = trial.suggest_float('gae_lambda', 0.90, 0.99)
        else:
            lr_actor = fixed_params.get('lr_actor', 3e-4)
            lr_critic = fixed_params.get('lr_critic', 1e-3)
            gamma = fixed_params.get('gamma', 0.99)
            epsilon_clip = fixed_params.get('epsilon_clip', 0.2)
            K_epochs = fixed_params.get('K_epochs', 4)
            gae_lambda = fixed_params.get('gae_lambda', 0.95)
        
        return PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            epsilon_clip=epsilon_clip,
            K_epochs=K_epochs,
            gae_lambda=gae_lambda
        )
    
    else:
        raise ValueError(f"Algorithm '{algorithm}' not supported for tuning")


def evaluate_hyperparameters(algorithm, env_name, trial, num_episodes=1000, 
                            eval_frequency=100, seed=42):
    """
    Objective function for Optuna: Train agent and return performance metric.
    
    This function:
    1. Creates environment and agent with suggested hyperparameters
    2. Trains the agent for specified episodes
    3. Reports intermediate results to Optuna (for pruning)
    4. Returns final performance metric
    
    Args:
        algorithm: Algorithm name
        env_name: Environment name
        trial: Optuna trial object
        num_episodes: Number of training episodes
        eval_frequency: How often to report intermediate results
        seed: Random seed for reproducibility
    
    Returns:
        Average reward over last 100 episodes (objective to maximize)
    """
    # Set random seeds for reproducibility
    set_global_seeds(seed)
    
    # Create environment
    env = create_env(env_name)
    seed_env(env, seed)
    state_dim, action_dim = get_env_dimensions(env)
    
    # Create agent with suggested hyperparameters
    agent = create_agent(algorithm, state_dim, action_dim, trial)
    
    # Track rewards for evaluation
    episode_rewards = []
    recent_rewards = []  # Last 100 episodes
    
    try:
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store transition based on agent type
                if hasattr(agent, 'memory'):
                    # For DQN agents with replay buffer
                    next_state_for_buffer = None if done else next_state
                    agent.memory.push(state, action, reward, next_state_for_buffer, done)
                    agent.learn()
                
                state = next_state
                episode_reward += reward
            
            # End episode for algorithms that need it (like REINFORCE)
            if hasattr(agent, 'end_episode'):
                agent.end_episode()
            
            episode_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            # Report intermediate results to Optuna for pruning
            if episode % eval_frequency == 0:
                avg_reward = np.mean(recent_rewards)
                trial.report(avg_reward, episode)
                
                # Check if trial should be pruned (stopped early)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    
    finally:
        env.close()
    
    # Return average reward over last 100 episodes
    final_score = np.mean(episode_rewards[-100:])
    return final_score


def tune_hyperparameters(algorithm, env_name, n_trials=50, num_episodes=1000,
                        timeout=None, seed=42):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        algorithm: Algorithm to tune
        env_name: Environment to tune on
        n_trials: Number of hyperparameter combinations to try
        num_episodes: Episodes per trial
        timeout: Maximum time in seconds (None for no limit)
        seed: Random seed
    
    Returns:
        study: Optuna study object with results
    """
    print("\n" + "="*70)
    print(f"HYPERPARAMETER TUNING: {algorithm.upper()} on {env_name}")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Number of trials: {n_trials}")
    print(f"  - Episodes per trial: {num_episodes}")
    print(f"  - Timeout: {timeout if timeout else 'None'}")
    print(f"  - Random seed: {seed}")
    print("="*70 + "\n")
    
    # Create Optuna study
    # - TPESampler: Tree-structured Parzen Estimator (smart search)
    # - MedianPruner: Stops unpromising trials early
    study = optuna.create_study(
        direction='maximize',  # We want to maximize reward
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=200)
    )
    
    # Define objective function with fixed arguments
    def objective(trial):
        return evaluate_hyperparameters(
            algorithm=algorithm,
            env_name=env_name,
            trial=trial,
            num_episodes=num_episodes,
            seed=seed + trial.number  # Different seed per trial
        )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    return study


def print_results(study, algorithm):
    """Print optimization results in a clear, pedagogical format."""
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\n✓ Completed {len(study.trials)} trials")
    print(f"✓ Best trial: #{study.best_trial.number}")
    print(f"✓ Best score: {study.best_value:.2f}")
    
    print(f"\n{'Best Hyperparameters:'}")
    print("-" * 40)
    for param, value in study.best_params.items():
        print(f"  {param:20s} = {value}")
    
    # Show improvement over trials
    print(f"\n{'Trial Progress:'}")
    print("-" * 40)
    print(f"  First trial score:  {study.trials[0].value:.2f}")
    print(f"  Best trial score:   {study.best_value:.2f}")
    print(f"  Improvement:        {study.best_value - study.trials[0].value:+.2f}")
    
    # Statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 1:
        scores = [t.value for t in completed_trials]
        print(f"\n{'Score Statistics:'}")
        print("-" * 40)
        print(f"  Mean:   {np.mean(scores):.2f}")
        print(f"  Std:    {np.std(scores):.2f}")
        print(f"  Min:    {np.min(scores):.2f}")
        print(f"  Max:    {np.max(scores):.2f}")
    
    print("\n" + "="*70 + "\n")


def save_best_config(study, algorithm, env_name, output_file=None):
    """Save best hyperparameters to a file."""
    if output_file is None:
        output_file = f"best_config_{algorithm}_{env_name}.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"# Best hyperparameters for {algorithm.upper()} on {env_name}\n")
        f.write(f"# Found by Optuna after {len(study.trials)} trials\n")
        f.write(f"# Best score: {study.best_value:.2f}\n\n")
        
        for param, value in study.best_params.items():
            f.write(f"{param} = {value}\n")
    
    print(f"✓ Saved best configuration to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Tune hyperparameters for RL algorithms using Optuna',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune REINFORCE on CartPole with 50 trials
  python tune_hyperparameters.py reinforce CartPole-v1 --n-trials 50
  
  # Quick tuning with fewer episodes and trials
  python tune_hyperparameters.py dqn_experience_replay CartPole-v1 --n-trials 20 --episodes 500
  
  # Tune DQN with prioritised experience replay
  python tune_hyperparameters.py dqn_prioritised_experience_replay CartPole-v1 --n-trials 30
  
  # Set timeout instead of number of trials
  python tune_hyperparameters.py ppo CartPole-v1 --timeout 3600
        """
    )
    
    parser.add_argument('algorithm', type=str, 
                       choices=['reinforce', 'dqn_experience_replay', 
                               'dqn_prioritised_experience_replay', 'actor_critic', 'ppo'],
                       help='Algorithm to tune')
    parser.add_argument('env', type=str,
                       help='Environment name (e.g., CartPole-v1)')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials (default: 50)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Episodes per trial (default: 1000)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds (default: None)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--save', type=str, default=None,
                       help='File to save best config (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Run optimization
    study = tune_hyperparameters(
        algorithm=args.algorithm,
        env_name=args.env,
        n_trials=args.n_trials,
        num_episodes=args.episodes,
        timeout=args.timeout,
        seed=args.seed
    )
    
    # Print results
    print_results(study, args.algorithm)
    
    # Save best configuration
    save_best_config(study, args.algorithm, args.env, args.save)
    
    # Optional: Show parameter importance (requires scikit-learn and more trials)
    if len(study.trials) >= 10:
        try:
            print("\nParameter Importance:")
            print("-" * 40)
            importances = optuna.importance.get_param_importances(study)
            for param, importance in importances.items():
                print(f"  {param:20s} = {importance:.3f}")
            print("\nNote: Higher importance = more impact on performance")
        except ImportError:
            print("\nParameter Importance: Skipped")
            print("(Install scikit-learn to enable: uv pip install scikit-learn)")
        except Exception as e:
            print(f"\nCould not compute parameter importance: {e}")


if __name__ == "__main__":
    main()
