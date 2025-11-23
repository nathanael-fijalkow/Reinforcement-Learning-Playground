"""Benchmark script to run RL algorithms across multiple seeds.

This script runs all compatible algorithms (discrete or continuous based on the
environment) with multiple random seeds and stores results in per-algorithm CSV files.
Each run includes training scores per episode and final evaluation scores.
"""
import argparse
import csv
import importlib
import inspect
import os
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from src.utils.environments import create_env, get_env_dimensions
from src.utils.env_config import get_env_params
from src.utils.seeding import seed_env, set_global_seeds


def discover_algorithms() -> Dict:
    """Discover algorithm agent classes dynamically from src/ directory."""
    from src.base_agent import BaseAgent

    algorithms = {}
    for file_path in Path("src").glob("*.py"):
        if file_path.name in {"__init__.py", "base_agent.py"}:
            continue
        if file_path.name.startswith("_"):
            continue
        module_name = file_path.stem
        
        # Skip random agent (not useful for benchmarking)
        if "random" in module_name.lower():
            continue
        
        try:
            module = importlib.import_module(f"src.{module_name}")
            agent_classes = [
                (name, cls)
                for name, cls in inspect.getmembers(module, inspect.isclass)
                if issubclass(cls, BaseAgent) and cls is not BaseAgent
            ]
            if not agent_classes:
                continue
            train_func = getattr(module, "train", None)
            if not train_func:
                continue  # Skip algorithms without train function
            lname = module_name.lower()
            setting = "any"
            if any(k in lname for k in ["dqn", "actor", "critic", "ppo"]):
                setting = "continuous"
            elif any(k in lname for k in ["q_learning", "sarsa", "monte", "dyna"]):
                setting = "discrete"
            for class_name, agent_class in agent_classes:
                algorithms[module_name] = {
                    "agent": agent_class,
                    "trainer": train_func,
                    "setting": setting,
                    "class_name": class_name,
                }
                print(f"✓ Found: {module_name} ({class_name}) - {setting}")
        except Exception as e:
            print(f"✗ Could not import {module_name}: {e}")
    return algorithms


def get_compatible_algorithms(algorithms: Dict, setting: str) -> Dict:
    """Filter algorithms compatible with the given setting (discrete/continuous)."""
    return {
        name: params
        for name, params in algorithms.items()
        if params["setting"] == "any" or params["setting"] == setting
    }


def run_algorithm_with_seed(
    algo_name: str,
    algo_params: Dict,
    env_name: str,
    seed: int,
    num_episodes: int,
    max_steps: int,
    target_score: float,
    num_eval_episodes: int = 10,
) -> Tuple[List[float], float, bool]:
    """Run algorithm with seed and return training + eval scores.
    
    Returns:
        Tuple of (training_scores_list, eval_score_mean, success_flag)
    """
    print(f"  Running {algo_name} with seed {seed}...")
    
    # Set global seeds
    set_global_seeds(seed)
    
    # Create and seed environment
    env = create_env(env_name)
    seed_env(env, seed, seed_spaces=True, only_once=True)
    state_dim, action_dim = get_env_dimensions(env)
    
    try:
        trainer = algo_params["trainer"]
        agent, training_scores = trainer(
            env, state_dim, action_dim, num_episodes, max_steps, target_score
        )
        
        # Evaluate trained agent
        eval_env = create_env(env_name)
        seed_env(eval_env, seed + 999999, seed_spaces=True, only_once=True)
        eval_scores = []
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = eval_env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                action = agent.select_action(state, greedy=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
                steps += 1
            eval_scores.append(ep_reward)
        eval_mean = float(np.mean(eval_scores))
        
        env.close()
        eval_env.close()
        return training_scores, eval_mean, True
    except Exception as e:
        print(f"    Error running {algo_name} with seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return [], 0.0, False


def save_algorithm_results(
    algo_name: str,
    env_name: str,
    setting: str,
    training_results: List[Dict],
    eval_results: List[Dict],
    output_dir: str,
):
    """Save results for a single algorithm to CSV files.
    
    Creates two files:
    - <algo>_<env>_training.csv: seed, episode, score
    - <algo>_<env>_eval.csv: seed, eval_score
    """
    # Save training scores
    training_path = os.path.join(
        output_dir, f"{algo_name}_{env_name}_training.csv"
    )
    with open(training_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "episode", "score"])
        writer.writeheader()
        writer.writerows(training_results)
    
    # Save evaluation scores
    eval_path = os.path.join(output_dir, f"{algo_name}_{env_name}_eval.csv")
    with open(eval_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "eval_score"])
        writer.writeheader()
        writer.writerows(eval_results)
    
    # Save metadata
    meta_path = os.path.join(output_dir, f"{algo_name}_{env_name}_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"algorithm: {algo_name}\n")
        f.write(f"environment: {env_name}\n")
        f.write(f"setting: {setting}\n")
    
    print(f"  ✓ Saved results for {algo_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RL algorithms with multiple seeds"
    )
    parser.add_argument("environment", help="Gymnasium environment name")
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Specific algorithm to benchmark (if not specified, runs all compatible algorithms)",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=10, help="Number of random seeds to run"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=None, help="Number of training episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Max steps per episode"
    )
    parser.add_argument(
        "--target-score", type=float, default=None, help="Target score threshold"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42, help="Base seed for generating seeds"
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for final evaluation",
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover algorithms
    print("Discovering algorithms...")
    all_algorithms = discover_algorithms()
    
    # Determine environment setting
    print(f"\nAnalyzing environment: {args.environment}")
    test_env = create_env(args.environment)
    if isinstance(test_env.observation_space, gym.spaces.Discrete):
        setting = "discrete"
    elif isinstance(
        test_env.observation_space,
        (gym.spaces.Box, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary),
    ):
        setting = "continuous"
    else:
        raise ValueError(
            f"Unsupported observation space: {type(test_env.observation_space)}"
        )
    test_env.close()
    print(f"Environment type: {setting}")
    
    # Get compatible algorithms
    compatible_algos = get_compatible_algorithms(all_algorithms, setting)
    if not compatible_algos:
        print(f"No compatible algorithms found for {setting} environment!")
        return
    
    # Filter to single algorithm if specified
    if args.algorithm:
        if args.algorithm not in compatible_algos:
            if args.algorithm in all_algorithms:
                print(f"Error: Algorithm '{args.algorithm}' is not compatible with {setting} environment!")
                print(f"This algorithm requires: {all_algorithms[args.algorithm]['setting']}")
            else:
                print(f"Error: Algorithm '{args.algorithm}' not found!")
                print(f"\nAvailable algorithms:")
                for name in sorted(all_algorithms.keys()):
                    print(f"  - {name} ({all_algorithms[name]['setting']})")
            return
        compatible_algos = {args.algorithm: compatible_algos[args.algorithm]}
        print(f"\nRunning single algorithm: {args.algorithm}")
    else:
        print(f"\nCompatible algorithms ({len(compatible_algos)}):")
        for name in compatible_algos.keys():
            print(f"  - {name}")
    
    # Get environment parameters from config
    num_episodes, max_steps, target_score = get_env_params(
        args.environment, args.num_episodes, args.max_steps, args.target_score
    )
    
    print("\nBenchmark settings:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Target score: {target_score}")
    print(f"  Seeds: {args.num_seeds}")
    print(f"  Eval episodes: {args.num_eval_episodes}")
    
    # Generate seeds
    rng = np.random.RandomState(args.base_seed)
    seeds = rng.randint(0, 100000, size=args.num_seeds).tolist()
    print(f"  Using seeds: {seeds}")
    
    # Run benchmarks
    print("\n" + "=" * 60)
    print("Starting benchmark runs...")
    print("=" * 60 + "\n")
    
    for algo_name, algo_params in compatible_algos.items():
        print(f"Algorithm: {algo_name}")
        
        training_results = []
        eval_results = []
        
        for seed_idx, seed in enumerate(seeds, 1):
            training_scores, eval_score, success = run_algorithm_with_seed(
                algo_name,
                algo_params,
                args.environment,
                seed,
                num_episodes,
                max_steps,
                target_score,
                args.num_eval_episodes,
            )
            if success and training_scores:
                # Store training scores
                for episode_idx, score in enumerate(training_scores):
                    training_results.append({
                        "seed": seed,
                        "episode": episode_idx,
                        "score": score,
                    })
                # Store eval score
                eval_results.append({
                    "seed": seed,
                    "eval_score": eval_score,
                })
                eval_msg = f"(eval: {eval_score:.2f})"
                print(f"  Seed {seed_idx}/{args.num_seeds} complete {eval_msg}")
            else:
                print(f"  Seed {seed_idx}/{args.num_seeds} failed")
        
        # Save results for this algorithm immediately
        if training_results and eval_results:
            save_algorithm_results(
                algo_name,
                args.environment,
                setting,
                training_results,
                eval_results,
                args.output_dir,
            )
        print()
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
