import argparse
import os
import importlib
import inspect
from pathlib import Path

import gymnasium as gym

from src.utils.environments import create_env, get_env_dimensions
from src.utils.env_config import get_env_params
from src.utils.policy_evaluation import evaluate_policy, run_simulation
from src.utils.plotting import plot_scores
from src.utils.seeding import seed_env, set_global_seeds


def discover_algorithms():
    from src.base_agent import BaseAgent

    AVAILABLE_ALGORITHMS = {}
    src_path = Path("src")

    python_files = [
        f
        for f in src_path.glob("*.py")
        if f.name not in ["__init__.py", "base_agent.py"]
        and not f.name.startswith("_")
    ]

    for file_path in python_files:
        module_name = file_path.stem
        try:
            module = importlib.import_module(f"src.{module_name}")

            agent_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseAgent) and obj is not BaseAgent:
                    agent_classes.append((name, obj))

            train_func = getattr(module, "train", None)

            for class_name, agent_class in agent_classes:
                algo_key = module_name

                setting = "any"
                if any(
                    kw in module_name.lower() for kw in ["dqn", "actor", "critic", "ppo"]
                ):
                    setting = "continuous"
                elif any(
                    kw in module_name.lower() for kw in ["q_learning", "sarsa", "monte", "dyna"]
                ):
                    setting = "discrete"

                AVAILABLE_ALGORITHMS[algo_key] = {
                    "agent": agent_class,
                    "trainer": train_func,
                    "setting": setting,
                    "class_name": class_name,
                }
                print(f"✓ Found: {algo_key} ({class_name}) - {setting}")

        except Exception as e:
            print(f"✗ Could not import {module_name}: {e}")

    return AVAILABLE_ALGORITHMS


AVAILABLE_ALGORITHMS = discover_algorithms()

if not AVAILABLE_ALGORITHMS:
    print("Warning: No algorithms were found!")

print(f"\nTotal algorithms available: {len(AVAILABLE_ALGORITHMS)}")
print(f"Algorithm keys: {list(AVAILABLE_ALGORITHMS.keys())}\n")


def main():
    parser = argparse.ArgumentParser(description="Run RL algorithms")
    parser.add_argument(
        "algorithm",
        choices=list(AVAILABLE_ALGORITHMS.keys()),
        help="Algorithm key to run",
    )
    parser.add_argument("environment", help="Gymnasium environment name")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--num-episodes", type=int, default=None, help="Number of training episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Max steps per episode"
    )
    parser.add_argument(
        "--target-score", type=float, default=None, help="Target score to solve"
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Render one simulation after training"
    )
    parser.add_argument(
        "--save-model", action="store_true", help="Persist trained model to disk"
    )
    parser.add_argument(
        "--load-model", type=str, default=None, help="Path to load a trained model"
    )
    parser.add_argument("--plot", action="store_true", help="Plot training score curve")
    args = parser.parse_args()

    # Global seeding
    set_global_seeds(args.seed)

    # Create and seed environment
    env_name = args.environment
    env = create_env(env_name)
    seed_env(env, args.seed, seed_spaces=True, only_once=True)

    state_dim, action_dim = get_env_dimensions(env)

    # Get environment parameters from config
    num_episodes, max_steps_per_episode, target_score = get_env_params(
        env_name, args.num_episodes, args.max_steps, args.target_score
    )

    # Determine whether the state space is discrete or continuous
    if isinstance(env.observation_space, gym.spaces.Discrete):
        setting = "discrete"
    elif isinstance(
        env.observation_space,
        (gym.spaces.Box, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary),
    ):
        setting = "continuous"
    else:
        raise ValueError(
            f"Unsupported observation space type: {type(env.observation_space)}."
        )

    agent = None
    scores = []

    algo_params = AVAILABLE_ALGORITHMS[args.algorithm]
    AgentClass = algo_params["agent"]
    trainer = algo_params["trainer"]
    required_setting = algo_params["setting"]

    if required_setting != "any" and setting != required_setting:
        raise ValueError(
            "Algorithm '{algo}' requires a {req} environment, but got a {got} one.".format(
                algo=args.algorithm, req=required_setting, got=setting
            )
        )

    if args.load_model:
        print(f"Loading agent for {args.algorithm} from {args.load_model}...")
        agent = AgentClass(state_dim, action_dim)
        if args.algorithm != "random":
            agent.load(args.load_model)
    else:
        if trainer:
            agent, scores = trainer(
                env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
            )
        else:
            agent = AgentClass(action_dim)
            print(f"Created {AgentClass.__name__} with action_dim={action_dim}")
            scores = []

    if args.save_model:
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if args.algorithm != "random":  # Random agent has no parameters to save
            extension = (
                ".pth"
                if AVAILABLE_ALGORITHMS[args.algorithm]["setting"] == "continuous"
                else ".npy"
            )
            model_path = os.path.join(
                model_dir, f"{args.algorithm}_{args.environment}{extension}"
            )
            agent.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("RandomAgent: No parameters to save.")

    if args.plot and scores:
        plot_scores(scores, args.algorithm, args.environment)

    # Evaluate the trained agent
    avg_reward = evaluate_policy(agent, env)
    print(f"Average reward: {avg_reward:.2f}")

    if args.simulate:
        sim_env = create_env(env_name, render_mode="human")
        seed_env(sim_env, args.seed, seed_spaces=True, only_once=True)
        run_simulation(agent, sim_env)
        sim_env.close()

    env.close()


if __name__ == "__main__":
    main()
