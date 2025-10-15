import argparse
import os
import gymnasium as gym

AVAILABLE_ALGORITHMS = {}

try:
    from src.random_agent import RandomAgent
    AVAILABLE_ALGORITHMS["random"] = {
        "agent": RandomAgent,
        "trainer": None,  # Random agent doesn't need training
        "setting": "any"
    }
except ImportError as e:
    print(f"Random agent not yet implemented")


try:
    from src.sarsa import train as train_sarsa, SarsaAgent
    AVAILABLE_ALGORITHMS["sarsa"] = {
        "agent": SarsaAgent,
        "trainer": train_sarsa,  # Random agent doesn't need training
        "setting": "discrete"
    }
except ImportError as e:
    print(f"Sarsa not yet implemented")

try:
    from src.monte_carlo import train as train_monte_carlo, MonteCarlo
    AVAILABLE_ALGORITHMS["monte_carlo"] = {
        "agent": MonteCarlo,
        "trainer": train_monte_carlo,
        "setting": "discrete"
    }
except ImportError as e:
    print(f"Monte Carlo not yet implemented")

try:
    from src.q_learning import train as train_q_learning, QLearningAgent
    AVAILABLE_ALGORITHMS["q_learning"] = {
        "agent": QLearningAgent,
        "trainer": train_q_learning,
        "setting": "discrete"
    }
except ImportError as e:
    print(f"Q-Learning not yet implemented")

try:
    from src.q_learning_experience_replay import train as train_q_learning_experience_replay, QLearningExpReplayAgent
    AVAILABLE_ALGORITHMS["q_learning_experience_replay"] = {
        "agent": QLearningExpReplayAgent,
        "trainer": train_q_learning_experience_replay,
        "setting": "discrete"
    }
except ImportError as e:
    print(f"Q-Learning not yet implemented")

try:
    from src.dyna_q import train as train_dyna_q, DynaQAgent
    AVAILABLE_ALGORITHMS["dyna_q"] = {
        "agent": DynaQAgent,
        "trainer": train_dyna_q,
        "setting": "discrete"
    }
except ImportError as e:
    print(f"Dyna-Q not yet implemented")

try:
    from src.dqn import train as train_dqn, DQNAgent
    AVAILABLE_ALGORITHMS["dqn"] = {
        "agent": DQNAgent,
        "trainer": train_dqn,
        "setting": "continuous"
    }
except ImportError as e:
    print(f"DQN not yet implemented")

try:
    from src.actor_critic import train as train_actor_critic, ActorCriticAgent
    AVAILABLE_ALGORITHMS["actor_critic"] = {
        "agent": ActorCriticAgent,
        "trainer": train_actor_critic,
        "setting": "continuous"
    }
except ImportError as e:
    print(f"Actor-Critic not yet implemented")

try:
    from src.ppo import train as train_ppo, PPOAgent
    AVAILABLE_ALGORITHMS["ppo"] = {
        "agent": PPOAgent,
        "trainer": train_ppo,
        "setting": "continuous"
    }
except ImportError as e:
    print(f"PPO not yet implemented")

from src.environments import create_env, get_env_dimensions
from src.policy_evaluation import evaluate_policy, run_simulation
from src.plotting import plot_scores

def main():
    parser = argparse.ArgumentParser(description="Run RL algorithms")
    parser.add_argument("algorithm", choices=list(AVAILABLE_ALGORITHMS.keys()), help="The algorithm to run")
    parser.add_argument("environment", help="The environment name from Gymnasium")
    parser.add_argument("--num-episodes", type=int, default=None, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps per episode")
    parser.add_argument("--target-score", type=float, default=None, help="Target score to solve the environment")
    parser.add_argument("--simulate", action="store_true", help="Run a simulation with rendering after training")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model")
    parser.add_argument("--load-model", type=str, default=None, help="Load a trained model from the specified path")
    parser.add_argument("--plot", action="store_true", help="Plot the training scores")
    args = parser.parse_args()

    # Create the environment
    env_name = args.environment
    env = create_env(env_name)

    state_dim, action_dim = get_env_dimensions(env)

    # Hyperparameter settings
    if env_name == "CartPole-v1":
        num_episodes = args.num_episodes if args.num_episodes is not None else 2000
        max_steps_per_episode = args.max_steps if args.max_steps is not None else 500
        target_score = args.target_score if args.target_score is not None else 475.0
    elif env_name == "FrozenLake-v1":
        num_episodes = args.num_episodes if args.num_episodes is not None else 10000
        max_steps_per_episode = args.max_steps if args.max_steps is not None else 100
        target_score = args.target_score if args.target_score is not None else 0.95
    else:
        print(f"Using custom settings for environment: {env_name}")
        num_episodes = args.num_episodes if args.num_episodes is not None else 1000
        max_steps_per_episode = args.max_steps if args.max_steps is not None else 200
        target_score = args.target_score if args.target_score is not None else 195.0 

    # Determine whether the state space is discrete or continuous
    if isinstance(env.observation_space, gym.spaces.Discrete):
        setting = "discrete"
    elif isinstance(env.observation_space, (gym.spaces.Box, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
        setting = "continuous"
    else:
        raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}.")

    agent = None
    scores = []
    
    algo_params = AVAILABLE_ALGORITHMS[args.algorithm]
    AgentClass = algo_params["agent"]
    trainer = algo_params["trainer"]
    required_setting = algo_params["setting"]

    if required_setting != "any" and setting != required_setting:
        raise ValueError(f"Algorithm '{args.algorithm}' requires a {required_setting} environment, but got a {setting} one.")

    if args.load_model:
        print(f"Loading agent for {args.algorithm} from {args.load_model}...")
        agent = AgentClass(state_dim, action_dim)
        if args.algorithm != "random":
            agent.load(args.load_model)
    else:
        if trainer:
            agent, scores = trainer(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score)
        else:
            agent = AgentClass(action_dim)
            print(f"Created {AgentClass.__name__} with action_dim={action_dim}")
            scores = []

    if args.save_model:
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        if args.algorithm != "random":  # Random agent has no parameters to save
            extension = ".pth" if AVAILABLE_ALGORITHMS[args.algorithm]['setting'] == 'continuous' else ".npy"
            model_path = os.path.join(model_dir, f"{args.algorithm}_{args.environment}{extension}")
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
        run_simulation(agent, sim_env)
        sim_env.close()

    env.close()

if __name__ == "__main__":
    main()
