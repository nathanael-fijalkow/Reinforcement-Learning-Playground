import argparse
import os
import gymnasium as gym
from src.q_learning import train as train_q_learning, QLearningAgent
from src.dqn import train as train_dqn, DQNAgent
from src.actor_critic import train as train_actor_critic, ActorCriticAgent
from src.ppo import train as train_ppo, PPOAgent
from src.dyna_q import train as train_dyna_q, DynaQAgent
from src.environments import create_env, get_env_dimensions
from src.policy_evaluation import evaluate_policy, run_simulation
from src.plotting import plot_scores

def main():
    parser = argparse.ArgumentParser(description="Run RL algorithms")
    parser.add_argument("algorithm", choices=["q_learning", "dqn", "actor_critic", "ppo", "dyna_q"], help="The algorithm to run")
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
    if args.load_model:
        print(f"Loading agent for {args.algorithm} from {args.load_model}...")
        if args.algorithm == "q_learning":
            agent = QLearningAgent(state_dim, action_dim)
        elif args.algorithm == "dqn":
            agent = DQNAgent(state_dim, action_dim)
        elif args.algorithm == "actor_critic":
            agent = ActorCriticAgent(state_dim, action_dim)
        elif args.algorithm == "ppo":
            agent = PPOAgent(state_dim, action_dim)
        elif args.algorithm == "dyna_q":
            agent = DynaQAgent(state_dim, action_dim)
        agent.load(args.load_model)
    else:
        if args.algorithm == "q_learning":
            assert(setting == "discrete")
            agent, scores = train_q_learning(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score)
        elif args.algorithm == "dqn":
            assert(setting == "continuous")
            agent, scores = train_dqn(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score)
        elif args.algorithm == "actor_critic":
            assert(setting == "continuous")
            agent, scores = train_actor_critic(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score)
        elif args.algorithm == "ppo":
            assert(setting == "continuous")
            agent, scores = train_ppo(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score)
        elif args.algorithm == "dyna_q":
            assert(setting == "discrete")
            agent, scores = train_dyna_q(env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score)
        else:
            raise ValueError("Algorithm not supported")

    if args.save_model:
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        extension = ".npy" if args.algorithm in ["q_learning", "dyna_q"] else ".pth"
        model_path = os.path.join(model_dir, f"{args.algorithm}_{args.environment}{extension}")
        agent.save(model_path)
        print(f"Model saved to {model_path}")

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
