import time

import numpy as np


def evaluate_policy(agent, env, episodes=100):
    """
    Evaluates the policy of an agent.
    """
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            action = agent.select_action(state, greedy=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, truncated, _ = step_result
            else:
                next_state, reward, done, _ = step_result
                truncated = False

            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


def run_simulation(agent, env):
    """
    Runs a single episode of the agent in the environment with rendering.

    :param agent: The agent to simulate.
    :param env: The environment to run the simulation in (must have rendering enabled).
    """
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, greedy=True)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        env.render()
        time.sleep(0.05)  # Small delay to make rendering watchable
