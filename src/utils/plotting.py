import matplotlib.pyplot as plt
import numpy as np
import os

def plot_scores(scores, algorithm_name, env_name):
    """
    Plots the training scores and saves the plot to a file.

    :param scores: A list of scores from training.
    :param algorithm_name: The name of the algorithm used.
    :param env_name: The name of the environment.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title(f'Training Scores for {algorithm_name} on {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Calculate and plot a rolling average
    if len(scores) >= 100:
        rolling_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(scores)), rolling_avg, label='100-episode rolling average')
        plt.legend()

    # Save the plot
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    plot_path = os.path.join(plots_dir, f"{algorithm_name}_{env_name}_scores.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()
