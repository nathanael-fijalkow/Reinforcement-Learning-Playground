"""Generate comparison plots from benchmark results.

This script reads benchmark results from CSV files and generates two types
of plots: rolling average training scores and final evaluation scores.
Plots are generated separately for discrete and continuous environments.
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def find_result_files(results_dir: str) -> Dict[str, Dict[str, List[Path]]]:
    """Find all result CSV files and organize by environment and algorithm.
    
    Returns:
        Dict structure: {env_name: {"training": [...], "eval": [...]}}
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_dir}")
    
    # Find all training and eval CSV files
    training_files = list(results_path.glob("*_training.csv"))
    
    # Organize by environment
    env_results = {}
    
    for train_file in training_files:
        # Parse filename: {algo}_{env}_training.csv
        filename = train_file.stem  # Remove .csv
        parts = filename.rsplit("_", 2)  # Split from right: algo, env, "training"
        if len(parts) != 3 or parts[2] != "training":
            continue
        
        algo_name = parts[0]
        env_name = parts[1]
        
        # Find corresponding eval file
        eval_file = train_file.parent / f"{algo_name}_{env_name}_eval.csv"
        if not eval_file.exists():
            print(f"Warning: No eval file found for {algo_name} on {env_name}")
            continue
        
        # Add to results
        if env_name not in env_results:
            env_results[env_name] = {"training": [], "eval": [], "algos": []}
        env_results[env_name]["training"].append(train_file)
        env_results[env_name]["eval"].append(eval_file)
        env_results[env_name]["algos"].append(algo_name)
    
    return env_results


def detect_environment_type(env_name: str) -> str:
    """Detect if environment is discrete or continuous.
    
    Uses heuristics based on common environment names.
    """
    discrete_keywords = ["frozenlake", "taxi", "cliffwalking"]
    continuous_keywords = ["cartpole", "mountaincar", "pendulum", "acrobot"]
    
    env_lower = env_name.lower()
    
    for keyword in discrete_keywords:
        if keyword in env_lower:
            return "discrete"
    
    for keyword in continuous_keywords:
        if keyword in env_lower:
            return "continuous"
    
    # Default to continuous
    print(f"Warning: Could not detect type for {env_name}, assuming continuous")
    return "continuous"


def load_training_data(
    training_files: List[Path], algo_names: List[str]
) -> pd.DataFrame:
    """Load and combine training data from multiple algorithm CSV files."""
    all_data = []
    
    for train_file, algo_name in zip(training_files, algo_names):
        df = pd.read_csv(train_file)
        df["algorithm"] = algo_name
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def load_eval_data(eval_files: List[Path], algo_names: List[str]) -> pd.DataFrame:
    """Load and combine evaluation data from multiple algorithm CSV files."""
    all_data = []
    
    for eval_file, algo_name in zip(eval_files, algo_names):
        df = pd.read_csv(eval_file)
        df["algorithm"] = algo_name
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def plot_rolling_average(
    df: pd.DataFrame,
    env_name: str,
    env_type: str,
    output_dir: str,
    window: int = 50,
):
    """Plot rolling average of training scores."""
    sns.set_theme(style="darkgrid")
    
    # Calculate rolling average for each algorithm-seed combination
    df_rolling = df.copy()
    df_rolling = df_rolling.sort_values(["algorithm", "seed", "episode"])
    df_rolling["rolling_score"] = df_rolling.groupby(
        ["algorithm", "seed"]
    )["score"].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(
        data=df_rolling,
        x="episode",
        y="rolling_score",
        hue="algorithm",
        errorbar=("ci", 95),
        ax=ax,
    )
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Rolling Average Score (window={window})", fontsize=12)
    ax.set_title(
        f"Rolling Average: {env_name} ({env_type})\n(Mean ± 95% CI)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(
        output_dir, f"{env_name}_{env_type}_rolling.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Saved rolling average plot: {output_file}")
    plt.close()


def plot_eval_scores(
    df: pd.DataFrame,
    env_name: str,
    env_type: str,
    output_dir: str,
):
    """Plot final evaluation scores as a bar chart with error bars."""
    sns.set_theme(style="darkgrid")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use barplot to show mean with 95% CI
    sns.barplot(
        data=df,
        x="algorithm",
        y="eval_score",
        hue="algorithm",
        errorbar=("ci", 95),
        ax=ax,
        palette="Set2",
        legend=False,
    )
    
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Evaluation Score", fontsize=12)
    ax.set_title(
        f"Final Evaluation: {env_name} ({env_type})\n(Mean ± 95% CI)",
        fontsize=14,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    output_file = os.path.join(
        output_dir, f"{env_name}_{env_type}_evaluation.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Saved evaluation plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark_results",
        help="Directory containing benchmark result CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save generated plots",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Window size for rolling average",
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all result files
    print("Scanning for result files...")
    env_results = find_result_files(args.results_dir)
    
    if not env_results:
        print("No result files found!")
        return
    
    print(f"Found results for {len(env_results)} environment(s):")
    for env_name, data in env_results.items():
        print(f"  - {env_name}: {len(data['algos'])} algorithm(s)")
    
    # Generate plots for each environment
    print("\nGenerating plots...")
    for env_name, data in env_results.items():
        print(f"\nEnvironment: {env_name}")
        
        # Detect environment type
        env_type = detect_environment_type(env_name)
        print(f"  Type: {env_type}")
        
        # Load training data
        training_df = load_training_data(data["training"], data["algos"])
        
        # Load eval data
        eval_df = load_eval_data(data["eval"], data["algos"])
        
        # Generate two plots
        plot_rolling_average(
            training_df, env_name, env_type, args.output_dir, args.rolling_window
        )
        plot_eval_scores(eval_df, env_name, env_type, args.output_dir)
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
