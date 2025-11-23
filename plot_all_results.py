"""Generate comparison plots for all benchmark results (both discrete and continuous)."""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def detect_environment_type(env_name: str) -> str:
    """Detect if environment is discrete or continuous."""
    discrete_keywords = ["frozenlake", "taxi", "cliffwalking"]
    continuous_keywords = ["cartpole", "mountaincar", "pendulum", "acrobot"]
    
    env_lower = env_name.lower()
    
    for keyword in discrete_keywords:
        if keyword in env_lower:
            return "discrete"
    
    for keyword in continuous_keywords:
        if keyword in env_lower:
            return "continuous"
    
    return "continuous"


def load_and_plot_environment(env_name, eval_files, training_files):
    """Load results and generate plots for a single environment."""
    env_type = detect_environment_type(env_name)
    
    print(f"\nEnvironment: {env_name}")
    print(f"  Type: {env_type}")
    print(f"  Algorithms: {len(eval_files)}")
    
    # Load evaluation data
    eval_data = []
    for eval_file in eval_files:
        algo_name = eval_file.stem.replace(f"_{env_name}_eval", "")
        df = pd.read_csv(eval_file)
        df["algorithm"] = algo_name
        eval_data.append(df)
    
    eval_df = pd.concat(eval_data, ignore_index=True)
    
    # Load training data
    training_data = []
    for train_file in training_files:
        algo_name = train_file.stem.replace(f"_{env_name}_training", "")
        df = pd.read_csv(train_file)
        df["algorithm"] = algo_name
        training_data.append(df)
    
    training_df = pd.concat(training_data, ignore_index=True)
    
    # Plot 1: Evaluation scores
    fig, ax = plt.subplots(figsize=(14, 6))
    
    eval_summary = eval_df.groupby("algorithm")["eval_score"].agg(
        ["mean", "std"]
    ).reset_index()
    eval_summary = eval_summary.sort_values("mean", ascending=False)
    
    x = np.arange(len(eval_summary))
    ax.bar(x, eval_summary["mean"], yerr=eval_summary["std"],
           capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(eval_summary["algorithm"], rotation=45, ha='right')
    ax.set_ylabel("Evaluation Score", fontsize=12)
    ax.set_title(
        f"{env_name}: Final Evaluation Scores ({env_type})\n"
        f"Mean ± Std across seeds",
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = f"plots/{env_name}_{env_type}_evaluation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved evaluation plot: {output_file}")
    plt.close()
    
    # Plot 2: Rolling average training curves
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Determine window size based on environment
    window = 100 if env_type == "discrete" else 50
    
    # Use more distinct colors for better visibility
    colors = plt.cm.tab10(np.linspace(0, 1, len(eval_summary)))
    
    for idx, algo in enumerate(eval_summary["algorithm"]):
        algo_data = training_df[training_df["algorithm"] == algo]
        
        # Calculate rolling average per seed, then average across seeds
        rolling_means = []
        valid_seeds = []
        for seed in sorted(algo_data["seed"].unique()):
            seed_data = algo_data[algo_data["seed"] == seed].sort_values(
                "episode"
            )
            # Skip seeds with too few episodes (early termination/failure)
            if len(seed_data) < 100:
                continue
            rolling = seed_data["score"].rolling(
                window=window, min_periods=1
            ).mean()
            rolling_means.append(rolling.values)
            valid_seeds.append(seed)
        
        # Skip algorithm if no valid seeds
        if len(rolling_means) == 0:
            continue
        
        # For continuous environments, use max length and pad shorter runs with NaN
        # For discrete environments, use min length to avoid extrapolation
        if env_type == "continuous":
            max_len = max(len(r) for r in rolling_means)
            # Pad shorter runs with NaN (will be ignored in nanmean)
            rolling_means_padded = []
            for r in rolling_means:
                if len(r) < max_len:
                    padded = np.full(max_len, np.nan)
                    padded[:len(r)] = r
                    rolling_means_padded.append(padded)
                else:
                    rolling_means_padded.append(r)
            avg_rolling = np.nanmean(rolling_means_padded, axis=0)
            std_rolling = np.nanstd(rolling_means_padded, axis=0)
            n_seeds_per_point = np.sum(~np.isnan(rolling_means_padded), axis=0)
            ci_95 = 1.96 * std_rolling / np.sqrt(n_seeds_per_point)
            ref_len = max_len
        else:
            # For discrete: use min length to avoid sparse data at end
            min_len = min(len(r) for r in rolling_means)
            rolling_means_trimmed = [r[:min_len] for r in rolling_means]
            avg_rolling = np.mean(rolling_means_trimmed, axis=0)
            std_rolling = np.std(rolling_means_trimmed, axis=0)
            n_seeds = len(rolling_means_trimmed)
            ci_95 = 1.96 * std_rolling / np.sqrt(n_seeds)
            ref_len = min_len
        
        # Get episode numbers - use longest seed for continuous, first seed for discrete
        if env_type == "continuous":
            # Find the seed with the most episodes
            longest_seed = max(valid_seeds, key=lambda s: len(algo_data[algo_data["seed"] == s]))
            episodes = algo_data[
                algo_data["seed"] == longest_seed
            ].sort_values("episode")["episode"].values[:ref_len]
        else:
            episodes = algo_data[
                algo_data["seed"] == valid_seeds[0]
            ].sort_values("episode")["episode"].values[:ref_len]
        
        # Subsample for smoother plotting (every 10th point for discrete)
        # Keep episode numbers aligned with data
        if env_type == "discrete" and len(episodes) > 1000:
            step = 10
            episodes_sub = episodes[::step]
            avg_rolling_sub = avg_rolling[::step]
            ci_95_sub = ci_95[::step]
        else:
            episodes_sub = episodes
            avg_rolling_sub = avg_rolling
            ci_95_sub = ci_95
        
        # Plot mean with shaded 95% CI
        ax.plot(episodes_sub, avg_rolling_sub, label=algo, linewidth=3,
                alpha=0.95, color=colors[idx])
        ax.fill_between(episodes_sub, 
                        np.maximum(0, avg_rolling_sub - ci_95_sub),
                        np.minimum(1 if env_type == "discrete" else np.inf,
                                   avg_rolling_sub + ci_95_sub),
                        alpha=0.15, color=colors[idx])
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Rolling Average Score (window={window})", fontsize=12)
    ax.set_title(
        f"{env_name}: Training Progress ({env_type})\n"
        f"Mean ± 95% CI across seeds",
        fontsize=14, fontweight='bold'
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # For discrete environments, set y-axis to [0, 1] for better visibility
    if env_type == "discrete":
        ax.set_ylim(0, 1.0)
        # Add horizontal reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    output_file = f"plots/{env_name}_{env_type}_rolling.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved rolling average plot: {output_file}")
    plt.close()
    
    # Print summary statistics
    print(f"\n  Results Summary:")
    print("  " + "=" * 58)
    for _, row in eval_summary.iterrows():
        print(f"  {row['algorithm']:40s}: "
              f"{row['mean']:6.2f} ± {row['std']:5.2f}")


def main():
    """Main function to discover and plot all benchmark results."""
    results_dir = Path("benchmark_results")
    
    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        return
    
    # Find all eval files and group by environment
    eval_files = list(results_dir.glob("*_eval.csv"))
    
    if not eval_files:
        print("No benchmark results found")
        return
    
    # Group by environment
    env_data = {}
    for eval_file in eval_files:
        # Parse: {algo}_{env}_eval.csv
        parts = eval_file.stem.rsplit("_", 2)
        if len(parts) != 3 or parts[2] != "eval":
            continue
        
        algo_name = parts[0]
        env_name = parts[1]
        
        # Find training file
        train_file = results_dir / f"{algo_name}_{env_name}_training.csv"
        if not train_file.exists():
            print(f"Warning: Missing training file for {algo_name} on "
                  f"{env_name}")
            continue
        
        if env_name not in env_data:
            env_data[env_name] = {"eval": [], "training": []}
        
        env_data[env_name]["eval"].append(eval_file)
        env_data[env_name]["training"].append(train_file)
    
    print(f"Found {len(env_data)} environment(s):")
    for env_name, data in env_data.items():
        print(f"  - {env_name}: {len(data['eval'])} algorithm(s)")
    
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    
    # Generate plots for each environment
    for env_name, data in env_data.items():
        load_and_plot_environment(
            env_name, data["eval"], data["training"]
        )
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
