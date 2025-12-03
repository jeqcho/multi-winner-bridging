"""
Plot histograms of alpha metrics for each voting method across all PB elections.

Creates a 2x2 subplot figure where each subplot shows the distribution of one alpha metric
(alpha_AV, alpha_CC, alpha_PAIRS, alpha_CONS) with overlapping transparent histograms
for each voting method.

Output: analysis/alpha_histograms.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    pb_output_dir = base_dir / "output" / "pb"
    analysis_dir = base_dir / "analysis"
    
    # Create analysis directory if it doesn't exist
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all voting_results.csv files
    voting_files = list(pb_output_dir.rglob("voting_results.csv"))
    print(f"Found {len(voting_files)} voting_results.csv files")
    
    if not voting_files:
        print("No voting_results.csv files found in output/pb/")
        return
    
    # Load and concatenate all data
    dfs = []
    for csv_path in voting_files:
        df = pd.read_csv(csv_path)
        df["source"] = csv_path.parent.name  # Add source election name
        dfs.append(df)
    
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(all_data)}")
    
    # Alpha metrics and voting methods
    alpha_metrics = ["alpha_AV", "alpha_CC", "alpha_PAIRS", "alpha_CONS"]
    methods = ["MES", "greedy-AV", "greedy-AV/cost", "greedy-AV/cost^2", "greedy-CC", "greedy-PAV"]
    
    # Colors for each method
    colors = {
        "MES": "#e41a1c",           # red
        "greedy-AV": "#377eb8",     # blue
        "greedy-AV/cost": "#4daf4a",    # green
        "greedy-AV/cost^2": "#984ea3",  # purple
        "greedy-CC": "#ff7f00",     # orange
        "greedy-PAV": "#a65628",    # brown
    }
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(alpha_metrics):
        ax = axes[idx]
        
        for method in methods:
            method_data = all_data[all_data["method"] == method][metric].dropna()
            if len(method_data) > 0:
                ax.hist(
                    method_data,
                    bins=30,
                    alpha=0.5,
                    color=colors[method],
                    label=method,
                    edgecolor=colors[method],
                    linewidth=0.5,
                )
        
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Distribution of {metric}", fontsize=14)
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    fig.suptitle("Alpha Metrics Distribution by Voting Method\n(across all PB elections)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "alpha_histograms.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nHistogram saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    for metric in alpha_metrics:
        print(f"\n{metric}:")
        for method in methods:
            method_data = all_data[all_data["method"] == method][metric].dropna()
            if len(method_data) > 0:
                print(f"  {method:20s}: n={len(method_data):4d}, mean={method_data.mean():.4f}, min={method_data.min():.4f}, max={method_data.max():.4f}")


if __name__ == "__main__":
    main()

