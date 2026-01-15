"""
Plot CDFs of alpha metrics for each voting method across all PB elections.
Excludes values equal to 1.0 to show distribution of non-perfect scores.

Creates a 3x2 subplot figure where each subplot shows the CDF of one alpha metric
(alpha_AV, alpha_CC, alpha_PAIRS, alpha_CONS, alpha_EJR) with overlapping lines
for each voting method.

Output: analysis/alpha_cdf_no_ones.png
"""

import pandas as pd
import numpy as np
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
    alpha_metrics = ["alpha_AV", "alpha_CC", "alpha_PAIRS", "alpha_CONS", "alpha_EJR"]
    alpha_labels = [r"$\alpha_{AV}$", r"$\alpha_{CC}$", r"$\alpha_{PAIRS}$", r"$\alpha_{CONS}$", r"$\alpha_{EJR}$"]
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
    
    # Create 3x2 subplot figure (5 metrics + 1 empty)
    fig, axes = plt.subplots(3, 2, figsize=(3.5, 4.5))
    axes = axes.flatten()
    
    for idx, metric in enumerate(alpha_metrics):
        ax = axes[idx]
        
        for method in methods:
            method_data = all_data[all_data["method"] == method][metric].dropna()
            # Filter out values equal to 1.0
            method_data = method_data[method_data < 1.0]
            if len(method_data) > 0:
                # Sort data for CDF
                sorted_data = np.sort(method_data)
                # Compute cumulative proportions
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.plot(sorted_data, cdf, color=colors[method], label=method, linewidth=1)
        
        ax.set_xlabel(alpha_labels[idx], fontsize=6)
        ax.set_ylabel("Cumulative Proportion", fontsize=6)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='both', labelsize=5)
    
    # Hide the 6th (empty) subplot
    axes[5].axis("off")
    
    fig.suptitle("Alpha Metrics CDF by Voting Method\n(excluding perfect scores)", fontsize=9, fontweight="bold")
    
    # Single shared legend in the empty subplot area
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc="center", ncol=2, fontsize=6, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_path = analysis_dir / "alpha_cdf_no_ones.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nCDF plot saved to: {output_path}")


if __name__ == "__main__":
    main()
