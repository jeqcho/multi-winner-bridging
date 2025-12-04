"""
Plot grouped bar chart showing (1.0) achievement for each individual metric.

X-axis: 4 individual metrics (alpha_AV, alpha_CC, alpha_PAIRS, alpha_CONS)
Bars: Best Possible (always 1.0) + 6 voting methods showing proportion achieving 1.0

Output: analysis/metrics_bar.png
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
    
    # Metrics (including EJR)
    metrics = ["alpha_AV", "alpha_CC", "alpha_PAIRS", "alpha_CONS", "EJR"]
    metric_labels = ["AV", "CC", "PAIRS", "CONS", "EJR"]
    
    # Voting methods (ordered for plot)
    methods = ["MES", "greedy-CC", "greedy-PAV", "greedy-AV", "greedy-AV/cost", "greedy-AV/cost^2"]
    
    # Load all voting results
    dfs = []
    for csv_path in voting_files:
        df = pd.read_csv(csv_path)
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)
    
    # Compute proportions for each voting method
    method_props = {method: {} for method in methods}
    for method in methods:
        method_data = all_data[all_data["method"] == method]
        total = len(method_data)
        for metric in metrics:
            if metric == "EJR":
                # EJR is boolean True/False
                achieved = method_data[metric].sum()
            else:
                # Alpha metrics: check if == 1.0
                achieved = (method_data[metric] == 1.0).sum()
            method_props[method][metric] = achieved / total if total > 0 else 0
    
    print("\nProportions achieving 1.0 for each metric:")
    for method in methods:
        print(f"  {method}: {method_props[method]}")
    
    # ===== Create Grouped Bar Chart =====
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n_metrics = len(metrics)
    n_bars = len(methods)  # Just voting methods
    
    # Bar positions
    x = np.arange(n_metrics)
    width = 0.12
    
    # Colors
    colors = {
        "MES": "#e74c3c",            # red
        "greedy-CC": "#1abc9c",      # teal
        "greedy-PAV": "#e67e22",     # dark orange
        "greedy-AV": "#3498db",      # blue
        "greedy-AV/cost": "#9b59b6", # purple
        "greedy-AV/cost^2": "#f39c12", # orange
    }
    
    # Plot voting methods
    for i, method in enumerate(methods):
        heights = [method_props[method][metric] for metric in metrics]
        offset = -width * (n_bars - 1) / 2 + width * i
        ax.bar(x + offset, heights, width, label=method,
               color=colors[method], edgecolor="black", linewidth=0.5)
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=18)
    ax.set_ylabel("Proportion of Elections Achieving 1.0", fontsize=20)
    ax.set_xlabel("Metric", fontsize=20)
    ax.set_title("Metric Achievement by Voting Method", fontsize=22, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis='both', labelsize=16)
    
    # Legend below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "metrics_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nBar chart saved to: {output_path}")


if __name__ == "__main__":
    main()

