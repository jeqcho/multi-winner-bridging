"""
Plot PAIRS vs EJR and CONS vs EJR scatter plots.
Each point is a voting method result from an election.

Output: 
  - analysis/pairs_vs_ejr_scatter.png
  - analysis/cons_vs_ejr_scatter.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Color scheme for voting methods
METHOD_COLORS = {
    "MES": "#1f77b4",
    "greedy-CC": "#ff7f0e", 
    "greedy-PAV": "#2ca02c",
    "greedy-AV": "#d62728",
    "greedy-AV/cost": "#9467bd",
    "greedy-AV/cost^2": "#8c564b",
}


def load_voting_results():
    """Load all voting results and return combined dataframe."""
    base_dir = Path(__file__).parent.parent
    pb_output_dir = base_dir / "output" / "pb"
    
    # Find all voting_results.csv files
    result_files = list(pb_output_dir.rglob("voting_results.csv"))
    print(f"Found {len(result_files)} voting_results.csv files")
    
    all_data = []
    for csv_path in result_files:
        election = csv_path.parent.name
        df = pd.read_csv(csv_path)
        df["election"] = election
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def plot_scatter(df, x_col, y_col, x_label, y_label, title, output_path):
    """Create a scatter plot with points colored by voting method."""
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Plot each method with different color
    for method, color in METHOD_COLORS.items():
        method_data = df[df["method"] == method]
        if len(method_data) > 0:
            ax.scatter(
                method_data[x_col], 
                method_data[y_col],
                alpha=0.5, 
                color=color, 
                s=15, 
                edgecolors="none",
                label=method
            )
    
    # Formatting
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_voting_results()
    print(f"Total data points: {len(df)}")
    
    # Filter out rows with missing alpha_EJR
    df = df.dropna(subset=["alpha_EJR", "alpha_PAIRS", "alpha_CONS"])
    print(f"Points with alpha_EJR: {len(df)}")
    
    # Plot 1: PAIRS vs EJR
    plot_scatter(
        df,
        x_col="alpha_PAIRS",
        y_col="alpha_EJR",
        x_label=r"$\alpha_{PAIRS}$",
        y_label=r"$\alpha_{EJR}$",
        title="PAIRS vs EJR\n(voting method outcomes)",
        output_path=analysis_dir / "pairs_vs_ejr_scatter.png"
    )
    
    # Plot 2: CONS vs EJR
    plot_scatter(
        df,
        x_col="alpha_CONS",
        y_col="alpha_EJR",
        x_label=r"$\alpha_{CONS}$",
        y_label=r"$\alpha_{EJR}$",
        title="CONS vs EJR\n(voting method outcomes)",
        output_path=analysis_dir / "cons_vs_ejr_scatter.png"
    )
    
    # Print some stats
    print("\nCorrelations:")
    print(f"  PAIRS vs EJR: {df['alpha_PAIRS'].corr(df['alpha_EJR']):.3f}")
    print(f"  CONS vs EJR: {df['alpha_CONS'].corr(df['alpha_EJR']):.3f}")


if __name__ == "__main__":
    main()
