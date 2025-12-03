"""
Plot Pareto front scatter plots across all PB elections.

Creates a 2x2 subplot figure showing all Pareto-optimal committees from each election:
- Row 1: alpha_PAIRS vs alpha_AV, alpha_PAIRS vs alpha_CC
- Row 2: alpha_CONS vs alpha_AV, alpha_CONS vs alpha_CC

Points are colored by election ID to distinguish clusters.

Output: analysis/pareto_front_scatter.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_pareto_front(df: pd.DataFrame, col_x: str, col_y: str) -> pd.DataFrame:
    """
    Find Pareto-optimal points for two metrics (both to be maximized).
    
    A point is Pareto-optimal if no other point dominates it (i.e., is better
    or equal in both metrics and strictly better in at least one).
    
    Args:
        df: DataFrame with scores
        col_x: Column name for first metric
        col_y: Column name for second metric
    
    Returns:
        DataFrame with only Pareto-optimal rows
    """
    # Extract the two columns as numpy array for efficiency
    points = df[[col_x, col_y]].values
    n = len(points)
    
    # Find non-dominated points
    is_pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_pareto[i]:
            continue
        # Check if point i is dominated by any other point
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j >= i in both and j > i in at least one
            if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                is_pareto[i] = False
                break
    
    return df[is_pareto]


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    pb_output_dir = base_dir / "output" / "pb"
    analysis_dir = base_dir / "analysis"
    
    # Create analysis directory if it doesn't exist
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all raw_scores.csv files
    raw_files = sorted(list(pb_output_dir.rglob("raw_scores.csv")))
    print(f"Found {len(raw_files)} raw_scores.csv files")
    
    if not raw_files:
        print("No raw_scores.csv files found in output/pb/")
        return
    
    # Collect Pareto front points from each election
    results = {
        "pairs_av": [],  # [(alpha_PAIRS, alpha_AV, election_id), ...]
        "pairs_cc": [],
        "cons_av": [],
        "cons_cc": [],
    }
    
    for election_id, csv_path in enumerate(raw_files):
        df = pd.read_csv(csv_path)
        
        # Skip empty or trivial elections
        if len(df) < 2:
            continue
        
        # Get max values for normalization
        max_av = df["AV"].max()
        max_cc = df["CC"].max()
        max_pairs = df["PAIRS"].max()
        max_cons = df["CONS"].max()
        
        # Skip if any max is 0 (can't normalize)
        if max_av == 0 or max_cc == 0 or max_pairs == 0 or max_cons == 0:
            continue
        
        # Compute alpha values
        df["alpha_AV"] = df["AV"] / max_av
        df["alpha_CC"] = df["CC"] / max_cc
        df["alpha_PAIRS"] = df["PAIRS"] / max_pairs
        df["alpha_CONS"] = df["CONS"] / max_cons
        
        # Get Pareto fronts for each metric pair
        pareto_pairs_av = get_pareto_front(df, "alpha_PAIRS", "alpha_AV")
        pareto_pairs_cc = get_pareto_front(df, "alpha_PAIRS", "alpha_CC")
        pareto_cons_av = get_pareto_front(df, "alpha_CONS", "alpha_AV")
        pareto_cons_cc = get_pareto_front(df, "alpha_CONS", "alpha_CC")
        
        # Store results with election_id
        for _, row in pareto_pairs_av.iterrows():
            results["pairs_av"].append((row["alpha_PAIRS"], row["alpha_AV"], election_id))
        for _, row in pareto_pairs_cc.iterrows():
            results["pairs_cc"].append((row["alpha_PAIRS"], row["alpha_CC"], election_id))
        for _, row in pareto_cons_av.iterrows():
            results["cons_av"].append((row["alpha_CONS"], row["alpha_AV"], election_id))
        for _, row in pareto_cons_cc.iterrows():
            results["cons_cc"].append((row["alpha_CONS"], row["alpha_CC"], election_id))
    
    n_elections = len(raw_files)
    print(f"Processed {n_elections} elections")
    print(f"Pareto front sizes: PAIRS-AV={len(results['pairs_av'])}, PAIRS-CC={len(results['pairs_cc'])}, "
          f"CONS-AV={len(results['cons_av'])}, CONS-CC={len(results['cons_cc'])}")
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Pareto Fronts: Alpha-Approximation Trade-offs\n(colored by election)", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    # Reference line x values
    x_ref = np.linspace(0, 1, 100)
    
    # Use a colormap with enough colors
    cmap = plt.cm.hsv
    
    # Plot settings
    point_alpha = 0.5
    point_size = 15
    
    def plot_subplot(ax, data, xlabel, ylabel, title, ref_y, ref_label):
        data_arr = np.array(data)
        x = data_arr[:, 0]
        y = data_arr[:, 1]
        election_ids = data_arr[:, 2].astype(int)
        
        # Normalize election IDs to [0, 1] for colormap
        colors = cmap(election_ids / n_elections)
        
        ax.scatter(x, y, c=colors, alpha=point_alpha, s=point_size, edgecolors="none")
        ax.plot(x_ref, ref_y, "k-", linewidth=2, alpha=0.5, label=ref_label)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Reference lines
    ref_linear = 1 - x_ref           # a + b = 1
    ref_quadratic = 1 - x_ref**2     # a² + b = 1
    
    # Plot 1: PAIRS vs AV (top-left)
    plot_subplot(axes[0, 0], results["pairs_av"], "alpha_PAIRS", "alpha_AV", "PAIRS vs AV", ref_linear, "a + b = 1")
    
    # Plot 2: PAIRS vs CC (top-right)
    plot_subplot(axes[0, 1], results["pairs_cc"], "alpha_PAIRS", "alpha_CC", "PAIRS vs CC", ref_linear, "a + b = 1")
    
    # Plot 3: CONS vs AV (bottom-left)
    plot_subplot(axes[1, 0], results["cons_av"], "alpha_CONS", "alpha_AV", "CONS vs AV", ref_quadratic, "a² + b = 1")
    
    # Plot 4: CONS vs CC (bottom-right)
    plot_subplot(axes[1, 1], results["cons_cc"], "alpha_CONS", "alpha_CC", "CONS vs CC", ref_quadratic, "a² + b = 1")
    
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "pareto_front_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nScatter plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    for name, data in [("PAIRS-AV", results["pairs_av"]), ("PAIRS-CC", results["pairs_cc"]),
                       ("CONS-AV", results["cons_av"]), ("CONS-CC", results["cons_cc"])]:
        data_arr = np.array(data)
        print(f"\n{name} ({len(data)} points):")
        print(f"  X: mean={data_arr[:, 0].mean():.4f}, min={data_arr[:, 0].min():.4f}, max={data_arr[:, 0].max():.4f}")
        print(f"  Y: mean={data_arr[:, 1].mean():.4f}, min={data_arr[:, 1].min():.4f}, max={data_arr[:, 1].max():.4f}")
        # Count how many points are above the a + b = 1 line
        above_line = np.sum(data_arr[:, 0] + data_arr[:, 1] > 1)
        print(f"  Points above a+b=1: {above_line}/{len(data)} ({100*above_line/len(data):.1f}%)")


if __name__ == "__main__":
    main()

