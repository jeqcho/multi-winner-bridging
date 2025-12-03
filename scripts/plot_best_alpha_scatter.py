"""
Plot best alpha scatter plots across all PB elections.

Creates a 2x2 subplot figure showing one point per election:
- Row 1: alpha_PAIRS vs alpha_AV, alpha_PAIRS vs alpha_CC
- Row 2: alpha_CONS vs alpha_AV, alpha_CONS vs alpha_CC

Selection logic (lexicographic):
- PAIRS-AV: max PAIRS, then max AV as tiebreaker
- PAIRS-CC: max PAIRS, then max CC as tiebreaker
- CONS-AV: max CONS, then max AV as tiebreaker
- CONS-CC: max CONS, then max CC as tiebreaker

Output: analysis/best_alpha_scatter.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_best_committee(df: pd.DataFrame, primary: str, secondary: str) -> pd.Series:
    """
    Select the best committee using lexicographic ordering.
    
    Args:
        df: DataFrame with raw scores
        primary: Primary metric to maximize (e.g., 'PAIRS')
        secondary: Secondary metric to maximize as tiebreaker (e.g., 'AV')
    
    Returns:
        Series with the best committee's data
    """
    # Find max primary value
    max_primary = df[primary].max()
    # Filter to committees with max primary
    best_primary = df[df[primary] == max_primary]
    # Among those, find max secondary
    best_idx = best_primary[secondary].idxmax()
    return df.loc[best_idx]


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    pb_output_dir = base_dir / "output" / "pb"
    analysis_dir = base_dir / "analysis"
    
    # Create analysis directory if it doesn't exist
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all raw_scores.csv files
    raw_files = list(pb_output_dir.rglob("raw_scores.csv"))
    print(f"Found {len(raw_files)} raw_scores.csv files")
    
    if not raw_files:
        print("No raw_scores.csv files found in output/pb/")
        return
    
    # Collect best points from each election
    results = {
        "pairs_av": [],  # (alpha_PAIRS, alpha_AV)
        "pairs_cc": [],  # (alpha_PAIRS, alpha_CC)
        "cons_av": [],   # (alpha_CONS, alpha_AV)
        "cons_cc": [],   # (alpha_CONS, alpha_CC)
    }
    
    for csv_path in raw_files:
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
        
        # Get best committees for each metric pair
        best_pairs_av = get_best_committee(df, "PAIRS", "AV")
        best_pairs_cc = get_best_committee(df, "PAIRS", "CC")
        best_cons_av = get_best_committee(df, "CONS", "AV")
        best_cons_cc = get_best_committee(df, "CONS", "CC")
        
        # Compute alpha values and store
        results["pairs_av"].append((
            best_pairs_av["PAIRS"] / max_pairs,
            best_pairs_av["AV"] / max_av
        ))
        results["pairs_cc"].append((
            best_pairs_cc["PAIRS"] / max_pairs,
            best_pairs_cc["CC"] / max_cc
        ))
        results["cons_av"].append((
            best_cons_av["CONS"] / max_cons,
            best_cons_av["AV"] / max_av
        ))
        results["cons_cc"].append((
            best_cons_cc["CONS"] / max_cons,
            best_cons_cc["CC"] / max_cc
        ))
    
    print(f"Processed {len(results['pairs_av'])} elections")
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Best Committees: Alpha-Approximation Trade-offs\n(one point per election)", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    # Reference line: a + b = 1
    x_ref = np.linspace(0, 1, 100)
    
    # Plot settings
    point_alpha = 0.4
    point_color = "black"
    point_size = 30
    
    # Plot 1: PAIRS vs AV (top-left)
    ax = axes[0, 0]
    pairs_av_data = np.array(results["pairs_av"])
    ax.scatter(pairs_av_data[:, 0], pairs_av_data[:, 1], 
               c=point_color, alpha=point_alpha, s=point_size, edgecolors="none")
    ax.plot(x_ref, 1 - x_ref, "k-", linewidth=2, alpha=0.5, label="a + b = 1")
    ax.set_xlabel("alpha_PAIRS", fontsize=11)
    ax.set_ylabel("alpha_AV", fontsize=11)
    ax.set_title("PAIRS vs AV", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: PAIRS vs CC (top-right)
    ax = axes[0, 1]
    pairs_cc_data = np.array(results["pairs_cc"])
    ax.scatter(pairs_cc_data[:, 0], pairs_cc_data[:, 1],
               c=point_color, alpha=point_alpha, s=point_size, edgecolors="none")
    ax.plot(x_ref, 1 - x_ref, "k-", linewidth=2, alpha=0.5, label="a + b = 1")
    ax.set_xlabel("alpha_PAIRS", fontsize=11)
    ax.set_ylabel("alpha_CC", fontsize=11)
    ax.set_title("PAIRS vs CC", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: CONS vs AV (bottom-left)
    ax = axes[1, 0]
    cons_av_data = np.array(results["cons_av"])
    ax.scatter(cons_av_data[:, 0], cons_av_data[:, 1],
               c=point_color, alpha=point_alpha, s=point_size, edgecolors="none")
    ax.plot(x_ref, 1 - x_ref**2, "k-", linewidth=2, alpha=0.5, label="a² + b = 1")
    ax.set_xlabel("alpha_CONS", fontsize=11)
    ax.set_ylabel("alpha_AV", fontsize=11)
    ax.set_title("CONS vs AV", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: CONS vs CC (bottom-right)
    ax = axes[1, 1]
    cons_cc_data = np.array(results["cons_cc"])
    ax.scatter(cons_cc_data[:, 0], cons_cc_data[:, 1],
               c=point_color, alpha=point_alpha, s=point_size, edgecolors="none")
    ax.plot(x_ref, 1 - x_ref**2, "k-", linewidth=2, alpha=0.5, label="a² + b = 1")
    ax.set_xlabel("alpha_CONS", fontsize=11)
    ax.set_ylabel("alpha_CC", fontsize=11)
    ax.set_title("CONS vs CC", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "best_alpha_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nScatter plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    for name, data in [("PAIRS-AV", pairs_av_data), ("PAIRS-CC", pairs_cc_data),
                       ("CONS-AV", cons_av_data), ("CONS-CC", cons_cc_data)]:
        print(f"\n{name}:")
        print(f"  X (primary): mean={data[:, 0].mean():.4f}, min={data[:, 0].min():.4f}, max={data[:, 0].max():.4f}")
        print(f"  Y (secondary): mean={data[:, 1].mean():.4f}, min={data[:, 1].min():.4f}, max={data[:, 1].max():.4f}")
        # Count how many points are above the a + b = 1 line
        above_line = np.sum(data[:, 0] + data[:, 1] > 1)
        print(f"  Points above a+b=1: {above_line}/{len(data)} ({100*above_line/len(data):.1f}%)")


if __name__ == "__main__":
    main()

