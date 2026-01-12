"""
Plot CONS vs CC scatter plot for a single election.

Output: analysis/cons_vs_cc_single_election.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    analysis_dir = base_dir / "analysis"
    
    # Use poland_warszawa_2019_rejon-poludniowy (8191 committees, max_error=0.000269)
    election_path = base_dir / "output" / "pb" / "poland_warszawa_2019_rejon-poludniowy" / "raw_scores.csv"
    election_name = "Poland Warszawa 2019 Rejon Poludniowy"
    
    print(f"Loading {election_name}...")
    df = pd.read_csv(election_path)
    
    # Get max values for normalization
    max_cons = df["CONS"].max()
    max_cc = df["CC"].max()
    
    # Filter out empty committee and normalize
    df_valid = df[df["subset_size"] > 0]
    alpha_cons = df_valid["CONS"] / max_cons
    alpha_cc = df_valid["CC"] / max_cc
    
    print(f"Total committees: {len(alpha_cons)}")
    
    # Compute error stats
    predicted_cons = alpha_cc ** 2
    errors = np.abs(alpha_cons - predicted_cons)
    print(f"Max error: {errors.max():.6f}")
    print(f"Mean error: {errors.mean():.6f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    # Plot all points
    ax.scatter(alpha_cons, alpha_cc, alpha=0.2, color="black", s=3, edgecolors="none")
    
    # Formatting
    ax.set_xlabel(r"$\alpha_{CONS}$", fontsize=8)
    ax.set_ylabel(r"$\alpha_{CC}$", fontsize=8)
    ax.set_title(f"CONS vs CC\n{election_name}", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=7)
    
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "cons_vs_cc_single_election.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nScatter plot saved to: {output_path}")


if __name__ == "__main__":
    main()

