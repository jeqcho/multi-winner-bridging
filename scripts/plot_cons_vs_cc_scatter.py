"""
Plot CONS vs CC scatter plot where each point is a committee from an election.
All elections are plotted on the same figure.

Includes the theory line: alpha_CONS^2 + alpha_CC = 1

Output: analysis/cons_vs_cc_scatter.png
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
    
    # Find all raw_scores.csv files
    raw_files = list(pb_output_dir.rglob("raw_scores.csv"))
    print(f"Found {len(raw_files)} raw_scores.csv files")
    
    if not raw_files:
        print("No raw_scores.csv files found in output/pb/")
        return
    
    # Collect all normalized points
    all_cons = []
    all_cc = []
    
    for csv_path in raw_files:
        df = pd.read_csv(csv_path)
        
        # Skip if empty or only has empty committee
        if len(df) < 2:
            continue
        
        # Get max values for normalization
        max_cons = df["CONS"].max()
        max_cc = df["CC"].max()
        
        if max_cons == 0 or max_cc == 0:
            continue
        
        # Normalize and collect points (excluding empty committee)
        df_valid = df[df["subset_size"] > 0]
        alpha_cons = df_valid["CONS"] / max_cons
        alpha_cc = df_valid["CC"] / max_cc
        
        all_cons.extend(alpha_cons.tolist())
        all_cc.extend(alpha_cc.tolist())
    
    print(f"Total points: {len(all_cons)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot all points
    ax.scatter(all_cons, all_cc, alpha=0.2, color="blue", s=10, edgecolors="none",
               label="Committee from an election")
    
    # Add theory line: alpha_CC = 1 - alpha_CONS^2
    x_theory = np.linspace(0, 1, 100)
    y_theory = 1 - x_theory**2
    ax.plot(x_theory, y_theory, color="red", linewidth=2, linestyle="--", 
            label=r"$\alpha_{CC} = 1 - \alpha_{CONS}^2$")
    
    # Formatting
    ax.set_xlabel(r"$\alpha_{CONS}$", fontsize=22)
    ax.set_ylabel(r"$\alpha_{CC}$", fontsize=22)
    ax.set_title("CONS vs CC (all committees from all elections)", fontsize=24, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16, loc="lower left")
    ax.tick_params(axis='both', labelsize=18)
    
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "cons_vs_cc_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nScatter plot saved to: {output_path}")


if __name__ == "__main__":
    main()

