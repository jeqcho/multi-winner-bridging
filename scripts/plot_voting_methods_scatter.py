"""
Plot alpha scatter plots for each voting method.

Creates one PNG per voting method showing 2x2 alpha trade-off plots:
- Row 1: alpha_PAIRS vs alpha_AV, alpha_PAIRS vs alpha_CC
- Row 2: alpha_CONS vs alpha_AV, alpha_CONS vs alpha_CC

Each point represents one election's result for that voting method.

Output: analysis/voting-methods/{method}.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    pb_output_dir = base_dir / "output" / "pb"
    output_dir = base_dir / "analysis" / "voting-methods"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        df["election"] = csv_path.parent.name
        dfs.append(df)
    
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(all_data)}")
    
    # Voting methods
    methods = ["MES", "greedy-AV", "greedy-AV/cost", "greedy-AV/cost^2", "greedy-CC", "greedy-PAV"]
    
    # Plot settings
    point_alpha = 0.4
    point_color = "black"
    point_size = 30
    
    # Reference lines
    x_ref = np.linspace(0, 1, 100)
    
    for method in methods:
        print(f"\nProcessing {method}...")
        method_data = all_data[all_data["method"] == method]
        
        if len(method_data) == 0:
            print(f"  No data found for {method}, skipping")
            continue
        
        print(f"  {len(method_data)} elections")
        
        # Create 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{method}: Alpha-Approximation Trade-offs\n(one point per election)", 
                     fontsize=16, fontweight="bold", y=0.995)
        
        # Plot 1: PAIRS vs AV (top-left)
        ax = axes[0, 0]
        ax.scatter(method_data["alpha_PAIRS"], method_data["alpha_AV"],
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
        ax.scatter(method_data["alpha_PAIRS"], method_data["alpha_CC"],
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
        ax.scatter(method_data["alpha_CONS"], method_data["alpha_AV"],
                   c=point_color, alpha=point_alpha, s=point_size, edgecolors="none")
        # CONS reference line: a² + b = 1
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
        ax.scatter(method_data["alpha_CONS"], method_data["alpha_CC"],
                   c=point_color, alpha=point_alpha, s=point_size, edgecolors="none")
        # CONS reference line: a² + b = 1
        ax.plot(x_ref, 1 - x_ref**2, "k-", linewidth=2, alpha=0.5, label="a² + b = 1")
        ax.set_xlabel("alpha_CONS", fontsize=11)
        ax.set_ylabel("alpha_CC", fontsize=11)
        ax.set_title("CONS vs CC", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure - replace special characters for filename
        safe_method = method.replace("/", "_").replace("^", "")
        output_path = output_dir / f"{safe_method}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"  Saved to: {output_path}")
    
    print(f"\n\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

