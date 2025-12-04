"""
Plot bar charts showing (1,1) achievement for each voting method.

For each voting method, creates a bar chart showing the proportion of elections
where the method achieved (1,1) for each trade-off pair (PAIRS-AV, PAIRS-CC,
CONS-AV, CONS-CC).

Output: analysis/voting-methods/{method}_tradeoff.png
"""

import pandas as pd
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
    
    # Trade-off pairs to check
    pairs = [
        ("alpha_PAIRS", "alpha_AV", "PAIRS-AV"),
        ("alpha_PAIRS", "alpha_CC", "PAIRS-CC"),
        ("alpha_CONS", "alpha_AV", "CONS-AV"),
        ("alpha_CONS", "alpha_CC", "CONS-CC"),
    ]
    
    for method in methods:
        print(f"\nProcessing {method}...")
        method_data = all_data[all_data["method"] == method]
        total_elections = len(method_data)
        
        if total_elections == 0:
            print(f"  No data found for {method}, skipping")
            continue
        
        # Count (1,1) achievements for each pair
        counts = {}
        for col1, col2, name in pairs:
            achieved = ((method_data[col1] == 1.0) & (method_data[col2] == 1.0)).sum()
            counts[name] = achieved
            print(f"  {name}: {achieved}/{total_elections} = {achieved/total_elections:.4f}")
        
        # Compute proportions
        proportions = {name: counts[name] / total_elections for name in counts}
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        names = [name for _, _, name in pairs]
        heights = [proportions[name] for name in names]
        
        bars = ax.bar(names, heights, color="#4a90d9", edgecolor="black", linewidth=0.5)
        
        # Add value labels on bars
        for bar, height in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=11)
        
        ax.set_ylabel("Proportion of Elections", fontsize=12)
        ax.set_xlabel("Trade-off Pair", fontsize=12)
        ax.set_title(f"{method}: (1,1) Achievement by Trade-off Pair", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        # Save figure - replace special characters for filename
        safe_method = method.replace("/", "_").replace("^", "")
        output_path = output_dir / f"{safe_method}_tradeoff.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"  Saved to: {output_path}")
    
    print(f"\n\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()


