"""
Plot bar chart showing proportion of elections where each voting method achieved EJR.

Output: analysis/ejr_bar.png
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
        df["election_id"] = str(csv_path)  # Add election identifier
        dfs.append(df)
    
    all_data = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(all_data)}")
    
    # Voting methods
    methods = ["MES", "greedy-AV", "greedy-AV/cost", "greedy-AV/cost^2", "greedy-CC", "greedy-PAV"]
    
    # Count EJR achievements per method
    ejr_proportions = {}
    for method in methods:
        method_data = all_data[all_data["method"] == method]
        total = len(method_data)
        achieved = method_data["EJR"].sum()
        proportion = achieved / total if total > 0 else 0
        ejr_proportions[method] = proportion
        print(f"{method}: {achieved}/{total} = {proportion:.4f}")
    
    # Compute best-possible (union of EJR across methods per election)
    # For each election, check if ANY method achieved EJR
    election_ejr = all_data.groupby("election_id")["EJR"].any()
    best_possible_achieved = election_ejr.sum()
    best_possible_total = len(election_ejr)
    best_possible_proportion = election_ejr.mean()
    ejr_proportions["best-possible"] = best_possible_proportion
    print(f"best-possible: {best_possible_achieved}/{best_possible_total} = {best_possible_proportion:.4f}")
    
    # Add best-possible to methods list
    methods = methods + ["best-possible"]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(methods))
    heights = [ejr_proportions[m] for m in methods]
    
    bars = ax.bar(x, heights, color="#4a90d9", edgecolor="black", linewidth=0.5)
    
    # Add value labels on bars
    for bar, height in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f"{height:.2f}", ha="center", va="bottom", fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Proportion of Elections with EJR", fontsize=12)
    ax.set_xlabel("Voting Method", fontsize=12)
    ax.set_title("EJR Achievement by Voting Method", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "ejr_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nBar chart saved to: {output_path}")


if __name__ == "__main__":
    main()

