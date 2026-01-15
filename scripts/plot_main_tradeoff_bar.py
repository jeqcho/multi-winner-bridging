"""
Plot main grouped bar chart showing (1,1) achievement for best-possible and all voting methods.

X-axis: 6 trade-off pairs (PAIRS-AV, PAIRS-CC, CONS-AV, CONS-CC, PAIRS-EJR, CONS-EJR)
Bars: 7 colors (Best Possible + 6 voting methods)

Output: analysis/main_tradeoff_bar.png
"""

import json
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
    
    # Find all files
    raw_files = list(pb_output_dir.rglob("raw_scores.csv"))
    voting_files = list(pb_output_dir.rglob("voting_results.csv"))
    print(f"Found {len(raw_files)} raw_scores.csv files")
    print(f"Found {len(voting_files)} voting_results.csv files")
    
    # Trade-off pairs
    pairs_raw = [
        ("PAIRS", "AV", "PAIRS-AV"),
        ("PAIRS", "CC", "PAIRS-CC"),
        ("CONS", "AV", "CONS-AV"),
        ("CONS", "CC", "CONS-CC"),
    ]
    pairs_voting = [
        ("alpha_PAIRS", "alpha_AV", "PAIRS-AV"),
        ("alpha_PAIRS", "alpha_CC", "PAIRS-CC"),
        ("alpha_CONS", "alpha_AV", "CONS-AV"),
        ("alpha_CONS", "alpha_CC", "CONS-CC"),
    ]
    # EJR trade-off pairs (using budget-aware alpha_EJR)
    ejr_pairs = [
        ("alpha_PAIRS", "alpha_EJR", "PAIRS-EJR"),
        ("alpha_CONS", "alpha_EJR", "CONS-EJR"),
    ]
    
    # Voting methods (ordered for plot)
    methods = ["MES", "greedy-CC", "greedy-PAV", "greedy-AV", "greedy-AV/cost", "greedy-AV/cost^2"]
    
    # ===== Compute Best Possible (from raw_scores.csv) =====
    best_possible = {name: 0 for _, _, name in pairs_raw}
    total_elections = 0
    
    for csv_path in raw_files:
        df = pd.read_csv(csv_path)
        if len(df) < 2:
            continue
        total_elections += 1
        
        max_vals = {col: df[col].max() for col in ["AV", "CC", "PAIRS", "CONS"]}
        if any(v == 0 for v in max_vals.values()):
            continue
        
        for metric1, metric2, name in pairs_raw:
            exists_11 = ((df[metric1] == max_vals[metric1]) & 
                         (df[metric2] == max_vals[metric2])).any()
            if exists_11:
                best_possible[name] += 1
    
    best_possible_prop = {name: best_possible[name] / total_elections for name in best_possible}
    
    # Load correct PAIRS-EJR and CONS-EJR best-possible values
    # These check if committees with alpha_PAIRS=1 or alpha_CONS=1 also achieve alpha_EJR=1
    ejr_tradeoffs_json = analysis_dir / "best_possible_ejr_tradeoffs.json"
    
    if ejr_tradeoffs_json.exists():
        with open(ejr_tradeoffs_json) as f:
            ejr_tradeoffs_data = json.load(f)
        best_possible_prop["PAIRS-EJR"] = ejr_tradeoffs_data["pairs_ejr_11_proportion"]
        best_possible_prop["CONS-EJR"] = ejr_tradeoffs_data["cons_ejr_11_proportion"]
    else:
        print(f"Warning: {ejr_tradeoffs_json} not found. Run compute_best_possible_ejr_tradeoffs.py first.")
        best_possible_prop["PAIRS-EJR"] = 0.0
        best_possible_prop["CONS-EJR"] = 0.0
    
    print(f"\nBest Possible (out of {total_elections} elections):")
    for name, prop in best_possible_prop.items():
        print(f"  {name}: {prop:.4f}")
    
    # ===== Compute Voting Methods (from voting_results.csv) =====
    dfs = []
    for csv_path in voting_files:
        df = pd.read_csv(csv_path)
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)
    
    method_props = {method: {} for method in methods}
    for method in methods:
        method_data = all_data[all_data["method"] == method]
        total = len(method_data)
        for col1, col2, name in pairs_voting:
            achieved = ((method_data[col1] == 1.0) & (method_data[col2] == 1.0)).sum()
            method_props[method][name] = achieved / total if total > 0 else 0
        # EJR trade-off pairs (alpha_EJR is continuous, check >= 0.9999)
        for col1, col2, name in ejr_pairs:
            achieved = ((method_data[col1] >= 0.9999) & (method_data[col2] >= 0.9999)).sum()
            method_props[method][name] = achieved / total if total > 0 else 0
    
    print("\nVoting Methods:")
    for method in methods:
        print(f"  {method}: {method_props[method]}")
    
    # ===== Create Grouped Bar Chart =====
    fig, ax = plt.subplots(figsize=(5, 4))
    
    pair_names = ["PAIRS-AV", "PAIRS-CC", "PAIRS-EJR", "CONS-AV", "CONS-CC", "CONS-EJR"]
    n_pairs = len(pair_names)
    n_bars = 1 + len(methods)  # Best Possible + voting methods
    
    # Bar positions
    x = np.arange(n_pairs)
    width = 0.10
    
    # Colors
    colors = {
        "Best Possible": "#2ecc71",  # green
        "MES": "#e74c3c",            # red
        "greedy-AV": "#3498db",      # blue
        "greedy-AV/cost": "#9b59b6", # purple
        "greedy-AV/cost^2": "#f39c12", # orange
        "greedy-CC": "#1abc9c",      # teal
        "greedy-PAV": "#e67e22",     # dark orange
    }
    
    # Plot Best Possible
    heights = [best_possible_prop[name] for name in pair_names]
    offset = -width * (n_bars - 1) / 2
    bars = ax.bar(x + offset, heights, width, label="Best Possible", 
                  color=colors["Best Possible"], edgecolor="black", linewidth=0.3)
    
    # Plot voting methods
    for i, method in enumerate(methods):
        heights = [method_props[method][name] for name in pair_names]
        offset = -width * (n_bars - 1) / 2 + width * (i + 1)
        ax.bar(x + offset, heights, width, label=method,
               color=colors[method], edgecolor="black", linewidth=0.3)
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Proportion of Elections that\nMaximize Both Metrics", fontsize=8)
    ax.set_xlabel("Trade-off Pair", fontsize=8)
    ax.set_title("Proportion of Elections with Metric Pairs\nMaximized by Voting Methods", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax.tick_params(axis='both', labelsize=7)
    
    # Legend below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4, fontsize=6)
    
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "main_tradeoff_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nBar chart saved to: {output_path}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

