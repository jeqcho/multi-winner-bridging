"""
Plot bar chart showing proportion of elections where a (1,1) committee exists.

For each trade-off pair (PAIRS-AV, PAIRS-CC, CONS-AV, CONS-CC), shows the
proportion of elections where there exists at least one committee that
achieves the maximum value for both metrics simultaneously.

Output: analysis/tradeoff_bar.png
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
    
    # Find all raw_scores.csv files
    raw_files = list(pb_output_dir.rglob("raw_scores.csv"))
    print(f"Found {len(raw_files)} raw_scores.csv files")
    
    if not raw_files:
        print("No raw_scores.csv files found in output/pb/")
        return
    
    # Trade-off pairs to check
    pairs = [
        ("PAIRS", "AV", "PAIRS-AV"),
        ("PAIRS", "CC", "PAIRS-CC"),
        ("CONS", "AV", "CONS-AV"),
        ("CONS", "CC", "CONS-CC"),
    ]
    
    # Count elections where (1,1) exists for each pair
    counts = {name: 0 for _, _, name in pairs}
    total_elections = 0
    
    for csv_path in raw_files:
        df = pd.read_csv(csv_path)
        
        # Skip empty or trivial elections
        if len(df) < 2:
            continue
        
        total_elections += 1
        
        # Get max values for each metric
        max_vals = {
            "AV": df["AV"].max(),
            "CC": df["CC"].max(),
            "PAIRS": df["PAIRS"].max(),
            "CONS": df["CONS"].max(),
        }
        
        # Skip if any max is 0
        if any(v == 0 for v in max_vals.values()):
            continue
        
        # Check each trade-off pair
        for metric1, metric2, name in pairs:
            # Check if any committee achieves max for both metrics
            exists_11 = ((df[metric1] == max_vals[metric1]) & 
                         (df[metric2] == max_vals[metric2])).any()
            if exists_11:
                counts[name] += 1
    
    print(f"Total elections: {total_elections}")
    
    # Compute proportions
    proportions = {name: counts[name] / total_elections for name in counts}
    
    for name in counts:
        print(f"{name}: {counts[name]}/{total_elections} = {proportions[name]:.4f}")
    
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
    ax.set_title("Existence of (1,1) Committee by Trade-off Pair", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "tradeoff_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nBar chart saved to: {output_path}")


if __name__ == "__main__":
    main()

