"""
Plot histogram of connected components for Set A and Set B committees.

X-axis: number of connected components (under CONS)
Red: Lower CONS (Set A)
Blue: Higher CONS (Set B)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import sys
from pathlib import Path

# Add src to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from pb_data_loader import load_pb_file
from scoring import UnionFind

# Paths
PB_FILE = BASE_DIR / "data" / "pb_selected_10_20251202_023743" / "poland_warszawa_2017_wawrzyszew.pb"
SET_A_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_A.csv"
SET_B_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_B.csv"
OUTPUT_DIR = BASE_DIR / "output" / "warsaw-2017" / "reports"


def count_connected_components_all(M: np.ndarray, W: list) -> int:
    """
    Count the number of connected components under CONS for a committee W.
    Includes all voters (unrepresented voters are their own component).
    
    Args:
        M: Boolean approval matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        
    Returns:
        Number of connected components (all voters)
    """
    if len(W) == 0:
        return M.shape[0]  # Each voter is their own component
    
    n_voters = M.shape[0]
    uf = UnionFind(n_voters)
    
    # For each candidate in W, union all their supporters
    for c in W:
        supporters = np.where(M[:, c])[0]
        for i in range(len(supporters) - 1):
            uf.union(int(supporters[i]), int(supporters[i + 1]))
    
    # Return number of components
    return len(uf.get_component_sizes())


def count_connected_components_represented(M: np.ndarray, W: list) -> int:
    """
    Count the number of connected components under CONS for a committee W.
    Only counts voters who are represented (approve at least one candidate in W).
    
    Args:
        M: Boolean approval matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        
    Returns:
        Number of connected components among represented voters
    """
    if len(W) == 0:
        return 0  # No one is represented
    
    # Find represented voters (those who approve at least one candidate in W)
    represented = np.where(M[:, W].sum(axis=1) > 0)[0]
    
    if len(represented) == 0:
        return 0
    
    # Create mapping from original voter index to compressed index
    voter_to_idx = {v: i for i, v in enumerate(represented)}
    
    n_represented = len(represented)
    uf = UnionFind(n_represented)
    
    # For each candidate in W, union all their supporters (among represented voters)
    for c in W:
        supporters = np.where(M[:, c])[0]
        # Map to compressed indices
        compressed = [voter_to_idx[v] for v in supporters if v in voter_to_idx]
        for i in range(len(compressed) - 1):
            uf.union(compressed[i], compressed[i + 1])
    
    # Return number of components
    return len(uf.get_component_sizes())


def parse_subset(subset_str):
    """Parse subset_indices string to a list of integers."""
    return list(ast.literal_eval(subset_str))


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load approval matrix
    print("Loading approval matrix...")
    M, project_ids, project_costs, budget = load_pb_file(str(PB_FILE))
    
    # Load Set A and Set B
    print("Loading Set A and Set B...")
    df_a = pd.read_csv(SET_A_FILE)
    df_b = pd.read_csv(SET_B_FILE)
    
    print(f"Set A: {len(df_a)} committees")
    print(f"Set B: {len(df_b)} committees")
    
    from collections import Counter
    
    # Compute connected components for each committee (all voters)
    print("Computing connected components (all voters) for Set A...")
    components_all_a = []
    for subset_str in df_a["subset_indices"]:
        W = parse_subset(subset_str)
        components_all_a.append(count_connected_components_all(M, W))
    
    print("Computing connected components (all voters) for Set B...")
    components_all_b = []
    for subset_str in df_b["subset_indices"]:
        W = parse_subset(subset_str)
        components_all_b.append(count_connected_components_all(M, W))
    
    # Compute connected components for each committee (represented voters only)
    print("Computing connected components (represented only) for Set A...")
    components_rep_a = []
    for subset_str in df_a["subset_indices"]:
        W = parse_subset(subset_str)
        components_rep_a.append(count_connected_components_represented(M, W))
    
    print("Computing connected components (represented only) for Set B...")
    components_rep_b = []
    for subset_str in df_b["subset_indices"]:
        W = parse_subset(subset_str)
        components_rep_b.append(count_connected_components_represented(M, W))
    
    # Print summary stats
    print(f"\n=== All Voters ===")
    print(f"Set A (Lower CONS): min={min(components_all_a)}, max={max(components_all_a)}, mean={np.mean(components_all_a):.1f}")
    print(f"Set B (Higher CONS): min={min(components_all_b)}, max={max(components_all_b)}, mean={np.mean(components_all_b):.1f}")
    
    print(f"\n=== Represented Voters Only ===")
    print(f"Set A (Lower CONS): min={min(components_rep_a)}, max={max(components_rep_a)}, mean={np.mean(components_rep_a):.1f}")
    print(f"Set B (Higher CONS): min={min(components_rep_b)}, max={max(components_rep_b)}, mean={np.mean(components_rep_b):.1f}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: All voters
    ax1 = axes[0]
    counts_a = Counter(components_all_a)
    counts_b = Counter(components_all_b)
    all_values = sorted(set(components_all_a + components_all_b))
    
    # Use bins for histogram since there are many values
    ax1.hist(components_all_a, bins=30, alpha=0.5, color='red', label='Lower CONS', edgecolor='darkred')
    ax1.hist(components_all_b, bins=30, alpha=0.5, color='blue', label='Higher CONS', edgecolor='darkblue')
    
    ax1.set_xlabel('Number of Connected Components', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('All Voters\n(unrepresented = own component)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Represented voters only
    ax2 = axes[1]
    counts_a = Counter(components_rep_a)
    counts_b = Counter(components_rep_b)
    all_values = sorted(set(components_rep_a + components_rep_b))
    
    width = 0.35
    x = np.arange(len(all_values))
    heights_a = [counts_a.get(v, 0) for v in all_values]
    heights_b = [counts_b.get(v, 0) for v in all_values]
    
    ax2.bar(x - width/2, heights_a, width, alpha=0.5, color='red', label='Lower CONS', edgecolor='darkred')
    ax2.bar(x + width/2, heights_b, width, alpha=0.5, color='blue', label='Higher CONS', edgecolor='darkblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_values)
    
    ax2.set_xlabel('Number of Connected Components', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Represented Voters Only\n(excludes unrepresented)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Distribution of Connected Components - Warsaw 2017 Wawrzyszew', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = OUTPUT_DIR / "components_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nHistogram saved to: {output_path}")


if __name__ == "__main__":
    main()

