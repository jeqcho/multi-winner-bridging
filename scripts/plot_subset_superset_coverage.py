"""
Plot histogram of subset-superset coverage proportions across all PB elections.

For each election:
1. Enumerate all budget-feasible candidate subsets (committees)
2. For each subset, check if any voter's approval set is a superset of it
3. Compute the proportion of subsets that are "covered" by some voter

Output:
- Histogram showing distribution of coverage proportions across elections
- Summary table with statistics
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pb_data_loader import parse_pb_file, enumerate_valid_committees


def compute_coverage_proportion(data: dict, show_progress: bool = False) -> tuple[float, int, int]:
    """
    Compute the proportion of budget-feasible subsets covered by some voter.
    
    A subset S is "covered" if there exists a voter whose approval set A satisfies S ⊆ A.
    
    Args:
        data: Parsed PB data from parse_pb_file()
        show_progress: Whether to show progress bar for committee enumeration
        
    Returns:
        tuple: (proportion, covered_count, total_count)
    """
    # Extract project info
    project_ids = list(data["projects"].keys())
    n_projects = len(project_ids)
    project_costs = [data["projects"][pid]["cost"] for pid in project_ids]
    
    # Get budget (handle different formats)
    budget_raw = data["meta"].get("budget", 0)
    budget = int(float(budget_raw)) if isinstance(budget_raw, (int, float, str)) else 0
    
    # Create project_id to index mapping
    project_to_idx = {pid: idx for idx, pid in enumerate(project_ids)}
    
    # Convert voter approvals to sets of indices
    voter_sets = []
    for voter_id, approved_projects in data["votes"].items():
        approved_indices = frozenset(
            project_to_idx[pid] for pid in approved_projects if pid in project_to_idx
        )
        voter_sets.append(approved_indices)
    
    # Enumerate all budget-feasible committees
    committees = enumerate_valid_committees(project_costs, budget, show_progress=show_progress)
    total_count = len(committees)
    
    if total_count == 0:
        return 0.0, 0, 0
    
    # Count how many committees are covered by some voter
    covered_count = 0
    for committee in committees:
        committee_set = frozenset(committee)
        # Check if any voter's approval set is a superset of this committee
        if any(committee_set <= voter_set for voter_set in voter_sets):
            covered_count += 1
    
    proportion = covered_count / total_count
    return proportion, covered_count, total_count


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    analysis_dir = base_dir / "analysis"
    
    # Create analysis directory if it doesn't exist
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .pb files
    pb_files = list(data_dir.rglob("*.pb"))
    print(f"Found {len(pb_files)} .pb files")
    
    if not pb_files:
        print("No .pb files found in data/")
        return
    
    # Process each election
    proportions = []
    election_names = []
    skipped_count = 0
    
    for pb_file in tqdm(pb_files, desc="Processing elections"):
        try:
            data = parse_pb_file(str(pb_file))
        except Exception as e:
            print(f"  Warning: Failed to parse {pb_file.name}: {e}")
            continue
        
        n_projects = len(data["projects"])
        
        # Skip elections with too many candidates
        if n_projects > 13:
            print(f"  Warning: Skipping {pb_file.name} with {n_projects} candidates (>13)")
            skipped_count += 1
            continue
        
        if n_projects == 0:
            continue
        
        # Compute coverage proportion
        proportion, covered, total = compute_coverage_proportion(data, show_progress=False)
        
        if total > 0:
            proportions.append(proportion)
            election_names.append(pb_file.name)
    
    if not proportions:
        print("No valid elections found!")
        return
    
    proportions = np.array(proportions)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Subset-Superset Coverage Statistics")
    print("=" * 70)
    print(f"{'Metric':<40} {'Value':>25}")
    print("-" * 70)
    print(f"{'Number of elections processed':<40} {len(proportions):>25d}")
    print(f"{'Elections skipped (>13 candidates)':<40} {skipped_count:>25d}")
    print("-" * 70)
    print(f"{'Mean proportion':<40} {proportions.mean():>25.4f}")
    print(f"{'Median proportion':<40} {np.median(proportions):>25.4f}")
    print(f"{'Standard deviation':<40} {proportions.std():>25.4f}")
    print("-" * 70)
    print(f"{'Minimum proportion':<40} {proportions.min():>25.4f}")
    print(f"{'25th percentile':<40} {np.percentile(proportions, 25):>25.4f}")
    print(f"{'75th percentile':<40} {np.percentile(proportions, 75):>25.4f}")
    print(f"{'Maximum proportion':<40} {proportions.max():>25.4f}")
    print("-" * 70)
    pct_full_coverage = (proportions == 1.0).sum() / len(proportions) * 100
    print(f"{'Elections with 100% coverage':<40} {pct_full_coverage:>24.2f}%")
    print("=" * 70)
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(
        proportions,
        bins=30,
        alpha=0.7,
        color="#377eb8",
        edgecolor="black",
        linewidth=0.5,
    )
    
    ax.set_xlabel("Proportion of Committees Covered", fontsize=14)
    ax.set_ylabel("Number of Elections", fontsize=14)
    ax.set_title(
        "Proportion of Budget-Feasible Committees\nCovered by Some Voter's Approval Set",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.05)
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="100% coverage")
    ax.axvline(x=proportions.mean(), color="green", linestyle="-", alpha=0.7, label=f"Mean ({proportions.mean():.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_path = analysis_dir / "subset_superset_coverage.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nHistogram saved to: {output_path}")


if __name__ == "__main__":
    main()
