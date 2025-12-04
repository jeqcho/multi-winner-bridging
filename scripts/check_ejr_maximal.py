#!/usr/bin/env python3
"""
Check EJR for maximal valid subsets of two elections.

For each election:
1. Load valid subsets from raw_scores.csv
2. Filter to maximal subsets (not strict subsets of any other valid subset)
3. Check EJR for each maximal subset, terminating early if one satisfies EJR

Usage:
    python scripts/check_ejr_maximal.py
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pb_data_loader import load_pb_file
from scoring import ejr_satisfied


def load_subsets_from_csv(csv_path: str) -> list:
    """Load valid subsets from raw_scores.csv."""
    df = pd.read_csv(csv_path)
    subsets = []
    for _, row in df.iterrows():
        subset_str = row['subset_indices']
        # Parse JSON list
        subset = json.loads(subset_str) if isinstance(subset_str, str) else []
        subsets.append(frozenset(subset))
    return subsets


def find_maximal_subsets(subsets: list) -> list:
    """
    Find maximal subsets (those not strictly contained in any other subset).
    
    A subset A is maximal if there is no subset B where A ⊂ B (strict subset).
    """
    subset_set = set(subsets)
    maximal = []
    
    for s in subsets:
        is_maximal = True
        for other in subset_set:
            # Check if s is a strict subset of other
            if s != other and s < other:
                is_maximal = False
                break
        if is_maximal:
            maximal.append(s)
    
    return maximal


def check_ejr_for_election(election_name: str, data_path: str, csv_path: str) -> dict:
    """
    Check EJR for maximal subsets of an election.
    
    Returns dict with results.
    """
    print(f"\n{'='*60}")
    print(f"Election: {election_name}")
    print(f"{'='*60}")
    
    # Load election data
    M, project_ids, project_costs, budget = load_pb_file(data_path)
    n_voters, n_projects = M.shape
    
    # Load valid subsets
    print(f"\nLoading subsets from {csv_path}...")
    subsets = load_subsets_from_csv(csv_path)
    print(f"Total valid subsets: {len(subsets)}")
    
    # Find maximal subsets
    print("\nFinding maximal subsets...")
    maximal = find_maximal_subsets(subsets)
    print(f"Maximal subsets: {len(maximal)}")
    
    # Sort by size descending (larger committees first)
    maximal_sorted = sorted(maximal, key=lambda s: -len(s))
    
    # Check EJR for each maximal subset
    print("\nChecking EJR for maximal subsets...")
    ejr_committee = None
    checked = 0
    
    for subset in maximal_sorted:
        W = list(subset)
        k = len(W)
        
        if k == 0:
            # Empty committee - skip
            continue
        
        checked += 1
        satisfies = ejr_satisfied(M, W, k)
        
        if satisfies:
            total_cost = sum(project_costs[j] for j in W)
            print(f"\n*** FOUND EJR COMMITTEE ***")
            print(f"  Committee indices: {sorted(W)}")
            print(f"  Committee size (k): {k}")
            print(f"  Total cost: {total_cost:,}")
            print(f"  Checked {checked} maximal subsets before finding EJR")
            ejr_committee = {
                'indices': sorted(W),
                'size': k,
                'cost': total_cost,
                'checked_count': checked
            }
            break
    
    if ejr_committee is None:
        print(f"\nNo EJR committee found after checking {checked} maximal subsets")
    
    return {
        'election': election_name,
        'n_voters': n_voters,
        'n_projects': n_projects,
        'budget': budget,
        'total_subsets': len(subsets),
        'maximal_subsets': len(maximal),
        'ejr_committee': ejr_committee
    }


def main():
    base_dir = Path(__file__).parent.parent
    
    # Define elections to check
    elections = [
        {
            'name': 'poland_lodz_2023_baluty-zachodnie',
            'data_path': base_dir / 'data' / 'poland_lodz_2023_baluty-zachodnie.pb',
            'csv_path': base_dir / 'output' / 'pb' / 'poland_lodz_2023_baluty-zachodnie' / 'raw_scores.csv'
        },
        {
            'name': 'us_stanford-dataset_2022-jersey-city-ward-f_vote-knapsacks',
            'data_path': base_dir / 'data' / 'us_stanford-dataset_2022-jersey-city-ward-f_vote-knapsacks.pb',
            'csv_path': base_dir / 'output' / 'pb' / 'us_stanford-dataset_2022-jersey-city-ward-f_vote-knapsacks' / 'raw_scores.csv'
        }
    ]
    
    results = []
    
    for election in elections:
        result = check_ejr_for_election(
            election['name'],
            str(election['data_path']),
            str(election['csv_path'])
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for r in results:
        print(f"\n{r['election']}:")
        print(f"  Voters: {r['n_voters']}, Projects: {r['n_projects']}, Budget: {r['budget']:,}")
        print(f"  Valid subsets: {r['total_subsets']}, Maximal: {r['maximal_subsets']}")
        if r['ejr_committee']:
            ejr = r['ejr_committee']
            print(f"  EJR Committee: {ejr['indices']} (size={ejr['size']}, cost={ejr['cost']:,})")
            print(f"  Found after checking {ejr['checked_count']} maximal subsets")
        else:
            print(f"  No EJR committee found")
    
    # Save results to analysis/
    output_dir = base_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'ejr_maximal_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()


