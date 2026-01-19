#!/usr/bin/env python3
"""
Compare ILP vs Enumeration for PAIRS objective only on larger elections.

Tests elections with 10 projects and 150-1000 voters to see where ILP
starts to outperform enumeration.

Usage:
    python scripts/compare_ilp_vs_enum_pairs_only.py
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pb_data_loader import load_pb_file, enumerate_valid_committees, count_valid_committees
from pb_objectives_ilp import maximize_pairs_ilp
from scoring import pairs_score

# Elections with 10 projects and varying voter counts (154 to 972)
ELECTIONS = [
    ("data/France_Toulouse_2022_7_-_Sept_Deniers_Ginestous-Sesquieres_Lalande.pb", 154),
    ("data/France_Toulouse_2022_13_-_Rangueil_Sauzelong_Jules-Julien_Pech-David_Pouvourville.pb", 304),
    ("data/France_Toulouse_2022_8_-_Minimes_Barriere_de_Paris_Ponts-Jumeaux_La_Vache_Raisins_Fondeyre.pb", 512),
    ("data/France_Toulouse_2022_12_-_Pont_des_Demoiselles_Ormeau_Montaudran_La_Terrasse_Malepere.pb", 659),
    ("data/France_Toulouse_2022_1_-_Capitole_Arnaud_Bernard_Carmes.pb", 972),
]


def run_enumeration_pairs(M, project_costs, budget) -> Dict[str, Any]:
    """Find optimal PAIRS score by enumeration."""
    print("  Running enumeration method...")
    
    num_committees = count_valid_committees(project_costs, budget)
    print(f"    Valid committees: {num_committees:,}")
    
    start_total = time.time()
    committees = enumerate_valid_committees(project_costs, budget, show_progress=False)
    enum_time = time.time() - start_total
    
    start_pairs = time.time()
    max_pairs = 0
    best_committee = []
    for committee in committees:
        score = pairs_score(M, committee)
        if score > max_pairs:
            max_pairs = score
            best_committee = committee
    pairs_time = time.time() - start_pairs
    
    total_time = enum_time + pairs_time
    print(f"    Enumeration: {enum_time:.4f}s, Scoring: {pairs_time:.4f}s, Total: {total_time:.4f}s")
    
    return {
        'max_pairs': max_pairs,
        'committee': best_committee,
        'enum_time': enum_time,
        'pairs_time': pairs_time,
        'total_time': total_time,
        'num_committees': num_committees,
    }


def run_ilp_pairs(M, project_costs, budget, time_limit: int = 120) -> Dict[str, Any]:
    """Find optimal PAIRS score using ILP."""
    print("  Running ILP method...")
    
    start = time.time()
    committee, score = maximize_pairs_ilp(
        M, project_costs, budget, verbose=False, time_limit=time_limit
    )
    ilp_time = time.time() - start
    print(f"    ILP time: {ilp_time:.4f}s (score={score})")
    
    return {
        'max_pairs': score,
        'committee': committee,
        'ilp_time': ilp_time,
    }


def process_election(filepath: str, expected_voters: int, time_limit: int = 120) -> Dict[str, Any]:
    """Process a single election."""
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # Load data
    M, project_ids, project_costs, budget = load_pb_file(filepath)
    n_voters, n_projects = M.shape
    
    print(f"\n  Stats: {n_projects} projects, {n_voters} voters, budget={budget:,}")
    
    # Run both methods
    print()
    enum_results = run_enumeration_pairs(M, project_costs, budget)
    print()
    ilp_results = run_ilp_pairs(M, project_costs, budget, time_limit)
    
    # Compare
    match = enum_results['max_pairs'] == ilp_results['max_pairs']
    speedup = enum_results['total_time'] / ilp_results['ilp_time'] if ilp_results['ilp_time'] > 0 else float('inf')
    
    print(f"\n  Results: Enum={enum_results['max_pairs']}, ILP={ilp_results['max_pairs']} {'[MATCH]' if match else '[MISMATCH!]'}")
    print(f"  Speedup: {speedup:.2f}x (Enum/ILP) - {'ILP faster' if speedup > 1 else 'Enum faster'}")
    
    return {
        'filepath': filepath,
        'name': os.path.basename(filepath).replace('.pb', ''),
        'n_projects': n_projects,
        'n_voters': n_voters,
        'num_committees': enum_results['num_committees'],
        'optimal_pairs': enum_results['max_pairs'],
        'enum_time': enum_results['total_time'],
        'ilp_time': ilp_results['ilp_time'],
        'match': match,
        'speedup': speedup,
    }


def generate_report(results: List[Dict[str, Any]], output_path: str) -> None:
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# PAIRS: ILP vs Enumeration Comparison (Larger Elections)

**Generated:** {timestamp}

## Overview

This experiment compares ILP vs enumeration for the **PAIRS objective only** on elections with 10 projects and 150-1000 voters.

## Elections Tested

| Election | Projects | Voters | Valid Committees |
|----------|----------|--------|------------------|
"""
    
    for r in results:
        report += f"| {r['name'][:50]}{'...' if len(r['name']) > 50 else ''} | {r['n_projects']} | {r['n_voters']} | {r['num_committees']:,} |\n"
    
    report += """
## PAIRS Results

| Voters | Committees | Optimal PAIRS | Enum Time (s) | ILP Time (s) | Speedup | Match |
|--------|------------|---------------|---------------|--------------|---------|-------|
"""
    
    for r in results:
        match_str = "Yes" if r['match'] else "**NO**"
        speedup_str = f"{r['speedup']:.2f}x"
        winner = "ILP" if r['speedup'] > 1 else "Enum"
        report += f"| {r['n_voters']} | {r['num_committees']:,} | {r['optimal_pairs']:,} | {r['enum_time']:.4f} | {r['ilp_time']:.4f} | {speedup_str} ({winner}) | {match_str} |\n"
    
    # Summary
    total_enum = sum(r['enum_time'] for r in results)
    total_ilp = sum(r['ilp_time'] for r in results)
    all_match = all(r['match'] for r in results)
    
    report += f"""
## Summary

- **Total Enumeration Time:** {total_enum:.4f}s
- **Total ILP Time:** {total_ilp:.4f}s
- **Overall:** {'ILP' if total_ilp < total_enum else 'Enum'} is {max(total_enum/total_ilp, total_ilp/total_enum):.1f}x faster overall
- **All Results Match:** {"Yes" if all_match else "NO - check for bugs!"}

## Analysis

"""
    
    # Find crossover point
    ilp_wins = [r for r in results if r['speedup'] > 1]
    if ilp_wins:
        crossover = min(r['n_voters'] for r in ilp_wins)
        report += f"- ILP becomes faster than enumeration at around **{crossover} voters** for 10-project elections\n"
    else:
        report += "- Enumeration was faster for all tested elections\n"
    
    report += f"""- PAIRS ILP scales with O(|V|^2) variables, while enumeration scales with number of valid committees
- For 10 projects with typical PB budgets, there are ~500-1000 valid committees regardless of voter count
- ILP time increases with voters (more pair variables), enumeration time increases with committees
"""
    
    if all_match:
        report += "\nBoth methods produce identical optimal scores, confirming correctness.\n"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PAIRS-only ILP vs Enum comparison')
    parser.add_argument('--output-dir', type=str, default='analysis')
    parser.add_argument('--time-limit', type=int, default=120)
    args = parser.parse_args()
    
    print("="*70)
    print("PAIRS: ILP vs Enumeration (10 projects, 150-1000 voters)")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    results = []
    for election_path, expected_voters in ELECTIONS:
        full_path = os.path.join(base_dir, election_path)
        result = process_election(full_path, expected_voters, args.time_limit)
        results.append(result)
    
    # Generate report
    report_path = os.path.join(base_dir, args.output_dir, 'ilp_vs_enum_pairs_large.md')
    generate_report(results, report_path)
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
