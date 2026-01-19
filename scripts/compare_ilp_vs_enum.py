#!/usr/bin/env python3
"""
Compare ILP vs Enumeration for computing optimal PAIRS and CONS scores.

This script runs an experiment comparing:
1. ILP-based optimization using Gurobi
2. Brute-force enumeration over all valid committees

on 5 small PB elections and generates a markdown report with timing results.

Usage:
    python scripts/compare_ilp_vs_enum.py
    python scripts/compare_ilp_vs_enum.py --output-dir analysis/experiments
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pb_data_loader import load_pb_file, enumerate_valid_committees, count_valid_committees
from pb_objectives_ilp import maximize_pairs_ilp, maximize_cons_ilp
from scoring import pairs_score, cons_score

# Elections to test (selected based on <10 projects AND <100 voters criteria, plus one with exactly 10)
# Ordered from smallest to largest to get quick results first
ELECTIONS = [
    "data/poland_warszawa_2018_przyczolek-grochowski.pb",  # 1 project, 94 voters
    "data/poland_warszawa_2018_sadul.pb",                   # 2 projects, 91 voters
    "data/poland_warszawa_2019_marysin-wawerski-poludniowy.pb",  # 2 projects, 85 voters
    "data/poland_warszawa_2017_plac-wojska-polskiego.pb",   # 4 projects, 27 voters
    "data/France_Toulouse_2022_17_-_Mirail-Universite_Reynerie_Bellefontaine.pb",  # 10 projects, 93 voters (slowest)
]

# Maximum voters for CONS ILP (skip CONS ILP for larger elections due to O(|V|^4) complexity)
MAX_VOTERS_FOR_CONS_ILP = 50


def run_enumeration(M, project_costs, budget) -> Dict[str, Any]:
    """
    Find optimal PAIRS and CONS scores by enumerating all valid committees.
    
    Returns:
        Dict with max_pairs, max_cons, pairs_committee, cons_committee, 
        enum_time, num_committees
    """
    print("  Running enumeration method...")
    
    # Count committees first
    num_committees = count_valid_committees(project_costs, budget)
    print(f"    Valid committees to enumerate: {num_committees:,}")
    
    # Enumerate all valid committees
    start_enum = time.time()
    committees = enumerate_valid_committees(project_costs, budget, show_progress=False)
    enum_time = time.time() - start_enum
    print(f"    Enumeration time: {enum_time:.4f}s")
    
    # Find max PAIRS
    start_pairs = time.time()
    max_pairs = 0
    pairs_committee = []
    for committee in committees:
        score = pairs_score(M, committee)
        if score > max_pairs:
            max_pairs = score
            pairs_committee = committee
    pairs_time = time.time() - start_pairs
    print(f"    PAIRS scoring time: {pairs_time:.4f}s")
    
    # Find max CONS
    start_cons = time.time()
    max_cons = 0
    cons_committee = []
    for committee in committees:
        score = cons_score(M, committee)
        if score > max_cons:
            max_cons = score
            cons_committee = committee
    cons_time = time.time() - start_cons
    print(f"    CONS scoring time: {cons_time:.4f}s")
    
    total_time = enum_time + pairs_time + cons_time
    
    return {
        'max_pairs': max_pairs,
        'max_cons': max_cons,
        'pairs_committee': pairs_committee,
        'cons_committee': cons_committee,
        'enum_time': enum_time,
        'pairs_scoring_time': pairs_time,
        'cons_scoring_time': cons_time,
        'total_time': total_time,
        'num_committees': num_committees,
    }


def run_ilp(M, project_costs, budget, time_limit: int = 300, skip_cons: bool = False) -> Dict[str, Any]:
    """
    Find optimal PAIRS and CONS scores using ILP.
    
    Args:
        M: Approval matrix
        project_costs: List of project costs
        budget: Budget constraint
        time_limit: Time limit per ILP solve
        skip_cons: If True, skip CONS ILP (useful for large instances)
    
    Returns:
        Dict with max_pairs, max_cons, pairs_committee, cons_committee,
        pairs_time, cons_time, total_time
    """
    print("  Running ILP method...")
    
    # Run PAIRS ILP
    start_pairs = time.time()
    pairs_committee, pairs_score_ilp = maximize_pairs_ilp(
        M, project_costs, budget, verbose=False, time_limit=time_limit
    )
    pairs_time = time.time() - start_pairs
    print(f"    PAIRS ILP time: {pairs_time:.4f}s (score={pairs_score_ilp})")
    
    # Run CONS ILP (skip if too many voters - O(|V|^4) complexity)
    if skip_cons:
        print(f"    CONS ILP: SKIPPED (too many voters, would be too slow)")
        cons_committee = []
        cons_score_ilp = -1  # Sentinel value indicating skipped
        cons_time = 0
    else:
        start_cons = time.time()
        cons_committee, cons_score_ilp = maximize_cons_ilp(
            M, project_costs, budget, verbose=False, time_limit=time_limit
        )
        cons_time = time.time() - start_cons
        print(f"    CONS ILP time: {cons_time:.4f}s (score={cons_score_ilp})")
    
    total_time = pairs_time + cons_time
    
    return {
        'max_pairs': pairs_score_ilp,
        'max_cons': cons_score_ilp,
        'pairs_committee': pairs_committee,
        'cons_committee': cons_committee,
        'pairs_time': pairs_time,
        'cons_time': cons_time,
        'total_time': total_time,
        'cons_skipped': skip_cons,
    }


def process_election(filepath: str, time_limit: int = 300) -> Dict[str, Any]:
    """
    Process a single election, comparing ILP vs enumeration.
    
    Returns:
        Dict with all results for this election
    """
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # Load election data
    M, project_ids, project_costs, budget = load_pb_file(filepath)
    n_voters, n_projects = M.shape
    
    print(f"\n  Election stats:")
    print(f"    Projects: {n_projects}")
    print(f"    Voters: {n_voters}")
    print(f"    Budget: {budget:,}")
    
    # Run enumeration
    print()
    enum_results = run_enumeration(M, project_costs, budget)
    
    # Run ILP (skip CONS for large instances)
    skip_cons = n_voters > MAX_VOTERS_FOR_CONS_ILP
    if skip_cons:
        print(f"\n  Note: Skipping CONS ILP because {n_voters} voters > {MAX_VOTERS_FOR_CONS_ILP} threshold")
    print()
    ilp_results = run_ilp(M, project_costs, budget, time_limit, skip_cons=skip_cons)
    
    # Compare results
    pairs_match = enum_results['max_pairs'] == ilp_results['max_pairs']
    cons_skipped = ilp_results.get('cons_skipped', False)
    cons_match = cons_skipped or (enum_results['max_cons'] == ilp_results['max_cons'])
    
    print(f"\n  Results comparison:")
    print(f"    PAIRS - Enum: {enum_results['max_pairs']}, ILP: {ilp_results['max_pairs']} {'[MATCH]' if pairs_match else '[MISMATCH!]'}")
    if cons_skipped:
        print(f"    CONS  - Enum: {enum_results['max_cons']}, ILP: SKIPPED")
    else:
        print(f"    CONS  - Enum: {enum_results['max_cons']}, ILP: {ilp_results['max_cons']} {'[MATCH]' if cons_match else '[MISMATCH!]'}")
    
    # Calculate speedups
    pairs_speedup = enum_results['pairs_scoring_time'] / ilp_results['pairs_time'] if ilp_results['pairs_time'] > 0 else float('inf')
    cons_speedup = enum_results['cons_scoring_time'] / ilp_results['cons_time'] if ilp_results['cons_time'] > 0 and not cons_skipped else float('inf')
    total_speedup = enum_results['total_time'] / ilp_results['total_time'] if ilp_results['total_time'] > 0 else float('inf')
    
    print(f"\n  Speedup (Enum/ILP):")
    print(f"    PAIRS: {pairs_speedup:.2f}x")
    print(f"    CONS: {cons_speedup:.2f}x")
    print(f"    Total: {total_speedup:.2f}x")
    
    return {
        'filepath': filepath,
        'name': os.path.basename(filepath).replace('.pb', ''),
        'n_projects': n_projects,
        'n_voters': n_voters,
        'budget': budget,
        'num_committees': enum_results['num_committees'],
        'enum_results': enum_results,
        'ilp_results': ilp_results,
        'pairs_match': pairs_match,
        'cons_match': cons_match,
        'cons_skipped': cons_skipped,
        'pairs_speedup': pairs_speedup,
        'cons_speedup': cons_speedup,
        'total_speedup': total_speedup,
    }


def generate_report(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Generate a markdown report from the experiment results.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# ILP vs Enumeration Comparison Report

**Generated:** {timestamp}

## Overview

This report compares two methods for finding optimal PAIRS and CONS scores in participatory budgeting elections:

1. **ILP Method**: Uses Integer Linear Programming (Gurobi) to directly optimize the objective
2. **Enumeration Method**: Enumerates all budget-feasible committees and computes scores for each

## Elections Tested

| Election | Projects | Voters | Budget | Valid Committees |
|----------|----------|--------|--------|------------------|
"""
    
    for r in results:
        report += f"| {r['name'][:50]}{'...' if len(r['name']) > 50 else ''} | {r['n_projects']} | {r['n_voters']} | {r['budget']:,} | {r['num_committees']:,} |\n"
    
    report += """
## Results Summary

### PAIRS Objective

| Election | Optimal Score | ILP Time (s) | Enum Time (s) | Speedup | Match |
|----------|---------------|--------------|---------------|---------|-------|
"""
    
    for r in results:
        match_icon = "Yes" if r['pairs_match'] else "**NO**"
        enum_time = r['enum_results']['enum_time'] + r['enum_results']['pairs_scoring_time']
        speedup_str = f"{r['pairs_speedup']:.2f}x" if r['pairs_speedup'] != float('inf') else "N/A"
        report += f"| {r['name'][:40]}{'...' if len(r['name']) > 40 else ''} | {r['enum_results']['max_pairs']} | {r['ilp_results']['pairs_time']:.4f} | {enum_time:.4f} | {speedup_str} | {match_icon} |\n"
    
    report += """
### CONS Objective

| Election | Optimal Score | ILP Time (s) | Enum Time (s) | Speedup | Match |
|----------|---------------|--------------|---------------|---------|-------|
"""
    
    for r in results:
        if r.get('cons_skipped', False):
            match_icon = "SKIPPED"
            ilp_time_str = "SKIPPED"
            speedup_str = "N/A"
        else:
            match_icon = "Yes" if r['cons_match'] else "**NO**"
            ilp_time_str = f"{r['ilp_results']['cons_time']:.4f}"
            speedup_str = f"{r['cons_speedup']:.2f}x" if r['cons_speedup'] != float('inf') else "N/A"
        enum_time = r['enum_results']['enum_time'] + r['enum_results']['cons_scoring_time']
        report += f"| {r['name'][:40]}{'...' if len(r['name']) > 40 else ''} | {r['enum_results']['max_cons']} | {ilp_time_str} | {enum_time:.4f} | {speedup_str} | {match_icon} |\n"
    
    report += """
## Total Time Comparison

| Election | ILP Total (s) | Enum Total (s) | Speedup |
|----------|---------------|----------------|---------|
"""
    
    for r in results:
        speedup_str = f"{r['total_speedup']:.2f}x" if r['total_speedup'] != float('inf') else "N/A"
        report += f"| {r['name'][:40]}{'...' if len(r['name']) > 40 else ''} | {r['ilp_results']['total_time']:.4f} | {r['enum_results']['total_time']:.4f} | {speedup_str} |\n"
    
    # Summary statistics
    all_pairs_match = all(r['pairs_match'] for r in results)
    cons_tested = [r for r in results if not r.get('cons_skipped', False)]
    all_cons_match = all(r['cons_match'] for r in cons_tested) if cons_tested else True
    cons_skipped_count = sum(1 for r in results if r.get('cons_skipped', False))
    
    total_ilp_time = sum(r['ilp_results']['total_time'] for r in results)
    total_enum_time = sum(r['enum_results']['total_time'] for r in results)
    
    cons_match_str = "All matched" if all_cons_match else "Some mismatches!"
    if cons_skipped_count > 0:
        cons_match_str += f" ({cons_skipped_count} skipped due to size)"
    
    report += f"""
## Summary Statistics

- **Total ILP Time:** {total_ilp_time:.4f}s
- **Total Enumeration Time:** {total_enum_time:.4f}s
- **Overall Speedup:** {total_enum_time/total_ilp_time:.2f}x (Enum/ILP)
- **PAIRS Results Match:** {"All matched" if all_pairs_match else "Some mismatches!"}
- **CONS Results Match:** {cons_match_str}

## Conclusion

"""
    
    if all_pairs_match and all_cons_match:
        report += "Both ILP and enumeration methods produced identical optimal scores for all elections, confirming the correctness of the ILP formulations.\n"
    else:
        report += "**WARNING:** Some results did not match between methods. This may indicate a bug in the implementation or a timeout in the ILP solver.\n"
    
    if total_ilp_time < total_enum_time:
        report += f"\nThe ILP method was **{total_enum_time/total_ilp_time:.1f}x faster** overall than enumeration for these small elections.\n"
    else:
        report += f"\nThe enumeration method was **{total_ilp_time/total_enum_time:.1f}x faster** overall than ILP for these small elections. This is expected for very small instances where ILP overhead dominates.\n"
    
    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare ILP vs Enumeration for PAIRS and CONS optimization'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='analysis',
        help='Output directory for the report (default: analysis)'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=600,
        help='Time limit for each ILP solve in seconds (default: 600)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ILP vs Enumeration Comparison Experiment")
    print("="*70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elections to test: {len(ELECTIONS)}")
    print(f"ILP time limit: {args.time_limit}s per solve")
    
    # Check that all election files exist
    base_dir = os.path.dirname(os.path.dirname(__file__))
    missing = []
    for election in ELECTIONS:
        full_path = os.path.join(base_dir, election)
        if not os.path.exists(full_path):
            missing.append(election)
    
    if missing:
        print(f"\nERROR: The following election files were not found:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    
    # Process each election
    results = []
    for election in ELECTIONS:
        full_path = os.path.join(base_dir, election)
        result = process_election(full_path, args.time_limit)
        results.append(result)
    
    # Generate report
    report_path = os.path.join(base_dir, args.output_dir, 'ilp_vs_enum_report.md')
    generate_report(results, report_path)
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final summary
    all_match = all(r['pairs_match'] and r['cons_match'] for r in results)
    if all_match:
        print("\nAll results matched between ILP and enumeration methods.")
    else:
        print("\nWARNING: Some results did not match!")
        for r in results:
            if not r['pairs_match']:
                print(f"  - {r['name']}: PAIRS mismatch")
            if not r['cons_match']:
                print(f"  - {r['name']}: CONS mismatch")


if __name__ == '__main__':
    main()
