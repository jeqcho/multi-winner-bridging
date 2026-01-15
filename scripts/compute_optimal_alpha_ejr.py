#!/usr/bin/env python3
"""
Compute optimal alpha-EJR for each election (best possible outcome).

For each election, runs the full cutting-plane ILP optimization to find
the best (alpha, W) pair that maximizes alpha while satisfying budget-aware EJR.

This is used for the "Best Possible" bars in the tradeoff charts.

Output: analysis/optimal_alpha_ejr.json

Usage:
    uv run python scripts/compute_optimal_alpha_ejr.py
    uv run python scripts/compute_optimal_alpha_ejr.py --limit 10
    uv run python scripts/compute_optimal_alpha_ejr.py --workers 8
"""

import sys
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import argparse
import fnmatch

# Limit numpy thread usage
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pb_data_loader import load_pb_file
from alpha_ejr_pb_ilp import compute_alpha_ejr_pb_full_optimization
from scoring import pairs_score, cons_score


def find_elections(base_dir: Path, pattern: str = None) -> list:
    """Find all elections with .pb files and voting_results.csv."""
    pb_output_dir = base_dir / "output" / "pb"
    data_dir = base_dir / "data"
    
    elections = []
    
    for csv_path in pb_output_dir.rglob("voting_results.csv"):
        election_name = csv_path.parent.name
        
        if pattern and not fnmatch.fnmatch(election_name.lower(), pattern.lower()):
            continue
        
        # Look for .pb file
        pb_path = None
        for candidate in data_dir.rglob(f"{election_name}.pb"):
            pb_path = candidate
            break
        
        if pb_path is None:
            continue
        
        elections.append({
            'name': election_name,
            'pb_path': str(pb_path)
        })
    
    return sorted(elections, key=lambda x: x['name'])


def process_election(args: tuple) -> dict:
    """
    Compute optimal alpha-EJR for a single election.
    
    Args:
        args: (election_name, pb_path)
    
    Returns:
        dict with optimal alpha, committee, and scores
    """
    election_name, pb_path = args
    
    try:
        import io
        import contextlib
        
        with contextlib.redirect_stdout(io.StringIO()):
            M, project_ids, project_costs, budget = load_pb_file(pb_path)
        
        n_voters, n_projects = M.shape
        
        # Run full optimization to find best (alpha, W)
        optimal_alpha, optimal_W = compute_alpha_ejr_pb_full_optimization(
            M, project_costs, budget, verbose=False, max_iterations=50
        )
        
        # Compute scores for optimal committee
        if optimal_W:
            pairs = pairs_score(M, optimal_W)
            cons = cons_score(M, optimal_W)
            total_cost = sum(project_costs[c] for c in optimal_W)
        else:
            pairs = 0
            cons = 0
            total_cost = 0
        
        return {
            'election': election_name,
            'n_voters': n_voters,
            'n_projects': n_projects,
            'budget': budget,
            'optimal_alpha': optimal_alpha,
            'optimal_W': optimal_W,
            'committee_size': len(optimal_W) if optimal_W else 0,
            'total_cost': total_cost,
            'PAIRS': pairs,
            'CONS': cons,
            'error': None
        }
    
    except Exception as e:
        import traceback
        return {
            'election': election_name,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(
        description='Compute optimal alpha-EJR for each election'
    )
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of elections')
    parser.add_argument('--election', type=str, default=None,
                       help='Filter by election name pattern')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--sequential', action='store_true',
                       help='Run sequentially')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("COMPUTE OPTIMAL ALPHA-EJR (BEST POSSIBLE)")
    print("=" * 80)
    
    # Find elections
    print("\nFinding elections...")
    elections = find_elections(base_dir, args.election)
    
    if args.limit:
        elections = elections[:args.limit]
    
    print(f"Found {len(elections)} elections")
    
    if not elections:
        print("No elections found.")
        sys.exit(1)
    
    # Prepare args
    args_list = [(e['name'], e['pb_path']) for e in elections]
    
    # Process
    results = []
    errors = []
    
    if args.sequential:
        print("\nProcessing sequentially...")
        for election_args in tqdm(args_list, desc="Optimizing", unit="election"):
            result = process_election(election_args)
            if result.get('error'):
                errors.append(result)
            else:
                results.append(result)
    else:
        n_workers = args.workers or min(8, multiprocessing.cpu_count())
        print(f"\nProcessing with {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_election, a): a[0] for a in args_list}
            
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc="Optimizing", unit="election"):
                result = future.result()
                if result.get('error'):
                    errors.append(result)
                else:
                    results.append(result)
    
    # Sort results
    results.sort(key=lambda x: x['election'])
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nElections processed: {len(results)}")
    print(f"Errors: {len(errors)}")
    
    if results:
        alphas = [r['optimal_alpha'] for r in results]
        alpha_1_count = sum(1 for a in alphas if a >= 0.9999)
        
        print(f"\nOptimal alpha statistics:")
        print(f"  alpha=1: {alpha_1_count} ({100*alpha_1_count/len(alphas):.1f}%)")
        print(f"  alpha<1: {len(alphas) - alpha_1_count} ({100*(len(alphas)-alpha_1_count)/len(alphas):.1f}%)")
        print(f"  min: {min(alphas):.4f}")
        print(f"  max: {max(alphas):.4f}")
        print(f"  mean: {sum(alphas)/len(alphas):.4f}")
    
    # Save results
    output_data = {
        'total_elections': len(results),
        'alpha_1_count': sum(1 for r in results if r['optimal_alpha'] >= 0.9999),
        'alpha_1_proportion': sum(1 for r in results if r['optimal_alpha'] >= 0.9999) / len(results) if results else 0,
        'results': results,
        'errors': [{'election': e['election'], 'error': e['error']} for e in errors]
    }
    
    output_file = analysis_dir / 'optimal_alpha_ejr.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors[:5]:
            print(f"  {e['election']}: {e['error']}")


if __name__ == "__main__":
    main()
