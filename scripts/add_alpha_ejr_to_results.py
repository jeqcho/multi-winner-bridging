#!/usr/bin/env python3
"""
Add alpha_EJR column to all voting_results.csv files.

For each election:
1. Load the .pb file to get approval matrix, costs, and budget
2. Load voting_results.csv
3. For each voting method, compute budget-aware alpha-EJR using the ILP
4. Add alpha_EJR column and save the updated CSV

Usage:
    uv run python scripts/add_alpha_ejr_to_results.py
    uv run python scripts/add_alpha_ejr_to_results.py --limit 5  # Test on 5 elections
    uv run python scripts/add_alpha_ejr_to_results.py --election "France_Toulouse*"
    uv run python scripts/add_alpha_ejr_to_results.py --dry-run  # Don't write files
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import argparse
import fnmatch

# Limit numpy thread usage to avoid oversubscription with multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pb_data_loader import load_pb_file
from alpha_ejr_pb_ilp import compute_alpha_ejr_pb


def find_elections(base_dir: Path, pattern: str = None) -> list:
    """
    Find all elections with voting_results.csv and corresponding .pb files.
    """
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
            'voting_csv': str(csv_path),
            'pb_path': str(pb_path)
        })
    
    return sorted(elections, key=lambda x: x['name'])


def process_election(args: tuple) -> dict:
    """
    Process a single election: compute alpha_EJR for all methods.
    
    Args:
        args: (election_name, voting_csv, pb_path, dry_run)
    
    Returns:
        dict with results
    """
    election_name, voting_csv, pb_path, dry_run = args
    
    try:
        # Suppress output from load_pb_file
        import io
        import contextlib
        
        with contextlib.redirect_stdout(io.StringIO()):
            M, project_ids, project_costs, budget = load_pb_file(pb_path)
        
        n_voters, n_projects = M.shape
        
        # Load voting results
        df = pd.read_csv(voting_csv)
        
        # Compute alpha_EJR for each method
        alpha_ejrs = []
        method_results = []
        
        for idx, row in df.iterrows():
            method = row['method']
            committee_str = row['subset_indices']
            committee = json.loads(committee_str) if isinstance(committee_str, str) else []
            
            # Compute alpha-EJR
            alpha = compute_alpha_ejr_pb(M, project_costs, budget, committee, verbose=False)
            alpha_ejrs.append(alpha)
            
            method_results.append({
                'method': method,
                'committee_size': len(committee),
                'alpha_EJR': alpha
            })
        
        # Add column
        df['alpha_EJR'] = alpha_ejrs
        
        # Save (unless dry run)
        if not dry_run:
            df.to_csv(voting_csv, index=False)
        
        return {
            'election': election_name,
            'n_voters': n_voters,
            'n_projects': n_projects,
            'budget': budget,
            'methods': method_results,
            'saved': not dry_run,
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
        description='Add alpha_EJR column to all voting_results.csv files'
    )
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of elections to process')
    parser.add_argument('--election', type=str, default=None,
                       help='Filter elections by name pattern')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--sequential', action='store_true',
                       help='Run sequentially (for debugging)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Compute but do not write files')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    
    print("=" * 80)
    print("ADD alpha_EJR COLUMN TO VOTING RESULTS")
    print("=" * 80)
    
    if args.dry_run:
        print("\n*** DRY RUN - Files will NOT be modified ***\n")
    
    # Find elections
    print("Finding elections...")
    elections = find_elections(base_dir, args.election)
    
    if args.limit:
        elections = elections[:args.limit]
    
    print(f"Found {len(elections)} elections to process")
    
    if not elections:
        print("No elections found.")
        sys.exit(1)
    
    # Count total methods
    total_methods = len(elections) * 6  # Approximately 6 methods per election
    print(f"Estimated {total_methods} committees to evaluate")
    
    # Prepare args
    args_list = [(e['name'], e['voting_csv'], e['pb_path'], args.dry_run) 
                 for e in elections]
    
    # Process
    results = []
    errors = []
    
    if args.sequential:
        print("\nProcessing sequentially...")
        for election_args in tqdm(args_list, desc="Processing", unit="election"):
            result = process_election(election_args)
            if result.get('error'):
                errors.append(result)
                if args.verbose:
                    print(f"\nError in {result['election']}: {result['error']}")
            else:
                results.append(result)
                if args.verbose:
                    print(f"\n{result['election']}:")
                    for m in result['methods']:
                        print(f"  {m['method']}: alpha_EJR={m['alpha_EJR']:.4f}")
    else:
        n_workers = args.workers or min(8, multiprocessing.cpu_count())
        print(f"\nProcessing with {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_election, a): a[0] for a in args_list}
            
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc="Processing", unit="election"):
                result = future.result()
                if result.get('error'):
                    errors.append(result)
                else:
                    results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nElections processed: {len(results)}")
    print(f"Errors: {len(errors)}")
    
    if results:
        # Collect all alpha values
        all_alphas = []
        for r in results:
            for m in r['methods']:
                all_alphas.append(m['alpha_EJR'])
        
        alpha_1_count = sum(1 for a in all_alphas if a >= 0.9999)
        alpha_below_1 = [a for a in all_alphas if a < 0.9999]
        
        print(f"\nTotal committees evaluated: {len(all_alphas)}")
        print(f"  alpha_EJR = 1: {alpha_1_count} ({100*alpha_1_count/len(all_alphas):.1f}%)")
        print(f"  alpha_EJR < 1: {len(alpha_below_1)} ({100*len(alpha_below_1)/len(all_alphas):.1f}%)")
        
        if alpha_below_1:
            print(f"    min: {min(alpha_below_1):.4f}")
            print(f"    max: {max(alpha_below_1):.4f}")
            print(f"    mean: {sum(alpha_below_1)/len(alpha_below_1):.4f}")
    
    # Show sample results
    if results and args.verbose:
        print("\n--- Sample Results ---")
        for r in results[:3]:
            print(f"\n{r['election']}:")
            for m in r['methods']:
                print(f"  {m['method']}: alpha_EJR={m['alpha_EJR']:.4f}")
    
    # Report errors
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors[:5]:
            print(f"  {e['election']}: {e['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    if not args.dry_run and results:
        print(f"\n✓ Updated {len(results)} voting_results.csv files with alpha_EJR column")


if __name__ == "__main__":
    main()
