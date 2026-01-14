#!/usr/bin/env python3
"""
Compare alpha-EJR ILP results with greedy-CC EJR outcomes.

For each election with processed output:
1. Load the greedy-CC committee from voting_results.csv
2. Load the .pb file to get approval matrix and costs
3. Run budget-aware alpha-EJR ILP to compute optimal alpha
4. Compare with the EJR column (True/False from abcvoting's check_EJR)

Expected outcomes:
- If greedy-CC EJR=True and ILP alpha=1: Formulation is correct
- If greedy-CC EJR=True but ILP alpha<1: Bug in implementation or reference doc
- If greedy-CC EJR=False: ILP alpha shows actual alpha-EJR level achieved

Usage:
    uv run python scripts/compare_alpha_ejr_ilp.py
    uv run python scripts/compare_alpha_ejr_ilp.py --limit 10  # Test with first 10
    uv run python scripts/compare_alpha_ejr_ilp.py --election "france_toulouse*"
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
    Find all elections with processed output and corresponding .pb files.
    
    Args:
        base_dir: Project base directory
        pattern: Optional glob pattern to filter elections by name
    
    Returns:
        List of dicts with election info
    """
    pb_output_dir = base_dir / "output" / "pb"
    data_dir = base_dir / "data"
    
    elections = []
    
    # Find all voting_results.csv files
    for csv_path in pb_output_dir.rglob("voting_results.csv"):
        election_name = csv_path.parent.name
        
        # Apply pattern filter if provided
        if pattern and not fnmatch.fnmatch(election_name.lower(), pattern.lower()):
            continue
        
        # Look for .pb file in data directory (including subdirectories)
        pb_path = None
        for candidate in data_dir.rglob(f"{election_name}.pb"):
            pb_path = candidate
            break
        
        if pb_path is None:
            # Try without exact match (some files might have different names)
            continue
        
        elections.append({
            'name': election_name,
            'voting_csv': str(csv_path),
            'pb_path': str(pb_path)
        })
    
    return sorted(elections, key=lambda x: x['name'])


def get_greedy_cc_result(voting_csv: str) -> dict:
    """
    Extract greedy-CC result from voting_results.csv.
    
    Returns:
        dict with 'committee', 'ejr', 'total_cost', 'av', 'cc' etc.
    """
    df = pd.read_csv(voting_csv)
    
    # Find greedy-CC row
    cc_row = df[df['method'] == 'greedy-CC']
    if len(cc_row) == 0:
        raise ValueError("greedy-CC not found in voting_results.csv")
    
    cc_row = cc_row.iloc[0]
    
    # Parse committee
    committee_str = cc_row['subset_indices']
    committee = json.loads(committee_str) if isinstance(committee_str, str) else []
    
    # Get EJR status (may be True/False string, bool, or numpy bool)
    ejr_val = cc_row.get('EJR', None)
    if pd.isna(ejr_val):
        ejr = None
    elif isinstance(ejr_val, str):
        ejr = ejr_val.lower() == 'true'
    else:
        # Handle bool, numpy.bool_, etc.
        ejr = bool(ejr_val)
    
    return {
        'committee': committee,
        'ejr': ejr,
        'total_cost': int(cc_row['total_cost']),
        'subset_size': int(cc_row['subset_size']),
        'AV': int(cc_row['AV']),
        'CC': int(cc_row['CC']),
    }


def process_election(args: tuple) -> dict:
    """
    Worker function to process a single election.
    
    Args:
        args: (election_name, voting_csv, pb_path)
    
    Returns:
        dict with comparison results
    """
    election_name, voting_csv, pb_path = args
    
    try:
        # Suppress output from load_pb_file
        import io
        import contextlib
        
        with contextlib.redirect_stdout(io.StringIO()):
            M, project_ids, project_costs, budget = load_pb_file(pb_path)
        
        n_voters, n_projects = M.shape
        
        # Get greedy-CC result
        cc_result = get_greedy_cc_result(voting_csv)
        committee = cc_result['committee']
        ejr_abcvoting = cc_result['ejr']
        
        # Compute budget-aware alpha-EJR using ILP
        alpha_ilp = compute_alpha_ejr_pb(M, project_costs, budget, committee, verbose=False)
        
        # Determine validation status
        if ejr_abcvoting is True:
            if alpha_ilp >= 0.9999:
                validation = "PASS"  # EJR=True and alpha=1
            else:
                validation = "MISMATCH"  # EJR=True but alpha<1 (unexpected!)
        elif ejr_abcvoting is False:
            if alpha_ilp < 1.0:
                validation = "EXPECTED"  # EJR=False and alpha<1
            else:
                validation = "SURPRISING"  # EJR=False but alpha=1 (interesting)
        else:
            validation = "NO_EJR_DATA"
        
        return {
            'election': election_name,
            'n_voters': n_voters,
            'n_projects': n_projects,
            'budget': budget,
            'committee_size': len(committee),
            'total_cost': cc_result['total_cost'],
            'ejr_abcvoting': ejr_abcvoting,
            'alpha_ilp': alpha_ilp,
            'validation': validation,
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
        description='Compare alpha-EJR ILP results with greedy-CC EJR outcomes'
    )
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of elections to process')
    parser.add_argument('--election', type=str, default=None,
                       help='Filter elections by name pattern (e.g., "france*")')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--sequential', action='store_true',
                       help='Run sequentially (for debugging)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    
    print("=" * 80)
    print("ALPHA-EJR ILP COMPARISON: greedy-CC vs Budget-Aware ILP")
    print("=" * 80)
    
    # Find elections
    print("\nFinding elections with processed output...")
    elections = find_elections(base_dir, args.election)
    
    if args.limit:
        elections = elections[:args.limit]
    
    print(f"Found {len(elections)} elections to process")
    
    if not elections:
        print("No elections found. Run main_pb_batch.py first.")
        sys.exit(1)
    
    # Prepare args for processing
    args_list = [(e['name'], e['voting_csv'], e['pb_path']) for e in elections]
    
    # Process elections
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
                    print(f"\n{result['election']}: EJR={result['ejr_abcvoting']}, "
                          f"alpha={result['alpha_ilp']:.4f}, {result['validation']}")
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
    
    # Sort results by election name
    results.sort(key=lambda x: x['election'])
    
    # Calculate statistics
    total = len(results)
    
    # Validation counts
    pass_count = sum(1 for r in results if r['validation'] == 'PASS')
    mismatch_count = sum(1 for r in results if r['validation'] == 'MISMATCH')
    expected_count = sum(1 for r in results if r['validation'] == 'EXPECTED')
    surprising_count = sum(1 for r in results if r['validation'] == 'SURPRISING')
    
    # EJR counts
    ejr_true = sum(1 for r in results if r['ejr_abcvoting'] is True)
    ejr_false = sum(1 for r in results if r['ejr_abcvoting'] is False)
    
    # Alpha statistics
    alphas = [r['alpha_ilp'] for r in results]
    alpha_1_count = sum(1 for a in alphas if a >= 0.9999)
    alpha_below_1 = [a for a in alphas if a < 0.9999]
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal elections processed: {total}")
    print(f"Errors: {len(errors)}")
    
    print(f"\n--- EJR Status (from abcvoting, committee-size based, NOT budget-aware) ---")
    print(f"  NOTE: abcvoting uses committee size k, not budget B!")
    print(f"  EJR=True:  {ejr_true} ({100*ejr_true/total:.1f}%)")
    print(f"  EJR=False: {ejr_false} ({100*ejr_false/total:.1f}%)")
    
    print(f"\n--- Alpha-EJR ILP (budget-aware) ---")
    print(f"  alpha=1:   {alpha_1_count} ({100*alpha_1_count/total:.1f}%)")
    print(f"  alpha<1:   {len(alpha_below_1)} ({100*len(alpha_below_1)/total:.1f}%)")
    if alpha_below_1:
        print(f"    min alpha: {min(alpha_below_1):.4f}")
        print(f"    max alpha: {max(alpha_below_1):.4f}")
        print(f"    mean alpha: {sum(alpha_below_1)/len(alpha_below_1):.4f}")
    
    print(f"\n--- Validation ---")
    print(f"  PASS (EJR=True, alpha=1):        {pass_count}")
    print(f"  MISMATCH (EJR=True, alpha<1):    {mismatch_count}  <- Check these!")
    print(f"  EXPECTED (EJR=False, alpha<1):   {expected_count}")
    print(f"  SURPRISING (EJR=False, alpha=1): {surprising_count}")
    
    # Show MISMATCH cases (most important for validation)
    mismatch_results = [r for r in results if r['validation'] == 'MISMATCH']
    if mismatch_results:
        print(f"\n--- MISMATCH Cases (EJR=True but alpha<1) ---")
        for r in mismatch_results:
            print(f"  {r['election']}")
            print(f"    alpha={r['alpha_ilp']:.4f}, |W|={r['committee_size']}, "
                  f"cost={r['total_cost']}, budget={r['budget']}")
    
    # Show SURPRISING cases
    surprising_results = [r for r in results if r['validation'] == 'SURPRISING']
    if surprising_results:
        print(f"\n--- SURPRISING Cases (EJR=False but alpha=1) ---")
        for r in surprising_results:
            print(f"  {r['election']}")
            print(f"    |W|={r['committee_size']}, cost={r['total_cost']}, budget={r['budget']}")
    
    # Show some EXPECTED cases (EJR=False with their alpha values)
    expected_results = [r for r in results if r['validation'] == 'EXPECTED']
    if expected_results:
        print(f"\n--- Sample EXPECTED Cases (EJR=False, alpha<1) ---")
        # Sort by alpha to show worst cases first
        expected_sorted = sorted(expected_results, key=lambda x: x['alpha_ilp'])
        for r in expected_sorted[:10]:  # Show top 10 lowest alphas
            print(f"  {r['election']}: alpha={r['alpha_ilp']:.4f}")
    
    # Save detailed results
    analysis_dir = base_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    output_file = analysis_dir / 'alpha_ejr_ilp_comparison.json'
    output_data = {
        'summary': {
            'total_elections': total,
            'errors': len(errors),
            'ejr_true': ejr_true,
            'ejr_false': ejr_false,
            'alpha_1_count': alpha_1_count,
            'alpha_below_1_count': len(alpha_below_1),
            'pass_count': pass_count,
            'mismatch_count': mismatch_count,
            'expected_count': expected_count,
            'surprising_count': surprising_count,
        },
        'results': results,
        'errors': errors
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Also save as CSV for easy viewing
    csv_file = analysis_dir / 'alpha_ejr_ilp_comparison.csv'
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"CSV results saved to: {csv_file}")
    
    # Report errors if any
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors[:5]:
            print(f"  {e['election']}: {e['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


if __name__ == "__main__":
    main()
