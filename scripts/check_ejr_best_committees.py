#!/usr/bin/env python3
"""
Check EJR satisfaction for best PAIRS and CONS committees across all elections.

For each election with processed output:
1. Find the best PAIRS-maximizing committee from raw_scores.csv
2. Find the best CONS-maximizing committee from raw_scores.csv
3. Load the approval matrix from the corresponding .pb file
4. Check EJR satisfaction for both committees

Outputs:
    analysis/ejr_best_pairs.json - EJR results for best PAIRS committees
    analysis/ejr_best_cons.json - EJR results for best CONS committees

Usage:
    uv run python scripts/check_ejr_best_committees.py
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

# Limit numpy thread usage to avoid oversubscription with multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pb_data_loader import load_pb_file
from scoring import ejr_satisfied


def find_elections(base_dir: Path) -> list:
    """
    Find all elections with processed output.
    
    Returns list of dicts with election info:
        - name: election name
        - csv_path: path to raw_scores.csv
        - pb_path: path to .pb file
    """
    pb_output_dir = base_dir / "output" / "pb"
    data_dir = base_dir / "data"
    
    elections = []
    
    # Find all raw_scores.csv files
    for csv_path in pb_output_dir.rglob("raw_scores.csv"):
        election_name = csv_path.parent.name
        pb_path = data_dir / f"{election_name}.pb"
        
        # Only include if .pb file exists
        if pb_path.exists():
            elections.append({
                'name': election_name,
                'csv_path': str(csv_path),
                'pb_path': str(pb_path)
            })
    
    return sorted(elections, key=lambda x: x['name'])


def get_best_committee(csv_path: str, metric: str) -> dict:
    """
    Get the best committee for a given metric from raw_scores.csv.
    
    Args:
        csv_path: Path to raw_scores.csv
        metric: Column name ('PAIRS' or 'CONS')
    
    Returns:
        dict with 'committee' (list of indices) and 'score' (int)
    """
    df = pd.read_csv(csv_path)
    
    # Find row with maximum score for the metric
    best_idx = df[metric].idxmax()
    best_row = df.loc[best_idx]
    
    # Parse committee indices from JSON
    subset_str = best_row['subset_indices']
    committee = json.loads(subset_str) if isinstance(subset_str, str) else []
    
    return {
        'committee': committee,
        'score': int(best_row[metric])
    }


def get_all_best_committees(csv_path: str, metric: str) -> dict:
    """
    Get ALL committees that achieve the maximum score for a given metric.
    
    Args:
        csv_path: Path to raw_scores.csv
        metric: Column name ('PAIRS' or 'CONS')
    
    Returns:
        dict with 'committees' (list of committee lists), 'score' (int), and 'count' (int)
    """
    df = pd.read_csv(csv_path)
    
    # Find maximum score
    max_score = df[metric].max()
    
    # Get all rows with the maximum score
    best_rows = df[df[metric] == max_score]
    
    committees = []
    for _, row in best_rows.iterrows():
        subset_str = row['subset_indices']
        committee = json.loads(subset_str) if isinstance(subset_str, str) else []
        committees.append(committee)
    
    return {
        'committees': committees,
        'score': int(max_score),
        'count': len(committees)
    }


def check_election_ejr(args: tuple) -> dict:
    """
    Worker function to check EJR for best PAIRS and CONS committees of an election.
    
    For each metric, checks all committees that achieve the maximum score (handles ties).
    Reports if ANY of the tied committees satisfies EJR.
    
    Args:
        args: tuple of (election_name, csv_path, pb_path)
    
    Returns:
        dict with election results for both metrics
    """
    election_name, csv_path, pb_path = args
    
    try:
        # Load approval matrix (suppress print output)
        import io
        import contextlib
        
        with contextlib.redirect_stdout(io.StringIO()):
            M, project_ids, project_costs, budget = load_pb_file(pb_path)
        
        n_voters, n_projects = M.shape
        
        # Get ALL best committees (handles ties)
        all_pairs = get_all_best_committees(csv_path, 'PAIRS')
        all_cons = get_all_best_committees(csv_path, 'CONS')
        
        # Check EJR for ALL PAIRS committees, track which satisfy EJR
        pairs_ejr_results = []
        for committee in all_pairs['committees']:
            k = len(committee)
            ejr = ejr_satisfied(M, committee, k) if k > 0 else True
            pairs_ejr_results.append({
                'committee': committee,
                'committee_size': k,
                'ejr_satisfied': ejr
            })
        
        # Check EJR for ALL CONS committees, track which satisfy EJR
        cons_ejr_results = []
        for committee in all_cons['committees']:
            k = len(committee)
            ejr = ejr_satisfied(M, committee, k) if k > 0 else True
            cons_ejr_results.append({
                'committee': committee,
                'committee_size': k,
                'ejr_satisfied': ejr
            })
        
        # Summary: any of the tied committees satisfy EJR?
        pairs_any_ejr = any(r['ejr_satisfied'] for r in pairs_ejr_results)
        cons_any_ejr = any(r['ejr_satisfied'] for r in cons_ejr_results)
        
        pairs_ejr_count = sum(1 for r in pairs_ejr_results if r['ejr_satisfied'])
        cons_ejr_count = sum(1 for r in cons_ejr_results if r['ejr_satisfied'])
        
        return {
            'election': election_name,
            'n_voters': n_voters,
            'n_projects': n_projects,
            'budget': budget,
            'pairs': {
                'score': all_pairs['score'],
                'num_tied_committees': all_pairs['count'],
                'ejr_satisfied_any': pairs_any_ejr,
                'ejr_satisfied_count': pairs_ejr_count,
                'committees': pairs_ejr_results
            },
            'cons': {
                'score': all_cons['score'],
                'num_tied_committees': all_cons['count'],
                'ejr_satisfied_any': cons_any_ejr,
                'ejr_satisfied_count': cons_ejr_count,
                'committees': cons_ejr_results
            },
            'error': None
        }
    
    except Exception as e:
        return {
            'election': election_name,
            'error': str(e)
        }


def main():
    base_dir = Path(__file__).parent.parent
    
    print("=" * 70)
    print("EJR CHECK FOR BEST PAIRS AND CONS COMMITTEES")
    print("=" * 70)
    
    # Find all elections
    print("\nFinding elections with processed output...")
    elections = find_elections(base_dir)
    print(f"Found {len(elections)} elections")
    
    if not elections:
        print("No elections found. Run main_pb_batch.py first to process elections.")
        sys.exit(1)
    
    # Prepare args for parallel processing
    args_list = [(e['name'], e['csv_path'], e['pb_path']) for e in elections]
    
    # Process in parallel
    n_workers = min(32, multiprocessing.cpu_count())
    print(f"\nProcessing {len(elections)} elections using {n_workers} workers...")
    
    results = []
    errors = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(check_election_ejr, args): args[0] 
                   for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Checking EJR", unit="election"):
            result = future.result()
            if result.get('error'):
                errors.append(result)
            else:
                results.append(result)
    
    # Sort results by election name
    results.sort(key=lambda x: x['election'])
    
    # Calculate statistics (any tied committee satisfies EJR)
    pairs_ejr_count = sum(1 for r in results if r['pairs']['ejr_satisfied_any'])
    cons_ejr_count = sum(1 for r in results if r['cons']['ejr_satisfied_any'])
    total = len(results)
    
    # Count elections with ties
    pairs_ties = sum(1 for r in results if r['pairs']['num_tied_committees'] > 1)
    cons_ties = sum(1 for r in results if r['cons']['num_tied_committees'] > 1)
    
    # Elections where NO tied committee satisfies EJR
    pairs_none_ejr = [r for r in results if not r['pairs']['ejr_satisfied_any']]
    cons_none_ejr = [r for r in results if not r['cons']['ejr_satisfied_any']]
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal elections processed: {total}")
    print(f"Errors: {len(errors)}")
    print(f"\nPAIRS:")
    print(f"  Elections with ties: {pairs_ties}")
    print(f"  Elections where ANY best committee satisfies EJR: {pairs_ejr_count}/{total} ({100*pairs_ejr_count/total:.1f}%)")
    print(f"  Elections where NO best committee satisfies EJR: {len(pairs_none_ejr)}")
    if pairs_none_ejr:
        print(f"    Elections failing EJR:")
        for r in pairs_none_ejr:
            print(f"      - {r['election']} ({r['pairs']['num_tied_committees']} tied committees)")
    
    print(f"\nCONS:")
    print(f"  Elections with ties: {cons_ties}")
    print(f"  Elections where ANY best committee satisfies EJR: {cons_ejr_count}/{total} ({100*cons_ejr_count/total:.1f}%)")
    print(f"  Elections where NO best committee satisfies EJR: {len(cons_none_ejr)}")
    if cons_none_ejr:
        print(f"    Elections failing EJR:")
        for r in cons_none_ejr:
            print(f"      - {r['election']} ({r['cons']['num_tied_committees']} tied committees)")
    
    # Prepare output for PAIRS
    pairs_output = {
        'metric': 'PAIRS',
        'total_elections': total,
        'ejr_satisfied_count': pairs_ejr_count,
        'ejr_satisfied_proportion': pairs_ejr_count / total if total > 0 else 0,
        'elections_with_ties': pairs_ties,
        'results': [
            {
                'election': r['election'],
                'score': r['pairs']['score'],
                'num_tied_committees': r['pairs']['num_tied_committees'],
                'ejr_satisfied_any': r['pairs']['ejr_satisfied_any'],
                'ejr_satisfied_count': r['pairs']['ejr_satisfied_count'],
                'committees': r['pairs']['committees']
            }
            for r in results
        ]
    }
    
    # Prepare output for CONS
    cons_output = {
        'metric': 'CONS',
        'total_elections': total,
        'ejr_satisfied_count': cons_ejr_count,
        'ejr_satisfied_proportion': cons_ejr_count / total if total > 0 else 0,
        'elections_with_ties': cons_ties,
        'results': [
            {
                'election': r['election'],
                'score': r['cons']['score'],
                'num_tied_committees': r['cons']['num_tied_committees'],
                'ejr_satisfied_any': r['cons']['ejr_satisfied_any'],
                'ejr_satisfied_count': r['cons']['ejr_satisfied_count'],
                'committees': r['cons']['committees']
            }
            for r in results
        ]
    }
    
    # Save results
    analysis_dir = base_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    pairs_file = analysis_dir / 'ejr_best_pairs.json'
    cons_file = analysis_dir / 'ejr_best_cons.json'
    
    with open(pairs_file, 'w') as f:
        json.dump(pairs_output, f, indent=2)
    print(f"\nSaved PAIRS results to {pairs_file}")
    
    with open(cons_file, 'w') as f:
        json.dump(cons_output, f, indent=2)
    print(f"Saved CONS results to {cons_file}")
    
    # Report errors if any
    if errors:
        print(f"\n{len(errors)} elections had errors:")
        for e in errors[:10]:  # Show first 10
            print(f"  {e['election']}: {e['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()

