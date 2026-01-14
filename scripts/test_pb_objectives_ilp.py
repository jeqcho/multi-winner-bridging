#!/usr/bin/env python3
"""
Test script to validate ILP implementations against ground truth.

Loads elections from output/pb/ directories, runs ILP for each objective,
and compares results with ground truth from raw_scores.csv (computed by 
exhaustive enumeration).

Usage:
    python scripts/test_pb_objectives_ilp.py
    python scripts/test_pb_objectives_ilp.py --election "France_Toulouse_2022_1_*"
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pb_objectives_ilp import (
    maximize_av_ilp,
    maximize_cc_ilp,
    maximize_pairs_ilp,
    maximize_cons_ilp
)
from pb_data_loader import load_pb_file
from scoring import av_score, cc_score, pairs_score, cons_score


def load_ground_truth(output_dir: str) -> dict:
    """
    Load ground truth max scores from raw_scores.csv.
    
    Returns dict with max_AV, max_CC, max_PAIRS, max_CONS.
    """
    raw_file = os.path.join(output_dir, 'raw_scores.csv')
    if not os.path.exists(raw_file):
        return None
    
    df = pd.read_csv(raw_file)
    return {
        'max_AV': df['AV'].max(),
        'max_CC': df['CC'].max(),
        'max_PAIRS': df['PAIRS'].max(),
        'max_CONS': df['CONS'].max(),
    }


def find_pb_file(output_dir: str) -> str:
    """Find the corresponding .pb file for an output directory."""
    # The output dir name matches the pb file name (without .pb)
    name = os.path.basename(output_dir)
    
    # Search in data/ directory
    data_dirs = ['data', 'data/pb_selected_10_20251202_023743']
    for data_dir in data_dirs:
        pb_path = os.path.join(os.path.dirname(output_dir), '..', '..', data_dir, f'{name}.pb')
        if os.path.exists(pb_path):
            return pb_path
    
    # Search more broadly
    for pb_file in glob.glob(f'data/**/{name}.pb', recursive=True):
        return pb_file
    
    return None


def test_election(output_dir: str, verbose: bool = True, time_limit: int = 300) -> dict:
    """
    Test ILP implementations on a single election.
    
    Args:
        output_dir: Path to output directory with raw_scores.csv
        verbose: Whether to print progress
        time_limit: Time limit for each ILP
        
    Returns:
        Dict with test results
    """
    name = os.path.basename(output_dir)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
    
    # Load ground truth
    ground_truth = load_ground_truth(output_dir)
    if ground_truth is None:
        if verbose:
            print(f"  SKIP: No raw_scores.csv found")
        return {'name': name, 'status': 'skipped', 'reason': 'no raw_scores.csv'}
    
    # Find and load the .pb file
    pb_file = find_pb_file(output_dir)
    if pb_file is None:
        if verbose:
            print(f"  SKIP: No .pb file found")
        return {'name': name, 'status': 'skipped', 'reason': 'no .pb file'}
    
    # Suppress load_pb_file output
    import io
    from contextlib import redirect_stdout
    
    if not verbose:
        f = io.StringIO()
        with redirect_stdout(f):
            M, project_ids, project_costs, budget = load_pb_file(pb_file)
    else:
        M, project_ids, project_costs, budget = load_pb_file(pb_file)
    
    n_voters, n_projects = M.shape
    if verbose:
        print(f"\n  Voters: {n_voters}, Projects: {n_projects}, Budget: {budget:,}")
        print(f"  Ground truth: AV={ground_truth['max_AV']}, CC={ground_truth['max_CC']}, "
              f"PAIRS={ground_truth['max_PAIRS']}, CONS={ground_truth['max_CONS']}")
    
    results = {
        'name': name,
        'n_voters': n_voters,
        'n_projects': n_projects,
        'budget': budget,
        'ground_truth': ground_truth,
        'ilp_results': {},
        'errors': []
    }
    
    # Test each objective
    objectives = [
        ('AV', maximize_av_ilp, 'max_AV'),
        ('CC', maximize_cc_ilp, 'max_CC'),
        ('PAIRS', maximize_pairs_ilp, 'max_PAIRS'),
        ('CONS', maximize_cons_ilp, 'max_CONS'),
    ]
    
    for obj_name, ilp_func, gt_key in objectives:
        if verbose:
            print(f"\n  Testing {obj_name}...")
        
        try:
            committee, score = ilp_func(M, project_costs, budget, verbose=False, time_limit=time_limit)
            
            # Verify the score matches what scoring function computes
            if obj_name == 'AV':
                verify_score = av_score(M, committee)
            elif obj_name == 'CC':
                verify_score = cc_score(M, committee)
            elif obj_name == 'PAIRS':
                verify_score = pairs_score(M, committee)
            else:  # CONS
                verify_score = cons_score(M, committee)
            
            gt_score = ground_truth[gt_key]
            
            results['ilp_results'][obj_name] = {
                'committee': committee,
                'ilp_score': score,
                'verify_score': verify_score,
                'ground_truth': gt_score,
                'match': score == gt_score,
                'verify_match': score == verify_score
            }
            
            if verbose:
                status = "✓" if score == gt_score else "✗"
                verify_status = "✓" if score == verify_score else "✗"
                print(f"    ILP score: {score}, Ground truth: {gt_score} {status}")
                print(f"    Verified: {verify_score} {verify_status}")
                if score != gt_score:
                    print(f"    ERROR: ILP found suboptimal solution! Gap: {gt_score - score}")
            
            if score != gt_score:
                results['errors'].append(f"{obj_name}: ILP={score}, GT={gt_score}")
                
        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")
            results['ilp_results'][obj_name] = {'error': str(e)}
            results['errors'].append(f"{obj_name}: {e}")
    
    results['status'] = 'passed' if not results['errors'] else 'failed'
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test PB objectives ILP implementations')
    parser.add_argument('--election', type=str, default=None,
                        help='Glob pattern for election names (e.g., "France_Toulouse*")')
    parser.add_argument('--output-dir', type=str, default='output/pb',
                        help='Base output directory')
    parser.add_argument('--time-limit', type=int, default=300,
                        help='Time limit per ILP in seconds')
    parser.add_argument('--max-elections', type=int, default=3,
                        help='Maximum number of elections to test')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Find election directories
    output_base = args.output_dir
    if args.election:
        pattern = os.path.join(output_base, args.election)
    else:
        pattern = os.path.join(output_base, '*')
    
    election_dirs = sorted(glob.glob(pattern))
    election_dirs = [d for d in election_dirs if os.path.isdir(d)]
    
    if not election_dirs:
        print(f"No election directories found matching {pattern}")
        return 1
    
    # Filter to elections with raw_scores.csv
    election_dirs = [d for d in election_dirs 
                     if os.path.exists(os.path.join(d, 'raw_scores.csv'))]
    
    print(f"Found {len(election_dirs)} elections with ground truth")
    
    # Limit number of elections
    if args.max_elections > 0 and len(election_dirs) > args.max_elections:
        print(f"Testing first {args.max_elections} elections")
        election_dirs = election_dirs[:args.max_elections]
    
    # Test each election
    all_results = []
    passed = 0
    failed = 0
    skipped = 0
    
    for election_dir in election_dirs:
        result = test_election(election_dir, verbose=True, time_limit=args.time_limit)
        all_results.append(result)
        
        if result['status'] == 'passed':
            passed += 1
        elif result['status'] == 'failed':
            failed += 1
        else:
            skipped += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed > 0:
        print("\nFailed elections:")
        for result in all_results:
            if result['status'] == 'failed':
                print(f"  {result['name']}")
                for error in result['errors']:
                    print(f"    - {error}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
