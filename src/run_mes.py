"""
Script to run Method of Equal Shares and other voting methods.

Computes MES, AV, greedy-CC, greedy-PAV, and max-score committees for each size k=1..n 
and saves results with scores and alpha-approximations.
"""

import numpy as np
import pandas as pd
import json
import time

from data_loader import load_and_combine_data, load_preflib_file
from mes import method_of_equal_shares
from scoring import av_score, cc_score, pairs_score, cons_score
from voting_methods import approval_voting, chamberlin_courant_greedy, pav_greedy, select_max_committee


def run_mes_all_sizes(output_file='output/french_election/voting_results.csv', M=None, candidates=None, output_dir=None):
    """
    Run MES and other voting methods for all committee sizes and save results.
    
    Args:
        output_file: Path to save results CSV (or filename if output_dir provided)
        M: Optional pre-loaded approval matrix
        candidates: Optional pre-loaded candidate names
        output_dir: Optional directory prefix for input/output files
    """
    import os
    
    start_time = time.time()
    
    # Handle output_dir prefixing
    max_by_size_file = 'output/french_election/max_scores_by_size.csv'
    raw_scores_file = 'output/french_election/raw_scores.csv'
    if output_dir:
        output_file = os.path.join(output_dir, 'voting_results.csv')
        max_by_size_file = os.path.join(output_dir, 'max_scores_by_size.csv')
        raw_scores_file = os.path.join(output_dir, 'raw_scores.csv')
    
    print("="*70)
    print("RUNNING VOTING METHODS (MES, AV, greedy-CC, greedy-PAV, PAIRS-AV, PAIRS-CC, CONS-AV, CONS-CC)")
    print("="*70)
    
    # Load data if not provided
    if M is None or candidates is None:
        print("\nLoading data...")
        M, candidates = load_and_combine_data()
    n_voters, n_candidates = M.shape
    print(f"Dataset: {n_voters} voters, {n_candidates} candidates")
    
    # Load max scores for alpha normalization (by size)
    print("\nLoading max scores for normalization...")
    max_by_size = pd.read_csv(max_by_size_file)
    
    # Load raw scores for max-score methods
    print("Loading raw scores for max-score methods...")
    raw_scores = pd.read_csv(raw_scores_file)
    
    # Define greedy voting methods (computed from M)
    greedy_methods = {
        'MES': method_of_equal_shares,
        'AV': approval_voting,
        'greedy-CC': chamberlin_courant_greedy,
        'greedy-PAV': pav_greedy,
    }
    
    # Define max-score methods (computed from raw_scores.csv)
    max_score_methods = {
        'PAIRS-AV': ('PAIRS', 'AV'),
        'PAIRS-CC': ('PAIRS', 'CC'),
        'CONS-AV': ('CONS', 'AV'),
        'CONS-CC': ('CONS', 'CC'),
    }
    
    # Run all methods for each committee size
    results = []
    timing = {name: 0 for name in list(greedy_methods.keys()) + list(max_score_methods.keys())}
    
    print("\nRunning voting methods for each committee size...")
    for k in range(1, n_candidates + 1):
        print(f"\n  k={k}:")
        
        # Get max scores for this size
        size_row = max_by_size[max_by_size['subset_size'] == k].iloc[0]
        max_av_size = size_row['max_AV']
        max_cc_size = size_row['max_CC']
        max_pairs_size = size_row['max_PAIRS']
        max_cons_size = size_row['max_CONS']
        
        # Run greedy methods
        for method_name, method_func in greedy_methods.items():
            t0 = time.time()
            committee = method_func(M, k)
            timing[method_name] += time.time() - t0
            
            # Calculate scores
            av = av_score(M, committee)
            cc = cc_score(M, committee)
            pairs = pairs_score(M, committee)
            cons = cons_score(M, committee)
            
            print(f"    {method_name}: {committee} | AV={av}, CC={cc}, PAIRS={pairs}, CONS={cons}")
            
            # Calculate alpha approximations (by size)
            alpha_av_size = av / max_av_size if max_av_size > 0 else 0
            alpha_cc_size = cc / max_cc_size if max_cc_size > 0 else 0
            alpha_pairs_size = pairs / max_pairs_size if max_pairs_size > 0 else 0
            alpha_cons_size = cons / max_cons_size if max_cons_size > 0 else 0
            
            results.append({
                'method': method_name,
                'subset_size': k,
                'subset_indices': json.dumps(committee),
                'AV': av,
                'CC': cc,
                'PAIRS': pairs,
                'CONS': cons,
                # Size-normalized alpha values (for alpha_plots_by_size.png and by_size/)
                'alpha_AV': alpha_av_size,
                'alpha_CC': alpha_cc_size,
                'alpha_PAIRS': alpha_pairs_size,
                'alpha_CONS': alpha_cons_size,
            })
        
        # Run max-score methods
        for method_name, (primary, secondary) in max_score_methods.items():
            t0 = time.time()
            committee = select_max_committee(raw_scores, k, primary, secondary)
            timing[method_name] += time.time() - t0
            
            # Calculate scores
            av = av_score(M, committee)
            cc = cc_score(M, committee)
            pairs = pairs_score(M, committee)
            cons = cons_score(M, committee)
            
            print(f"    {method_name}: {committee} | AV={av}, CC={cc}, PAIRS={pairs}, CONS={cons}")
            
            # Calculate alpha approximations (by size)
            alpha_av_size = av / max_av_size if max_av_size > 0 else 0
            alpha_cc_size = cc / max_cc_size if max_cc_size > 0 else 0
            alpha_pairs_size = pairs / max_pairs_size if max_pairs_size > 0 else 0
            alpha_cons_size = cons / max_cons_size if max_cons_size > 0 else 0
            
            results.append({
                'method': method_name,
                'subset_size': k,
                'subset_indices': json.dumps(committee),
                'AV': av,
                'CC': cc,
                'PAIRS': pairs,
                'CONS': cons,
                # Size-normalized alpha values (for alpha_plots_by_size.png and by_size/)
                'alpha_AV': alpha_av_size,
                'alpha_CC': alpha_cc_size,
                'alpha_PAIRS': alpha_pairs_size,
                'alpha_CONS': alpha_cons_size,
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    
    print(f"\nSaving results to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Timing breakdown
    total_elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("TIMING BREAKDOWN")
    print("="*70)
    for method_name, elapsed in sorted(timing.items(), key=lambda x: -x[1]):
        pct = elapsed / total_elapsed * 100
        print(f"  {method_name}: {elapsed:.2f}s ({pct:.1f}%)")
    
    # Summary
    print("\n" + "="*70)
    print("VOTING METHODS RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Output file: {output_file}")
    
    return df


if __name__ == "__main__":
    run_mes_all_sizes()

