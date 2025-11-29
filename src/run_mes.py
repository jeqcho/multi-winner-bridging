"""
Script to run Method of Equal Shares on the French 2007 approval voting data.

Computes MES committees for each size k=1..12 and saves results with scores
and alpha-approximations.
"""

import numpy as np
import pandas as pd
import json

from data_loader import load_and_combine_data, load_preflib_file
from mes import method_of_equal_shares
from scoring import av_score, cc_score, pairs_score, cons_score, ejr_satisfied, beta_ejr


def run_mes_all_sizes(output_file='output/french_election/mes_results.csv', M=None, candidates=None, output_dir=None):
    """
    Run MES for all committee sizes and save results.
    
    Args:
        output_file: Path to save results CSV (or filename if output_dir provided)
        M: Optional pre-loaded approval matrix
        candidates: Optional pre-loaded candidate names
        output_dir: Optional directory prefix for input/output files
    """
    import os
    
    # Handle output_dir prefixing
    alpha_scores_file = 'output/french_election/alpha_scores.csv'
    max_by_size_file = 'output/french_election/max_scores_by_size.csv'
    if output_dir:
        output_file = os.path.join(output_dir, 'mes_results.csv')
        alpha_scores_file = os.path.join(output_dir, 'alpha_scores.csv')
        max_by_size_file = os.path.join(output_dir, 'max_scores_by_size.csv')
    
    print("="*70)
    print("RUNNING METHOD OF EQUAL SHARES")
    print("="*70)
    
    # Load data if not provided
    if M is None or candidates is None:
        print("\nLoading data...")
        M, candidates = load_and_combine_data()
    n_voters, n_candidates = M.shape
    print(f"Dataset: {n_voters} voters, {n_candidates} candidates")
    
    # Load max scores for alpha normalization
    print("\nLoading max scores for normalization...")
    max_global = pd.read_csv(alpha_scores_file)
    max_by_size = pd.read_csv(max_by_size_file)
    
    # Global max values
    global_max_av = max_global['AV'].max()
    global_max_cc = max_global['CC'].max()
    global_max_pairs = max_global['PAIRS'].max()
    global_max_cons = max_global['CONS'].max()
    
    print(f"Global max - AV: {global_max_av}, CC: {global_max_cc}, PAIRS: {global_max_pairs}, CONS: {global_max_cons}")
    
    # Run MES for each committee size
    results = []
    
    print("\nRunning MES for each committee size...")
    for k in range(1, n_candidates + 1):
        print(f"\n  k={k}:")
        
        # Run MES
        committee = method_of_equal_shares(M, k)
        print(f"    Committee: {committee}")
        
        # Calculate scores
        av = av_score(M, committee)
        cc = cc_score(M, committee)
        pairs = pairs_score(M, committee)
        cons = cons_score(M, committee)
        ejr = ejr_satisfied(M, committee, k)
        beta = beta_ejr(M, committee, k)
        
        print(f"    AV={av}, CC={cc}, PAIRS={pairs}, CONS={cons}")
        print(f"    EJR={ejr}, beta_EJR={beta:.3f}")
        
        # Get max scores for this size
        size_row = max_by_size[max_by_size['subset_size'] == k].iloc[0]
        max_av_size = size_row['max_AV']
        max_cc_size = size_row['max_CC']
        max_pairs_size = size_row['max_PAIRS']
        max_cons_size = size_row['max_CONS']
        
        # Calculate alpha approximations (global)
        alpha_av_global = av / global_max_av if global_max_av > 0 else 0
        alpha_cc_global = cc / global_max_cc if global_max_cc > 0 else 0
        alpha_pairs_global = pairs / global_max_pairs if global_max_pairs > 0 else 0
        alpha_cons_global = cons / global_max_cons if global_max_cons > 0 else 0
        
        # Calculate alpha approximations (by size)
        alpha_av_size = av / max_av_size if max_av_size > 0 else 0
        alpha_cc_size = cc / max_cc_size if max_cc_size > 0 else 0
        alpha_pairs_size = pairs / max_pairs_size if max_pairs_size > 0 else 0
        alpha_cons_size = cons / max_cons_size if max_cons_size > 0 else 0
        
        results.append({
            'subset_size': k,
            'subset_indices': json.dumps(committee),
            'AV': av,
            'CC': cc,
            'PAIRS': pairs,
            'CONS': cons,
            'EJR': ejr,
            'beta_EJR': beta,
            # Global alpha values (for alpha_plots.png)
            'alpha_AV_global': alpha_av_global,
            'alpha_CC_global': alpha_cc_global,
            'alpha_PAIRS_global': alpha_pairs_global,
            'alpha_CONS_global': alpha_cons_global,
            'alpha_EJR': beta,  # Same as beta_EJR
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
    
    # Summary
    print("\n" + "="*70)
    print("MES RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Output file: {output_file}")
    
    return df


if __name__ == "__main__":
    run_mes_all_sizes()

