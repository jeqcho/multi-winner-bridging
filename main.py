"""
Main script to calculate scores for all 2^12 = 4,096 candidate subsets.

DO NOT RUN YET - This will take approximately 34 minutes.
"""

import numpy as np
import pandas as pd
from itertools import combinations
import json
import time
from src.data_loader import load_and_combine_data
from src.scoring import av_score, cc_score, pairs_score, cons_score, ejr_satisfied, beta_ejr


def calculate_all_scores():
    """Calculate scores for all possible subsets of candidates."""
    print("="*70)
    print("CALCULATING SCORES FOR ALL SUBSETS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    M, candidates = load_and_combine_data()
    n_candidates = len(candidates)
    total_subsets = 2 ** n_candidates
    
    print(f"\nDataset: {M.shape[0]} voters, {n_candidates} candidates")
    print(f"Candidates: {candidates}")
    print(f"Total subsets to process: {total_subsets:,}")
    print(f"Estimated time: ~34 minutes")
    
    # Prepare results storage
    results = []
    
    # Start timing
    start_time = time.time()
    
    # Iterate through all subset sizes
    for k in range(n_candidates + 1):
        print(f"\n{'='*70}")
        print(f"Processing subsets of size {k}...")
        
        if k == 0:
            # Empty subset
            subsets = [[]]
        else:
            # All combinations of size k
            subsets = list(combinations(range(n_candidates), k))
        
        print(f"  Number of subsets: {len(subsets):,}")
        
        subset_start = time.time()
        
        for i, W in enumerate(subsets):
            W_list = list(W)
            
            # Calculate all scores
            av = av_score(M, W_list)
            cc = cc_score(M, W_list)
            pairs = pairs_score(M, W_list)
            cons = cons_score(M, W_list)
            ejr = ejr_satisfied(M, W_list, k)
            beta = beta_ejr(M, W_list, k)
            
            # Store result
            results.append({
                'subset_size': k,
                'subset_indices': json.dumps(W_list),
                'AV': av,
                'CC': cc,
                'PAIRS': pairs,
                'CONS': cons,
                'EJR': ejr,
                'beta_EJR': beta
            })
            
            # Progress update every 100 subsets or at the end
            if (i + 1) % 100 == 0 or i + 1 == len(subsets):
                elapsed = time.time() - subset_start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(subsets) - i - 1)
                print(f"    {i+1:4d}/{len(subsets):4d} subsets | "
                      f"Avg: {avg_time*1000:6.1f} ms | "
                      f"Remaining: {remaining:5.1f}s")
        
        subset_elapsed = time.time() - subset_start
        print(f"  Completed in {subset_elapsed:.2f}s ({subset_elapsed/60:.2f} min)")
    
    # Convert to DataFrame
    print("\n" + "="*70)
    print("Converting to DataFrame...")
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = "output/french_election/raw_scores.csv"
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Summary statistics
    total_elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min, {total_elapsed/3600:.2f} hr)")
    print(f"Total subsets processed: {len(results):,}")
    print(f"Average time per subset: {total_elapsed/len(results)*1000:.2f} ms")
    print(f"Output file: {output_file}")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Display some statistics
    print("\n" + "="*70)
    print("SCORE STATISTICS")
    print("="*70)
    print(f"\nAV score range: {df['AV'].min()} - {df['AV'].max()}")
    print(f"CC score range: {df['CC'].min()} - {df['CC'].max()}")
    print(f"PAIRS score range: {df['PAIRS'].min()} - {df['PAIRS'].max()}")
    print(f"CONS score range: {df['CONS'].min()} - {df['CONS'].max()}")
    print(f"EJR satisfaction rate: {df['EJR'].sum() / len(df) * 100:.1f}%")
    print(f"Beta-EJR range: {df['beta_EJR'].min():.3f} - {df['beta_EJR'].max():.3f}")
    
    print("\n" + "="*70)
    print("Next step: Run src/alpha_approx.py to compute alpha-approximations")
    print("="*70)
    
    return df


if __name__ == "__main__":
    print("\n" + "="*70)
    print("⚠️  WARNING: This script will take approximately 34 minutes to run.")
    print("="*70)
    response = input("\nAre you sure you want to proceed? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        df = calculate_all_scores()
    else:
        print("\nAborted. When ready, run: uv run python main.py")





