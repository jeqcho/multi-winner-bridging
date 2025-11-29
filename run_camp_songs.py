"""
Runner script to process Camp Songs dataset (00059) from PrefLib.

Processes both feasible files:
- 00059-00000002.cat: 8 candidates (2022 Second Question)
- 00059-00000004.cat: 10 candidates (2023 Second Question)

Outputs to output_camp_songs/file_02/ and output_camp_songs/file_04/
"""

import numpy as np
import pandas as pd
from itertools import combinations
import json
import time
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_preflib_file
from scoring import av_score, cc_score, pairs_score, cons_score, ejr_satisfied, beta_ejr
from alpha_approx import calculate_alpha_approximations
from alpha_approx_by_size import calculate_alpha_by_size
from plot_results import plot_results
from plot_results_by_size import plot_results_by_size
from plot_individual_sizes import plot_all_sizes
from run_mes import run_mes_all_sizes


# Camp Songs dataset files
CAMP_SONGS_BASE_URL = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/00059%20-%20campsongs/"

CAMP_SONGS_FILES = {
    'file_02': {
        'url': CAMP_SONGS_BASE_URL + "00059-00000002.cat",
        'description': "2022 Second Question (8 candidates, 39 voters)"
    },
    'file_04': {
        'url': CAMP_SONGS_BASE_URL + "00059-00000004.cat",
        'description': "2023 Second Question (10 candidates, 56 voters)"
    }
}


def calculate_all_scores_for_dataset(M, candidates, output_dir):
    """Calculate scores for all possible subsets of candidates."""
    n_candidates = len(candidates)
    total_subsets = 2 ** n_candidates
    
    print(f"\nDataset: {M.shape[0]} voters, {n_candidates} candidates")
    print(f"Candidates: {candidates}")
    print(f"Total subsets to process: {total_subsets:,}")
    
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
    output_file = os.path.join(output_dir, "raw_scores.csv")
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Summary statistics
    total_elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print(f"Total subsets processed: {len(results):,}")
    print(f"Average time per subset: {total_elapsed/len(results)*1000:.2f} ms")
    print(f"Output file: {output_file}")
    
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
    
    return df


def process_single_file(file_key, file_info, base_output_dir='output/camp_songs'):
    """Process a single Camp Songs file through the full pipeline."""
    print("\n" + "="*70)
    print(f"PROCESSING: {file_info['description']}")
    print("="*70)
    
    # Create output directory
    output_dir = os.path.join(base_output_dir, file_key)
    os.makedirs(output_dir, exist_ok=True)
    by_size_dir = os.path.join(output_dir, 'by_size')
    os.makedirs(by_size_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Load data
    print("\n" + "="*70)
    print("STEP 1: Loading data")
    print("="*70)
    M, candidates = load_preflib_file(file_info['url'])
    n_candidates = len(candidates)
    
    # Step 2: Calculate raw scores
    print("\n" + "="*70)
    print("STEP 2: Calculating raw scores")
    print("="*70)
    calculate_all_scores_for_dataset(M, candidates, output_dir)
    
    # Step 3: Calculate alpha approximations (global)
    print("\n" + "="*70)
    print("STEP 3: Calculating alpha approximations (global)")
    print("="*70)
    calculate_alpha_approximations(
        input_file='raw_scores.csv',
        output_file='alpha_scores.csv',
        output_dir=output_dir
    )
    
    # Step 4: Calculate alpha approximations (by size)
    print("\n" + "="*70)
    print("STEP 4: Calculating alpha approximations (by size)")
    print("="*70)
    calculate_alpha_by_size(
        input_file='raw_scores.csv',
        output_file='alpha_scores_by_size.csv',
        max_file='max_scores_by_size.csv',
        output_dir=output_dir
    )
    
    # Step 5: Run MES
    print("\n" + "="*70)
    print("STEP 5: Running Method of Equal Shares")
    print("="*70)
    run_mes_all_sizes(
        output_file='mes_results.csv',
        M=M,
        candidates=candidates,
        output_dir=output_dir
    )
    
    # Step 6: Create plots (global)
    print("\n" + "="*70)
    print("STEP 6: Creating plots (global)")
    print("="*70)
    plot_results(
        input_file='alpha_scores.csv',
        output_file='alpha_plots.png',
        mes_file='mes_results.csv',
        output_dir=output_dir,
        n_candidates=n_candidates
    )
    
    # Step 7: Create plots (by size)
    print("\n" + "="*70)
    print("STEP 7: Creating plots (by size)")
    print("="*70)
    plot_results_by_size(
        input_file='alpha_scores_by_size.csv',
        output_file='alpha_plots_by_size.png',
        mes_file='mes_results.csv',
        output_dir=output_dir,
        n_candidates=n_candidates
    )
    
    # Step 8: Create individual size plots
    print("\n" + "="*70)
    print("STEP 8: Creating individual size plots")
    print("="*70)
    plot_all_sizes(
        input_file='alpha_scores_by_size.csv',
        output_dir='by_size',
        mes_file='mes_results.csv',
        base_dir=output_dir
    )
    
    print("\n" + "="*70)
    print(f"COMPLETED: {file_info['description']}")
    print(f"Output saved to: {output_dir}")
    print("="*70)


def main():
    """Process all Camp Songs dataset files."""
    print("\n" + "="*70)
    print("CAMP SONGS DATASET PROCESSING")
    print("="*70)
    print("\nThis script processes the Camp Songs dataset (00059) from PrefLib.")
    print("Files to process:")
    for key, info in CAMP_SONGS_FILES.items():
        print(f"  - {key}: {info['description']}")
    
    start_time = time.time()
    
    for file_key, file_info in CAMP_SONGS_FILES.items():
        process_single_file(file_key, file_info)
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("ALL PROCESSING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("\nOutput directories:")
    for key in CAMP_SONGS_FILES:
        print(f"  - output/camp_songs/{key}/")
    print("\nYou can view the plots with:")
    for key in CAMP_SONGS_FILES:
        print(f"  open output/camp_songs/{key}/alpha_plots.png")


if __name__ == "__main__":
    main()

