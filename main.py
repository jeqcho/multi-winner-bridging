"""
Unified runner script for multi-winner bridging analysis.

Supports two datasets:
- French Election (2007): 12 candidates, 2836 voters from 6 polling stations
- Camp Songs: PrefLib dataset 00059 with two files (8 and 10 candidates)

Usage:
    python main.py french_election           # Run French election dataset
    python main.py camp_songs                # Run all Camp Songs files
    python main.py camp_songs --file file_02 # Run specific Camp Songs file
"""

import numpy as np
import pandas as pd
from itertools import combinations
import json
import time
import os
import sys
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_preflib_file, load_and_combine_data
from scoring import av_score, cc_score, pairs_score, cons_score
from alpha_approx_by_size import calculate_alpha_by_size
from plot_results_by_size import plot_results_by_size
from plot_individual_sizes import plot_all_sizes
from run_mes import run_mes_all_sizes
from plot_ejr import plot_ejr_results


# Dataset configurations
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

FRENCH_ELECTION = {
    'description': "2007 French Presidential Election (12 candidates, 2836 voters)"
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
    
    # Timing breakdown
    timing = {'AV': 0, 'CC': 0, 'PAIRS': 0, 'CONS': 0}
    
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
            
            # Calculate all scores with timing
            t0 = time.time()
            av = av_score(M, W_list)
            timing['AV'] += time.time() - t0
            
            t0 = time.time()
            cc = cc_score(M, W_list)
            timing['CC'] += time.time() - t0
            
            t0 = time.time()
            pairs = pairs_score(M, W_list)
            timing['PAIRS'] += time.time() - t0
            
            t0 = time.time()
            cons = cons_score(M, W_list)
            timing['CONS'] += time.time() - t0
            
            # Store result (EJR removed)
            results.append({
                'subset_size': k,
                'subset_indices': json.dumps(W_list),
                'AV': av,
                'CC': cc,
                'PAIRS': pairs,
                'CONS': cons,
            })
            
            # Progress update every 100 subsets or at the end
            if (i + 1) % 100 == 0 or i + 1 == len(subsets):
                elapsed = time.time() - subset_start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(subsets) - i - 1)
                # Calculate per-score averages
                n_done = i + 1
                avg_av = timing['AV'] / n_done * 1000
                avg_cc = timing['CC'] / n_done * 1000
                avg_pairs = timing['PAIRS'] / n_done * 1000
                avg_cons = timing['CONS'] / n_done * 1000
                print(f"    {i+1:4d}/{len(subsets):4d} | "
                      f"Avg: {avg_time*1000:5.1f}ms | "
                      f"AV:{avg_av:4.1f} CC:{avg_cc:4.1f} PAIRS:{avg_pairs:5.1f} CONS:{avg_cons:5.1f} | "
                      f"ETA: {remaining:5.1f}s")
        
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
    
    # Timing breakdown
    print("\n" + "="*70)
    print("TIMING BREAKDOWN")
    print("="*70)
    for score_name, elapsed in sorted(timing.items(), key=lambda x: -x[1]):
        pct = elapsed / total_elapsed * 100
        print(f"  {score_name}: {elapsed:.2f}s ({pct:.1f}%)")
    
    # Display some statistics
    print("\n" + "="*70)
    print("SCORE STATISTICS")
    print("="*70)
    print(f"\nAV score range: {df['AV'].min()} - {df['AV'].max()}")
    print(f"CC score range: {df['CC'].min()} - {df['CC'].max()}")
    print(f"PAIRS score range: {df['PAIRS'].min()} - {df['PAIRS'].max()}")
    print(f"CONS score range: {df['CONS'].min()} - {df['CONS'].max()}")
    
    return df


def process_dataset(M, candidates, output_dir, description):
    """Process a dataset through the full pipeline."""
    print("\n" + "="*70)
    print(f"PROCESSING: {description}")
    print("="*70)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    by_size_dir = os.path.join(output_dir, 'by_size')
    os.makedirs(by_size_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    n_candidates = len(candidates)
    
    # Step 1: Calculate raw scores
    print("\n" + "="*70)
    print("STEP 1: Calculating raw scores")
    print("="*70)
    calculate_all_scores_for_dataset(M, candidates, output_dir)
    
    # Step 2: Calculate alpha approximations (by size)
    print("\n" + "="*70)
    print("STEP 2: Calculating alpha approximations (by size)")
    print("="*70)
    calculate_alpha_by_size(
        input_file='raw_scores.csv',
        output_file='alpha_scores_by_size.csv',
        max_file='max_scores_by_size.csv',
        output_dir=output_dir
    )
    
    # Step 3: Run voting methods (MES, AV, greedy-CC, greedy-PAV)
    print("\n" + "="*70)
    print("STEP 3: Running voting methods (MES, AV, greedy-CC, greedy-PAV)")
    print("="*70)
    run_mes_all_sizes(
        output_file='voting_results.csv',
        M=M,
        candidates=candidates,
        output_dir=output_dir
    )
    
    # Step 4: Create plots (by size)
    print("\n" + "="*70)
    print("STEP 4: Creating plots (by size)")
    print("="*70)
    plot_results_by_size(
        input_file='alpha_scores_by_size.csv',
        output_file='alpha_plots_by_size.png',
        mes_file='voting_results.csv',
        output_dir=output_dir,
        n_candidates=n_candidates
    )
    
    # Step 5: Create individual size plots
    print("\n" + "="*70)
    print("STEP 5: Creating individual size plots")
    print("="*70)
    plot_all_sizes(
        input_file='alpha_scores_by_size.csv',
        output_dir='by_size',
        mes_file='voting_results.csv',
        base_dir=output_dir
    )
    
    # Step 6: Create EJR-specific plots with voting methods
    print("\n" + "="*70)
    print("STEP 6: Creating EJR plots with voting methods")
    print("="*70)
    plot_ejr_results(
        M=M,
        candidates=candidates,
        output_dir=output_dir
    )
    
    print("\n" + "="*70)
    print(f"COMPLETED: {description}")
    print(f"Output saved to: {output_dir}")
    print("="*70)


def run_french_election():
    """Process the French Election dataset."""
    print("\n" + "="*70)
    print("FRENCH ELECTION DATASET PROCESSING")
    print("="*70)
    print(f"\n{FRENCH_ELECTION['description']}")
    
    start_time = time.time()
    
    # Load data
    print("\n" + "="*70)
    print("Loading French Election data...")
    print("="*70)
    M, candidates = load_and_combine_data()
    
    # Process through full pipeline
    process_dataset(
        M=M,
        candidates=candidates,
        output_dir='output/french_election',
        description=FRENCH_ELECTION['description']
    )
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("FRENCH ELECTION PROCESSING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("\nOutput directory: output/french_election/")
    print("View plots with: open output/french_election/alpha_plots_by_size.png")


def run_camp_songs(file_key=None):
    """Process Camp Songs dataset files."""
    print("\n" + "="*70)
    print("CAMP SONGS DATASET PROCESSING")
    print("="*70)
    print("\nThis script processes the Camp Songs dataset (00059) from PrefLib.")
    
    # Determine which files to process
    if file_key:
        if file_key not in CAMP_SONGS_FILES:
            print(f"Error: Unknown file key '{file_key}'")
            print(f"Available options: {list(CAMP_SONGS_FILES.keys())}")
            sys.exit(1)
        files_to_process = {file_key: CAMP_SONGS_FILES[file_key]}
    else:
        files_to_process = CAMP_SONGS_FILES
    
    print("Files to process:")
    for key, info in files_to_process.items():
        print(f"  - {key}: {info['description']}")
    
    start_time = time.time()
    
    for file_key, file_info in files_to_process.items():
        # Load data
        print("\n" + "="*70)
        print(f"Loading {file_key}...")
        print("="*70)
        M, candidates = load_preflib_file(file_info['url'])
        
        # Process through full pipeline
        process_dataset(
            M=M,
            candidates=candidates,
            output_dir=f'output/camp_songs/{file_key}',
            description=file_info['description']
        )
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("CAMP SONGS PROCESSING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("\nOutput directories:")
    for key in files_to_process:
        print(f"  - output/camp_songs/{key}/")
    print("\nView plots with:")
    for key in files_to_process:
        print(f"  open output/camp_songs/{key}/alpha_plots_by_size.png")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-winner bridging analysis for voting datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py french_election           # Run French election dataset
  python main.py camp_songs                # Run all Camp Songs files
  python main.py camp_songs --file file_02 # Run specific Camp Songs file
        """
    )
    
    parser.add_argument(
        'dataset',
        choices=['french_election', 'camp_songs'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--file',
        choices=['file_02', 'file_04'],
        help='Specific Camp Songs file to process (only for camp_songs dataset)'
    )
    
    args = parser.parse_args()
    
    if args.dataset == 'french_election':
        if args.file:
            print("Warning: --file option is ignored for french_election dataset")
        run_french_election()
    elif args.dataset == 'camp_songs':
        run_camp_songs(file_key=args.file)


if __name__ == "__main__":
    main()
