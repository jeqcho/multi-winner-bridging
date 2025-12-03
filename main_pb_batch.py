"""
Batch runner script for multi-winner bridging analysis on pabulib PB data.

Streamlined version of main_pb.py that:
- Outputs only raw_scores.csv and voting_results.csv (no plots)
- Includes 6 greedy voting methods: MES, greedy-AV, greedy-AV/cost, greedy-AV/cost^2, greedy-CC, greedy-PAV
- Adds boolean EJR satisfaction check

Usage:
    python main_pb_batch.py path/to/file.pb           # Run single file
    python main_pb_batch.py path/to/directory/        # Run all .pb files in directory
"""

import numpy as np
import pandas as pd
from itertools import combinations
import json
import time
import os
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Limit numpy thread usage to avoid oversubscription with multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pb_data_loader import load_pb_file, get_pb_files, enumerate_valid_committees, count_valid_committees
from scoring import av_score, cc_score, pairs_score, cons_score, ejr_satisfied
from mes import method_of_equal_shares_budget
from voting_methods import (
    approval_voting_budget,
    approval_voting_cost_ratio_budget,
    approval_voting_cost_squared_ratio_budget,
    chamberlin_courant_greedy_budget, 
    pav_greedy_budget,
)


def _score_committee(args):
    """
    Worker function for parallel scoring of a single committee.
    
    Must be at module level to be picklable for multiprocessing.
    """
    M, project_costs, W = args
    W_list = list(W)
    return {
        'subset_size': len(W_list),
        'total_cost': sum(project_costs[j] for j in W_list),
        'subset_indices': json.dumps(W_list),
        'AV': av_score(M, W_list),
        'CC': cc_score(M, W_list),
        'PAIRS': pairs_score(M, W_list),
        'CONS': cons_score(M, W_list),
    }


def calculate_all_scores_for_pb(M, project_costs, budget, output_dir):
    """Calculate scores for all valid (budget-feasible) committees."""
    n_voters, n_projects = M.shape
    
    print(f"\nDataset: {n_voters} voters, {n_projects} projects")
    print(f"Budget: {budget:,}")
    
    # Count valid committees first
    print("\nCounting valid committees...")
    num_valid = count_valid_committees(project_costs, budget)
    print(f"Valid committees: {num_valid:,}")
    print(f"Total possible (2^n): {2**n_projects:,}")
    print(f"Reduction: {2**n_projects / num_valid:.1f}x")
    
    # Enumerate valid committees
    print("\nEnumerating valid committees...")
    valid_committees = enumerate_valid_committees(project_costs, budget, show_progress=True)
    print(f"Enumerated {len(valid_committees):,} committees")
    
    # Start timing
    start_time = time.time()
    
    # Parallel scoring using all available cores
    n_workers = min(32, multiprocessing.cpu_count())
    print(f"\nProcessing {len(valid_committees):,} committees using {n_workers} workers...")
    
    # Prepare args for each committee
    args_list = [(M, project_costs, W) for W in valid_committees]
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_score_committee, args): i 
                   for i, args in enumerate(args_list)}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), 
                          total=len(futures),
                          desc="Scoring committees",
                          unit="committee"):
            results.append(future.result())
    
    # Convert to DataFrame
    print("\nConverting to DataFrame...")
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
    print(f"Total committees processed: {len(results):,}")
    print(f"Average time per committee: {total_elapsed/len(results)*1000:.2f} ms")
    print(f"Effective throughput: {len(results)/total_elapsed:.1f} committees/sec")
    print(f"Workers used: {n_workers}")
    
    return df


def run_voting_methods_pb(M, project_costs, budget, output_dir):
    """
    Run 6 greedy voting methods for PB data with EJR check.
    
    Methods: MES, greedy-AV, greedy-AV/cost, greedy-AV/cost^2, greedy-CC, greedy-PAV
    """
    print("="*70)
    print("RUNNING VOTING METHODS")
    print("="*70)
    
    start_time = time.time()
    
    # Load raw scores to get max values for alpha calculations
    raw_file = os.path.join(output_dir, 'raw_scores.csv')
    raw_df = pd.read_csv(raw_file)
    
    max_av = raw_df['AV'].max()
    max_cc = raw_df['CC'].max()
    max_pairs = raw_df['PAIRS'].max()
    max_cons = raw_df['CONS'].max()
    
    n_voters, n_projects = M.shape
    print(f"Dataset: {n_voters} voters, {n_projects} projects")
    print(f"Budget: {budget:,}")
    print(f"Max scores: AV={max_av}, CC={max_cc}, PAIRS={max_pairs}, CONS={max_cons}")
    
    # Define 6 greedy voting methods (budget-aware)
    greedy_methods = {
        'MES': method_of_equal_shares_budget,
        'greedy-AV': approval_voting_budget,
        'greedy-AV/cost': approval_voting_cost_ratio_budget,
        'greedy-AV/cost^2': approval_voting_cost_squared_ratio_budget,
        'greedy-CC': chamberlin_courant_greedy_budget,
        'greedy-PAV': pav_greedy_budget,
    }
    
    results = []
    
    print("\nRunning greedy methods...")
    for method_name, method_func in tqdm(greedy_methods.items(), desc="Voting methods", unit="method"):
        t0 = time.time()
        committee = method_func(M, project_costs, budget)
        elapsed = time.time() - t0
        
        # Calculate scores
        av = av_score(M, committee)
        cc = cc_score(M, committee)
        pairs = pairs_score(M, committee)
        cons = cons_score(M, committee)
        total_cost = sum(project_costs[j] for j in committee)
        k = len(committee)
        
        # Calculate alpha approximations
        alpha_av = av / max_av if max_av > 0 else 0
        alpha_cc = cc / max_cc if max_cc > 0 else 0
        alpha_pairs = pairs / max_pairs if max_pairs > 0 else 0
        alpha_cons = cons / max_cons if max_cons > 0 else 0
        
        # Calculate EJR satisfaction
        ejr = ejr_satisfied(M, committee, k) if k > 0 else True
        
        tqdm.write(f"  {method_name}: {k} projects, cost={total_cost:,}, EJR={ejr}")
        tqdm.write(f"    AV={av}, CC={cc}, PAIRS={pairs}, CONS={cons}")
        
        # Convert to Python int for JSON serialization
        committee_list = [int(x) for x in sorted(committee)]
        
        results.append({
            'method': method_name,
            'subset_size': k,
            'total_cost': total_cost,
            'subset_indices': json.dumps(committee_list),
            'AV': av,
            'CC': cc,
            'PAIRS': pairs,
            'CONS': cons,
            'alpha_AV': alpha_av,
            'alpha_CC': alpha_cc,
            'alpha_PAIRS': alpha_pairs,
            'alpha_CONS': alpha_cons,
            'EJR': ejr,
        })
    
    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, 'voting_results.csv')
    df.to_csv(output_file, index=False)
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal time: {total_elapsed:.2f}s")
    print(f"Saved to {output_file}")
    
    return df


def process_pb_file(filepath, output_base_dir='output/pb'):
    """
    Process a single PB file through the streamlined pipeline.
    
    Outputs only raw_scores.csv and voting_results.csv.
    """
    print("\n" + "="*70)
    print(f"PROCESSING: {filepath}")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    M, project_ids, project_costs, budget = load_pb_file(filepath)
    
    # Create output directory
    filename = os.path.basename(filepath)
    dataset_name = os.path.splitext(filename)[0]
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Step 1: Calculate raw scores for all valid committees
    print("\n" + "="*70)
    print("STEP 1: Calculating raw scores for valid committees")
    print("="*70)
    calculate_all_scores_for_pb(M, project_costs, budget, output_dir)
    
    # Step 2: Run voting methods with EJR
    print("\n" + "="*70)
    print("STEP 2: Running voting methods with EJR check")
    print("="*70)
    run_voting_methods_pb(M, project_costs, budget, output_dir)
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"COMPLETED: {filepath}")
    print(f"Output saved to: {output_dir}")
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Batch multi-winner bridging analysis for pabulib PB datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_pb_batch.py data/poland_warszawa_2017_marymont-potok-zoliborz-dziennikarski.pb
  python main_pb_batch.py data/
        """
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        help='Path to .pb file or directory containing .pb files'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output/pb',
        help='Base output directory (default: output/pb)'
    )
    
    args = parser.parse_args()
    
    if args.path:
        path = args.path
        if os.path.isfile(path):
            # Single file
            process_pb_file(path, args.output_dir)
        elif os.path.isdir(path):
            # Directory - process all .pb files
            pb_files = get_pb_files(path)
            if not pb_files:
                print(f"No .pb files found in {path}")
                sys.exit(1)
            
            print(f"Found {len(pb_files)} .pb files to process")
            
            # Track overall progress
            start_time = time.time()
            success_count = 0
            error_count = 0
            errors = []
            
            for i, filepath in enumerate(pb_files):
                print(f"\n[{i+1}/{len(pb_files)}] Processing {os.path.basename(filepath)}...")
                try:
                    process_pb_file(filepath, args.output_dir)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    errors.append((filepath, str(e)))
                    print(f"ERROR processing {filepath}: {e}")
            
            total_elapsed = time.time() - start_time
            
            print("\n" + "="*70)
            print("BATCH PROCESSING COMPLETE!")
            print("="*70)
            print(f"Total files: {len(pb_files)}")
            print(f"Successful: {success_count}")
            print(f"Errors: {error_count}")
            print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
            
            if errors:
                print("\nFailed files:")
                for fp, err in errors:
                    print(f"  {os.path.basename(fp)}: {err}")
        else:
            print(f"Error: {path} is not a valid file or directory")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


