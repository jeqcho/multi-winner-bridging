"""
Estimate computation time for calculating all scores on all subsets.

Samples 50 random subsets and measures time for each scoring function,
then extrapolates to 2^12 = 4,096 subsets.
"""

import numpy as np
import time
import random
from itertools import combinations
from data_loader import load_and_combine_data
from scoring import av_score, cc_score, pairs_score, cons_score, ejr_satisfied, alpha_ejr


def time_single_subset(M, W, k):
    """Time all scoring functions on a single subset."""
    times = {}
    
    # Time AV
    start = time.time()
    av_score(M, W)
    times['AV'] = time.time() - start
    
    # Time CC
    start = time.time()
    cc_score(M, W)
    times['CC'] = time.time() - start
    
    # Time PAIRS
    start = time.time()
    pairs_score(M, W)
    times['PAIRS'] = time.time() - start
    
    # Time CONS
    start = time.time()
    cons_score(M, W)
    times['CONS'] = time.time() - start
    
    # Time EJR (boolean)
    start = time.time()
    ejr_satisfied(M, W, k)
    times['EJR'] = time.time() - start
    
    # Time alpha-EJR
    start = time.time()
    alpha_ejr(M, W, k)
    times['alpha_EJR'] = time.time() - start
    
    # Total time
    times['TOTAL'] = sum(times.values())
    
    return times


def estimate_time():
    """Estimate total computation time by sampling random subsets."""
    print("Loading data...")
    M, candidates = load_and_combine_data()
    n_candidates = len(candidates)
    
    print(f"\nEstimating computation time...")
    print(f"Dataset: {M.shape[0]} voters, {n_candidates} candidates")
    print(f"Total subsets to compute: 2^{n_candidates} = {2**n_candidates:,}")
    
    # Sample 50 random subsets of varying sizes
    n_samples = 50
    random.seed(42)
    
    print(f"\nTiming {n_samples} random subsets...")
    
    all_times = {
        'AV': [],
        'CC': [],
        'PAIRS': [],
        'CONS': [],
        'EJR': [],
        'alpha_EJR': [],
        'TOTAL': []
    }
    
    # Generate random subsets of different sizes
    for i in range(n_samples):
        # Random size from 0 to n_candidates
        k = random.randint(0, n_candidates)
        
        if k == 0:
            W = []
        else:
            # Random subset of size k
            W = sorted(random.sample(range(n_candidates), k))
        
        # Time this subset
        subset_times = time_single_subset(M, W, k)
        
        for key in all_times:
            all_times[key].append(subset_times[key])
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_samples} samples...")
    
    # Calculate averages
    print("\n" + "="*70)
    print("TIMING RESULTS")
    print("="*70)
    
    avg_times = {}
    for key in ['AV', 'CC', 'PAIRS', 'CONS', 'EJR', 'alpha_EJR', 'TOTAL']:
        avg = np.mean(all_times[key])
        std = np.std(all_times[key])
        avg_times[key] = avg
        print(f"{key:12s}: {avg*1000:8.3f} ms ± {std*1000:6.3f} ms per subset")
    
    # Extrapolate to full dataset
    total_subsets = 2 ** n_candidates
    estimated_total = avg_times['TOTAL'] * total_subsets
    
    print("\n" + "="*70)
    print("ESTIMATED TOTAL TIME FOR ALL SUBSETS")
    print("="*70)
    print(f"Total subsets: {total_subsets:,}")
    print(f"Average time per subset: {avg_times['TOTAL']*1000:.3f} ms")
    print(f"\nEstimated total time:")
    print(f"  {estimated_total:.2f} seconds")
    print(f"  {estimated_total/60:.2f} minutes")
    print(f"  {estimated_total/3600:.2f} hours")
    
    # Breakdown by function
    print("\nEstimated time breakdown:")
    for key in ['AV', 'CC', 'PAIRS', 'CONS', 'EJR', 'alpha_EJR']:
        func_total = avg_times[key] * total_subsets
        percentage = (avg_times[key] / avg_times['TOTAL']) * 100
        print(f"  {key:12s}: {func_total:8.2f} s ({func_total/60:6.2f} min) - {percentage:5.1f}%")
    
    print("\n" + "="*70)
    
    # Check if alpha_EJR is the bottleneck
    if avg_times['alpha_EJR'] > avg_times['TOTAL'] * 0.5:
        print("\n⚠️  WARNING: alpha_EJR is the bottleneck!")
        print("   Consider optimizing or using sampling.")
    
    return avg_times, estimated_total


if __name__ == "__main__":
    estimate_time()





