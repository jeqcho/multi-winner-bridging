"""
Post-processing script to calculate alpha-approximations.

Takes raw_scores.csv and produces alpha_scores.csv with normalized scores.
"""

import pandas as pd
import numpy as np


def calculate_alpha_approximations(input_file='raw_scores.csv', output_file='alpha_scores.csv'):
    """
    Calculate alpha-approximation for all scores.
    
    Args:
        input_file: Path to raw scores CSV
        output_file: Path to output alpha scores CSV
    """
    print("="*70)
    print("CALCULATING ALPHA-APPROXIMATIONS")
    print("="*70)
    
    # Load raw scores
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} subsets")
    
    # Find maximum scores
    print("\nFinding maximum scores...")
    max_av = df['AV'].max()
    max_cc = df['CC'].max()
    max_pairs = df['PAIRS'].max()
    max_cons = df['CONS'].max()
    
    print(f"  Max AV: {max_av}")
    print(f"  Max CC: {max_cc}")
    print(f"  Max PAIRS: {max_pairs}")
    print(f"  Max CONS: {max_cons}")
    
    # Calculate alpha-approximations
    print("\nCalculating alpha values...")
    
    # Handle division by zero for empty subsets
    df['alpha_AV'] = df['AV'] / max_av if max_av > 0 else 0
    df['alpha_CC'] = df['CC'] / max_cc if max_cc > 0 else 0
    df['alpha_PAIRS'] = df['PAIRS'] / max_pairs if max_pairs > 0 else 0
    df['alpha_CONS'] = df['CONS'] / max_cons if max_cons > 0 else 0
    
    # For EJR, alpha_EJR is the same as beta_EJR (already calculated)
    df['alpha_EJR'] = df['beta_EJR']
    
    # Save to CSV
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Display statistics
    print("\n" + "="*70)
    print("ALPHA-APPROXIMATION STATISTICS")
    print("="*70)
    
    print("\nAlpha AV:")
    print(f"  Range: [{df['alpha_AV'].min():.4f}, {df['alpha_AV'].max():.4f}]")
    print(f"  Mean: {df['alpha_AV'].mean():.4f}")
    print(f"  Median: {df['alpha_AV'].median():.4f}")
    
    print("\nAlpha CC:")
    print(f"  Range: [{df['alpha_CC'].min():.4f}, {df['alpha_CC'].max():.4f}]")
    print(f"  Mean: {df['alpha_CC'].mean():.4f}")
    print(f"  Median: {df['alpha_CC'].median():.4f}")
    
    print("\nAlpha PAIRS:")
    print(f"  Range: [{df['alpha_PAIRS'].min():.4f}, {df['alpha_PAIRS'].max():.4f}]")
    print(f"  Mean: {df['alpha_PAIRS'].mean():.4f}")
    print(f"  Median: {df['alpha_PAIRS'].median():.4f}")
    
    print("\nAlpha CONS:")
    print(f"  Range: [{df['alpha_CONS'].min():.4f}, {df['alpha_CONS'].max():.4f}]")
    print(f"  Mean: {df['alpha_CONS'].mean():.4f}")
    print(f"  Median: {df['alpha_CONS'].median():.4f}")
    
    print("\nAlpha EJR (= Beta EJR):")
    print(f"  Range: [{df['alpha_EJR'].min():.4f}, {df['alpha_EJR'].max():.4f}]")
    print(f"  Mean: {df['alpha_EJR'].mean():.4f}")
    print(f"  Median: {df['alpha_EJR'].median():.4f}")
    print(f"  Full EJR (alpha=1.0): {(df['alpha_EJR'] >= 0.999).sum()} subsets ({(df['alpha_EJR'] >= 0.999).sum() / len(df) * 100:.1f}%)")
    
    # Find best subsets
    print("\n" + "="*70)
    print("BEST SUBSETS")
    print("="*70)
    
    # Best by each metric
    best_av_idx = df['AV'].idxmax()
    best_cc_idx = df['CC'].idxmax()
    best_pairs_idx = df['PAIRS'].idxmax()
    best_cons_idx = df['CONS'].idxmax()
    best_ejr_idx = df.loc[df['alpha_EJR'] >= 0.999, 'AV'].idxmax() if (df['alpha_EJR'] >= 0.999).any() else None
    
    print(f"\nBest AV (score={df.loc[best_av_idx, 'AV']}):")
    print(f"  Committee: {df.loc[best_av_idx, 'subset_indices']}")
    
    print(f"\nBest CC (score={df.loc[best_cc_idx, 'CC']}):")
    print(f"  Committee: {df.loc[best_cc_idx, 'subset_indices']}")
    
    print(f"\nBest PAIRS (score={df.loc[best_pairs_idx, 'PAIRS']}):")
    print(f"  Committee: {df.loc[best_pairs_idx, 'subset_indices']}")
    
    print(f"\nBest CONS (score={df.loc[best_cons_idx, 'CONS']}):")
    print(f"  Committee: {df.loc[best_cons_idx, 'subset_indices']}")
    
    if best_ejr_idx is not None:
        print(f"\nBest AV among EJR-satisfying subsets (score={df.loc[best_ejr_idx, 'AV']}):")
        print(f"  Committee: {df.loc[best_ejr_idx, 'subset_indices']}")
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Output file: {output_file}")
    print("\nNext step: Run src/plot_results.py to create visualizations")
    print("="*70)
    
    return df


if __name__ == "__main__":
    calculate_alpha_approximations()





