"""
Post-processing script to calculate alpha-approximations relative to each subset size.

Takes raw_scores.csv and produces alpha_scores_by_size.csv with normalized scores
where each score is normalized by the maximum for that specific subset size.

Also produces max_scores_by_size.csv showing the maximum scores for each k.
"""

import pandas as pd
import numpy as np


def calculate_alpha_by_size(input_file='output/french_election/raw_scores.csv', 
                            output_file='output/french_election/alpha_scores_by_size.csv',
                            max_file='output/french_election/max_scores_by_size.csv',
                            output_dir=None):
    """
    Calculate alpha-approximation for all scores relative to max within each subset size.
    
    Args:
        input_file: Path to raw scores CSV (or filename if output_dir provided)
        output_file: Path to output alpha scores CSV (or filename if output_dir provided)
        max_file: Path to output max scores by size CSV (or filename if output_dir provided)
        output_dir: Optional directory prefix for input/output files
    """
    import os
    if output_dir:
        input_file = os.path.join(output_dir, input_file)
        output_file = os.path.join(output_dir, output_file)
        max_file = os.path.join(output_dir, max_file)
    print("="*70)
    print("CALCULATING ALPHA-APPROXIMATIONS BY SUBSET SIZE")
    print("="*70)
    
    # Load raw scores
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} subsets")
    
    # Calculate max scores for each subset size
    print("\nCalculating max scores for each subset size...")
    max_by_size = df.groupby('subset_size').agg({
        'AV': 'max',
        'CC': 'max',
        'PAIRS': 'max',
        'CONS': 'max'
    }).reset_index()
    
    max_by_size.columns = ['subset_size', 'max_AV', 'max_CC', 'max_PAIRS', 'max_CONS']
    
    # Display max scores by size
    print("\nMax scores by subset size:")
    print(max_by_size.to_string(index=False))
    
    # Save max scores
    print(f"\nSaving max scores to {max_file}...")
    max_by_size.to_csv(max_file, index=False)
    
    # Merge max scores back to original dataframe
    df = df.merge(max_by_size, on='subset_size', how='left')
    
    # Calculate alpha-approximations relative to size-specific max
    print("\nCalculating alpha values relative to subset size...")
    
    # Handle division by zero
    df['alpha_AV'] = df.apply(
        lambda row: row['AV'] / row['max_AV'] if row['max_AV'] > 0 else 0, axis=1
    )
    df['alpha_CC'] = df.apply(
        lambda row: row['CC'] / row['max_CC'] if row['max_CC'] > 0 else 0, axis=1
    )
    df['alpha_PAIRS'] = df.apply(
        lambda row: row['PAIRS'] / row['max_PAIRS'] if row['max_PAIRS'] > 0 else 0, axis=1
    )
    df['alpha_CONS'] = df.apply(
        lambda row: row['CONS'] / row['max_CONS'] if row['max_CONS'] > 0 else 0, axis=1
    )
    
    # For EJR, alpha_EJR is the same as beta_EJR (already calculated)
    df['alpha_EJR'] = df['beta_EJR']
    
    # Drop the max_ columns before saving (keep output clean)
    output_df = df.drop(columns=['max_AV', 'max_CC', 'max_PAIRS', 'max_CONS'])
    
    # Save to CSV
    print(f"\nSaving to {output_file}...")
    output_df.to_csv(output_file, index=False)
    
    # Display statistics
    print("\n" + "="*70)
    print("ALPHA-APPROXIMATION STATISTICS (BY SIZE)")
    print("="*70)
    
    print("\nAlpha AV:")
    print(f"  Range: [{output_df['alpha_AV'].min():.4f}, {output_df['alpha_AV'].max():.4f}]")
    print(f"  Mean: {output_df['alpha_AV'].mean():.4f}")
    print(f"  Median: {output_df['alpha_AV'].median():.4f}")
    
    print("\nAlpha CC:")
    print(f"  Range: [{output_df['alpha_CC'].min():.4f}, {output_df['alpha_CC'].max():.4f}]")
    print(f"  Mean: {output_df['alpha_CC'].mean():.4f}")
    print(f"  Median: {output_df['alpha_CC'].median():.4f}")
    
    print("\nAlpha PAIRS:")
    print(f"  Range: [{output_df['alpha_PAIRS'].min():.4f}, {output_df['alpha_PAIRS'].max():.4f}]")
    print(f"  Mean: {output_df['alpha_PAIRS'].mean():.4f}")
    print(f"  Median: {output_df['alpha_PAIRS'].median():.4f}")
    
    print("\nAlpha CONS:")
    print(f"  Range: [{output_df['alpha_CONS'].min():.4f}, {output_df['alpha_CONS'].max():.4f}]")
    print(f"  Mean: {output_df['alpha_CONS'].mean():.4f}")
    print(f"  Median: {output_df['alpha_CONS'].median():.4f}")
    
    print("\nAlpha EJR (= Beta EJR):")
    print(f"  Range: [{output_df['alpha_EJR'].min():.4f}, {output_df['alpha_EJR'].max():.4f}]")
    print(f"  Mean: {output_df['alpha_EJR'].mean():.4f}")
    print(f"  Median: {output_df['alpha_EJR'].median():.4f}")
    
    # Count how many subsets achieve max for each metric at each size
    print("\n" + "="*70)
    print("OPTIMAL SUBSETS BY SIZE")
    print("="*70)
    
    for k in sorted(output_df['subset_size'].unique()):
        size_df = output_df[output_df['subset_size'] == k]
        n_optimal_av = (size_df['alpha_AV'] >= 0.999).sum()
        n_optimal_cc = (size_df['alpha_CC'] >= 0.999).sum()
        n_optimal_pairs = (size_df['alpha_PAIRS'] >= 0.999).sum()
        n_optimal_cons = (size_df['alpha_CONS'] >= 0.999).sum()
        total = len(size_df)
        
        print(f"\nSize k={k:2d} ({total:4d} subsets):")
        print(f"  Optimal AV:    {n_optimal_av:4d} ({n_optimal_av/total*100:5.1f}%)")
        print(f"  Optimal CC:    {n_optimal_cc:4d} ({n_optimal_cc/total*100:5.1f}%)")
        print(f"  Optimal PAIRS: {n_optimal_pairs:4d} ({n_optimal_pairs/total*100:5.1f}%)")
        print(f"  Optimal CONS:  {n_optimal_cons:4d} ({n_optimal_cons/total*100:5.1f}%)")
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Output file: {output_file}")
    print(f"Max scores file: {max_file}")
    print("\nNext step: Run src/plot_results_by_size.py to create visualizations")
    print("="*70)
    
    return output_df, max_by_size


if __name__ == "__main__":
    calculate_alpha_by_size()





