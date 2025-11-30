"""
Create EJR-specific plots for voting methods.

Creates 2 plots (1x2 grid):
- Plot 1: alpha_PAIRS (x-axis) vs alpha_EJR (y-axis)
- Plot 2: alpha_CONS (x-axis) vs alpha_EJR (y-axis)

Points are shown for MES, AV, CC, PAV, and max-score methods at each committee size.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import time

from scoring import av_score, cc_score, pairs_score, cons_score, ejr_satisfied, alpha_ejr
from mes import method_of_equal_shares
from voting_methods import approval_voting, chamberlin_courant_greedy, pav_greedy, select_max_committee


def plot_ejr_results(M, candidates, output_dir):
    """
    Create EJR plots showing voting methods' performance.
    
    Args:
        M: Boolean approval matrix (n_voters, n_candidates)
        candidates: List of candidate names
        output_dir: Directory to save outputs
    """
    start_time = time.time()
    
    print("="*70)
    print("CREATING EJR PLOTS")
    print("="*70)
    
    n_voters, n_candidates = M.shape
    print(f"Dataset: {n_voters} voters, {n_candidates} candidates")
    
    # Load max scores BY SIZE for alpha normalization (consistent with alpha_plots_by_size.png)
    max_by_size_file = os.path.join(output_dir, 'max_scores_by_size.csv')
    print(f"\nLoading max scores by size from {max_by_size_file}...")
    max_by_size = pd.read_csv(max_by_size_file)
    
    print("Max scores by size:")
    print(max_by_size.to_string(index=False))
    
    # Load raw scores for max-score methods
    raw_scores_file = os.path.join(output_dir, 'raw_scores.csv')
    print(f"\nLoading raw scores from {raw_scores_file}...")
    raw_scores = pd.read_csv(raw_scores_file)
    
    # Define greedy voting methods with their visual properties
    # Format: (function, marker, color, size)
    greedy_methods = {
        'MES': (method_of_equal_shares, '*', 'gold', 200),      # Star, gold
        'AV': (approval_voting, 's', 'red', 100),                # Square, red
        'CC': (chamberlin_courant_greedy, '^', 'blue', 100),     # Triangle up, blue
        'PAV': (pav_greedy, 'D', 'green', 100),                  # Diamond, green
    }
    
    # Define max-score methods with their visual properties
    # Format: (primary_score, secondary_score, marker, color, size)
    max_score_methods = {
        'PAIRS-AV': ('PAIRS', 'AV', 'o', 'purple', 100),        # Circle, purple
        'PAIRS-CC': ('PAIRS', 'CC', 'p', 'magenta', 100),       # Pentagon, magenta
        'CONS-AV': ('CONS', 'AV', 'h', 'orange', 100),          # Hexagon, orange
        'CONS-CC': ('CONS', 'CC', 'v', 'cyan', 100),            # Triangle down, cyan
    }
    
    # Compute scores and EJR for each method at each k
    results = []
    timing = {'scoring': 0, 'ejr': 0}
    
    print("\nComputing scores and EJR for each voting method...")
    for k in range(1, n_candidates + 1):
        # Get max scores for this committee size
        size_row = max_by_size[max_by_size['subset_size'] == k].iloc[0]
        max_pairs_k = size_row['max_PAIRS']
        max_cons_k = size_row['max_CONS']
        
        print(f"  k={k} (max_PAIRS={max_pairs_k}, max_CONS={max_cons_k}):", end=" ")
        
        # Run greedy methods
        for method_name, (method_func, marker, color, size) in greedy_methods.items():
            # Get committee
            committee = method_func(M, k)
            
            # Calculate scores
            t0 = time.time()
            pairs = pairs_score(M, committee)
            cons = cons_score(M, committee)
            timing['scoring'] += time.time() - t0
            
            # Calculate EJR
            t0 = time.time()
            alpha = alpha_ejr(M, committee, k)
            timing['ejr'] += time.time() - t0
            
            # Normalize BY SIZE (consistent with alpha_plots_by_size.png)
            alpha_pairs = pairs / max_pairs_k if max_pairs_k > 0 else 0
            alpha_cons = cons / max_cons_k if max_cons_k > 0 else 0
            
            results.append({
                'method': method_name,
                'k': k,
                'committee': committee,
                'alpha_PAIRS': alpha_pairs,
                'alpha_CONS': alpha_cons,
                'alpha_EJR': alpha,
                'marker': marker,
                'color': color,
                'size': size,
            })
            
            print(f"{method_name}(EJR={alpha:.2f})", end=" ")
        
        # Run max-score methods
        for method_name, (primary, secondary, marker, color, size) in max_score_methods.items():
            # Get committee from raw scores
            committee = select_max_committee(raw_scores, k, primary, secondary)
            
            # Calculate scores
            t0 = time.time()
            pairs = pairs_score(M, committee)
            cons = cons_score(M, committee)
            timing['scoring'] += time.time() - t0
            
            # Calculate EJR
            t0 = time.time()
            alpha = alpha_ejr(M, committee, k)
            timing['ejr'] += time.time() - t0
            
            # Normalize BY SIZE (consistent with alpha_plots_by_size.png)
            alpha_pairs = pairs / max_pairs_k if max_pairs_k > 0 else 0
            alpha_cons = cons / max_cons_k if max_cons_k > 0 else 0
            
            results.append({
                'method': method_name,
                'k': k,
                'committee': committee,
                'alpha_PAIRS': alpha_pairs,
                'alpha_CONS': alpha_cons,
                'alpha_EJR': alpha,
                'marker': marker,
                'color': color,
                'size': size,
            })
            
            print(f"{method_name}(EJR={alpha:.2f})", end=" ")
        print()
    
    # Print timing
    total_time = time.time() - start_time
    print(f"\nTiming: scoring={timing['scoring']:.2f}s, EJR={timing['ejr']:.2f}s")
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Voting Methods: Bridging vs EJR (Normalized by Size)', fontsize=16, fontweight='bold', y=1.02)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Plot 1: alpha_PAIRS vs alpha_EJR
    ax = axes[0]
    for method_name, group in df.groupby('method'):
        marker = group['marker'].iloc[0]
        color = group['color'].iloc[0]
        size = group['size'].iloc[0]
        
        ax.scatter(group['alpha_PAIRS'], group['alpha_EJR'],
                  marker=marker, c=color, s=size,
                  edgecolors='black', linewidths=0.5,
                  label=method_name, alpha=0.8)
        
        # Connect points with lines for each method
        sorted_group = group.sort_values('k')
        ax.plot(sorted_group['alpha_PAIRS'], sorted_group['alpha_EJR'],
               c=color, alpha=0.3, linewidth=1)
    
    ax.set_xlabel('alpha_PAIRS', fontsize=12)
    ax.set_ylabel('alpha_EJR', fontsize=12)
    ax.set_title('PAIRS vs EJR', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: alpha_CONS vs alpha_EJR
    ax = axes[1]
    for method_name, group in df.groupby('method'):
        marker = group['marker'].iloc[0]
        color = group['color'].iloc[0]
        size = group['size'].iloc[0]
        
        ax.scatter(group['alpha_CONS'], group['alpha_EJR'],
                  marker=marker, c=color, s=size,
                  edgecolors='black', linewidths=0.5,
                  label=method_name, alpha=0.8)
        
        # Connect points with lines for each method
        sorted_group = group.sort_values('k')
        ax.plot(sorted_group['alpha_CONS'], sorted_group['alpha_EJR'],
               c=color, alpha=0.3, linewidth=1)
    
    ax.set_xlabel('alpha_CONS', fontsize=12)
    ax.set_ylabel('alpha_EJR', fontsize=12)
    ax.set_title('CONS vs EJR', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'ejr_plots.png')
    print(f"\nSaving EJR plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully")
    
    # Create ZOOMED IN version (0.8-1.0 range with margins)
    fig_zoom, axes_zoom = plt.subplots(1, 2, figsize=(14, 6))
    fig_zoom.suptitle('Voting Methods: Bridging vs EJR (Zoomed 0.8-1.0)', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: alpha_PAIRS vs alpha_EJR (zoomed)
    ax = axes_zoom[0]
    for method_name, group in df.groupby('method'):
        marker = group['marker'].iloc[0]
        color = group['color'].iloc[0]
        size = group['size'].iloc[0]
        
        ax.scatter(group['alpha_PAIRS'], group['alpha_EJR'],
                  marker=marker, c=color, s=size,
                  edgecolors='black', linewidths=0.5,
                  label=method_name, alpha=0.8)
        
        sorted_group = group.sort_values('k')
        ax.plot(sorted_group['alpha_PAIRS'], sorted_group['alpha_EJR'],
               c=color, alpha=0.3, linewidth=1)
    
    ax.set_xlabel('alpha_PAIRS', fontsize=12)
    ax.set_ylabel('alpha_EJR', fontsize=12)
    ax.set_title('PAIRS vs EJR (Zoomed)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.78, 1.02)
    ax.set_ylim(0.78, 1.02)
    
    # Plot 2: alpha_CONS vs alpha_EJR (zoomed)
    ax = axes_zoom[1]
    for method_name, group in df.groupby('method'):
        marker = group['marker'].iloc[0]
        color = group['color'].iloc[0]
        size = group['size'].iloc[0]
        
        ax.scatter(group['alpha_CONS'], group['alpha_EJR'],
                  marker=marker, c=color, s=size,
                  edgecolors='black', linewidths=0.5,
                  label=method_name, alpha=0.8)
        
        sorted_group = group.sort_values('k')
        ax.plot(sorted_group['alpha_CONS'], sorted_group['alpha_EJR'],
               c=color, alpha=0.3, linewidth=1)
    
    ax.set_xlabel('alpha_CONS', fontsize=12)
    ax.set_ylabel('alpha_EJR', fontsize=12)
    ax.set_title('CONS vs EJR (Zoomed)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.78, 1.02)
    ax.set_ylim(0.78, 1.02)
    
    plt.tight_layout()
    
    # Save zoomed figure
    output_file_zoom = os.path.join(output_dir, 'ejr_plots_zoomed.png')
    print(f"Saving zoomed EJR plot to {output_file_zoom}...")
    plt.savefig(output_file_zoom, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zoomed plot saved successfully")
    
    # Also save the data
    data_file = os.path.join(output_dir, 'ejr_data.csv')
    df_save = df[['method', 'k', 'alpha_PAIRS', 'alpha_CONS', 'alpha_EJR']].copy()
    df_save['committee'] = df['committee'].apply(json.dumps)
    df_save.to_csv(data_file, index=False)
    print(f"EJR data saved to {data_file}")
    
    print(f"\nTotal time: {time.time() - start_time:.2f}s")
    print("="*70)
    
    return df


if __name__ == "__main__":
    # Test with French election data
    from data_loader import load_and_combine_data
    
    M, candidates = load_and_combine_data()
    plot_ejr_results(M, candidates, 'output/french_election')

