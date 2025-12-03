"""
Unified runner script for multi-winner bridging analysis on pabulib PB data.

Supports participatory budgeting datasets from pabulib (.pb files).
Unlike preflib data, PB data has:
- Budget constraint instead of committee size k
- Projects with costs (not unit cost)
- Valid committees are those fitting within budget

Usage:
    python main_pb.py path/to/file.pb           # Run single file
    python main_pb.py path/to/directory/        # Run all .pb files in directory
    python main_pb.py --test                    # Test on sample file
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
from scoring import av_score, cc_score, pairs_score, cons_score
from mes import method_of_equal_shares_budget
from voting_methods import (
    approval_voting_budget,
    approval_voting_cost_ratio_budget,
    approval_voting_cost_squared_ratio_budget,
    chamberlin_courant_greedy_budget, 
    pav_greedy_budget,
    select_max_committee_budget
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
    
    # Estimate time
    # PAIRS is O(n_voters^2) per committee, which dominates
    estimated_ops = num_valid * (n_voters ** 2)
    ops_per_second = 1e9  # rough estimate
    estimated_seconds = estimated_ops / ops_per_second
    print(f"\n⚠️  ESTIMATED TIME: {estimated_seconds:.1f}s ({estimated_seconds/60:.1f} min) for PAIRS scoring alone")
    print(f"    (Based on {num_valid:,} committees × {n_voters:,}² voters)")
    
    if estimated_seconds > 300:  # 5 minutes
        print(f"\n⚠️  WARNING: This will take a LONG time ({estimated_seconds/3600:.1f} hours estimated)")
        print("    Consider using a smaller dataset or implementing greedy-only mode")
    
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


def calculate_alpha_scores_pb(output_dir):
    """
    Calculate alpha-approximations for PB data.
    
    For PB, we normalize by the maximum score across ALL valid committees
    (not by committee size, since we're not iterating over sizes).
    """
    input_file = os.path.join(output_dir, 'raw_scores.csv')
    output_file = os.path.join(output_dir, 'alpha_scores.csv')
    max_file = os.path.join(output_dir, 'max_scores.csv')
    
    print("="*70)
    print("CALCULATING ALPHA-APPROXIMATIONS")
    print("="*70)
    
    # Load raw scores
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} committees")
    
    # Calculate max scores across ALL valid committees
    max_av = df['AV'].max()
    max_cc = df['CC'].max()
    max_pairs = df['PAIRS'].max()
    max_cons = df['CONS'].max()
    
    print(f"\nMax scores:")
    print(f"  AV: {max_av}")
    print(f"  CC: {max_cc}")
    print(f"  PAIRS: {max_pairs}")
    print(f"  CONS: {max_cons}")
    
    # Save max scores
    max_df = pd.DataFrame([{
        'max_AV': max_av,
        'max_CC': max_cc,
        'max_PAIRS': max_pairs,
        'max_CONS': max_cons
    }])
    max_df.to_csv(max_file, index=False)
    print(f"Saved max scores to {max_file}")
    
    # Calculate alpha-approximations
    df['alpha_AV'] = df['AV'] / max_av if max_av > 0 else 0
    df['alpha_CC'] = df['CC'] / max_cc if max_cc > 0 else 0
    df['alpha_PAIRS'] = df['PAIRS'] / max_pairs if max_pairs > 0 else 0
    df['alpha_CONS'] = df['CONS'] / max_cons if max_cons > 0 else 0
    
    # Save
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    
    return df, max_df


def run_voting_methods_pb(M, project_costs, budget, output_dir):
    """
    Run all 10 voting methods for PB data.
    
    Greedy methods (6): MES, AV, greedy-AV/cost, greedy-AV/cost^2, greedy-CC, greedy-PAV (budget-aware)
    Max-score methods (4): PAIRS-AV, PAIRS-CC, CONS-AV, CONS-CC (from exhaustive)
    """
    print("="*70)
    print("RUNNING VOTING METHODS")
    print("="*70)
    
    start_time = time.time()
    
    # Load max scores and raw scores
    max_file = os.path.join(output_dir, 'max_scores.csv')
    raw_file = os.path.join(output_dir, 'raw_scores.csv')
    
    max_df = pd.read_csv(max_file)
    raw_df = pd.read_csv(raw_file)
    
    max_av = max_df['max_AV'].iloc[0]
    max_cc = max_df['max_CC'].iloc[0]
    max_pairs = max_df['max_PAIRS'].iloc[0]
    max_cons = max_df['max_CONS'].iloc[0]
    
    n_voters, n_projects = M.shape
    print(f"Dataset: {n_voters} voters, {n_projects} projects")
    print(f"Budget: {budget:,}")
    
    # Define greedy voting methods (budget-aware)
    greedy_methods = {
        'MES': method_of_equal_shares_budget,
        'AV': approval_voting_budget,
        'greedy-AV/cost': approval_voting_cost_ratio_budget,
        'greedy-AV/cost^2': approval_voting_cost_squared_ratio_budget,
        'greedy-CC': chamberlin_courant_greedy_budget,
        'greedy-PAV': pav_greedy_budget,
    }
    
    # Define max-score methods
    max_score_methods = {
        'PAIRS-AV': ('PAIRS', 'AV'),
        'PAIRS-CC': ('PAIRS', 'CC'),
        'CONS-AV': ('CONS', 'AV'),
        'CONS-CC': ('CONS', 'CC'),
    }
    
    results = []
    timing = {name: 0 for name in list(greedy_methods.keys()) + list(max_score_methods.keys())}
    
    print("\nRunning greedy methods...")
    for method_name, method_func in tqdm(greedy_methods.items(), desc="Greedy methods", unit="method"):
        t0 = time.time()
        committee = method_func(M, project_costs, budget)
        timing[method_name] = time.time() - t0
        
        # Calculate scores
        av = av_score(M, committee)
        cc = cc_score(M, committee)
        pairs = pairs_score(M, committee)
        cons = cons_score(M, committee)
        total_cost = sum(project_costs[j] for j in committee)
        
        # Calculate alpha approximations
        alpha_av = av / max_av if max_av > 0 else 0
        alpha_cc = cc / max_cc if max_cc > 0 else 0
        alpha_pairs = pairs / max_pairs if max_pairs > 0 else 0
        alpha_cons = cons / max_cons if max_cons > 0 else 0
        
        tqdm.write(f"  {method_name}: {len(committee)} projects, cost={total_cost:,}")
        tqdm.write(f"    AV={av}, CC={cc}, PAIRS={pairs}, CONS={cons}")
        
        # Convert to Python int for JSON serialization
        committee_list = [int(x) for x in sorted(committee)]
        
        results.append({
            'method': method_name,
            'subset_size': len(committee),
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
        })
    
    print("\nRunning max-score methods...")
    for method_name, (primary, secondary) in tqdm(max_score_methods.items(), desc="Max-score methods", unit="method"):
        t0 = time.time()
        committee = select_max_committee_budget(raw_df, primary, secondary)
        timing[method_name] = time.time() - t0
        
        # Calculate scores
        av = av_score(M, committee)
        cc = cc_score(M, committee)
        pairs = pairs_score(M, committee)
        cons = cons_score(M, committee)
        total_cost = sum(project_costs[j] for j in committee)
        
        # Calculate alpha approximations
        alpha_av = av / max_av if max_av > 0 else 0
        alpha_cc = cc / max_cc if max_cc > 0 else 0
        alpha_pairs = pairs / max_pairs if max_pairs > 0 else 0
        alpha_cons = cons / max_cons if max_cons > 0 else 0
        
        tqdm.write(f"  {method_name}: {len(committee)} projects, cost={total_cost:,}")
        tqdm.write(f"    AV={av}, CC={cc}, PAIRS={pairs}, CONS={cons}")
        
        # Convert to Python int for JSON serialization
        committee_list = [int(x) for x in sorted(committee)]
        
        results.append({
            'method': method_name,
            'subset_size': len(committee),
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
        })
    
    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, 'voting_results.csv')
    df.to_csv(output_file, index=False)
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal time: {total_elapsed:.2f}s")
    print(f"Saved to {output_file}")
    
    return df


def create_pb_plots(M, project_costs, budget, output_dir):
    """
    Create plots for PB data.
    
    Since we don't iterate over committee sizes, we create simpler plots
    showing the alpha-approximations and voting method results.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scoring import alpha_ejr
    
    print("="*70)
    print("CREATING PLOTS")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    alpha_file = os.path.join(output_dir, 'alpha_scores.csv')
    voting_file = os.path.join(output_dir, 'voting_results.csv')
    
    df = pd.read_csv(alpha_file)
    voting_df = pd.read_csv(voting_file)
    
    n_voters, n_projects = M.shape
    
    # Method visual properties
    method_props = {
        'MES': {'marker': '*', 'color': 'gold', 'size': 200},
        'AV': {'marker': 's', 'color': 'red', 'size': 100},
        'greedy-AV/cost': {'marker': 'X', 'color': 'crimson', 'size': 100},
        'greedy-AV/cost^2': {'marker': '+', 'color': 'darkred', 'size': 100},
        'greedy-CC': {'marker': '^', 'color': 'blue', 'size': 100},
        'greedy-PAV': {'marker': 'D', 'color': 'green', 'size': 100},
        'PAIRS-AV': {'marker': 'o', 'color': 'purple', 'size': 100},
        'PAIRS-CC': {'marker': 'p', 'color': 'magenta', 'size': 100},
        'CONS-AV': {'marker': 'h', 'color': 'orange', 'size': 100},
        'CONS-CC': {'marker': 'v', 'color': 'cyan', 'size': 100},
    }
    
    # Create 2x2 alpha plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Alpha-Approximation Analysis (PB): Multi-Winner Bridging', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color by total_cost
    norm = plt.Normalize(vmin=df['total_cost'].min(), vmax=df['total_cost'].max())
    cmap = cm.viridis
    
    alpha_transparency = 0.35
    marker_size = 15
    
    def add_voting_methods(ax, x_col, y_col):
        for method_name, props in method_props.items():
            method_data = voting_df[voting_df['method'] == method_name]
            if len(method_data) > 0:
                ax.scatter(method_data[x_col], method_data[y_col],
                          marker=props['marker'], c=props['color'], s=props['size'],
                          edgecolors='black', linewidths=0.5,
                          label=method_name, zorder=10, alpha=0.8)
    
    # Group by alpha_PAIRS
    pairs_grouped = df.groupby('alpha_PAIRS').agg({
        'alpha_AV': 'mean',
        'alpha_CC': 'mean',
        'total_cost': 'mean'
    }).reset_index()
    
    # Plot 1: alpha_PAIRS vs beta_AV
    ax = axes[0, 0]
    scatter = ax.scatter(pairs_grouped['alpha_PAIRS'], pairs_grouped['alpha_AV'],
                        c=pairs_grouped['total_cost'], cmap=cmap, norm=norm,
                        alpha=alpha_transparency, s=marker_size)
    x_ref = np.linspace(0, 1, 100)
    ax.plot(x_ref, 1 - x_ref, 'k-', linewidth=2, alpha=0.7, label='a + b = 1')
    add_voting_methods(ax, 'alpha_PAIRS', 'alpha_AV')
    ax.set_xlabel('alpha_PAIRS', fontsize=11)
    ax.set_ylabel('beta_AV (avg)', fontsize=11)
    ax.set_title('PAIRS vs AV', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: alpha_PAIRS vs beta_CC
    ax = axes[0, 1]
    ax.scatter(pairs_grouped['alpha_PAIRS'], pairs_grouped['alpha_CC'],
              c=pairs_grouped['total_cost'], cmap=cmap, norm=norm,
              alpha=alpha_transparency, s=marker_size)
    ax.plot(x_ref, 1 - x_ref, 'k-', linewidth=2, alpha=0.7, label='a + b = 1')
    add_voting_methods(ax, 'alpha_PAIRS', 'alpha_CC')
    ax.set_xlabel('alpha_PAIRS', fontsize=11)
    ax.set_ylabel('beta_CC (avg)', fontsize=11)
    ax.set_title('PAIRS vs CC', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Group by alpha_CONS
    cons_grouped = df.groupby('alpha_CONS').agg({
        'alpha_AV': 'mean',
        'alpha_CC': 'mean',
        'total_cost': 'mean'
    }).reset_index()
    
    # Plot 3: alpha_CONS vs beta_AV
    ax = axes[1, 0]
    ax.scatter(cons_grouped['alpha_CONS'], cons_grouped['alpha_AV'],
              c=cons_grouped['total_cost'], cmap=cmap, norm=norm,
              alpha=alpha_transparency, s=marker_size)
    ax.plot(x_ref, 1 - x_ref**2, 'k-', linewidth=2, alpha=0.7, label='a² + b = 1')
    add_voting_methods(ax, 'alpha_CONS', 'alpha_AV')
    ax.set_xlabel('alpha_CONS', fontsize=11)
    ax.set_ylabel('beta_AV (avg)', fontsize=11)
    ax.set_title('CONS vs AV', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: alpha_CONS vs beta_CC
    ax = axes[1, 1]
    ax.scatter(cons_grouped['alpha_CONS'], cons_grouped['alpha_CC'],
              c=cons_grouped['total_cost'], cmap=cmap, norm=norm,
              alpha=alpha_transparency, s=marker_size)
    ax.plot(x_ref, 1 - x_ref**2, 'k-', linewidth=2, alpha=0.7, label='a² + b = 1')
    add_voting_methods(ax, 'alpha_CONS', 'alpha_CC')
    ax.set_xlabel('alpha_CONS', fontsize=11)
    ax.set_ylabel('beta_CC (avg)', fontsize=11)
    ax.set_title('CONS vs CC', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Total Cost', fontsize=12, rotation=270, labelpad=20)
    
    # Save
    output_file = os.path.join(output_dir, 'alpha_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved alpha plots to {output_file}")
    
    # Create EJR plots
    print("\nCalculating EJR for voting methods...")
    ejr_results = []
    
    for _, row in tqdm(voting_df.iterrows(), desc="Calculating EJR", total=len(voting_df), unit="method"):
        method_name = row['method']
        committee = json.loads(row['subset_indices'])
        k = len(committee)
        
        # Calculate alpha-EJR (this can be slow for large k)
        if k > 0:
            alpha = alpha_ejr(M, committee, k)
        else:
            alpha = 0.0
        
        ejr_results.append({
            'method': method_name,
            'committee': committee,
            'k': k,
            'alpha_PAIRS': row['alpha_PAIRS'],
            'alpha_CONS': row['alpha_CONS'],
            'alpha_EJR': alpha,
        })
        tqdm.write(f"  {method_name}: alpha_EJR = {alpha:.3f}")
    
    ejr_df = pd.DataFrame(ejr_results)
    
    # Save EJR data
    ejr_file = os.path.join(output_dir, 'ejr_data.csv')
    ejr_df.to_csv(ejr_file, index=False)
    print(f"Saved EJR data to {ejr_file}")
    
    # Create EJR plots (1x2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Voting Methods: Bridging vs EJR (PB)', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: alpha_PAIRS vs alpha_EJR
    ax = axes[0]
    for method_name, props in method_props.items():
        method_data = ejr_df[ejr_df['method'] == method_name]
        if len(method_data) > 0:
            ax.scatter(method_data['alpha_PAIRS'], method_data['alpha_EJR'],
                      marker=props['marker'], c=props['color'], s=props['size'],
                      edgecolors='black', linewidths=0.5,
                      label=method_name, alpha=0.8)
    
    ax.set_xlabel('alpha_PAIRS', fontsize=12)
    ax.set_ylabel('alpha_EJR', fontsize=12)
    ax.set_title('PAIRS vs EJR', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: alpha_CONS vs alpha_EJR
    ax = axes[1]
    for method_name, props in method_props.items():
        method_data = ejr_df[ejr_df['method'] == method_name]
        if len(method_data) > 0:
            ax.scatter(method_data['alpha_CONS'], method_data['alpha_EJR'],
                      marker=props['marker'], c=props['color'], s=props['size'],
                      edgecolors='black', linewidths=0.5,
                      label=method_name, alpha=0.8)
    
    ax.set_xlabel('alpha_CONS', fontsize=12)
    ax.set_ylabel('alpha_EJR', fontsize=12)
    ax.set_title('CONS vs EJR', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, 'ejr_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved EJR plots to {output_file}")
    
    # Create zoomed EJR plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Voting Methods: Bridging vs EJR (Zoomed 0.8-1.0)', fontsize=16, fontweight='bold', y=1.02)
    
    ax = axes[0]
    for method_name, props in method_props.items():
        method_data = ejr_df[ejr_df['method'] == method_name]
        if len(method_data) > 0:
            ax.scatter(method_data['alpha_PAIRS'], method_data['alpha_EJR'],
                      marker=props['marker'], c=props['color'], s=props['size'],
                      edgecolors='black', linewidths=0.5,
                      label=method_name, alpha=0.8)
    
    ax.set_xlabel('alpha_PAIRS', fontsize=12)
    ax.set_ylabel('alpha_EJR', fontsize=12)
    ax.set_title('PAIRS vs EJR (Zoomed)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.78, 1.02)
    ax.set_ylim(0.78, 1.02)
    
    ax = axes[1]
    for method_name, props in method_props.items():
        method_data = ejr_df[ejr_df['method'] == method_name]
        if len(method_data) > 0:
            ax.scatter(method_data['alpha_CONS'], method_data['alpha_EJR'],
                      marker=props['marker'], c=props['color'], s=props['size'],
                      edgecolors='black', linewidths=0.5,
                      label=method_name, alpha=0.8)
    
    ax.set_xlabel('alpha_CONS', fontsize=12)
    ax.set_ylabel('alpha_EJR', fontsize=12)
    ax.set_title('CONS vs EJR (Zoomed)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.78, 1.02)
    ax.set_ylim(0.78, 1.02)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'ejr_plots_zoomed.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved zoomed EJR plots to {output_file}")
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal plotting time: {total_elapsed:.2f}s")


def process_pb_file(filepath, output_base_dir='output/pb'):
    """
    Process a single PB file through the full pipeline.
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
    
    # Step 2: Calculate alpha approximations
    print("\n" + "="*70)
    print("STEP 2: Calculating alpha approximations")
    print("="*70)
    calculate_alpha_scores_pb(output_dir)
    
    # Step 3: Run voting methods
    print("\n" + "="*70)
    print("STEP 3: Running voting methods")
    print("="*70)
    run_voting_methods_pb(M, project_costs, budget, output_dir)
    
    # Step 4: Create plots
    print("\n" + "="*70)
    print("STEP 4: Creating plots")
    print("="*70)
    create_pb_plots(M, project_costs, budget, output_dir)
    
    total_elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"COMPLETED: {filepath}")
    print(f"Output saved to: {output_dir}")
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-winner bridging analysis for pabulib PB datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_pb.py data/pb_selected_10_20251202_023743/netherlands_amsterdam_524_.pb
  python main_pb.py data/pb_selected_10_20251202_023743/
  python main_pb.py --test
        """
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        help='Path to .pb file or directory containing .pb files'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run on test file (netherlands_amsterdam_524_.pb)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output/pb',
        help='Base output directory (default: output/pb)'
    )
    
    args = parser.parse_args()
    
    if args.test:
        filepath = "data/pb_selected_10_20251202_023743/poland_warszawa_2017_marymont-potok-zoliborz-dziennikarski.pb"
        process_pb_file(filepath, args.output_dir)
    elif args.path:
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
            for filepath in pb_files:
                process_pb_file(filepath, args.output_dir)
            
            print("\n" + "="*70)
            print("ALL FILES PROCESSED!")
            print("="*70)
        else:
            print(f"Error: {path} is not a valid file or directory")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

