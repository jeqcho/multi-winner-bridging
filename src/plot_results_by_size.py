"""
Create visualizations of alpha-approximation results (normalized by subset size).

Creates 4 scatter plots (2x2 grid):
- Row 1: alpha_PAIRS as x-axis vs beta_AV, beta_CC
- Row 2: alpha_CONS as x-axis vs beta_AV, beta_CC

Beta values are the average alpha values for each x-axis value.
Points are colored by subset_size using viridis colormap.
Reference lines show theoretical bounds.
Voting methods (MES, AV, CC, PAV) are shown as distinct markers.

Uses alpha scores normalized by subset size.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os


# Voting method visual properties
VOTING_METHODS = {
    'MES': {'marker': '*', 'color': 'gold', 'size': 200},
    'AV': {'marker': 's', 'color': 'red', 'size': 100},
    'greedy-CC': {'marker': '^', 'color': 'blue', 'size': 100},
    'greedy-PAV': {'marker': 'D', 'color': 'green', 'size': 100},
}


def plot_results_by_size(input_file='output/french_election/alpha_scores_by_size.csv', 
                         output_file='output/french_election/alpha_plots_by_size.png',
                         mes_file='output/french_election/voting_results.csv',
                         output_dir=None, n_candidates=12):
    """
    Create 4 plots showing relationships between alpha values (normalized by size).
    
    Args:
        input_file: Path to alpha scores CSV (or filename if output_dir provided)
        output_file: Path to output plot image (or filename if output_dir provided)
        mes_file: Path to voting methods results CSV (or filename if output_dir provided)
        output_dir: Optional directory prefix for input/output files
        n_candidates: Number of candidates (for colorbar normalization, default 12)
    """
    if output_dir:
        input_file = os.path.join(output_dir, input_file)
        output_file = os.path.join(output_dir, output_file)
        mes_file = os.path.join(output_dir, mes_file)
    print("="*70)
    print("CREATING PLOTS (ALPHA SCORES BY SIZE)")
    print("="*70)
    
    # Load data
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} subsets")
    
    # Load voting methods results
    methods_df = None
    if os.path.exists(mes_file):
        print(f"Loading voting methods results from {mes_file}...")
        methods_df = pd.read_csv(mes_file)
        print(f"Loaded {len(methods_df)} voting method results")
    else:
        print(f"Warning: Voting methods results file not found at {mes_file}")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Alpha-Approximation Analysis (Normalized by Subset Size): Multi-Winner Bridging', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color mapping by subset size
    norm = plt.Normalize(vmin=0, vmax=n_candidates)
    cmap = cm.viridis
    
    # Plotting parameters
    alpha_transparency = 0.35
    marker_size = 15
    
    # Helper function to add voting method points
    def add_voting_methods(ax, x_col, y_col):
        if methods_df is None:
            return
        for method_name, props in VOTING_METHODS.items():
            method_data = methods_df[methods_df['method'] == method_name]
            if len(method_data) > 0:
                ax.scatter(method_data[x_col], method_data[y_col],
                          marker=props['marker'], c=props['color'], s=props['size'],
                          edgecolors='black', linewidths=0.5,
                          label=method_name, zorder=10, alpha=0.8)
    
    # Row 1: alpha_PAIRS as x-axis
    print("\nProcessing row 1 (alpha_PAIRS)...")
    
    # Group by alpha_PAIRS and calculate mean (beta) values
    pairs_grouped = df.groupby('alpha_PAIRS').agg({
        'alpha_AV': 'mean',
        'alpha_CC': 'mean',
        'subset_size': 'mean'  # For coloring
    }).reset_index()
    
    # Rename for clarity
    pairs_grouped.rename(columns={
        'alpha_AV': 'beta_AV',
        'alpha_CC': 'beta_CC',
    }, inplace=True)
    
    # Plot 1: alpha_PAIRS vs beta_AV
    ax = axes[0, 0]
    scatter = ax.scatter(pairs_grouped['alpha_PAIRS'], pairs_grouped['beta_AV'],
                        c=pairs_grouped['subset_size'], cmap=cmap, norm=norm,
                        alpha=alpha_transparency, s=marker_size)
    # Reference line: beta = 1 - alpha (a + b = 1)
    x_ref = np.linspace(0, 1, 100)
    ax.plot(x_ref, 1 - x_ref, 'k-', linewidth=2, alpha=0.7, label='a + b = 1')
    add_voting_methods(ax, 'alpha_PAIRS', 'alpha_AV')
    ax.set_xlabel('alpha_PAIRS', fontsize=11)
    ax.set_ylabel('beta_AV (avg)', fontsize=11)
    ax.set_title('PAIRS vs AV (by size)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: alpha_PAIRS vs beta_CC
    ax = axes[0, 1]
    ax.scatter(pairs_grouped['alpha_PAIRS'], pairs_grouped['beta_CC'],
              c=pairs_grouped['subset_size'], cmap=cmap, norm=norm,
              alpha=alpha_transparency, s=marker_size)
    ax.plot(x_ref, 1 - x_ref, 'k-', linewidth=2, alpha=0.7, label='a + b = 1')
    add_voting_methods(ax, 'alpha_PAIRS', 'alpha_CC')
    ax.set_xlabel('alpha_PAIRS', fontsize=11)
    ax.set_ylabel('beta_CC (avg)', fontsize=11)
    ax.set_title('PAIRS vs CC (by size)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Row 2: alpha_CONS as x-axis
    print("Processing row 2 (alpha_CONS)...")
    
    # Group by alpha_CONS and calculate mean (beta) values
    cons_grouped = df.groupby('alpha_CONS').agg({
        'alpha_AV': 'mean',
        'alpha_CC': 'mean',
        'subset_size': 'mean'
    }).reset_index()
    
    cons_grouped.rename(columns={
        'alpha_AV': 'beta_AV',
        'alpha_CC': 'beta_CC',
    }, inplace=True)
    
    # Plot 3: alpha_CONS vs beta_AV
    ax = axes[1, 0]
    ax.scatter(cons_grouped['alpha_CONS'], cons_grouped['beta_AV'],
              c=cons_grouped['subset_size'], cmap=cmap, norm=norm,
              alpha=alpha_transparency, s=marker_size)
    # Reference line: beta = 1 - alpha^2 (a^2 + b = 1)
    ax.plot(x_ref, 1 - x_ref**2, 'k-', linewidth=2, alpha=0.7, label='a² + b = 1')
    add_voting_methods(ax, 'alpha_CONS', 'alpha_AV')
    ax.set_xlabel('alpha_CONS', fontsize=11)
    ax.set_ylabel('beta_AV (avg)', fontsize=11)
    ax.set_title('CONS vs AV (by size)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: alpha_CONS vs beta_CC
    ax = axes[1, 1]
    ax.scatter(cons_grouped['alpha_CONS'], cons_grouped['beta_CC'],
              c=cons_grouped['subset_size'], cmap=cmap, norm=norm,
              alpha=alpha_transparency, s=marker_size)
    ax.plot(x_ref, 1 - x_ref**2, 'k-', linewidth=2, alpha=0.7, label='a² + b = 1')
    add_voting_methods(ax, 'alpha_CONS', 'alpha_CC')
    ax.set_xlabel('alpha_CONS', fontsize=11)
    ax.set_ylabel('beta_CC (avg)', fontsize=11)
    ax.set_title('CONS vs CC (by size)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Adjust layout first
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # Add colorbar for subset size on the right side
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Committee Size (k)', fontsize=12, rotation=270, labelpad=20)
    
    # Save figure
    print(f"\nSaving plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully (DPI=300)")
    
    # Display summary statistics
    print("\n" + "="*70)
    print("PLOT STATISTICS (BY SIZE)")
    print("="*70)
    
    print(f"\nRow 1 (PAIRS):")
    print(f"  Unique alpha_PAIRS values: {len(pairs_grouped)}")
    print(f"  beta_AV range: [{pairs_grouped['beta_AV'].min():.4f}, {pairs_grouped['beta_AV'].max():.4f}]")
    print(f"  beta_CC range: [{pairs_grouped['beta_CC'].min():.4f}, {pairs_grouped['beta_CC'].max():.4f}]")
    
    print(f"\nRow 2 (CONS):")
    print(f"  Unique alpha_CONS values: {len(cons_grouped)}")
    print(f"  beta_AV range: [{cons_grouped['beta_AV'].min():.4f}, {cons_grouped['beta_AV'].max():.4f}]")
    print(f"  beta_CC range: [{cons_grouped['beta_CC'].min():.4f}, {cons_grouped['beta_CC'].max():.4f}]")
    
    if methods_df is not None:
        print(f"\nVoting methods plotted: {methods_df['method'].nunique()} methods, {len(methods_df)} total points")
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Output file: {output_file}")
    print(f"You can view the plot with: open {output_file}")
    print("="*70)


if __name__ == "__main__":
    plot_results_by_size()
