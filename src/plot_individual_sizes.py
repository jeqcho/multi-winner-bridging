"""
Create individual visualizations for each committee size (k=0 to 12).

For each size, creates a 2x3 grid of scatter plots showing relationships
between alpha values, filtered to only that specific committee size.
MES (Method of Equal Shares) committee is shown as a gold star.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os


def plot_single_size(df, k, output_dir='output/by_size', mes_df=None):
    """
    Create visualization for a single committee size.
    
    Args:
        df: DataFrame with alpha scores
        k: Committee size to plot
        output_dir: Directory to save plots
        mes_df: DataFrame with MES results (optional)
    """
    # Filter to only this size
    df_k = df[df['subset_size'] == k].copy()
    
    if len(df_k) == 0:
        print(f"  No data for size k={k}, skipping...")
        return
    
    print(f"  Size k={k}: {len(df_k)} subsets")
    
    # Get MES data for this size
    mes_k = None
    if mes_df is not None:
        mes_k = mes_df[mes_df['subset_size'] == k]
        if len(mes_k) > 0:
            mes_k = mes_k.iloc[0]  # Get the single row
        else:
            mes_k = None
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Alpha-Approximation Analysis: Committee Size k={k} ({len(df_k)} subsets)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Use single color for all points
    point_color = '#2E86AB'  # Nice blue color
    
    # Plotting parameters
    alpha_transparency = 0.6  # Less transparency since fewer points per size
    marker_size = 30  # Larger markers for individual size plots
    
    # MES marker parameters
    mes_marker = '*'
    mes_color = 'gold'
    mes_size = 400  # Larger for individual plots
    mes_edgecolor = 'black'
    mes_linewidth = 1.5
    
    # Row 1: alpha_PAIRS as x-axis
    
    # Group by alpha_PAIRS and calculate mean (beta) values
    pairs_grouped = df_k.groupby('alpha_PAIRS').agg({
        'alpha_AV': 'mean',
        'alpha_CC': 'mean',
        'alpha_EJR': 'mean'
    }).reset_index()
    
    pairs_grouped.rename(columns={
        'alpha_AV': 'beta_AV',
        'alpha_CC': 'beta_CC',
        'alpha_EJR': 'beta_EJR'
    }, inplace=True)
    
    # Plot 1: alpha_PAIRS vs beta_AV
    ax = axes[0, 0]
    ax.scatter(pairs_grouped['alpha_PAIRS'], pairs_grouped['beta_AV'],
              c=point_color, alpha=alpha_transparency, s=marker_size, 
              edgecolors='black', linewidths=0.5)
    # Reference line: beta = 1 - alpha (a + b = 1)
    x_ref = np.linspace(0, 1, 100)
    ax.plot(x_ref, 1 - x_ref, 'k--', linewidth=2, alpha=0.5, label='a + b = 1')
    # Add MES point
    if mes_k is not None:
        ax.scatter(mes_k['alpha_PAIRS'], mes_k['alpha_AV'],
                  marker=mes_marker, c=mes_color, s=mes_size,
                  edgecolors=mes_edgecolor, linewidths=mes_linewidth,
                  label='MES', zorder=10)
    ax.set_xlabel('alpha_PAIRS', fontsize=11)
    ax.set_ylabel('beta_AV (avg)', fontsize=11)
    ax.set_title('PAIRS vs AV', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: alpha_PAIRS vs beta_CC
    ax = axes[0, 1]
    ax.scatter(pairs_grouped['alpha_PAIRS'], pairs_grouped['beta_CC'],
              c=point_color, alpha=alpha_transparency, s=marker_size, 
              edgecolors='black', linewidths=0.5)
    ax.plot(x_ref, 1 - x_ref, 'k--', linewidth=2, alpha=0.5, label='a + b = 1')
    # Add MES point
    if mes_k is not None:
        ax.scatter(mes_k['alpha_PAIRS'], mes_k['alpha_CC'],
                  marker=mes_marker, c=mes_color, s=mes_size,
                  edgecolors=mes_edgecolor, linewidths=mes_linewidth,
                  label='MES', zorder=10)
    ax.set_xlabel('alpha_PAIRS', fontsize=11)
    ax.set_ylabel('beta_CC (avg)', fontsize=11)
    ax.set_title('PAIRS vs CC', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 3: alpha_PAIRS vs beta_EJR
    ax = axes[0, 2]
    ax.scatter(pairs_grouped['alpha_PAIRS'], pairs_grouped['beta_EJR'],
              c=point_color, alpha=alpha_transparency, s=marker_size, 
              edgecolors='black', linewidths=0.5)
    ax.plot(x_ref, 1 - x_ref, 'k--', linewidth=2, alpha=0.5, label='a + b = 1')
    # Add MES point
    if mes_k is not None:
        ax.scatter(mes_k['alpha_PAIRS'], mes_k['alpha_EJR'],
                  marker=mes_marker, c=mes_color, s=mes_size,
                  edgecolors=mes_edgecolor, linewidths=mes_linewidth,
                  label='MES', zorder=10)
    ax.set_xlabel('alpha_PAIRS', fontsize=11)
    ax.set_ylabel('beta_EJR (avg)', fontsize=11)
    ax.set_title('PAIRS vs EJR', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Row 2: alpha_CONS as x-axis
    
    # Group by alpha_CONS and calculate mean (beta) values
    cons_grouped = df_k.groupby('alpha_CONS').agg({
        'alpha_AV': 'mean',
        'alpha_CC': 'mean',
        'alpha_EJR': 'mean'
    }).reset_index()
    
    cons_grouped.rename(columns={
        'alpha_AV': 'beta_AV',
        'alpha_CC': 'beta_CC',
        'alpha_EJR': 'beta_EJR'
    }, inplace=True)
    
    # Plot 4: alpha_CONS vs beta_AV
    ax = axes[1, 0]
    ax.scatter(cons_grouped['alpha_CONS'], cons_grouped['beta_AV'],
              c=point_color, alpha=alpha_transparency, s=marker_size, 
              edgecolors='black', linewidths=0.5)
    # Reference line: beta = 1 - alpha^2 (a^2 + b = 1)
    ax.plot(x_ref, 1 - x_ref**2, 'k--', linewidth=2, alpha=0.5, label='a² + b = 1')
    # Add MES point
    if mes_k is not None:
        ax.scatter(mes_k['alpha_CONS'], mes_k['alpha_AV'],
                  marker=mes_marker, c=mes_color, s=mes_size,
                  edgecolors=mes_edgecolor, linewidths=mes_linewidth,
                  label='MES', zorder=10)
    ax.set_xlabel('alpha_CONS', fontsize=11)
    ax.set_ylabel('beta_AV (avg)', fontsize=11)
    ax.set_title('CONS vs AV', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 5: alpha_CONS vs beta_CC
    ax = axes[1, 1]
    ax.scatter(cons_grouped['alpha_CONS'], cons_grouped['beta_CC'],
              c=point_color, alpha=alpha_transparency, s=marker_size, 
              edgecolors='black', linewidths=0.5)
    ax.plot(x_ref, 1 - x_ref**2, 'k--', linewidth=2, alpha=0.5, label='a² + b = 1')
    # Add MES point
    if mes_k is not None:
        ax.scatter(mes_k['alpha_CONS'], mes_k['alpha_CC'],
                  marker=mes_marker, c=mes_color, s=mes_size,
                  edgecolors=mes_edgecolor, linewidths=mes_linewidth,
                  label='MES', zorder=10)
    ax.set_xlabel('alpha_CONS', fontsize=11)
    ax.set_ylabel('beta_CC (avg)', fontsize=11)
    ax.set_title('CONS vs CC', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 6: alpha_CONS vs beta_EJR
    ax = axes[1, 2]
    ax.scatter(cons_grouped['alpha_CONS'], cons_grouped['beta_EJR'],
              c=point_color, alpha=alpha_transparency, s=marker_size, 
              edgecolors='black', linewidths=0.5)
    ax.plot(x_ref, 1 - x_ref**2, 'k--', linewidth=2, alpha=0.5, label='a² + b = 1')
    # Add MES point
    if mes_k is not None:
        ax.scatter(mes_k['alpha_CONS'], mes_k['alpha_EJR'],
                  marker=mes_marker, c=mes_color, s=mes_size,
                  edgecolors=mes_edgecolor, linewidths=mes_linewidth,
                  label='MES', zorder=10)
    ax.set_xlabel('alpha_CONS', fontsize=11)
    ax.set_ylabel('beta_EJR (avg)', fontsize=11)
    ax.set_title('CONS vs EJR', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Adjust layout - no colorbar needed
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f'size_{k:02d}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_sizes(input_file='output/french_election/alpha_scores_by_size.csv', 
                   output_dir='output/french_election/by_size',
                   mes_file='output/french_election/mes_results.csv',
                   base_dir=None):
    """
    Create individual plots for all committee sizes.
    
    Args:
        input_file: Path to alpha scores CSV (or filename if base_dir provided)
        output_dir: Directory to save plots (or subdir name if base_dir provided)
        mes_file: Path to MES results CSV (or filename if base_dir provided)
        base_dir: Optional base directory prefix for all paths
    """
    if base_dir:
        input_file = os.path.join(base_dir, input_file)
        output_dir = os.path.join(base_dir, output_dir)
        mes_file = os.path.join(base_dir, mes_file)
    print("="*70)
    print("CREATING INDIVIDUAL PLOTS FOR EACH COMMITTEE SIZE")
    print("="*70)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} subsets")
    
    # Load MES results
    mes_df = None
    if os.path.exists(mes_file):
        print(f"Loading MES results from {mes_file}...")
        mes_df = pd.read_csv(mes_file)
        print(f"Loaded {len(mes_df)} MES committees")
    else:
        print(f"Warning: MES results file not found at {mes_file}")
    
    # Get all unique sizes
    sizes = sorted(df['subset_size'].unique())
    print(f"\nFound {len(sizes)} committee sizes: {list(sizes)}")
    
    # Create plot for each size
    print(f"\nGenerating plots...")
    for k in sizes:
        plot_single_size(df, k, output_dir, mes_df)
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Generated {len(sizes)} plots in {output_dir}/")
    print(f"Files: size_00.png to size_12.png")
    if mes_df is not None:
        print("MES committees marked with gold stars")
    print("="*70)


if __name__ == "__main__":
    plot_all_sizes()
