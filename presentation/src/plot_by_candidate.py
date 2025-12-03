"""
Generate PAIRS vs AV plot colored by candidate membership.

Creates a grid of subplots where each subplot highlights committees
containing a specific candidate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
PRESENTATION_OUTPUT = Path(__file__).parent.parent / "outputs"

# Presentation-sized fonts
FONT_SIZES = {
    "title": 24,
    "subplot_title": 14,
    "subplot_label": 12,
    "subplot_tick": 10,
}

# Colors
HIGHLIGHT_COLOR = '#E63946'  # Red for highlighted
DEFAULT_COLOR = '#ADB5BD'    # Gray for non-highlighted


def plot_pairs_vs_av_by_candidate(election_name="poland_warszawa_2018_wola"):
    """
    Create PAIRS vs AV plot with subplots for each candidate.
    
    Each subplot highlights committees containing that candidate.
    """
    print("="*70)
    print(f"GENERATING PAIRS vs AV BY CANDIDATE: {election_name}")
    print("="*70)
    
    # Load data
    data_file = OUTPUT_DIR / "pb" / election_name / "alpha_scores.csv"
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} committees")
    
    # Parse subset_indices to get candidate membership
    df['candidates'] = df['subset_indices'].apply(ast.literal_eval)
    
    # Get all unique candidates
    all_candidates = set()
    for candidates in df['candidates']:
        all_candidates.update(candidates)
    candidates = sorted(all_candidates)
    n_candidates = len(candidates)
    print(f"Found {n_candidates} candidates: {candidates}")
    
    # Determine grid layout
    if n_candidates <= 4:
        nrows, ncols = 1, n_candidates
    elif n_candidates <= 8:
        nrows, ncols = 2, 4
    elif n_candidates <= 12:
        nrows, ncols = 3, 4
    elif n_candidates <= 16:
        nrows, ncols = 4, 4
    else:
        nrows = (n_candidates + 4) // 5
        ncols = 5
    
    print(f"Using {nrows}x{ncols} grid")
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()
    
    # Reference line
    x_ref = np.linspace(-0.05, 1.05, 100)
    y_ref = 1 - x_ref  # a + b = 1
    
    # Create subplot for each candidate
    for idx, candidate in enumerate(candidates):
        ax = axes_flat[idx]
        
        # Create mask for committees containing this candidate
        has_candidate = df['candidates'].apply(lambda x: candidate in x)
        
        # Plot non-highlighted committees first (gray, in background)
        df_other = df[~has_candidate]
        ax.scatter(
            df_other['alpha_PAIRS'],
            df_other['alpha_AV'],
            c=DEFAULT_COLOR,
            alpha=0.3,
            s=10,
            label=f'Without candidate {candidate}'
        )
        
        # Plot highlighted committees (red, in foreground)
        df_with = df[has_candidate]
        ax.scatter(
            df_with['alpha_PAIRS'],
            df_with['alpha_AV'],
            c=HIGHLIGHT_COLOR,
            alpha=0.6,
            s=15,
            label=f'With candidate {candidate}'
        )
        
        # Reference line
        ax.plot(x_ref, y_ref, 'k-', linewidth=2, alpha=0.7)
        
        # Title
        ax.set_title(f'Candidate {candidate}', fontsize=FONT_SIZES["subplot_title"], fontweight='bold')
        
        # Axis labels (only on edges)
        if idx >= (nrows - 1) * ncols:  # Bottom row
            ax.set_xlabel('α_PAIRS', fontsize=FONT_SIZES["subplot_label"])
        if idx % ncols == 0:  # Left column
            ax.set_ylabel('α_AV', fontsize=FONT_SIZES["subplot_label"])
        
        # Axis limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Tick sizes
        ax.tick_params(axis='both', labelsize=FONT_SIZES["subplot_tick"])
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Count stats
        n_with = len(df_with)
        n_total = len(df)
        ax.text(0.02, 0.98, f'{n_with}/{n_total}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', color=HIGHLIGHT_COLOR, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_candidates, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Main title
    election_display = election_name.replace("_", " ").replace("-", " ").title()
    fig.suptitle(f'PAIRS vs AV by Candidate - {election_display}',
                 fontsize=FONT_SIZES["title"], fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    PRESENTATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_file = PRESENTATION_OUTPUT / f"pairs_vs_av_by_candidate_{election_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved to: {output_file}")
    print("="*70)


def main():
    plot_pairs_vs_av_by_candidate("poland_warszawa_2018_wola")


if __name__ == "__main__":
    main()

