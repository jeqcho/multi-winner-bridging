"""
Cluster Warsaw 2018 Wola committees to understand stripe patterns.

Uses band coordinate technique to project points perpendicular to stripes,
then analyzes which projects characterize each cluster.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path

# Base paths (following generate_plots.py pattern)
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data" / "pb_selected_10_20251202_023743"
PRESENTATION_OUTPUT = Path(__file__).parent.parent / "outputs"

# Election to analyze
ELECTION_NAME = "poland_warszawa_2018_wola"

# Project info from .pb file (project_id -> name)
PROJECT_INFO = {
    "314": {"name": "Wolskie lato filmowe", "cost": 47000, "votes": 3593, "category": "culture"},
    "2678": {"name": "Chronimy jerzyki i wróble na Woli", "cost": 49700, "votes": 3510, "category": "environmental"},
    "379": {"name": "Wola na 2 koła! (bicycle)", "cost": 30000, "votes": 3464, "category": "transit/sport"},
    "231": {"name": "Ćwiczenia dla seniorów", "cost": 50000, "votes": 2777, "category": "sport/health"},
    "402": {"name": "Rozwijamy wolskie maluchy", "cost": 24000, "votes": 2704, "category": "education"},
    "1668": {"name": "Skarby Woli", "cost": 11200, "votes": 2662, "category": "education"},
    "1412": {"name": "Tai Chi na świeżym powietrzu", "cost": 12350, "votes": 2567, "category": "sport"},
    "740": {"name": "Ogródek przy przystanku", "cost": 11294, "votes": 2529, "category": "public space"},
    "1595": {"name": "Virtualna Warszawa (tourism)", "cost": 81266, "votes": 2503, "category": "culture"},
    "576": {"name": "Nakarm dziecko zdrowiem", "cost": 6300, "votes": 2294, "category": "health"},
    "2700": {"name": "Moja Wola - Centrum Dialogu", "cost": 44000, "votes": 2286, "category": "education/culture"},
}

# Map index (0-10) to project_id (order from pb_data_loader)
INDEX_TO_PROJECT_ID = ["314", "2678", "379", "231", "402", "1668", "1412", "740", "1595", "576", "2700"]


def add_band_coord(df, x_col='alpha_PAIRS', y_col='alpha_AV', coord_col='band_coord'):
    """
    Add a 1-D coordinate (band_coord) that measures position
    perpendicular to the stripes.
    """
    X = df[x_col].to_numpy()
    Y = df[y_col].to_numpy()
    
    # least-squares fit: Y ≈ m * X + b
    A = np.vstack([X, np.ones(len(X))]).T
    m, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    
    # vector perpendicular to the stripes (normal vector)
    n = np.array([-m, 1.0], dtype=float)
    n /= np.linalg.norm(n)
    
    # projection on n → 1-D coordinate
    df = df.copy()
    df[coord_col] = df[[x_col, y_col]].to_numpy() @ n
    
    return df, m, b


def segment_into_clusters(df, coord_col='band_coord', n_clusters=None):
    """
    Segment data into clusters based on band coordinate.
    
    Uses histogram-based approach to find natural breaks in the distribution.
    """
    coords = df[coord_col].values
    
    if n_clusters is None:
        # Use histogram to find natural clusters
        # Try to detect peaks/valleys in the distribution
        hist, bin_edges = np.histogram(coords, bins=50)
        
        # Find valleys (local minima) as cluster boundaries
        valleys = []
        for i in range(1, len(hist) - 1):
            if hist[i] < hist[i-1] and hist[i] < hist[i+1]:
                if hist[i] < np.mean(hist) * 0.5:  # Significant valley
                    valleys.append(bin_edges[i+1])
        
        if len(valleys) > 0:
            boundaries = sorted(valleys)
            n_clusters = len(boundaries) + 1
        else:
            # Default to quantile-based if no clear valleys
            n_clusters = 5
            boundaries = [np.percentile(coords, p) for p in np.linspace(20, 80, n_clusters-1)]
    else:
        # Use quantiles for specified number of clusters
        boundaries = [np.percentile(coords, p) for p in np.linspace(100/n_clusters, 100*(n_clusters-1)/n_clusters, n_clusters-1)]
    
    # Assign cluster labels
    labels = np.zeros(len(df), dtype=int)
    for i, boundary in enumerate(boundaries):
        labels[coords > boundary] = i + 1
    
    df = df.copy()
    df['cluster'] = labels
    
    return df, boundaries


def analyze_clusters(df):
    """
    Analyze which projects characterize each cluster.
    
    Returns a summary of project membership rates per cluster.
    """
    # Parse subset_indices
    df['candidates'] = df['subset_indices'].apply(ast.literal_eval)
    
    n_projects = len(INDEX_TO_PROJECT_ID)
    clusters = sorted(df['cluster'].unique())
    
    # Calculate overall project frequency
    overall_freq = np.zeros(n_projects)
    for candidates in df['candidates']:
        for c in candidates:
            overall_freq[c] += 1
    overall_freq /= len(df)
    
    # Calculate per-cluster frequency
    cluster_analysis = {}
    for cluster_id in clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_df)
        
        freq = np.zeros(n_projects)
        for candidates in cluster_df['candidates']:
            for c in candidates:
                freq[c] += 1
        freq /= cluster_size
        
        # Calculate deviation from overall
        deviation = freq - overall_freq
        
        # Find most distinctive projects
        over_represented = np.argsort(deviation)[-3:][::-1]  # Top 3 over-represented
        under_represented = np.argsort(deviation)[:3]  # Top 3 under-represented
        
        cluster_analysis[cluster_id] = {
            'size': cluster_size,
            'freq': freq,
            'deviation': deviation,
            'over_represented': over_represented,
            'under_represented': under_represented,
            'mean_pairs': cluster_df['alpha_PAIRS'].mean(),
            'mean_av': cluster_df['alpha_AV'].mean(),
            'mean_band': cluster_df['band_coord'].mean(),
        }
    
    return cluster_analysis, overall_freq


def format_project_name(idx):
    """Format project name with index and short name."""
    project_id = INDEX_TO_PROJECT_ID[idx]
    info = PROJECT_INFO[project_id]
    return f"{idx}: {info['name'][:30]}"


def print_cluster_summary(cluster_analysis, overall_freq):
    """Print human-readable cluster interpretation."""
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS: Warsaw 2018 Wola")
    print("="*80)
    
    print("\n--- PROJECT REFERENCE ---")
    for idx, project_id in enumerate(INDEX_TO_PROJECT_ID):
        info = PROJECT_INFO[project_id]
        print(f"  {idx:2d}: {info['name'][:40]:<40} (cost: {info['cost']:,}, votes: {info['votes']:,})")
    
    print("\n--- OVERALL PROJECT FREQUENCY ---")
    for idx in range(len(overall_freq)):
        print(f"  Project {idx}: {overall_freq[idx]*100:.1f}%")
    
    print("\n--- CLUSTER INTERPRETATIONS ---")
    for cluster_id in sorted(cluster_analysis.keys()):
        analysis = cluster_analysis[cluster_id]
        
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id}")
        print(f"{'='*60}")
        print(f"  Size: {analysis['size']:,} committees ({analysis['size']/2048*100:.1f}% of total)")
        print(f"  Mean α_PAIRS: {analysis['mean_pairs']:.3f}")
        print(f"  Mean α_AV: {analysis['mean_av']:.3f}")
        print(f"  Mean band coord: {analysis['mean_band']:.3f}")
        
        print("\n  OVER-REPRESENTED projects (more frequent than average):")
        for idx in analysis['over_represented']:
            dev = analysis['deviation'][idx]
            if dev > 0.05:  # Only show if significant
                freq = analysis['freq'][idx]
                print(f"    - {format_project_name(idx)}")
                print(f"      Frequency: {freq*100:.1f}% (vs {overall_freq[idx]*100:.1f}% overall, +{dev*100:.1f}%)")
        
        print("\n  UNDER-REPRESENTED projects (less frequent than average):")
        for idx in analysis['under_represented']:
            dev = analysis['deviation'][idx]
            if dev < -0.05:  # Only show if significant
                freq = analysis['freq'][idx]
                print(f"    - {format_project_name(idx)}")
                print(f"      Frequency: {freq*100:.1f}% (vs {overall_freq[idx]*100:.1f}% overall, {dev*100:.1f}%)")
        
        # Generate interpretation
        print("\n  INTERPRETATION:")
        over = [idx for idx in analysis['over_represented'] if analysis['deviation'][idx] > 0.1]
        under = [idx for idx in analysis['under_represented'] if analysis['deviation'][idx] < -0.1]
        
        if over or under:
            interp_parts = []
            if over:
                over_names = [f"'{PROJECT_INFO[INDEX_TO_PROJECT_ID[i]]['name'][:25]}'" for i in over]
                interp_parts.append(f"committees that INCLUDE {', '.join(over_names)}")
            if under:
                under_names = [f"'{PROJECT_INFO[INDEX_TO_PROJECT_ID[i]]['name'][:25]}'" for i in under]
                interp_parts.append(f"EXCLUDE {', '.join(under_names)}")
            print(f"    → {' and '.join(interp_parts)}")
        else:
            print("    → No strongly distinctive project pattern")


def main():
    """Main clustering analysis."""
    print("="*80)
    print("WARSAW 2018 WOLA CLUSTER ANALYSIS")
    print("="*80)
    
    # Load data
    data_file = OUTPUT_DIR / "pb" / ELECTION_NAME / "alpha_scores.csv"
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} committees")
    
    # Step 1: Add band coordinate
    print("\n--- FITTING STRIPE DIRECTION ---")
    df, m, b = add_band_coord(df, x_col='alpha_PAIRS', y_col='alpha_AV')
    print(f"  Fitted line: α_AV = {m:.3f} * α_PAIRS + {b:.3f}")
    print(f"  Band coordinate range: [{df['band_coord'].min():.3f}, {df['band_coord'].max():.3f}]")
    
    # Step 2: Segment into clusters
    print("\n--- SEGMENTING INTO CLUSTERS ---")
    df, boundaries = segment_into_clusters(df, n_clusters=6)  # Try 6 clusters based on visual inspection
    print(f"  Cluster boundaries: {[f'{b:.3f}' for b in boundaries]}")
    print(f"  Cluster sizes: {df['cluster'].value_counts().sort_index().to_dict()}")
    
    # Step 3: Analyze clusters
    print("\n--- ANALYZING CLUSTER COMPOSITION ---")
    cluster_analysis, overall_freq = analyze_clusters(df)
    
    # Step 4: Print summary
    print_cluster_summary(cluster_analysis, overall_freq)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

