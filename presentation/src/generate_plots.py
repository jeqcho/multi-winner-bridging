"""
Generate presentation-ready plots for multi-winner bridging analysis.

Creates:
1. Empty theoretical trade-off plots (axes + black reference line only)
2. Featured big plots for poland_warszawa_2018_wawrzyszew
3. Mega plots (4x3 grid) combining remaining 12 elections
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
PRESENTATION_OUTPUT = Path(__file__).parent.parent / "outputs"

# Featured election
FEATURED_ELECTION = "poland_warszawa_2018_wawrzyszew"

# Plot configurations for 5 plot types
PLOT_CONFIGS = [
    {
        "name": "pairs_vs_av",
        "x_col": "alpha_PAIRS",
        "y_col": "alpha_AV",
        "x_label": "α_PAIRS",
        "y_label": "α_AV",
        "title": "PAIRS vs AV",
        "ref_func": lambda x: 1 - x,  # a + b = 1
        "ref_label": "a + b = 1"
    },
    {
        "name": "pairs_vs_cc",
        "x_col": "alpha_PAIRS",
        "y_col": "alpha_CC",
        "x_label": "α_PAIRS",
        "y_label": "α_CC",
        "title": "PAIRS vs CC",
        "ref_func": lambda x: 1 - x,  # a + b = 1
        "ref_label": "a + b = 1"
    },
    {
        "name": "cons_vs_av",
        "x_col": "alpha_CONS",
        "y_col": "alpha_AV",
        "x_label": "α_CONS",
        "y_label": "α_AV",
        "title": "CONS vs AV",
        "ref_func": lambda x: 1 - x**2,  # a² + b = 1
        "ref_label": "a² + b = 1"
    },
    {
        "name": "cons_vs_cc",
        "x_col": "alpha_CONS",
        "y_col": "alpha_CC",
        "x_label": "α_CONS",
        "y_label": "α_CC",
        "title": "CONS vs CC",
        "ref_func": lambda x: 1 - x**2,  # a² + b = 1
        "ref_label": "a² + b = 1"
    },
    {
        "name": "cc_vs_av",
        "x_col": "alpha_CC",
        "y_col": "alpha_AV",
        "x_label": "α_CC",
        "y_label": "α_AV",
        "title": "CC vs AV",
        "ref_func": None,  # No theoretical trade-off
        "ref_label": None
    },
    {
        "name": "pairs_vs_cons",
        "x_col": "alpha_PAIRS",
        "y_col": "alpha_CONS",
        "x_label": "α_PAIRS",
        "y_label": "α_CONS",
        "title": "PAIRS vs CONS",
        "ref_func": None,  # No theoretical trade-off
        "ref_label": None
    },
]

# Presentation-sized fonts
FONT_SIZES = {
    "title": 24,
    "label": 20,
    "tick": 16,
    "legend": 14,
    "subplot_title": 14,
    "subplot_label": 12,
    "subplot_tick": 10,
}

# Voting method visual properties
VOTING_METHODS = {
    'MES': {'marker': '*', 'color': 'gold', 'size': 150},
    'AV': {'marker': 's', 'color': 'red', 'size': 80},
    'greedy-CC': {'marker': '^', 'color': 'blue', 'size': 80},
    'greedy-PAV': {'marker': 'D', 'color': 'green', 'size': 80},
}


def format_pb_display_name(folder_name):
    """Create readable display name from pb folder name."""
    # Handle US Stanford dataset elections
    if folder_name.startswith("us_stanford-dataset"):
        # Extract ward and year: us_stanford-dataset_pb-chicago-33rd-ward-2021_vote-approvals
        parts = folder_name.split("_")
        # Find ward info
        for part in parts:
            if "ward" in part:
                # pb-chicago-33rd-ward-2021
                ward_parts = part.split("-")
                ward_num = ward_parts[2] if len(ward_parts) > 2 else ""
                year = ward_parts[-1] if len(ward_parts) > 4 else ""
                break
        # Find vote type
        vote_type = ""
        if "vote-approvals" in folder_name:
            vote_type = "Approvals"
        elif "vote-knapsacks" in folder_name:
            vote_type = "Knapsacks"
        return f"Chicago {ward_num} {year} {vote_type}".strip()
    
    # Handle Poland Warsaw elections
    if folder_name.startswith("poland_warszawa"):
        # poland_warszawa_2017_marymont-potok-zoliborz-dziennikarski
        parts = folder_name.split("_")
        year = parts[2] if len(parts) > 2 else ""
        district = parts[3] if len(parts) > 3 else ""
        # Clean up district name - take first part before hyphen, capitalize
        district_clean = district.split("-")[0].capitalize()
        return f"Warsaw {year} {district_clean}"
    
    # Fallback
    return folder_name.replace("_", " ").replace("-", " ")[:30]


def get_all_elections():
    """Get all election directories and their data files."""
    elections = []
    
    # french_election
    fe_path = OUTPUT_DIR / "french_election"
    if fe_path.exists():
        elections.append({
            "name": "french_election",
            "path": fe_path,
            "data_file": fe_path / "alpha_scores_by_size.csv",
            "display_name": "French Election"
        })
    
    # camp_songs
    for subdir in ["file_02", "file_04"]:
        cs_path = OUTPUT_DIR / "camp_songs" / subdir
        if cs_path.exists():
            elections.append({
                "name": f"camp_songs_{subdir}",
                "path": cs_path,
                "data_file": cs_path / "alpha_scores_by_size.csv",
                "display_name": f"Camp Songs {subdir[-2:]}"
            })
    
    # pb elections
    pb_path = OUTPUT_DIR / "pb"
    if pb_path.exists():
        for election_dir in sorted(pb_path.iterdir()):
            if election_dir.is_dir():
                elections.append({
                    "name": election_dir.name,
                    "path": election_dir,
                    "data_file": election_dir / "alpha_scores.csv",
                    "display_name": format_pb_display_name(election_dir.name)
                })
    
    return elections


def load_and_aggregate_data(data_file, x_col, y_col):
    """Load CSV and aggregate: group by x_col, calculate mean of y_col (beta)."""
    if not data_file.exists():
        return None
    
    df = pd.read_csv(data_file)
    
    # Group by x_col and calculate mean
    grouped = df.groupby(x_col).agg({
        y_col: 'mean',
        'subset_size': 'mean'
    }).reset_index()
    
    grouped.rename(columns={y_col: f'beta_{y_col.split("_")[1]}'}, inplace=True)
    
    return grouped


def generate_empty_plots():
    """Task 1: Generate 4 empty theoretical trade-off plots."""
    print("\n" + "="*70)
    print("TASK 1: GENERATING EMPTY THEORETICAL TRADE-OFF PLOTS")
    print("="*70)
    
    for config in PLOT_CONFIGS:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Reference line (extended for margins) - only if defined
        if config["ref_func"] is not None:
            x_ref = np.linspace(-0.05, 1.05, 100)
            y_ref = config["ref_func"](x_ref)
            ax.plot(x_ref, y_ref, 'k-', linewidth=3, alpha=0.9)
        
        # Axis labels
        ax.set_xlabel(config["x_label"], fontsize=FONT_SIZES["label"], fontweight='bold')
        ax.set_ylabel(config["y_label"], fontsize=FONT_SIZES["label"], fontweight='bold')
        ax.set_title(config["title"], fontsize=FONT_SIZES["title"], fontweight='bold')
        
        # Axis limits with margins
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Tick sizes
        ax.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        
        # Save
        output_file = PRESENTATION_OUTPUT / f"{config['name']}_empty.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_file.name}")
    
    print("Empty plots completed!")


def generate_featured_plots():
    """Task 2: Generate 4 featured big plots for poland_warszawa_2018_wawrzyszew."""
    print("\n" + "="*70)
    print("TASK 2: GENERATING FEATURED PLOTS (poland_warszawa_2018_wawrzyszew)")
    print("="*70)
    
    # Load featured election data
    data_file = OUTPUT_DIR / "pb" / FEATURED_ELECTION / "alpha_scores.csv"
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} data points")
    
    # Get max cost for colorbar
    max_cost = df['total_cost'].max()
    
    for config in PLOT_CONFIGS:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color mapping by cost
        norm = plt.Normalize(vmin=0, vmax=max_cost)
        cmap = cm.viridis
        
        # Raw scatter plot (no aggregation)
        scatter = ax.scatter(
            df[config["x_col"]], 
            df[config["y_col"]],
            c=df['total_cost'], 
            cmap=cmap, 
            norm=norm,
            alpha=0.35, 
            s=15
        )
        
        # Reference line (extended for margins) - only if defined
        if config["ref_func"] is not None:
            x_ref = np.linspace(-0.05, 1.05, 100)
            y_ref = config["ref_func"](x_ref)
            ax.plot(x_ref, y_ref, 'k-', linewidth=3, alpha=0.9)
        
        # Axis labels
        ax.set_xlabel(config["x_label"], fontsize=FONT_SIZES["label"], fontweight='bold')
        ax.set_ylabel(config["y_label"], fontsize=FONT_SIZES["label"], fontweight='bold')
        ax.set_title(config["title"], fontsize=FONT_SIZES["title"], fontweight='bold')
        
        # Axis limits with margins
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Tick sizes
        ax.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cost', fontsize=FONT_SIZES["label"], rotation=270, labelpad=25)
        cbar.ax.tick_params(labelsize=FONT_SIZES["tick"])
        
        plt.tight_layout()
        
        # Save
        output_file = PRESENTATION_OUTPUT / f"{config['name']}_featured.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_file.name}")
    
    print("Featured plots completed!")


def generate_mega_plots():
    """Task 3: Generate 4 mega plots (3x4 grid) with remaining 12 elections."""
    print("\n" + "="*70)
    print("TASK 3: GENERATING MEGA PLOTS (3x4 grid, 12 elections)")
    print("="*70)
    
    # Get all elections except the featured one
    all_elections = get_all_elections()
    elections = [e for e in all_elections if e["name"] != FEATURED_ELECTION]
    
    print(f"Found {len(elections)} elections (excluding featured):")
    for e in elections:
        print(f"  - {e['display_name']}")
    
    if len(elections) != 12:
        print(f"Warning: Expected 12 elections, found {len(elections)}")
    
    for config in PLOT_CONFIGS:
        # Create 3x4 grid (wide layout: 3 rows, 4 columns)
        fig, axes = plt.subplots(3, 4, figsize=(24, 16))
        axes = axes.flatten()
        
        for idx, election in enumerate(elections[:12]):
            ax = axes[idx]
            
            # Load data
            if not election["data_file"].exists():
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=FONT_SIZES["subplot_label"])
                ax.set_title(election["display_name"], fontsize=FONT_SIZES["subplot_title"], fontweight='bold')
                continue
            
            df = pd.read_csv(election["data_file"])
            max_size = df['subset_size'].max()
            
            # Color mapping
            norm = plt.Normalize(vmin=0, vmax=max_size)
            cmap = cm.viridis
            
            # Raw scatter plot (no aggregation)
            scatter = ax.scatter(
                df[config["x_col"]], 
                df[config["y_col"]],
                c=df['subset_size'], 
                cmap=cmap, 
                norm=norm,
                alpha=0.35, 
                s=8
            )
            
            # Reference line (extended for margins) - only if defined
            if config["ref_func"] is not None:
                x_ref = np.linspace(-0.05, 1.05, 100)
                y_ref = config["ref_func"](x_ref)
                ax.plot(x_ref, y_ref, 'k-', linewidth=2, alpha=0.9)
            
            # Title (shortened)
            ax.set_title(election["display_name"], fontsize=FONT_SIZES["subplot_title"], fontweight='bold')
            
            # Axis labels (only on edges)
            if idx >= 8:  # Bottom row (3x4 grid: indices 8, 9, 10, 11)
                ax.set_xlabel(config["x_label"], fontsize=FONT_SIZES["subplot_label"])
            if idx % 4 == 0:  # Left column (3x4 grid: indices 0, 4, 8)
                ax.set_ylabel(config["y_label"], fontsize=FONT_SIZES["subplot_label"])
            
            # Axis limits with margins to see edge points
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            
            # Tick sizes
            ax.tick_params(axis='both', labelsize=FONT_SIZES["subplot_tick"])
            
            # Grid
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots (if any)
        for idx in range(len(elections), 12):
            axes[idx].set_visible(False)
        
        # Main title
        fig.suptitle(f'{config["title"]} - All Elections', 
                     fontsize=FONT_SIZES["title"] + 4, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save
        output_file = PRESENTATION_OUTPUT / f"{config['name']}_mega.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_file.name}")
    
    print("Mega plots completed!")


def generate_mega_plots_with_methods():
    """Task 4: Generate 4 mega plots (3x4 grid) with voting methods overlaid."""
    print("\n" + "="*70)
    print("TASK 4: GENERATING MEGA PLOTS WITH VOTING METHODS (3x4 grid, 12 elections)")
    print("="*70)
    
    # Get all elections except the featured one
    all_elections = get_all_elections()
    elections = [e for e in all_elections if e["name"] != FEATURED_ELECTION]
    
    print(f"Found {len(elections)} elections (excluding featured):")
    for e in elections:
        print(f"  - {e['display_name']}")
    
    if len(elections) != 12:
        print(f"Warning: Expected 12 elections, found {len(elections)}")
    
    for config in PLOT_CONFIGS:
        # Create 3x4 grid (wide layout: 3 rows, 4 columns)
        fig, axes = plt.subplots(3, 4, figsize=(24, 16))
        axes = axes.flatten()
        
        for idx, election in enumerate(elections[:12]):
            ax = axes[idx]
            
            # Load data
            if not election["data_file"].exists():
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=FONT_SIZES["subplot_label"])
                ax.set_title(election["display_name"], fontsize=FONT_SIZES["subplot_title"], fontweight='bold')
                continue
            
            df = pd.read_csv(election["data_file"])
            max_size = df['subset_size'].max()
            
            # Color mapping
            norm = plt.Normalize(vmin=0, vmax=max_size)
            cmap = cm.viridis
            
            # Raw scatter plot (no aggregation)
            scatter = ax.scatter(
                df[config["x_col"]], 
                df[config["y_col"]],
                c=df['subset_size'], 
                cmap=cmap, 
                norm=norm,
                alpha=0.35, 
                s=8
            )
            
            # Load and plot voting methods
            voting_file = election["path"] / "voting_results.csv"
            if voting_file.exists():
                methods_df = pd.read_csv(voting_file)
                for method_name, props in VOTING_METHODS.items():
                    method_data = methods_df[methods_df['method'] == method_name]
                    if len(method_data) > 0:
                        ax.scatter(
                            method_data[config["x_col"]], 
                            method_data[config["y_col"]],
                            marker=props['marker'], 
                            c=props['color'], 
                            s=props['size'],
                            edgecolors='black', 
                            linewidths=0.5,
                            zorder=10, 
                            alpha=0.9
                        )
            
            # Reference line (extended for margins) - only if defined
            if config["ref_func"] is not None:
                x_ref = np.linspace(-0.05, 1.05, 100)
                y_ref = config["ref_func"](x_ref)
                ax.plot(x_ref, y_ref, 'k-', linewidth=2, alpha=0.9)
            
            # Title (shortened)
            ax.set_title(election["display_name"], fontsize=FONT_SIZES["subplot_title"], fontweight='bold')
            
            # Axis labels (only on edges)
            if idx >= 8:  # Bottom row (3x4 grid: indices 8, 9, 10, 11)
                ax.set_xlabel(config["x_label"], fontsize=FONT_SIZES["subplot_label"])
            if idx % 4 == 0:  # Left column (3x4 grid: indices 0, 4, 8)
                ax.set_ylabel(config["y_label"], fontsize=FONT_SIZES["subplot_label"])
            
            # Axis limits with margins to see edge points
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            
            # Tick sizes
            ax.tick_params(axis='both', labelsize=FONT_SIZES["subplot_tick"])
            
            # Grid
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots (if any)
        for idx in range(len(elections), 12):
            axes[idx].set_visible(False)
        
        # Main title
        fig.suptitle(f'{config["title"]} - All Elections (with Voting Methods)', 
                     fontsize=FONT_SIZES["title"] + 4, fontweight='bold', y=0.995)
        
        # Add legend for voting methods
        legend_elements = [
            plt.scatter([], [], marker=props['marker'], c=props['color'], s=props['size'], 
                       edgecolors='black', linewidths=0.5, label=method_name)
            for method_name, props in VOTING_METHODS.items()
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
                   fontsize=FONT_SIZES["legend"], bbox_to_anchor=(0.5, 0.01))
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])
        
        # Save
        output_file = PRESENTATION_OUTPUT / f"{config['name']}_mega_methods.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_file.name}")
    
    print("Mega plots with voting methods completed!")


def generate_mega_plots_by_cost():
    """Task 5: Generate mega plots (3x4 grid) colored by total cost."""
    print("\n" + "="*70)
    print("TASK 5: GENERATING MEGA PLOTS BY COST (3x4 grid, 12 elections)")
    print("="*70)
    
    # Get all elections except the featured one
    all_elections = get_all_elections()
    elections = [e for e in all_elections if e["name"] != FEATURED_ELECTION]
    
    print(f"Found {len(elections)} elections (excluding featured):")
    for e in elections:
        print(f"  - {e['display_name']}")
    
    if len(elections) != 12:
        print(f"Warning: Expected 12 elections, found {len(elections)}")
    
    for config in PLOT_CONFIGS:
        # Create 3x4 grid (wide layout: 3 rows, 4 columns)
        fig, axes = plt.subplots(3, 4, figsize=(24, 16))
        axes = axes.flatten()
        
        for idx, election in enumerate(elections[:12]):
            ax = axes[idx]
            
            # Load data
            if not election["data_file"].exists():
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=FONT_SIZES["subplot_label"])
                ax.set_title(election["display_name"], fontsize=FONT_SIZES["subplot_title"], fontweight='bold')
                continue
            
            df = pd.read_csv(election["data_file"])
            
            # Get cost column - use total_cost if available, otherwise use subset_size
            if 'total_cost' in df.columns:
                cost_col = 'total_cost'
            else:
                cost_col = 'subset_size'  # For french_election and camp_songs (cost = 1 per candidate)
            
            max_cost = df[cost_col].max()
            
            # Color mapping by cost
            norm = plt.Normalize(vmin=0, vmax=max_cost)
            cmap = cm.viridis
            
            # Raw scatter plot (no aggregation)
            scatter = ax.scatter(
                df[config["x_col"]], 
                df[config["y_col"]],
                c=df[cost_col], 
                cmap=cmap, 
                norm=norm,
                alpha=0.35, 
                s=8
            )
            
            # Reference line (extended for margins) - only if defined
            if config["ref_func"] is not None:
                x_ref = np.linspace(-0.05, 1.05, 100)
                y_ref = config["ref_func"](x_ref)
                ax.plot(x_ref, y_ref, 'k-', linewidth=2, alpha=0.9)
            
            # Title (shortened)
            ax.set_title(election["display_name"], fontsize=FONT_SIZES["subplot_title"], fontweight='bold')
            
            # Axis labels (only on edges)
            if idx >= 8:  # Bottom row (3x4 grid: indices 8, 9, 10, 11)
                ax.set_xlabel(config["x_label"], fontsize=FONT_SIZES["subplot_label"])
            if idx % 4 == 0:  # Left column (3x4 grid: indices 0, 4, 8)
                ax.set_ylabel(config["y_label"], fontsize=FONT_SIZES["subplot_label"])
            
            # Axis limits with margins to see edge points
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            
            # Tick sizes
            ax.tick_params(axis='both', labelsize=FONT_SIZES["subplot_tick"])
            
            # Grid
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots (if any)
        for idx in range(len(elections), 12):
            axes[idx].set_visible(False)
        
        # Main title
        fig.suptitle(f'{config["title"]} - All Elections (colored by cost)', 
                     fontsize=FONT_SIZES["title"] + 4, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save
        output_file = PRESENTATION_OUTPUT / f"{config['name']}_mega_cost.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {output_file.name}")
    
    print("Mega plots by cost completed!")


def main():
    """Generate all presentation plots."""
    print("="*70)
    print("PRESENTATION PLOTS GENERATOR")
    print("="*70)
    
    # Ensure output directory exists
    PRESENTATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {PRESENTATION_OUTPUT}")
    
    # Generate all plots
    generate_empty_plots()
    generate_featured_plots()
    generate_mega_plots()
    generate_mega_plots_with_methods()
    generate_mega_plots_by_cost()
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput files in: {PRESENTATION_OUTPUT}")
    for f in sorted(PRESENTATION_OUTPUT.iterdir()):
        if f.suffix == '.png':
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()

