"""
Plot CONS vs AV for Set C (without project 0) and Set D (with project 0).

C: Red (all committees without project 0)
D: Blue (all committees with project 0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
SET_C_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_C.csv"
SET_D_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_D.csv"
OUTPUT_DIR = BASE_DIR / "output" / "warsaw-2017" / "reports"

# Font sizes
FONT_SIZES = {
    "title": 24,
    "label": 20,
    "tick": 16,
    "legend": 14,
}


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading Set C and Set D...")
    df_c = pd.read_csv(SET_C_FILE)
    df_d = pd.read_csv(SET_D_FILE)
    
    print(f"Set C (without project 0): {len(df_c):,} committees")
    print(f"Set D (with project 0): {len(df_d):,} committees")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Set C (red) - without project 0
    ax.scatter(
        df_c["alpha_CONS"],
        df_c["alpha_AV"],
        c='red',
        alpha=0.35,
        s=15,
        label=f'Without project 0 (n={len(df_c):,})'
    )
    
    # Plot Set D (blue) - with project 0
    ax.scatter(
        df_d["alpha_CONS"],
        df_d["alpha_AV"],
        c='blue',
        alpha=0.35,
        s=15,
        label=f'With project 0 (n={len(df_d):,})'
    )
    
    # Reference line: a² + b = 1
    x_ref = np.linspace(-0.05, 1.05, 100)
    y_ref = 1 - x_ref**2
    ax.plot(x_ref, y_ref, 'k-', linewidth=3, alpha=0.9, label='α² + β = 1')
    
    # Axis labels
    ax.set_xlabel('α_CONS', fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_ylabel('α_AV', fontsize=FONT_SIZES["label"], fontweight='bold')
    ax.set_title('CONS vs AV - Warsaw 2017 Wawrzyszew\nBy Project 0 Presence', fontsize=FONT_SIZES["title"], fontweight='bold')
    
    # Axis limits with margins
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Tick sizes
    ax.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(fontsize=FONT_SIZES["legend"], loc='lower left')
    
    plt.tight_layout()
    
    # Save plot
    output_path = OUTPUT_DIR / "cons_vs_av_c_d.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()

