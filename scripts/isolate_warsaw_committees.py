"""
Isolate committees from Warsaw 2017 Wawrzyszew based on alpha score criteria.

Set A: CONS 0.4-0.6, AV 0.5-0.9
Set B: CONS 0.8-1.0, AV 0.6-1.0
Set A2: Set A without project 0
Set B2: Set B with project 0 only
Set C: All committees without project 0
Set D: All committees with project 0
"""

import pandas as pd
import ast
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "output" / "pb" / "poland_warszawa_2017_wawrzyszew" / "alpha_scores.csv"
OUTPUT_DIR = BASE_DIR / "output" / "warsaw-2017"


def contains_project(subset_str, project_id):
    """Check if a subset contains a specific project."""
    subset = set(ast.literal_eval(subset_str))
    return project_id in subset


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} committees")
    
    # Filter Set A: CONS 0.4-0.6, AV 0.5-0.9
    set_a = df[
        (df["alpha_CONS"] >= 0.4) & (df["alpha_CONS"] <= 0.6) &
        (df["alpha_AV"] >= 0.5) & (df["alpha_AV"] <= 0.9)
    ]
    
    # Filter Set B: CONS 0.8-1.0, AV 0.6-1.0
    set_b = df[
        (df["alpha_CONS"] >= 0.8) & (df["alpha_CONS"] <= 1.0) &
        (df["alpha_AV"] >= 0.6) & (df["alpha_AV"] <= 1.0)
    ]
    
    # Filter Set A2: Set A without project 0
    set_a2 = set_a[~set_a["subset_indices"].apply(lambda x: contains_project(x, 0))]
    
    # Filter Set B2: Set B with project 0 only
    set_b2 = set_b[set_b["subset_indices"].apply(lambda x: contains_project(x, 0))]
    
    # Filter Set C: All committees without project 0
    set_c = df[~df["subset_indices"].apply(lambda x: contains_project(x, 0))]
    
    # Filter Set D: All committees with project 0
    set_d = df[df["subset_indices"].apply(lambda x: contains_project(x, 0))]
    
    # Save results
    set_a_path = OUTPUT_DIR / "set_A.csv"
    set_b_path = OUTPUT_DIR / "set_B.csv"
    set_a2_path = OUTPUT_DIR / "set_A2.csv"
    set_b2_path = OUTPUT_DIR / "set_B2.csv"
    set_c_path = OUTPUT_DIR / "set_C.csv"
    set_d_path = OUTPUT_DIR / "set_D.csv"
    
    set_a.to_csv(set_a_path, index=False)
    set_b.to_csv(set_b_path, index=False)
    set_a2.to_csv(set_a2_path, index=False)
    set_b2.to_csv(set_b2_path, index=False)
    set_c.to_csv(set_c_path, index=False)
    set_d.to_csv(set_d_path, index=False)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Set A (CONS 0.4-0.6, AV 0.5-0.9): {len(set_a):,} committees")
    print(f"  Saved to: {set_a_path}")
    print(f"Set B (CONS 0.8-1.0, AV 0.6-1.0): {len(set_b):,} committees")
    print(f"  Saved to: {set_b_path}")
    print(f"Set A2 (Set A without project 0): {len(set_a2):,} committees")
    print(f"  Saved to: {set_a2_path}")
    print(f"Set B2 (Set B with project 0): {len(set_b2):,} committees")
    print(f"  Saved to: {set_b2_path}")
    print(f"Set C (All without project 0): {len(set_c):,} committees")
    print(f"  Saved to: {set_c_path}")
    print(f"Set D (All with project 0): {len(set_d):,} committees")
    print(f"  Saved to: {set_d_path}")


if __name__ == "__main__":
    main()

