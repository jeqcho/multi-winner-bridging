"""
Analyze the relationship between Set A and Set B from Warsaw 2017 Wawrzyszew.

Checks:
1. Match rate if candidate 0 is removed from Set B
2. % of subsets in A containing candidate 0
3. % of subsets in B containing candidate 0

Outputs a markdown report to output/warsaw-2017/reports/
"""

import pandas as pd
import numpy as np
import ast
import sys
from pathlib import Path

# Add src to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from pb_data_loader import load_pb_file

# Paths
PB_FILE = BASE_DIR / "data" / "pb_selected_10_20251202_023743" / "poland_warszawa_2017_wawrzyszew.pb"
FULL_SCORES_FILE = BASE_DIR / "output" / "pb" / "poland_warszawa_2017_wawrzyszew" / "alpha_scores.csv"
SET_A_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_A.csv"
SET_B_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_B.csv"
SET_A2_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_A2.csv"
SET_B2_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_B2.csv"
SET_C_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_C.csv"
SET_D_FILE = BASE_DIR / "output" / "warsaw-2017" / "set_D.csv"
REPORT_DIR = BASE_DIR / "output" / "warsaw-2017" / "reports"


def parse_subset(subset_str):
    """Parse subset_indices string to a set of integers."""
    return set(ast.literal_eval(subset_str))


def main():
    # Ensure report directory exists
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load PB data for general info
    print("Loading PB data...")
    M, project_ids, project_costs, budget = load_pb_file(str(PB_FILE))
    n_voters, n_candidates = M.shape
    total_approvals = int(M.sum())
    avg_approvals_per_voter = M.sum(axis=1).mean()
    
    # Compute per-project statistics
    project_approvals = M.sum(axis=0)  # Number of approvals per project
    project_approval_pct = 100 * project_approvals / n_voters
    
    # Find voters who approve only one project
    voter_approval_counts = M.sum(axis=1)  # Number of approvals per voter
    single_approval_voters = (voter_approval_counts == 1)
    
    # For each project, count voters who ONLY approve that project
    single_approval_by_project = []
    for c in range(n_candidates):
        # Voters who approve this project AND approve only one project total
        count = int((M[:, c] & single_approval_voters).sum())
        single_approval_by_project.append(count)
    
    total_single_approval = int(single_approval_voters.sum())
    
    # Load full scores to get max values for alpha conversion
    print("Loading full scores for max values...")
    df_full = pd.read_csv(FULL_SCORES_FILE)
    max_cons = df_full['CONS'].max()
    max_av = df_full['AV'].max()
    
    # Load data
    print("Loading Set A, B, A2, B2, C, D...")
    df_a = pd.read_csv(SET_A_FILE)
    df_b = pd.read_csv(SET_B_FILE)
    df_a2 = pd.read_csv(SET_A2_FILE)
    df_b2 = pd.read_csv(SET_B2_FILE)
    df_c = pd.read_csv(SET_C_FILE)
    df_d = pd.read_csv(SET_D_FILE)
    
    print(f"Set A: {len(df_a)} committees")
    print(f"Set B: {len(df_b)} committees")
    print(f"Set A2: {len(df_a2)} committees")
    print(f"Set B2: {len(df_b2)} committees")
    print(f"Set C: {len(df_c)} committees")
    print(f"Set D: {len(df_d)} committees")
    
    # Parse subsets
    subsets_a = [parse_subset(s) for s in df_a["subset_indices"]]
    subsets_b = [parse_subset(s) for s in df_b["subset_indices"]]
    subsets_a2 = [parse_subset(s) for s in df_a2["subset_indices"]]
    subsets_b2 = [parse_subset(s) for s in df_b2["subset_indices"]]
    subsets_c = [parse_subset(s) for s in df_c["subset_indices"]]
    subsets_d = [parse_subset(s) for s in df_d["subset_indices"]]
    
    # Count project appearances in each set
    project_count_a = [0] * n_candidates
    project_count_b = [0] * n_candidates
    project_count_a2 = [0] * n_candidates
    project_count_b2 = [0] * n_candidates
    project_count_c = [0] * n_candidates
    project_count_d = [0] * n_candidates
    
    for subset in subsets_a:
        for c in subset:
            project_count_a[c] += 1
    
    for subset in subsets_b:
        for c in subset:
            project_count_b[c] += 1
    
    for subset in subsets_a2:
        for c in subset:
            project_count_a2[c] += 1
    
    for subset in subsets_b2:
        for c in subset:
            project_count_b2[c] += 1
    
    for subset in subsets_c:
        for c in subset:
            project_count_c[c] += 1
    
    for subset in subsets_d:
        for c in subset:
            project_count_d[c] += 1
    
    # Convert to frozensets for set operations
    subsets_a_frozen = set(frozenset(s) for s in subsets_a)
    
    # Check % of subsets containing candidate 0
    a_with_0 = sum(1 for s in subsets_a if 0 in s)
    b_with_0 = sum(1 for s in subsets_b if 0 in s)
    
    a_with_0_pct = 100 * a_with_0 / len(subsets_a)
    b_with_0_pct = 100 * b_with_0 / len(subsets_b)
    
    print(f"\n=== Candidate 0 Presence ===")
    print(f"Set A with candidate 0: {a_with_0}/{len(subsets_a)} ({a_with_0_pct:.1f}%)")
    print(f"Set B with candidate 0: {b_with_0}/{len(subsets_b)} ({b_with_0_pct:.1f}%)")
    
    # Remove candidate 0 from Set B and check matches with Set A
    subsets_b_without_0 = [s - {0} for s in subsets_b]
    subsets_b_without_0_frozen = [frozenset(s) for s in subsets_b_without_0]
    
    # Count matches
    matches = sum(1 for s in subsets_b_without_0_frozen if s in subsets_a_frozen)
    matches_pct = 100 * matches / len(subsets_b)
    
    # Also check how many unique A subsets are covered
    matched_a_subsets = set(s for s in subsets_b_without_0_frozen if s in subsets_a_frozen)
    unique_matched_pct = 100 * len(matched_a_subsets) / len(subsets_a_frozen)
    
    print(f"\n=== Match Analysis (B minus candidate 0 → A) ===")
    print(f"Set B committees that match Set A after removing candidate 0: {matches}/{len(subsets_b)} ({matches_pct:.1f}%)")
    print(f"Unique Set A committees matched: {len(matched_a_subsets)}/{len(subsets_a_frozen)} ({unique_matched_pct:.1f}%)")
    
    # Compute score statistics
    # Set A
    a_cons_min, a_cons_max, a_cons_mean = df_a['CONS'].min(), df_a['CONS'].max(), df_a['CONS'].mean()
    a_av_min, a_av_max, a_av_mean = df_a['AV'].min(), df_a['AV'].max(), df_a['AV'].mean()
    a_alpha_cons_min, a_alpha_cons_max = df_a['alpha_CONS'].min(), df_a['alpha_CONS'].max()
    a_alpha_av_min, a_alpha_av_max = df_a['alpha_AV'].min(), df_a['alpha_AV'].max()
    
    # Set B
    b_cons_min, b_cons_max, b_cons_mean = df_b['CONS'].min(), df_b['CONS'].max(), df_b['CONS'].mean()
    b_av_min, b_av_max, b_av_mean = df_b['AV'].min(), df_b['AV'].max(), df_b['AV'].mean()
    b_alpha_cons_min, b_alpha_cons_max = df_b['alpha_CONS'].min(), df_b['alpha_CONS'].max()
    b_alpha_av_min, b_alpha_av_max = df_b['alpha_AV'].min(), df_b['alpha_AV'].max()
    
    # Set A2
    a2_cons_min, a2_cons_max, a2_cons_mean = df_a2['CONS'].min(), df_a2['CONS'].max(), df_a2['CONS'].mean()
    a2_av_min, a2_av_max, a2_av_mean = df_a2['AV'].min(), df_a2['AV'].max(), df_a2['AV'].mean()
    a2_alpha_cons_min, a2_alpha_cons_max = df_a2['alpha_CONS'].min(), df_a2['alpha_CONS'].max()
    a2_alpha_av_min, a2_alpha_av_max = df_a2['alpha_AV'].min(), df_a2['alpha_AV'].max()
    
    # Set B2
    b2_cons_min, b2_cons_max, b2_cons_mean = df_b2['CONS'].min(), df_b2['CONS'].max(), df_b2['CONS'].mean()
    b2_av_min, b2_av_max, b2_av_mean = df_b2['AV'].min(), df_b2['AV'].max(), df_b2['AV'].mean()
    b2_alpha_cons_min, b2_alpha_cons_max = df_b2['alpha_CONS'].min(), df_b2['alpha_CONS'].max()
    b2_alpha_av_min, b2_alpha_av_max = df_b2['alpha_AV'].min(), df_b2['alpha_AV'].max()
    
    # Set C
    c_cons_min, c_cons_max, c_cons_mean = df_c['CONS'].min(), df_c['CONS'].max(), df_c['CONS'].mean()
    c_av_min, c_av_max, c_av_mean = df_c['AV'].min(), df_c['AV'].max(), df_c['AV'].mean()
    c_alpha_cons_min, c_alpha_cons_max = df_c['alpha_CONS'].min(), df_c['alpha_CONS'].max()
    c_alpha_av_min, c_alpha_av_max = df_c['alpha_AV'].min(), df_c['alpha_AV'].max()
    
    # Set D
    d_cons_min, d_cons_max, d_cons_mean = df_d['CONS'].min(), df_d['CONS'].max(), df_d['CONS'].mean()
    d_av_min, d_av_max, d_av_mean = df_d['AV'].min(), df_d['AV'].max(), df_d['AV'].mean()
    d_alpha_cons_min, d_alpha_cons_max = df_d['alpha_CONS'].min(), df_d['alpha_CONS'].max()
    d_alpha_av_min, d_alpha_av_max = df_d['alpha_AV'].min(), df_d['alpha_AV'].max()
    
    # Max possible scores
    max_pairs = n_voters * (n_voters - 1) // 2
    
    # Compute alpha threshold conversions
    # Set A: CONS 0.4-0.6, AV 0.5-0.9
    a_cons_thresh_low = 0.4 * max_cons
    a_cons_thresh_high = 0.6 * max_cons
    a_av_thresh_low = 0.5 * max_av
    a_av_thresh_high = 0.9 * max_av
    
    # Set B: CONS 0.8-1.0, AV 0.6-1.0
    b_cons_thresh_low = 0.8 * max_cons
    b_cons_thresh_high = 1.0 * max_cons
    b_av_thresh_low = 0.6 * max_av
    b_av_thresh_high = 1.0 * max_av
    
    # Write markdown report
    report_path = REPORT_DIR / "analysis.md"
    report_content = f"""# Warsaw 2017 Wawrzyszew Committee Analysis

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | `poland_warszawa_2017_wawrzyszew.pb` |
| Number of Voters | {n_voters:,} |
| Number of Candidates | {n_candidates} |
| Total Budget | {budget:,} |
| Total Approvals | {total_approvals:,} |
| Avg Approvals per Voter | {avg_approvals_per_voter:.2f} |
| Max Possible Voter Pairs | {max_pairs:,} |
| Max CONS in dataset | {max_cons:,} |
| Max AV in dataset | {max_av:,} |
| Single-approval voters | {total_single_approval:,} ({100*total_single_approval/n_voters:.1f}%) |

## Project Statistics

| Project | Cost | Approvals | % of Voters | Single-Only Voters | % of Total |
|---------|------|-----------|-------------|-------------------|------------|
"""
    
    # Add project rows
    for c in range(n_candidates):
        pct_voters = project_approval_pct[c]
        single_only = single_approval_by_project[c]
        pct_total = 100 * single_only / n_voters
        report_content += f"| {c} | {project_costs[c]:,} | {int(project_approvals[c]):,} | {pct_voters:.1f}% | {single_only:,} | {pct_total:.1f}% |\n"
    
    report_content += f"""
## Project Appearances in Filtered Sets

| Project | In A | % of A | In B | % of B | In A2 | % of A2 | In B2 | % of B2 | In C | % of C | In D | % of D |
|---------|------|--------|------|--------|-------|---------|-------|---------|------|--------|------|--------|
"""
    
    # Add project appearance rows
    for c in range(n_candidates):
        pct_a = 100 * project_count_a[c] / len(subsets_a) if len(subsets_a) > 0 else 0
        pct_b = 100 * project_count_b[c] / len(subsets_b) if len(subsets_b) > 0 else 0
        pct_a2 = 100 * project_count_a2[c] / len(subsets_a2) if len(subsets_a2) > 0 else 0
        pct_b2 = 100 * project_count_b2[c] / len(subsets_b2) if len(subsets_b2) > 0 else 0
        pct_c = 100 * project_count_c[c] / len(subsets_c) if len(subsets_c) > 0 else 0
        pct_d = 100 * project_count_d[c] / len(subsets_d) if len(subsets_d) > 0 else 0
        report_content += f"| {c} | {project_count_a[c]:,} | {pct_a:.1f}% | {project_count_b[c]:,} | {pct_b:.1f}% | {project_count_a2[c]:,} | {pct_a2:.1f}% | {project_count_b2[c]:,} | {pct_b2:.1f}% | {project_count_c[c]:,} | {pct_c:.1f}% | {project_count_d[c]:,} | {pct_d:.1f}% |\n"
    
    report_content += f"""
## Filtered Sets

| Set | Description | α_CONS Range | α_AV Range | Committees |
|-----|-------------|--------------|------------|------------|
| A  | Lower CONS | 0.4 - 0.6  | 0.5 - 0.9 | {len(df_a)} |
| B  | Higher CONS | 0.8 - 1.0  | 0.6 - 1.0 | {len(df_b)} |
| A2 | A without project 0 | 0.4 - 0.6  | 0.5 - 0.9 | {len(df_a2)} |
| B2 | B with project 0 only | 0.8 - 1.0  | 0.6 - 1.0 | {len(df_b2)} |
| C  | All without project 0 | any | any | {len(df_c)} |
| D  | All with project 0 | any | any | {len(df_d)} |

### Alpha Threshold Conversions

| Threshold | α Value | Raw Score |
|-----------|---------|-----------|
| Set A CONS min | 0.4 | {a_cons_thresh_low:,.0f} |
| Set A CONS max | 0.6 | {a_cons_thresh_high:,.0f} |
| Set A AV min | 0.5 | {a_av_thresh_low:,.0f} |
| Set A AV max | 0.9 | {a_av_thresh_high:,.0f} |
| Set B CONS min | 0.8 | {b_cons_thresh_low:,.0f} |
| Set B CONS max | 1.0 | {b_cons_thresh_high:,.0f} |
| Set B AV min | 0.6 | {b_av_thresh_low:,.0f} |
| Set B AV max | 1.0 | {b_av_thresh_high:,.0f} |

## Score Statistics

### Set A (Lower CONS)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | {a_cons_min:,} | {a_cons_max:,} | {a_cons_mean:,.0f} |
| AV (raw) | {a_av_min:,} | {a_av_max:,} | {a_av_mean:,.0f} |
| α_CONS | {a_alpha_cons_min:.3f} | {a_alpha_cons_max:.3f} | - |
| α_AV | {a_alpha_av_min:.3f} | {a_alpha_av_max:.3f} | - |

### Set B (Higher CONS)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | {b_cons_min:,} | {b_cons_max:,} | {b_cons_mean:,.0f} |
| AV (raw) | {b_av_min:,} | {b_av_max:,} | {b_av_mean:,.0f} |
| α_CONS | {b_alpha_cons_min:.3f} | {b_alpha_cons_max:.3f} | - |
| α_AV | {b_alpha_av_min:.3f} | {b_alpha_av_max:.3f} | - |

### Set A2 (A without project 0)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | {a2_cons_min:,} | {a2_cons_max:,} | {a2_cons_mean:,.0f} |
| AV (raw) | {a2_av_min:,} | {a2_av_max:,} | {a2_av_mean:,.0f} |
| α_CONS | {a2_alpha_cons_min:.3f} | {a2_alpha_cons_max:.3f} | - |
| α_AV | {a2_alpha_av_min:.3f} | {a2_alpha_av_max:.3f} | - |

### Set B2 (B with project 0)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | {b2_cons_min:,} | {b2_cons_max:,} | {b2_cons_mean:,.0f} |
| AV (raw) | {b2_av_min:,} | {b2_av_max:,} | {b2_av_mean:,.0f} |
| α_CONS | {b2_alpha_cons_min:.3f} | {b2_alpha_cons_max:.3f} | - |
| α_AV | {b2_alpha_av_min:.3f} | {b2_alpha_av_max:.3f} | - |

### Set C (All without project 0)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | {c_cons_min:,} | {c_cons_max:,} | {c_cons_mean:,.0f} |
| AV (raw) | {c_av_min:,} | {c_av_max:,} | {c_av_mean:,.0f} |
| α_CONS | {c_alpha_cons_min:.3f} | {c_alpha_cons_max:.3f} | - |
| α_AV | {c_alpha_av_min:.3f} | {c_alpha_av_max:.3f} | - |

### Set D (All with project 0)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | {d_cons_min:,} | {d_cons_max:,} | {d_cons_mean:,.0f} |
| AV (raw) | {d_av_min:,} | {d_av_max:,} | {d_av_mean:,.0f} |
| α_CONS | {d_alpha_cons_min:.3f} | {d_alpha_cons_max:.3f} | - |
| α_AV | {d_alpha_av_min:.3f} | {d_alpha_av_max:.3f} | - |

## Candidate 0 Presence

| Set | With Candidate 0 | Total | Percentage |
|-----|------------------|-------|------------|
| A   | {a_with_0} | {len(subsets_a)} | {a_with_0_pct:.1f}% |
| B   | {b_with_0} | {len(subsets_b)} | {b_with_0_pct:.1f}% |

## Match Analysis

When removing candidate 0 from Set B committees:

- **{matches_pct:.1f}%** of Set B committees ({matches}/{len(subsets_b)}) match a Set A committee
- **{unique_matched_pct:.1f}%** of unique Set A committees ({len(matched_a_subsets)}/{len(subsets_a_frozen)}) are covered by this matching

## Interpretation

Candidate 0 appears to be a "consensus-building" candidate that significantly increases CONS scores. It is present in {b_with_0_pct:.1f}% of Set B (high CONS) but only {a_with_0_pct:.1f}% of Set A (moderate CONS). Removing candidate 0 from Set B reveals that most of these committees share their remaining structure with Set A committees.
"""
    
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()

