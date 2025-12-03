"""
Generate summary tables for voting methods performance.

Creates a markdown file with tables showing:
- Proportion of elections achieving α = 1.0
- Mean α values
- Median α values  
- Minimum α values

Only includes PB (participatory budgeting) instances.
"""

import pandas as pd
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
PRESENTATION_OUTPUT = Path(__file__).parent.parent / "outputs"

# Voting methods and metrics to analyze
VOTING_METHODS = ['MES', 'AV', 'greedy-CC', 'greedy-PAV']
METRICS = ['alpha_AV', 'alpha_CC', 'alpha_PAIRS', 'alpha_CONS', 'alpha_EJR']


def load_pb_voting_results():
    """Load voting results from all PB elections, merging voting_results.csv and ejr_data.csv."""
    all_results = []
    
    pb_path = OUTPUT_DIR / "pb"
    if pb_path.exists():
        for election_dir in sorted(pb_path.iterdir()):
            if election_dir.is_dir():
                vr_file = election_dir / "voting_results.csv"
                ejr_file = election_dir / "ejr_data.csv"
                
                if vr_file.exists():
                    df = pd.read_csv(vr_file)
                    df['election'] = election_dir.name
                    
                    # Merge alpha_EJR from ejr_data.csv if available
                    if ejr_file.exists():
                        ejr_df = pd.read_csv(ejr_file)
                        # Merge on method
                        df = df.merge(ejr_df[['method', 'alpha_EJR']], on='method', how='left')
                    else:
                        df['alpha_EJR'] = None
                    
                    all_results.append(df)
    
    if not all_results:
        raise ValueError("No PB voting results found")
    
    return pd.concat(all_results, ignore_index=True)


def generate_proportion_table(combined):
    """Generate table showing proportion of elections achieving α = 1.0."""
    lines = [
        "## Proportion of PB Elections Achieving α = 1.0",
        "",
        "| Method | α_AV = 1.0 | α_CC = 1.0 | α_PAIRS = 1.0 | α_CONS = 1.0 | α_EJR = 1.0 |",
        "|--------|------------|------------|---------------|--------------|-------------|",
    ]
    
    for method in VOTING_METHODS:
        method_data = combined[combined['method'] == method]
        row = f"| {method} |"
        for metric in METRICS:
            count_1 = (method_data[metric] == 1.0).sum()
            total = len(method_data)
            prop = count_1 / total if total > 0 else 0
            row += f" {count_1}/{total} ({prop:.1%}) |"
        lines.append(row)
    
    return "\n".join(lines)


def generate_mean_table(combined):
    """Generate table showing mean α values."""
    lines = [
        "## Mean α Values",
        "",
        "| Method | α_AV | α_CC | α_PAIRS | α_CONS | α_EJR |",
        "|--------|------|------|---------|--------|-------|",
    ]
    
    for method in VOTING_METHODS:
        method_data = combined[combined['method'] == method]
        row = f"| {method} |"
        for metric in METRICS:
            mean_val = method_data[metric].mean()
            row += f" {mean_val:.3f} |"
        lines.append(row)
    
    return "\n".join(lines)


def generate_median_table(combined):
    """Generate table showing median α values."""
    lines = [
        "## Median α Values",
        "",
        "| Method | α_AV | α_CC | α_PAIRS | α_CONS | α_EJR |",
        "|--------|------|------|---------|--------|-------|",
    ]
    
    for method in VOTING_METHODS:
        method_data = combined[combined['method'] == method]
        row = f"| {method} |"
        for metric in METRICS:
            median_val = method_data[metric].median()
            row += f" {median_val:.3f} |"
        lines.append(row)
    
    return "\n".join(lines)


def generate_min_table(combined):
    """Generate table showing minimum α values."""
    lines = [
        "## Minimum α Values",
        "",
        "| Method | α_AV | α_CC | α_PAIRS | α_CONS | α_EJR |",
        "|--------|------|------|---------|--------|-------|",
    ]
    
    for method in VOTING_METHODS:
        method_data = combined[combined['method'] == method]
        row = f"| {method} |"
        for metric in METRICS:
            min_val = method_data[metric].min()
            row += f" {min_val:.3f} |"
        lines.append(row)
    
    return "\n".join(lines)


def main():
    """Generate voting methods summary markdown file."""
    print("="*70)
    print("GENERATING VOTING METHODS SUMMARY TABLE")
    print("="*70)
    
    # Load data
    print("\nLoading PB voting results...")
    combined = load_pb_voting_results()
    
    total_elections = combined['election'].nunique()
    rows_per_method = len(combined[combined['method'] == VOTING_METHODS[0]])
    print(f"Total PB elections: {total_elections}")
    print(f"Rows per method: {rows_per_method}")
    
    # Generate markdown content
    md_content = "# Voting Methods Performance Summary (PB Instances Only)\n\n"
    md_content += generate_proportion_table(combined) + "\n\n"
    md_content += generate_mean_table(combined) + "\n\n"
    md_content += generate_median_table(combined) + "\n\n"
    md_content += generate_min_table(combined) + "\n"
    
    # Save to file
    PRESENTATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = PRESENTATION_OUTPUT / "voting_methods_summary.md"
    output_path.write_text(md_content)
    
    print(f"\nSaved to: {output_path}")
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()

