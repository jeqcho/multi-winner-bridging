"""
Script to create a summary table of minimum alpha scores observed for each voting method.
Rows: voting methods (MES, AV, greedy-AV/cost, greedy-AV/cost^2, greedy-CC, greedy-PAV, PAIRS-AV, PAIRS-CC, CONS-AV, CONS-CC)
Columns: metrics (alpha_AV, alpha_CC, alpha_PAIRS, alpha_CONS, alpha_EJR)
Cell values: minimum value observed across all files and k values
"""

import pandas as pd
from pathlib import Path


def main():
    output_dir = Path("output")
    
    # Find all voting_results.csv and ejr_data.csv files
    voting_files = list(output_dir.rglob("voting_results.csv"))
    ejr_files = list(output_dir.rglob("ejr_data.csv"))
    
    # Methods to analyze
    methods = ["MES", "AV", "greedy-AV/cost", "greedy-AV/cost^2", "greedy-CC", "greedy-PAV", "PAIRS-AV", "PAIRS-CC", "CONS-AV", "CONS-CC"]
    
    # Metrics from voting_results.csv
    voting_metrics = ["alpha_AV", "alpha_CC", "alpha_PAIRS", "alpha_CONS"]
    
    # Initialize results dictionary
    results = {method: {metric: [] for metric in voting_metrics + ["alpha_EJR"]} for method in methods}
    
    # Collect all alpha values from voting_results.csv files
    for csv_path in voting_files:
        df = pd.read_csv(csv_path)
        for method in methods:
            method_df = df[df["method"] == method]
            for metric in voting_metrics:
                if metric in method_df.columns:
                    results[method][metric].extend(method_df[metric].tolist())
    
    # Collect all alpha_EJR values from ejr_data.csv files
    for csv_path in ejr_files:
        df = pd.read_csv(csv_path)
        for method in methods:
            method_df = df[df["method"] == method]
            if "alpha_EJR" in method_df.columns:
                results[method]["alpha_EJR"].extend(method_df["alpha_EJR"].tolist())
    
    # Compute minimum values
    min_results = {}
    for method in methods:
        min_results[method] = {}
        for metric in voting_metrics + ["alpha_EJR"]:
            values = results[method][metric]
            if values:
                min_results[method][metric] = min(values)
            else:
                min_results[method][metric] = None
    
    # Create DataFrame
    df_result = pd.DataFrame(min_results).T
    df_result = df_result[voting_metrics + ["alpha_EJR"]]  # Reorder columns
    
    # Print table
    print("=" * 100)
    print("MINIMUM ALPHA SCORES OBSERVED FOR EACH VOTING METHOD")
    print("=" * 100)
    print()
    
    # Print as formatted table
    print(df_result.to_string(float_format=lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"))
    print()
    
    # Print as markdown table
    print("\nMarkdown format:")
    print("-" * 100)
    header = "| Method | " + " | ".join(voting_metrics + ["alpha_EJR"]) + " |"
    separator = "|--------|" + "|".join(["---------"] * (len(voting_metrics) + 1)) + "|"
    print(header)
    print(separator)
    for method in methods:
        row_values = []
        for metric in voting_metrics + ["alpha_EJR"]:
            val = min_results[method][metric]
            if val is not None:
                if val == 1.0:
                    row_values.append("1.0")
                else:
                    row_values.append(f"{val:.4f}")
            else:
                row_values.append("N/A")
        print(f"| {method} | " + " | ".join(row_values) + " |")
    
    # Save to CSV
    output_path = output_dir / "min_alpha_summary.csv"
    df_result.to_csv(output_path)
    print(f"\nSaved to {output_path}")
    
    # Also print summary of which methods achieve 1.0 for all k
    print("\n" + "=" * 100)
    print("METHODS THAT ALWAYS ACHIEVE 1.0:")
    print("=" * 100)
    for metric in voting_metrics + ["alpha_EJR"]:
        perfect_methods = [m for m in methods if min_results[m][metric] == 1.0]
        if perfect_methods:
            print(f"{metric}: {', '.join(perfect_methods)}")
        else:
            print(f"{metric}: None")


if __name__ == "__main__":
    main()

