"""
Script to check if hybrid voting methods achieve their expected alpha scores:
- PAIRS-AV should achieve alpha_AV = 1 and alpha_PAIRS = 1
- PAIRS-CC should achieve alpha_CC = 1 and alpha_PAIRS = 1
- CONS-AV should achieve alpha_AV = 1 and alpha_CONS = 1
- CONS-CC should achieve alpha_CC = 1 and alpha_CONS = 1
"""

import pandas as pd
from pathlib import Path


def check_alpha_scores(csv_path: Path) -> dict:
    """Check alpha scores for hybrid methods in a voting results CSV."""
    df = pd.read_csv(csv_path)
    
    results = {
        "file": str(csv_path),
        "violations": [],
        "summary": {}
    }
    
    # Define expected alpha = 1 conditions for each method
    expected_conditions = {
        "PAIRS-AV": ["alpha_AV", "alpha_PAIRS"],
        "PAIRS-CC": ["alpha_CC", "alpha_PAIRS"],
        "CONS-AV": ["alpha_AV", "alpha_CONS"],
        "CONS-CC": ["alpha_CC", "alpha_CONS"],
    }
    
    for method, alpha_cols in expected_conditions.items():
        method_df = df[df["method"] == method]
        
        for _, row in method_df.iterrows():
            k = row["subset_size"]
            for alpha_col in alpha_cols:
                alpha_value = row[alpha_col]
                if alpha_value != 1.0:
                    results["violations"].append({
                        "method": method,
                        "k": k,
                        "alpha_column": alpha_col,
                        "alpha_value": alpha_value,
                        "subset": row["subset_indices"]
                    })
    
    # Create summary for each method
    for method, alpha_cols in expected_conditions.items():
        method_df = df[df["method"] == method]
        all_k_values = sorted(method_df["subset_size"].unique())
        
        method_summary = {}
        for alpha_col in alpha_cols:
            # Check which k values have alpha = 1
            perfect_k = []
            imperfect_k = []
            for k in all_k_values:
                k_row = method_df[method_df["subset_size"] == k].iloc[0]
                if k_row[alpha_col] == 1.0:
                    perfect_k.append(k)
                else:
                    imperfect_k.append((k, k_row[alpha_col]))
            
            method_summary[alpha_col] = {
                "perfect_k": perfect_k,
                "imperfect_k": imperfect_k,
                "always_1": len(imperfect_k) == 0
            }
        
        results["summary"][method] = method_summary
    
    return results


def main():
    # Find all voting_results.csv files in output/
    output_dir = Path("output")
    csv_files = list(output_dir.rglob("voting_results.csv"))
    
    print("=" * 80)
    print("CHECKING ALPHA SCORES FOR HYBRID VOTING METHODS")
    print("=" * 80)
    
    all_results = []
    
    for csv_path in sorted(csv_files):
        results = check_alpha_scores(csv_path)
        all_results.append(results)
        
        print(f"\n{'='*80}")
        print(f"FILE: {csv_path}")
        print("=" * 80)
        
        for method, summary in results["summary"].items():
            print(f"\n{method}:")
            for alpha_col, data in summary.items():
                status = "✓ ALWAYS 1.0" if data["always_1"] else "✗ NOT ALWAYS 1.0"
                print(f"  {alpha_col}: {status}")
                if not data["always_1"]:
                    for k, val in data["imperfect_k"]:
                        print(f"    - k={k}: {val:.6f}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    expected_conditions = {
        "PAIRS-AV": ["alpha_AV", "alpha_PAIRS"],
        "PAIRS-CC": ["alpha_CC", "alpha_PAIRS"],
        "CONS-AV": ["alpha_AV", "alpha_CONS"],
        "CONS-CC": ["alpha_CC", "alpha_CONS"],
    }
    
    for method, alpha_cols in expected_conditions.items():
        print(f"\n{method}:")
        for alpha_col in alpha_cols:
            all_perfect = True
            violations = []
            for results in all_results:
                if not results["summary"][method][alpha_col]["always_1"]:
                    all_perfect = False
                    for k, val in results["summary"][method][alpha_col]["imperfect_k"]:
                        violations.append((results["file"], k, val))
            
            if all_perfect:
                print(f"  {alpha_col}: ✓ ALWAYS 1.0 across all files and k values")
            else:
                min_val = min(v[2] for v in violations)
                print(f"  {alpha_col}: ✗ NOT ALWAYS 1.0 (min observed: {min_val:.6f})")
                for file, k, val in violations:
                    print(f"    - {file}, k={k}: {val:.6f}")


if __name__ == "__main__":
    main()

