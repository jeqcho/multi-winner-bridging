"""
Compute the correct "Best Possible" proportions for PAIRS-EJR and CONS-EJR.

For each election:
1. Find committees achieving alpha_PAIRS = 1 or alpha_CONS = 1
2. Compute alpha_EJR for those committees
3. Check if any achieve (1,1) for PAIRS-EJR or CONS-EJR

Saves results to analysis/best_possible_ejr_tradeoffs.json
"""

import json
import ast
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pb_data_loader import load_pb_file
from alpha_ejr_pb_ilp import compute_alpha_ejr_pb


def process_election(election_dir: Path) -> dict:
    """Process a single election and return results."""
    election_name = election_dir.name
    
    try:
        # Load raw_scores.csv
        raw_scores_path = election_dir / "raw_scores.csv"
        if not raw_scores_path.exists():
            return {"election": election_name, "error": "No raw_scores.csv"}
        
        df = pd.read_csv(raw_scores_path)
        if len(df) < 2:
            return {"election": election_name, "error": "Too few committees"}
        
        # Find max values
        max_pairs = df["PAIRS"].max()
        max_cons = df["CONS"].max()
        
        if max_pairs == 0 or max_cons == 0:
            return {"election": election_name, "error": "Zero max values"}
        
        # Find committees with alpha_PAIRS=1 or alpha_CONS=1
        pairs1_mask = df["PAIRS"] == max_pairs
        cons1_mask = df["CONS"] == max_cons
        
        # Get unique committees to check (union of both)
        committees_to_check = df[pairs1_mask | cons1_mask].copy()
        
        # Load PB data for alpha_EJR computation
        data_dir = Path(__file__).parent.parent / "data"
        pb_file = data_dir / f"{election_name}.pb"
        
        if not pb_file.exists():
            return {"election": election_name, "error": f"No .pb file"}
        
        M, project_ids, costs, budget = load_pb_file(pb_file)
        
        # Compute alpha_EJR for each committee
        pairs_ejr_achieved = False
        cons_ejr_achieved = False
        
        for _, row in committees_to_check.iterrows():
            subset_str = row["subset_indices"]
            W = ast.literal_eval(subset_str) if isinstance(subset_str, str) else []
            
            if len(W) == 0:
                continue
            
            # Compute alpha_EJR
            alpha_ejr = compute_alpha_ejr_pb(M, costs, budget, W)
            
            is_pairs1 = row["PAIRS"] == max_pairs
            is_cons1 = row["CONS"] == max_cons
            is_ejr1 = alpha_ejr >= 0.9999
            
            if is_pairs1 and is_ejr1:
                pairs_ejr_achieved = True
            if is_cons1 and is_ejr1:
                cons_ejr_achieved = True
            
            # Early exit if both achieved
            if pairs_ejr_achieved and cons_ejr_achieved:
                break
        
        return {
            "election": election_name,
            "pairs_ejr_11": pairs_ejr_achieved,
            "cons_ejr_11": cons_ejr_achieved,
            "n_committees_checked": len(committees_to_check),
            "error": None
        }
        
    except Exception as e:
        return {"election": election_name, "error": str(e)}


def main():
    base_dir = Path(__file__).parent.parent
    pb_output_dir = base_dir / "output" / "pb"
    analysis_dir = base_dir / "analysis"
    
    # Find all election directories
    election_dirs = [d for d in pb_output_dir.iterdir() if d.is_dir()]
    print(f"Found {len(election_dirs)} elections")
    
    # Process elections in parallel
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_election, d): d for d in election_dirs}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results.append(result)
    
    # Compute summary statistics
    valid_results = [r for r in results if r.get("error") is None]
    total = len(valid_results)
    
    pairs_ejr_count = sum(1 for r in valid_results if r["pairs_ejr_11"])
    cons_ejr_count = sum(1 for r in valid_results if r["cons_ejr_11"])
    
    summary = {
        "total_elections": total,
        "pairs_ejr_11_count": pairs_ejr_count,
        "pairs_ejr_11_proportion": pairs_ejr_count / total if total > 0 else 0,
        "cons_ejr_11_count": cons_ejr_count,
        "cons_ejr_11_proportion": cons_ejr_count / total if total > 0 else 0,
        "errors": len(results) - total,
        "results": results
    }
    
    # Save results
    output_path = analysis_dir / "best_possible_ejr_tradeoffs.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults:")
    print(f"  Total elections: {total}")
    print(f"  PAIRS-EJR (1,1): {pairs_ejr_count} ({pairs_ejr_count/total:.1%})")
    print(f"  CONS-EJR (1,1): {cons_ejr_count} ({cons_ejr_count/total:.1%})")
    print(f"  Errors: {len(results) - total}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
