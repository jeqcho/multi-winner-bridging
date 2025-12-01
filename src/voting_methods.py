"""
Voting methods for committee selection.

Implements:
- Approval Voting (AV): Select top-k candidates by approval count
- Chamberlin-Courant (CC): Greedy coverage maximization  
- Proportional Approval Voting (PAV): Greedy harmonic weighting
- Method of Equal Shares (MES): Already in mes.py
- Max-score methods (PAIRS-AV, PAIRS-CC, CONS-AV, CONS-CC): Select from exhaustive scores
"""

import numpy as np
import pandas as pd
import json
import random
from typing import List, Tuple
from scoring import av_score, cc_score, pairs_score, cons_score


def select_max_committee(df: pd.DataFrame, size: int, primary_score: str, secondary_score: str) -> List[int]:
    """
    Select the committee that maximizes primary_score, using secondary_score as tiebreaker.
    
    Args:
        df: DataFrame with columns: subset_size, subset_indices, AV, CC, PAIRS, CONS
        size: Committee size k to filter for
        primary_score: Column name for primary score to maximize (e.g., 'PAIRS')
        secondary_score: Column name for secondary score for tiebreaking (e.g., 'AV')
        
    Returns:
        List of candidate indices for the selected committee
    """
    # Filter for the requested committee size
    size_df = df[df['subset_size'] == size].copy()
    
    if len(size_df) == 0:
        return []
    
    # Find max primary score
    max_primary = size_df[primary_score].max()
    
    # Filter to rows with max primary score
    primary_max_df = size_df[size_df[primary_score] == max_primary]
    
    # Find max secondary score among those
    max_secondary = primary_max_df[secondary_score].max()
    
    # Filter to rows with max secondary score
    final_df = primary_max_df[primary_max_df[secondary_score] == max_secondary]
    
    # If multiple rows remain, randomly select one
    if len(final_df) > 1:
        selected_row = final_df.sample(n=1, random_state=random.randint(0, 2**31-1)).iloc[0]
    else:
        selected_row = final_df.iloc[0]
    
    # Parse the subset_indices (stored as JSON string)
    committee = json.loads(selected_row['subset_indices'])
    
    return sorted(committee)


def approval_voting(M: np.ndarray, k: int) -> List[int]:
    """
    Select committee using Approval Voting.
    
    Selects the top-k candidates with the highest approval counts.
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        k: Committee size
        
    Returns:
        List of k candidate indices
    """
    # Count approvals for each candidate
    approval_counts = M.sum(axis=0)
    
    # Get top-k candidates
    top_k_indices = np.argsort(approval_counts)[::-1][:k]
    
    return sorted(top_k_indices.tolist())


def chamberlin_courant_greedy(M: np.ndarray, k: int) -> List[int]:
    """
    Select committee using greedy Chamberlin-Courant.
    
    Greedily selects candidates that maximize coverage (CC score).
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        k: Committee size
        
    Returns:
        List of k candidate indices
    """
    n_voters, n_candidates = M.shape
    committee = []
    covered = np.zeros(n_voters, dtype=bool)
    
    for _ in range(k):
        best_candidate = -1
        best_marginal = -1
        
        for c in range(n_candidates):
            if c in committee:
                continue
            
            # Marginal coverage: new voters covered by adding c
            marginal = ((M[:, c]) & (~covered)).sum()
            
            if marginal > best_marginal:
                best_marginal = marginal
                best_candidate = c
        
        if best_candidate >= 0:
            committee.append(best_candidate)
            covered = covered | M[:, best_candidate]
    
    return sorted(committee)


def pav_greedy(M: np.ndarray, k: int) -> List[int]:
    """
    Select committee using greedy Proportional Approval Voting.
    
    Uses harmonic weights: voter i contributes 1/(1 + |A_i ∩ W|) for each
    new candidate added.
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        k: Committee size
        
    Returns:
        List of k candidate indices
    """
    n_voters, n_candidates = M.shape
    committee = []
    
    # Track how many committee members each voter has approved so far
    voter_approvals = np.zeros(n_voters, dtype=int)
    
    for _ in range(k):
        best_candidate = -1
        best_score = -1
        
        for c in range(n_candidates):
            if c in committee:
                continue
            
            # PAV marginal gain for adding c:
            # For each voter i who approves c, add 1/(1 + current_approvals)
            approvers = M[:, c]
            marginal = (approvers / (1.0 + voter_approvals)).sum()
            
            if marginal > best_score:
                best_score = marginal
                best_candidate = c
        
        if best_candidate >= 0:
            committee.append(best_candidate)
            voter_approvals += M[:, best_candidate].astype(int)
    
    return sorted(committee)


def get_all_voting_methods() -> dict:
    """
    Return dictionary of all voting method functions.
    
    Returns:
        Dict mapping method name to (function, marker, color) tuple
    """
    return {
        'AV': (approval_voting, 's', 'red'),         # Square, red
        'greedy-CC': (chamberlin_courant_greedy, '^', 'blue'),  # Triangle, blue
        'greedy-PAV': (pav_greedy, 'D', 'green'),           # Diamond, green
    }


def run_voting_method(method_name: str, M: np.ndarray, k: int) -> List[int]:
    """
    Run a voting method by name.
    
    Args:
        method_name: One of 'AV', 'greedy-CC', 'greedy-PAV'
        M: Boolean matrix (n_voters, n_candidates)
        k: Committee size
        
    Returns:
        List of k candidate indices
    """
    methods = get_all_voting_methods()
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")
    
    func, _, _ = methods[method_name]
    return func(M, k)


if __name__ == "__main__":
    # Test with simple example
    M = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)
    
    k = 2
    
    print("Test matrix (4 voters, 4 candidates):")
    print(M.astype(int))
    print(f"\nCommittee size k={k}")
    
    print(f"\nAV committee: {approval_voting(M, k)}")
    print(f"CC committee: {chamberlin_courant_greedy(M, k)}")
    print(f"PAV committee: {pav_greedy(M, k)}")

