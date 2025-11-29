"""
Method of Equal Shares (MES) implementation for approval voting.

Implements the standard Equal Shares algorithm for selecting committees
from approval voting data.
"""

import numpy as np
from typing import List, Optional


def compute_price(M: np.ndarray, candidate: int, budgets: np.ndarray) -> Optional[float]:
    """
    Compute the minimum price rho for a candidate such that supporters can afford it.
    
    The price rho satisfies: sum(min(budget_v, rho) for v in supporters) = 1
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        candidate: Index of the candidate
        budgets: Current budget for each voter
        
    Returns:
        The price rho if affordable, None otherwise
    """
    # Get supporters of this candidate
    supporters = np.where(M[:, candidate])[0]
    
    if len(supporters) == 0:
        return None
    
    # Get budgets of supporters
    supporter_budgets = budgets[supporters]
    
    # Sort budgets in ascending order to find the right price
    sorted_budgets = np.sort(supporter_budgets)
    
    # Try to find rho such that sum(min(budget, rho)) = 1
    # We iterate through budget levels to find where the constraint is satisfied
    
    n_supporters = len(supporters)
    cumulative_budget = 0.0
    
    for i, budget in enumerate(sorted_budgets):
        # Number of supporters with budget >= current budget level
        remaining_supporters = n_supporters - i
        
        # If everyone from here pays `budget`, total contribution would be:
        # cumulative_budget (from those who can't afford more) + remaining_supporters * budget
        max_contribution = cumulative_budget + remaining_supporters * budget
        
        if max_contribution >= 1.0:
            # The price rho is such that: cumulative_budget + remaining_supporters * rho = 1
            rho = (1.0 - cumulative_budget) / remaining_supporters
            return rho
        
        # This supporter contributes their full budget
        cumulative_budget += budget
    
    # Check if total budget is enough
    if cumulative_budget >= 1.0 - 1e-9:
        # Edge case: everyone pays their full budget
        return sorted_budgets[-1] if len(sorted_budgets) > 0 else None
    
    # Not affordable
    return None


def method_of_equal_shares(M: np.ndarray, k: int) -> List[int]:
    """
    Select a committee of size k using Method of Equal Shares.
    
    Algorithm:
    1. Each voter starts with budget = k/n
    2. Each candidate costs 1 to add
    3. Iteratively select the candidate with minimum "price" rho where
       supporters can collectively afford to pay 1
    4. If MES doesn't fill k slots, complete with greedy AV
    
    Args:
        M: Boolean matrix (n_voters, n_candidates) where M[v][c] = 1 if voter v approves candidate c
        k: Target committee size
        
    Returns:
        List of candidate indices in the selected committee
    """
    if k == 0:
        return []
    
    n_voters, n_candidates = M.shape
    
    # Initialize budgets: each voter gets k/n
    budgets = np.full(n_voters, k / n_voters)
    
    # Track selected candidates and remaining candidates
    selected = []
    remaining = set(range(n_candidates))
    
    # Phase 1: MES selection
    while len(selected) < k and len(remaining) > 0:
        best_candidate = None
        best_price = float('inf')
        
        # Find candidate with minimum price
        for candidate in remaining:
            price = compute_price(M, candidate, budgets)
            
            if price is not None and price < best_price:
                best_price = price
                best_candidate = candidate
            elif price is not None and price == best_price:
                # Tie-breaking by candidate index (smaller index wins)
                if candidate < best_candidate:
                    best_candidate = candidate
        
        if best_candidate is None:
            # No affordable candidate, switch to greedy completion
            break
        
        # Select the best candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        
        # Deduct costs from supporters
        supporters = np.where(M[:, best_candidate])[0]
        for v in supporters:
            cost = min(budgets[v], best_price)
            budgets[v] -= cost
    
    # Phase 2: Greedy AV completion if needed
    while len(selected) < k and len(remaining) > 0:
        # Find candidate with most approvals among remaining
        best_candidate = None
        best_approvals = -1
        
        for candidate in remaining:
            approvals = M[:, candidate].sum()
            if approvals > best_approvals:
                best_approvals = approvals
                best_candidate = candidate
            elif approvals == best_approvals:
                # Tie-breaking by candidate index
                if candidate < best_candidate:
                    best_candidate = candidate
        
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
    
    return selected


if __name__ == "__main__":
    # Simple test
    M = np.array([
        [1, 1, 0, 0],  # Voter 0 approves candidates 0, 1
        [1, 1, 0, 0],  # Voter 1 approves candidates 0, 1
        [0, 0, 1, 1],  # Voter 2 approves candidates 2, 3
        [0, 0, 1, 1],  # Voter 3 approves candidates 2, 3
    ], dtype=bool)
    
    print("Testing MES on simple 4-voter, 4-candidate example")
    print("Voters 0,1 approve {0,1}; Voters 2,3 approve {2,3}")
    
    for k in range(1, 5):
        committee = method_of_equal_shares(M, k)
        print(f"  k={k}: Committee = {committee}")

