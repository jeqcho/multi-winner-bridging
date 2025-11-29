"""
Scoring functions for committee evaluation.

Implements AV, CC, PAIRS, CONS, and EJR scoring based on reference.md.
"""

import numpy as np
from itertools import combinations
from typing import Set, List, Tuple


def av_score(M: np.ndarray, W: List[int]) -> int:
    """
    Calculate Approval Voting (AV) score.
    
    AV(W) = Σ_{v ∈ V} |A_v ∩ W|
    
    Args:
        M: Boolean matrix (n_voters, n_candidates) where M[v][c] = 1 if voter v approves candidate c
        W: List of candidate indices in the committee
        
    Returns:
        Total number of approvals for committee members
    """
    if len(W) == 0:
        return 0
    return int(M[:, W].sum())


def cc_score(M: np.ndarray, W: List[int]) -> int:
    """
    Calculate Chamberlin-Courant Coverage (CC) score.
    
    CC(W) = |{ v ∈ V : (A_v ∩ W) ≠ ∅ }|
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        
    Returns:
        Number of voters who approve at least one member of W
    """
    if len(W) == 0:
        return 0
    return int((M[:, W].sum(axis=1) > 0).sum())


def pairs_score(M: np.ndarray, W: List[int]) -> int:
    """
    Calculate PAIRS score (direct pair coverage).
    
    PAIRS(W) = |{ {u, v} ⊆ V : (A_u ∩ A_v ∩ W) ≠ ∅ }|
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        
    Returns:
        Count of unordered voter pairs sharing at least one approved candidate in W
    """
    if len(W) == 0:
        return 0
    
    n_voters = M.shape[0]
    covered_pairs = set()
    
    # For each candidate in W, find voters who approve them
    for c in W:
        supporters = np.where(M[:, c])[0]
        
        # Add all pairs of supporters
        for i in range(len(supporters)):
            for j in range(i + 1, len(supporters)):
                pair = (int(supporters[i]), int(supporters[j]))
                covered_pairs.add(pair)
    
    return len(covered_pairs)


class UnionFind:
    """Union-Find data structure for connectivity."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_component_sizes(self) -> List[int]:
        """Return list of component sizes."""
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            components[root] = components.get(root, 0) + 1
        return list(components.values())


def cons_score(M: np.ndarray, W: List[int]) -> int:
    """
    Calculate CONS score (connectivity via selected candidates).
    
    CONS(W) = |{ {u, v} ⊆ V : u and v are connected by W }|
    
    Two voters are connected if there exists a path where each step shares
    a candidate in W.
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        
    Returns:
        Count of voter pairs in same connected component
    """
    if len(W) == 0:
        return 0
    
    n_voters = M.shape[0]
    uf = UnionFind(n_voters)
    
    # For each candidate in W, union all their supporters
    for c in W:
        supporters = np.where(M[:, c])[0]
        for i in range(len(supporters) - 1):
            uf.union(int(supporters[i]), int(supporters[i + 1]))
    
    # Count pairs in each component
    component_sizes = uf.get_component_sizes()
    total_pairs = sum(size * (size - 1) // 2 for size in component_sizes)
    
    return total_pairs


def ejr_satisfied(M: np.ndarray, W: List[int], k: int) -> bool:
    """
    Check if committee W satisfies Extended Justified Representation (EJR).
    
    EJR requires: For every ℓ ∈ {1,...,k} and every ℓ-cohesive group S with
    |S| ≥ (ℓ/k)·n, there exists some voter i ∈ S with |A_i ∩ W| ≥ ℓ.
    
    This implementation correctly checks ALL cohesive groups by focusing on
    unsatisfied voters. A violation exists iff there's an ℓ-cohesive group
    of unsatisfied voters that is large enough.
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        k: Committee size |W|
        
    Returns:
        True if W satisfies EJR, False otherwise
    """
    if k == 0:
        return len(W) == 0
    
    if len(W) != k:
        # Committee size doesn't match k
        return False
    
    n_voters = M.shape[0]
    n_candidates = M.shape[1]
    
    # Precompute approvals in W for each voter
    if len(W) > 0:
        approvals_in_W = M[:, W].sum(axis=1)
    else:
        approvals_in_W = np.zeros(n_voters, dtype=int)
    
    # For each ℓ from 1 to k
    for ell in range(1, k + 1):
        # Find unsatisfied voters for this ℓ (have < ℓ approved in W)
        unsatisfied = approvals_in_W < ell
        
        # For each ℓ-subset of candidates
        for T in combinations(range(n_candidates), ell):
            T_list = list(T)
            
            # Find voters who approve all candidates in T AND are unsatisfied
            # This gives us the unsatisfied ℓ-cohesive group for this T
            voters_approve_all_T = M[:, T_list].all(axis=1)
            cohesive_unsatisfied = voters_approve_all_T & unsatisfied
            group_size = cohesive_unsatisfied.sum()
            
            # Check if this unsatisfied cohesive group deserves ℓ seats
            # If so, it's an EJR violation (no one in the group is satisfied)
            if group_size * k >= ell * n_voters:
                return False
    
    return True


def beta_ejr(M: np.ndarray, W: List[int], k: int, precision: float = 0.01) -> float:
    """
    Calculate maximum β such that W satisfies β-EJR.
    
    β-EJR: For every ℓ-cohesive group S with |S| ≥ (ℓ/k)·n, there exists 
    some voter i ∈ S with |A_i ∩ W| ≥ ⌊β·ℓ⌋.
    
    This implementation correctly checks ALL cohesive groups by focusing on
    unsatisfied voters for each β threshold.
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        k: Committee size |W|
        precision: Step size for β search (default 0.01)
        
    Returns:
        Maximum β value (between 0 and 1) for which W satisfies β-EJR
    """
    if k == 0:
        return 1.0 if len(W) == 0 else 0.0
    
    if len(W) != k:
        return 0.0
    
    n_voters = M.shape[0]
    n_candidates = M.shape[1]
    
    # Precompute approvals in W for each voter
    if len(W) > 0:
        approvals_in_W = M[:, W].sum(axis=1)
    else:
        approvals_in_W = np.zeros(n_voters, dtype=int)
    
    # Binary search for maximum β
    beta_min, beta_max = 0.0, 1.0
    
    # First check if β=1 works (full EJR)
    if ejr_satisfied(M, W, k):
        return 1.0
    
    # Binary search with precision
    while beta_max - beta_min > precision / 2:
        beta = (beta_min + beta_max) / 2
        
        # Check if W satisfies β-EJR
        satisfies = True
        
        for ell in range(1, k + 1):
            threshold = int(beta * ell)  # ⌊β·ℓ⌋
            
            if threshold == 0:
                # Vacuous constraint (any voter trivially has ≥0 approvals)
                continue
            
            # Find unsatisfied voters for this threshold
            unsatisfied = approvals_in_W < threshold
            
            # For each ℓ-subset of candidates
            for T in combinations(range(n_candidates), ell):
                T_list = list(T)
                
                # Find voters who approve all candidates in T AND are unsatisfied
                voters_approve_all_T = M[:, T_list].all(axis=1)
                cohesive_unsatisfied = voters_approve_all_T & unsatisfied
                group_size = cohesive_unsatisfied.sum()
                
                # Check if this unsatisfied cohesive group deserves ℓ seats
                if group_size * k >= ell * n_voters:
                    # Violation found
                    satisfies = False
                    break
            
            if not satisfies:
                break
        
        if satisfies:
            beta_min = beta
        else:
            beta_max = beta
    
    return beta_min


if __name__ == "__main__":
    # Simple test
    M = np.array([
        [1, 1, 0, 0],  # Voter 0 approves candidates 0, 1
        [1, 1, 0, 0],  # Voter 1 approves candidates 0, 1
        [0, 0, 1, 1],  # Voter 2 approves candidates 2, 3
        [0, 0, 1, 1],  # Voter 3 approves candidates 2, 3
    ], dtype=bool)
    
    W = [0, 2]  # Committee with candidates 0 and 2
    k = 2
    
    print(f"Test committee W = {W}")
    print(f"AV score: {av_score(M, W)}")
    print(f"CC score: {cc_score(M, W)}")
    print(f"PAIRS score: {pairs_score(M, W)}")
    print(f"CONS score: {cons_score(M, W)}")
    print(f"EJR satisfied: {ejr_satisfied(M, W, k)}")
    print(f"Beta-EJR: {beta_ejr(M, W, k):.2f}")





