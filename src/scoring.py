"""
Scoring functions for committee evaluation.

Implements AV, CC, PAIRS, CONS, and EJR scoring based on reference.md.
"""

import numpy as np
from itertools import combinations
from typing import Set, List, Tuple, Iterator
import random
from math import comb


# Maximum number of candidate subsets to sample per coalition size for EJR checks
MAX_EJR_SAMPLES_PER_SIZE = 100


def _sample_combinations(n: int, r: int, max_samples: int) -> Iterator[Tuple[int, ...]]:
    """
    Yield combinations of r elements from range(n).
    If total combinations <= max_samples, yield all of them.
    Otherwise, randomly sample max_samples combinations.
    
    Args:
        n: Size of the set to choose from (range(n))
        r: Size of each combination
        max_samples: Maximum number of samples to return
        
    Yields:
        Tuples of r integers from range(n)
    """
    total = comb(n, r)
    
    if total <= max_samples:
        # Yield all combinations
        yield from combinations(range(n), r)
    else:
        # Random sampling - use a set to avoid duplicates
        seen = set()
        elements = list(range(n))
        
        while len(seen) < max_samples:
            # Generate a random combination
            combo = tuple(sorted(random.sample(elements, r)))
            if combo not in seen:
                seen.add(combo)
                yield combo


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
    Calculate PAIRS score (direct pair coverage) using vectorized operations.
    
    PAIRS(W) = |{ {u, v} ⊆ V : (A_u ∩ A_v ∩ W) ≠ ∅ }|
    
    Uses matrix multiplication for O(n² × k) vectorized computation instead of
    O(k × s²) Python loops, which is much faster for large voter counts.
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        
    Returns:
        Count of unordered voter pairs sharing at least one approved candidate in W
    """
    if len(W) == 0:
        return 0
    
    n_voters = M.shape[0]
    
    # Extract submatrix for committee candidates: n_voters × k
    M_W = M[:, W].astype(np.uint8)
    
    # Compute shared approvals matrix: shared[i,j] = # of W candidates both i and j approve
    # This is O(n² × k) but highly optimized in numpy
    shared = M_W @ M_W.T
    
    # Count pairs where shared > 0 (they share at least one candidate in W)
    # Use upper triangle only to avoid counting pairs twice
    # np.triu with k=1 excludes diagonal
    upper_shared = np.triu(shared, k=1)
    pairs_count = int((upper_shared > 0).sum())
    
    return pairs_count


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


def ejr_satisfied(M: np.ndarray, W: List[int], k: int, max_samples: int = MAX_EJR_SAMPLES_PER_SIZE) -> bool:
    """
    Check if committee W satisfies Extended Justified Representation (EJR).
    
    EJR requires: For every ℓ ∈ {1,...,k} and every ℓ-cohesive group S with
    |S| ≥ (ℓ/k)·n, there exists some voter i ∈ S with |A_i ∩ W| ≥ ℓ.
    
    This implementation correctly checks cohesive groups by focusing on
    unsatisfied voters. For efficiency, it samples at most max_samples
    candidate subsets per coalition size.
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        k: Committee size |W|
        max_samples: Maximum candidate subsets to check per coalition size
        
    Returns:
        True if W satisfies EJR (approximately), False if violation found
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
        
        # Sample ℓ-subsets of candidates (at most max_samples)
        for T in _sample_combinations(n_candidates, ell, max_samples):
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


def alpha_ejr(M: np.ndarray, W: List[int], k: int, 
              max_samples: int = MAX_EJR_SAMPLES_PER_SIZE) -> float:
    """
    Compute the maximum α such that committee W satisfies α-EJR.
    
    α-EJR: For every ℓ ∈ [k] and every ℓ-cohesive group S,
    if α·|S| ≥ (ℓ/k)·n, then some voter in S has ≥ℓ approved in W.
    
    For each violating group (ℓ-cohesive S where no voter has ≥ℓ in W),
    the violation threshold is α = (ℓ·n) / (k·|S|).
    Returns the minimum such threshold, or 1.0 if no violations (full EJR).
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        k: Committee size |W|
        max_samples: Maximum candidate subsets to check per coalition size
        
    Returns:
        Maximum α value in (0, 1] for which W satisfies α-EJR
    """
    n_voters, n_candidates = M.shape
    
    if k == 0 or len(W) == 0:
        return 0.0
    
    # Precompute how many W-candidates each voter approves
    approvals_in_W = M[:, W].sum(axis=1)
    
    min_alpha = 1.0  # Start assuming full EJR
    
    for ell in range(1, k + 1):
        # Find voters who are "unsatisfied" at level ℓ
        unsatisfied = approvals_in_W < ell
        
        # Check ℓ-sized candidate subsets T (with sampling)
        for T in _sample_combinations(n_candidates, ell, max_samples):
            T_list = list(T)
            # Voters who approve ALL candidates in T (ℓ-cohesive group)
            voters_approve_all_T = M[:, T_list].all(axis=1)
            
            # The violating group: ℓ-cohesive AND unsatisfied
            cohesive_unsatisfied = voters_approve_all_T & unsatisfied
            group_size = cohesive_unsatisfied.sum()
            
            if group_size > 0:
                # This group causes a violation when α ≥ (ℓ·n) / (k·|S|)
                threshold = (ell * n_voters) / (k * group_size)
                if threshold < min_alpha:
                    min_alpha = threshold
    
    return min_alpha


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
    print(f"Alpha-EJR: {alpha_ejr(M, W, k):.2f}")





