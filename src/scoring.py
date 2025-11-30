"""
Scoring functions for committee evaluation.

Implements AV, CC, PAIRS, CONS, and EJR scoring based on reference.md.
"""

import numpy as np
from typing import List

from abcvoting.preferences import Profile
from abcvoting.properties import check_EJR


def _matrix_to_profile(M: np.ndarray) -> Profile:
    """
    Convert numpy approval matrix to abcvoting Profile.
    
    Args:
        M: Boolean matrix (n_voters, n_candidates) where M[v][c] = 1 if voter v approves candidate c
        
    Returns:
        abcvoting Profile object
    """
    n_voters, n_candidates = M.shape
    profile = Profile(n_candidates)
    for voter_idx in range(n_voters):
        approved = list(np.where(M[voter_idx])[0])
        profile.add_voter(approved)
    return profile


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


def ejr_satisfied(M: np.ndarray, W: List[int], k: int) -> bool:
    """
    Check if committee W satisfies Extended Justified Representation (EJR).
    
    EJR requires: For every ℓ ∈ {1,...,k} and every ℓ-cohesive group S with
    |S| ≥ (ℓ/k)·n, there exists some voter i ∈ S with |A_i ∩ W| ≥ ℓ.
    
    Uses abcvoting's check_EJR for exact (non-sampling) verification.
    
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
    
    profile = _matrix_to_profile(M)
    return check_EJR(profile, set(W))


def alpha_ejr(M: np.ndarray, W: List[int], k: int, tolerance: float = 0.01) -> float:
    """
    Compute the maximum α such that committee W satisfies α-EJR.
    
    α-EJR: For every ℓ ∈ [k] and every ℓ-cohesive group S,
    if α·|S| ≥ (ℓ/k)·n, then some voter in S has ≥ℓ approved in W.
    
    Uses binary search with abcvoting's check_EJR and the quota parameter.
    For α-EJR, we set quota = n/(α·k) = (n/k)/α.
    
    Args:
        M: Boolean matrix (n_voters, n_candidates)
        W: List of candidate indices in the committee
        k: Committee size |W|
        tolerance: Binary search tolerance for α precision
        
    Returns:
        Maximum α value in (0, 1] for which W satisfies α-EJR
    """
    if k == 0 or len(W) == 0:
        return 0.0
    
    profile = _matrix_to_profile(M)
    n = len(profile)
    
    # Check full EJR first (α = 1)
    if check_EJR(profile, set(W)):
        return 1.0
    
    # Binary search for maximum α where EJR is satisfied
    # For α-EJR, quota = n/(α·k)
    lo, hi = 0.0, 1.0
    
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        if mid == 0:
            # Avoid division by zero
            lo = tolerance
            continue
        
        quota = n / (mid * k)
        if check_EJR(profile, set(W), quota=quota):
            lo = mid  # α works, try higher
        else:
            hi = mid  # α fails, try lower
    
    return lo


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





