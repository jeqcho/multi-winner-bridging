"""Tests for scoring functions."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from scoring import av_score, cc_score, pairs_score, cons_score, ejr_satisfied, beta_ejr


def test_av_score():
    """Test AV scoring."""
    # Simple 4 voters, 4 candidates
    M = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)
    
    # Empty committee
    assert av_score(M, []) == 0
    
    # Single candidate
    assert av_score(M, [0]) == 2
    assert av_score(M, [2]) == 2
    
    # Two candidates
    assert av_score(M, [0, 2]) == 4
    assert av_score(M, [0, 1]) == 4
    
    # All candidates
    assert av_score(M, [0, 1, 2, 3]) == 8
    
    print("✓ test_av_score passed")


def test_cc_score():
    """Test CC (Chamberlin-Courant) scoring."""
    M = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)
    
    # Empty committee covers no one
    assert cc_score(M, []) == 0
    
    # Single candidate covers 2 voters
    assert cc_score(M, [0]) == 2
    assert cc_score(M, [2]) == 2
    
    # Two candidates from same group still covers only 2
    assert cc_score(M, [0, 1]) == 2
    
    # Two candidates from different groups covers all 4
    assert cc_score(M, [0, 2]) == 4
    
    print("✓ test_cc_score passed")


def test_pairs_score():
    """Test PAIRS scoring."""
    M = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)
    
    # Empty committee
    assert pairs_score(M, []) == 0
    
    # Single candidate covers 1 pair (voters 0-1 or 2-3)
    assert pairs_score(M, [0]) == 1
    assert pairs_score(M, [2]) == 1
    
    # Two candidates from different groups: (0,1) + (2,3) = 2 pairs
    assert pairs_score(M, [0, 2]) == 2
    
    # Two candidates from same group still only 1 pair
    assert pairs_score(M, [0, 1]) == 1
    
    print("✓ test_pairs_score passed")


def test_cons_score():
    """Test CONS (connectivity) scoring."""
    M = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)
    
    # Empty committee - no connections
    assert cons_score(M, []) == 0
    
    # Single candidate connects voters who share it: 1 pair
    assert cons_score(M, [0]) == 1
    assert cons_score(M, [2]) == 1
    
    # Two candidates from different groups: two disconnected components
    # Component 1: voters 0,1 (1 pair), Component 2: voters 2,3 (1 pair)
    assert cons_score(M, [0, 2]) == 2
    
    # All four voters connected (if there's overlap)
    # In this case still 2 components
    assert cons_score(M, [0, 1]) == 1
    
    print("✓ test_cons_score passed")


def test_cons_score_fully_connected():
    """Test CONS with a fully connected graph."""
    M = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=bool)
    
    # Committee with candidates 0 and 1
    # Voter 0 approves 0, Voter 1 approves 0,1, Voter 2 approves 1,2, Voter 3 approves 2
    # With W=[0,1]: voters 0,1,2 are connected
    # Voter 3 is not connected to them (only approves 2, but 2 not in W)
    # So we have: component {0,1,2} = 3 pairs, component {3} = 0 pairs
    # Total = 3
    assert cons_score(M, [0, 1]) == 3
    
    # With W=[1,2]: voters 1,2,3 are connected
    # Voter 0 is not connected (only approves 0, not in W)
    # Component {1,2,3} = 3 pairs
    assert cons_score(M, [1, 2]) == 3
    
    # With W=[0,1,2]: all voters connected
    # Component {0,1,2,3} = 6 pairs
    assert cons_score(M, [0, 1, 2]) == 6
    
    print("✓ test_cons_score_fully_connected passed")


def test_ejr_satisfied():
    """Test EJR satisfaction."""
    # 4 voters, 4 candidates, k=2
    M = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)
    
    # W = [0, 2] should satisfy EJR with k=2
    # Each group of 2 voters shares 2 candidates and deserves 1 seat
    # Group {0,1} shares {0,1}, deserves 1 seat, and voter 0 approves 1 candidate in W (candidate 0)
    # Group {2,3} shares {2,3}, deserves 1 seat, and voter 2 approves 1 candidate in W (candidate 2)
    assert ejr_satisfied(M, [0, 2], k=2) == True
    
    # W = [0, 1] should not satisfy EJR
    # Group {2,3} shares {2,3}, deserves 1 seat, but no one in this group approves any candidate in W
    assert ejr_satisfied(M, [0, 1], k=2) == False
    
    # Empty committee with k=0
    assert ejr_satisfied(M, [], k=0) == True
    
    print("✓ test_ejr_satisfied passed")


def test_beta_ejr():
    """Test beta-EJR calculation."""
    M = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)
    
    # W = [0, 2] satisfies full EJR, so beta = 1.0
    beta = beta_ejr(M, [0, 2], k=2)
    assert beta >= 0.99, f"Expected beta ~1.0, got {beta}"
    
    # W = [0, 1] does not satisfy full EJR
    # Group {2,3} deserves 1 seat but gets 0 approvals
    # However, for β-EJR: threshold = ⌊β·1⌋
    # When β < 1.0, ⌊β⌋ = 0, which is vacuous (no constraint)
    # So W satisfies β-EJR for any β < 1.0
    # We expect beta to be just below 1.0
    beta = beta_ejr(M, [0, 1], k=2)
    assert 0.95 <= beta < 1.0, f"Expected beta ~0.99, got {beta}"
    
    # Test a case with more seats for clearer beta value
    # 6 voters, 3 groups of 2, committee size 3
    M2 = np.array([
        [1, 1, 1, 0, 0, 0],  # Group 1: voters 0,1
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],  # Group 2: voters 2,3
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],  # Group 3: voters 4,5 (approve nothing)
        [0, 0, 0, 0, 0, 0],
    ], dtype=bool)
    
    # W = [0, 3] with k=2
    # Group 1 (voters 0,1) share {0,1,2}, deserve 1 seat, get 1 approval each -> OK
    # Group 2 (voters 2,3) share {3,4,5}, deserve 1 seat, get 1 approval each -> OK
    beta = beta_ejr(M2, [0, 3], k=2)
    assert beta >= 0.99, f"Expected beta ~1.0 for fair committee, got {beta}"
    
    print("✓ test_beta_ejr passed")


def test_edge_cases():
    """Test edge cases."""
    # Single voter, single candidate
    M = np.array([[1]], dtype=bool)
    
    assert av_score(M, [0]) == 1
    assert cc_score(M, [0]) == 1
    assert pairs_score(M, [0]) == 0  # Only 1 voter, no pairs
    assert cons_score(M, [0]) == 0   # Only 1 voter, no pairs
    assert ejr_satisfied(M, [0], k=1) == True
    
    # No voters (edge case)
    M_empty = np.zeros((0, 4), dtype=bool)
    assert av_score(M_empty, [0, 1]) == 0
    assert cc_score(M_empty, [0, 1]) == 0
    assert pairs_score(M_empty, [0, 1]) == 0
    assert cons_score(M_empty, [0, 1]) == 0
    
    print("✓ test_edge_cases passed")


if __name__ == "__main__":
    test_av_score()
    test_cc_score()
    test_pairs_score()
    test_cons_score()
    test_cons_score_fully_connected()
    test_ejr_satisfied()
    test_beta_ejr()
    test_edge_cases()
    print("\n✅ All tests passed!")

