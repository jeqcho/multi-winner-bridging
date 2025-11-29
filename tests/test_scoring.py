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


def test_ejr_non_maximal_cohesive_group():
    """
    Test EJR detection for non-maximal cohesive groups.
    
    This tests a case where a cohesive subgroup S violates EJR, but the
    maximal group S_T (used by naive algorithms) includes additional
    satisfied voters. The correct algorithm must still detect the violation.
    """
    # Setup:
    # - Voters 0,1: approve {0,1} - form a cohesive group
    # - Voter 2: approves {0,1,2} - overlaps with group but also approves 2
    # - Voter 3: approves {2,3}
    M = np.array([
        [1, 1, 0, 0],  # Voter 0: approves {0, 1}
        [1, 1, 0, 0],  # Voter 1: approves {0, 1}
        [1, 1, 1, 0],  # Voter 2: approves {0, 1, 2}
        [0, 0, 1, 1],  # Voter 3: approves {2, 3}
    ], dtype=bool)
    
    k = 2
    W = [2, 3]  # Committee with candidates 2 and 3
    
    # Analysis for l=1:
    # - Group S = {voter 0, voter 1}:
    #   - Cohesive: ⋂ A_v = {0,1}, size=2 >= l=1 ✓
    #   - Large enough: |S|=2 >= (1/2)*4=2 ✓
    #   - Voter 0: |{0,1} ∩ {2,3}| = 0 < 1 ✗
    #   - Voter 1: |{0,1} ∩ {2,3}| = 0 < 1 ✗
    #   - NO voter in S is satisfied! This is an EJR VIOLATION.
    #
    # The naive algorithm would check S_T for T={0}, getting S_T = {0,1,2}.
    # Voter 2 in S_T has 1 approved in W, so naive algorithm says "satisfied".
    # But voter 2 is NOT in the cohesive group S={0,1}!
    
    result = ejr_satisfied(M, W, k)
    assert result == False, f"Expected EJR violation, but got {result}"
    
    # The beta value should be less than 1.0
    beta = beta_ejr(M, W, k)
    assert beta < 1.0, f"Expected beta < 1.0, but got {beta}"
    
    # Verify the good committee works
    W_good = [0, 2]  # One candidate from each group
    assert ejr_satisfied(M, W_good, k) == True, "Committee [0,2] should satisfy EJR"
    
    print("✓ test_ejr_non_maximal_cohesive_group passed")


def test_ejr_multiple_violations():
    """Test EJR with multiple potential violations."""
    # 6 voters split into 3 groups of 2
    M = np.array([
        [1, 1, 0, 0, 0, 0],  # Voters 0,1: approve {0,1}
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],  # Voters 2,3: approve {2,3}
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],  # Voters 4,5: approve {4,5}
        [0, 0, 0, 0, 1, 1],
    ], dtype=bool)
    
    k = 3
    
    # Committee with only candidates from first group - violates EJR for groups 2 and 3
    W_bad = [0, 1, 0]  # Invalid - duplicate, but let's use [0, 1, 2] instead
    # Actually, let's test [0, 1, 2] which only covers group 1 and part of group 2
    
    # Wait, let me think about this more carefully
    # With k=3 and n=6, threshold for l=1 is (1/3)*6 = 2
    # Each group has 2 voters, so each group deserves 1 seat
    
    # W = [0, 2, 4] - one from each group, should satisfy EJR
    W_good = [0, 2, 4]
    assert ejr_satisfied(M, W_good, k) == True
    
    # W = [0, 1, 2] - two from group 1, one from group 2, none from group 3
    # Group 3 (voters 4,5) shares {4,5}, deserves 1 seat, but gets 0 in W
    W_bad = [0, 1, 2]
    assert ejr_satisfied(M, W_bad, k) == False
    
    print("✓ test_ejr_multiple_violations passed")


if __name__ == "__main__":
    test_av_score()
    test_cc_score()
    test_pairs_score()
    test_cons_score()
    test_cons_score_fully_connected()
    test_ejr_satisfied()
    test_beta_ejr()
    test_edge_cases()
    test_ejr_non_maximal_cohesive_group()
    test_ejr_multiple_violations()
    print("\n✅ All tests passed!")

