"""Tests for data loader."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from data_loader import load_and_combine_data


def test_load_and_combine_data():
    """Test loading and combining PrefLib data."""
    M, candidates = load_and_combine_data()
    
    # Check shape
    assert M.shape[0] == 2836, f"Expected 2836 voters, got {M.shape[0]}"
    assert M.shape[1] == 12, f"Expected 12 candidates, got {M.shape[1]}"
    
    # Check data type
    assert M.dtype == bool, f"Expected bool dtype, got {M.dtype}"
    
    # Check candidates
    assert len(candidates) == 12, f"Expected 12 candidates, got {len(candidates)}"
    
    # Check that we have some approvals
    assert M.sum() > 0, "Expected some approvals"
    
    print(f"✓ Loaded {M.shape[0]} voters with {M.shape[1]} candidates")
    print(f"✓ Total approvals: {M.sum()}")
    print(f"✓ Average approvals per voter: {M.sum(axis=1).mean():.2f}")
    print("✓ test_load_and_combine_data passed")


if __name__ == "__main__":
    test_load_and_combine_data()
    print("\n✅ All data loader tests passed!")





