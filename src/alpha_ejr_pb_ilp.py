"""
Budget-aware alpha-EJR computation for Participatory Budgeting using ILP.

Implements the formulation from reference/alpha-ejr-pb-ilp.md.

For a fixed committee W, computes the maximum alpha such that W satisfies alpha-EJR
in the PB (cost-aware) setting.

Definition (alpha-EJR for PB):
Given alpha in (0,1], a funded set W satisfies alpha-EJR if for every subset S ⊆ V
and every q in [0,B]:
    If alpha * |S| >= (q/B) * |V| and S is q-cohesive (common approvals cost >= q),
    then exists i in S such that u_i(W) >= q.

Usage:
    from alpha_ejr_pb_ilp import compute_alpha_ejr_pb
    
    alpha = compute_alpha_ejr_pb(M, project_costs, budget, committee)
"""

import numpy as np
from typing import List, Tuple, Optional, Set
import gurobipy as gp
from gurobipy import GRB


def compute_utilities(M: np.ndarray, costs: List[int], W: List[int]) -> np.ndarray:
    """
    Compute utility for each voter given committee W.
    
    u_i(W) = sum_{c in W} cost(c) * A_{i,c}
    
    Args:
        M: Approval matrix (n_voters, n_projects)
        costs: Project costs
        W: List of project indices in the committee
        
    Returns:
        Array of utilities for each voter
    """
    n_voters = M.shape[0]
    utilities = np.zeros(n_voters, dtype=np.float64)
    
    for c in W:
        utilities += M[:, c].astype(np.float64) * costs[c]
    
    return utilities


def generate_q_candidates(costs: List[int], budget: int) -> List[float]:
    """
    Generate candidate q threshold values for the separation oracle.
    
    We use:
    1. All individual project costs (important thresholds)
    2. Prefix sums of sorted costs (cumulative thresholds)
    3. Budget fractions
    
    Args:
        costs: List of project costs
        budget: Total budget
        
    Returns:
        Sorted list of unique q values > 0 and <= budget
    """
    candidates = set()
    
    # Add all individual costs
    for c in costs:
        if 0 < c <= budget:
            candidates.add(float(c))
    
    # Add prefix sums of sorted costs
    sorted_costs = sorted(costs)
    cumsum = 0
    for c in sorted_costs:
        cumsum += c
        if 0 < cumsum <= budget:
            candidates.add(float(cumsum))
    
    # Add some budget fractions for finer granularity
    for frac in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
        q = budget * frac
        if q > 0:
            candidates.add(float(q))
    
    return sorted(candidates)


def find_largest_violating_group(
    M: np.ndarray,
    costs: List[int],
    utilities: np.ndarray,
    q: float,
    epsilon: float = 1e-6,
    verbose: bool = False
) -> Tuple[List[int], bool]:
    """
    Find the largest group S such that:
    1. All voters in S have utility < q (representation failure)
    2. S is q-cohesive (common approvals have total cost >= q)
    
    Uses Gurobi ILP (separation oracle from Section 5 of reference).
    
    Args:
        M: Approval matrix (n_voters, n_projects)
        costs: Project costs
        utilities: Pre-computed utilities for each voter
        q: Budget threshold
        epsilon: Small tolerance for utility comparison
        verbose: Whether to print Gurobi output
        
    Returns:
        (S, is_feasible): List of voter indices in S, and whether a valid S was found
    """
    n_voters, n_projects = M.shape
    
    # Check if any voter has utility < q (otherwise no violation possible)
    potential_voters = [i for i in range(n_voters) if utilities[i] < q - epsilon]
    if not potential_voters:
        return [], False
    
    try:
        model = gp.Model("separation")
        model.setParam('OutputFlag', 1 if verbose else 0)
        model.setParam('TimeLimit', 60)  # 60 second timeout per separation
        
        # Variables
        # s_i = 1 if voter i is in S
        s = model.addVars(n_voters, vtype=GRB.BINARY, name="s")
        # z_c = 1 if project c is approved by ALL voters in S
        z = model.addVars(n_projects, vtype=GRB.BINARY, name="z")
        
        # Objective: maximize |S| (find strongest violation)
        model.setObjective(gp.quicksum(s[i] for i in range(n_voters)), GRB.MAXIMIZE)
        
        # Constraint: only voters with utility < q can be in S (representation failure)
        for i in range(n_voters):
            if utilities[i] >= q - epsilon:
                model.addConstr(s[i] == 0, name=f"util_{i}")
        
        # Constraint: z_c <= A_{i,c} + (1 - s_i) for all i, c
        # This ensures z_c = 1 only if all voters in S approve project c
        for c in range(n_projects):
            for i in range(n_voters):
                model.addConstr(z[c] <= float(M[i, c]) + (1 - s[i]), name=f"common_{i}_{c}")
        
        # Constraint: S is q-cohesive (common approvals cost >= q)
        model.addConstr(
            gp.quicksum(costs[c] * z[c] for c in range(n_projects)) >= q,
            name="cohesive"
        )
        
        # Need at least one voter in S
        model.addConstr(gp.quicksum(s[i] for i in range(n_voters)) >= 1, name="nonempty")
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            if model.SolCount > 0:
                S = [i for i in range(n_voters) if s[i].X > 0.5]
                if len(S) > 0:
                    return S, True
        
        return [], False
        
    except gp.GurobiError as e:
        print(f"Gurobi error in separation: {e}")
        return [], False


def compute_alpha_ejr_pb(
    M: np.ndarray,
    costs: List[int],
    budget: int,
    W: List[int],
    verbose: bool = False
) -> float:
    """
    Compute the maximum alpha such that committee W satisfies alpha-EJR for PB.
    
    Uses a separation oracle approach:
    1. For each candidate threshold q
    2. Find the largest q-cohesive group S where all voters have utility < q
    3. Compute bound = q * |V| / (B * |S|)
    4. Return min(1, min over all bounds)
    
    Args:
        M: Approval matrix (n_voters, n_projects), boolean
        costs: List of project costs
        budget: Total budget B
        W: List of project indices in the funded committee
        verbose: Whether to print progress
        
    Returns:
        Maximum alpha in (0, 1] such that W satisfies alpha-EJR
    """
    if len(W) == 0:
        # Empty committee trivially satisfies alpha-EJR (no positive utility possible)
        return 1.0
    
    n_voters, n_projects = M.shape
    n = n_voters
    B = budget
    
    # Compute utilities for all voters
    utilities = compute_utilities(M, costs, W)
    
    if verbose:
        print(f"Computing alpha-EJR for committee of size {len(W)}")
        print(f"  Voters: {n}, Projects: {n_projects}, Budget: {B}")
        print(f"  Utilities: min={utilities.min():.0f}, max={utilities.max():.0f}, mean={utilities.mean():.0f}")
    
    # Generate candidate q values
    q_candidates = generate_q_candidates(costs, budget)
    
    if verbose:
        print(f"  Testing {len(q_candidates)} q thresholds...")
    
    min_bound = float('inf')
    worst_violation = None
    
    for q in q_candidates:
        if q <= 0:
            continue
        
        # Find largest q-cohesive group where all have utility < q
        S, is_feasible = find_largest_violating_group(M, costs, utilities, q, verbose=False)
        
        if is_feasible and len(S) > 0:
            # This group S violates alpha-EJR at alpha >= bound
            # bound = (q/B) * n / |S| = q * n / (B * |S|)
            bound = (q * n) / (B * len(S))
            
            if verbose:
                print(f"  q={q:.0f}: found |S|={len(S)}, bound={bound:.4f}")
            
            if bound < min_bound:
                min_bound = bound
                worst_violation = (S, q, bound)
    
    # The maximum valid alpha is just below the minimum bound
    # For practical purposes, we return the bound itself (the supremum)
    if min_bound == float('inf'):
        alpha = 1.0
    else:
        alpha = min(1.0, min_bound)
    
    if verbose:
        if worst_violation:
            S, q, bound = worst_violation
            print(f"  Worst violation: q={q:.0f}, |S|={len(S)}, bound={bound:.4f}")
        print(f"  Optimal alpha: {alpha:.4f}")
    
    return alpha


def check_alpha_ejr_pb(
    M: np.ndarray,
    costs: List[int],
    budget: int,
    W: List[int],
    alpha: float,
    verbose: bool = False
) -> Tuple[bool, Optional[Tuple[List[int], float]]]:
    """
    Check if committee W satisfies alpha-EJR for PB.
    
    Args:
        M: Approval matrix
        costs: Project costs
        budget: Total budget
        W: Committee (list of project indices)
        alpha: Alpha value to check
        verbose: Whether to print details
        
    Returns:
        (satisfied, violation): True if satisfied, else (False, (S, q)) for a witness
    """
    if len(W) == 0:
        return True, None
    
    n_voters, n_projects = M.shape
    n = n_voters
    B = budget
    
    utilities = compute_utilities(M, costs, W)
    q_candidates = generate_q_candidates(costs, budget)
    
    for q in q_candidates:
        if q <= 0:
            continue
        
        # For this q, find largest violating group
        S, is_feasible = find_largest_violating_group(M, costs, utilities, q, verbose=False)
        
        if is_feasible and len(S) > 0:
            # Check if this violates alpha-EJR
            # Violation occurs if: alpha * |S| >= (q/B) * n
            required_size = (q / B) * n / alpha
            
            if len(S) >= required_size:
                if verbose:
                    print(f"Violation found: q={q:.0f}, |S|={len(S)}, required={required_size:.2f}")
                return False, (S, q)
    
    return True, None


def compute_alpha_ejr_pb_full_optimization(
    M: np.ndarray,
    costs: List[int],
    budget: int,
    verbose: bool = False,
    max_iterations: int = 100
) -> Tuple[float, List[int]]:
    """
    Compute the optimal (alpha, W) pair using the full cutting-plane algorithm.
    
    This jointly optimizes both the committee W and the alpha value.
    
    WARNING: This can be computationally expensive for large instances.
    
    Args:
        M: Approval matrix
        costs: Project costs  
        budget: Total budget
        verbose: Whether to print progress
        max_iterations: Maximum cutting-plane iterations
        
    Returns:
        (optimal_alpha, optimal_W): Best alpha and corresponding committee
    """
    n_voters, n_projects = M.shape
    n = n_voters
    B = budget
    
    if verbose:
        print(f"Full alpha-EJR optimization")
        print(f"  Voters: {n}, Projects: {n_projects}, Budget: {B}")
    
    try:
        # Master problem
        master = gp.Model("alpha_ejr_master")
        master.setParam('OutputFlag', 1 if verbose else 0)
        
        # Variables
        x = master.addVars(n_projects, vtype=GRB.BINARY, name="x")  # Project selection
        alpha = master.addVar(lb=0, ub=1, name="alpha")  # Alpha value
        
        # Objective: maximize alpha
        master.setObjective(alpha, GRB.MAXIMIZE)
        
        # Budget constraint
        master.addConstr(
            gp.quicksum(costs[c] * x[c] for c in range(n_projects)) <= B,
            name="budget"
        )
        
        # Track cuts
        cuts = []  # List of (S, q) pairs
        cut_vars = {}  # y_{i,t} variables for each cut
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}")
            
            # Solve master
            master.optimize()
            
            if master.status != GRB.OPTIMAL:
                if verbose:
                    print(f"Master not optimal, status={master.status}")
                break
            
            # Extract solution
            alpha_star = alpha.X
            W = [c for c in range(n_projects) if x[c].X > 0.5]
            
            if verbose:
                total_cost = sum(costs[c] for c in W)
                print(f"  alpha*={alpha_star:.4f}, |W|={len(W)}, cost={total_cost}")
            
            # Compute utilities
            utilities = compute_utilities(M, costs, W)
            
            # Run separation oracle
            q_candidates = generate_q_candidates(costs, budget)
            violation_found = False
            
            for q in q_candidates:
                if q <= 0:
                    continue
                
                # Check if this q could cause a violation at current alpha
                S, is_feasible = find_largest_violating_group(M, costs, utilities, q, verbose=False)
                
                if is_feasible and len(S) > 0:
                    # Check if size condition is met at alpha*
                    required_size = (q / B) * n / alpha_star if alpha_star > 1e-9 else float('inf')
                    
                    if len(S) >= required_size - 1e-6:
                        # Add cut for (S, q)
                        if verbose:
                            print(f"  Adding cut: q={q:.0f}, |S|={len(S)}")
                        
                        t = len(cuts)
                        cuts.append((S, q))
                        
                        # Add y_{i,t} variables for i in S
                        y_t = {}
                        for i in S:
                            y_t[i] = master.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
                        cut_vars[t] = y_t
                        
                        # Constraint: sum_{i in S} y_{i,t} >= 1
                        master.addConstr(
                            gp.quicksum(y_t[i] for i in S) >= 1,
                            name=f"cut_sum_{t}"
                        )
                        
                        # Constraint: if y_{i,t} = 1, voter i must have utility >= q
                        # sum_c cost(c) * A_{i,c} * x_c >= q * y_{i,t}
                        for i in S:
                            master.addConstr(
                                gp.quicksum(costs[c] * float(M[i, c]) * x[c] 
                                           for c in range(n_projects)) >= q * y_t[i],
                                name=f"cut_util_{i}_{t}"
                            )
                        
                        violation_found = True
                        break  # Re-solve master with new cut
            
            if not violation_found:
                if verbose:
                    print(f"  No violations found - optimal!")
                return alpha_star, W
        
        if verbose:
            print(f"Max iterations reached")
        
        # Return best found solution
        alpha_star = alpha.X
        W = [c for c in range(n_projects) if x[c].X > 0.5]
        return alpha_star, W
        
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return 0.0, []


if __name__ == "__main__":
    # Simple test
    import sys
    sys.path.insert(0, '.')
    
    # Test case: 4 voters, 4 projects
    M = np.array([
        [1, 1, 0, 0],  # Voter 0 approves projects 0, 1
        [1, 1, 0, 0],  # Voter 1 approves projects 0, 1
        [0, 0, 1, 1],  # Voter 2 approves projects 2, 3
        [0, 0, 1, 1],  # Voter 3 approves projects 2, 3
    ], dtype=bool)
    
    costs = [100, 100, 100, 100]
    budget = 200
    
    # Committee that represents both groups
    W_good = [0, 2]
    # Committee that only represents first group
    W_bad = [0, 1]
    
    print("Test 1: Good committee [0, 2]")
    alpha_good = compute_alpha_ejr_pb(M, costs, budget, W_good, verbose=True)
    print(f"Alpha: {alpha_good:.4f}\n")
    
    print("Test 2: Bad committee [0, 1]")
    alpha_bad = compute_alpha_ejr_pb(M, costs, budget, W_bad, verbose=True)
    print(f"Alpha: {alpha_bad:.4f}\n")
    
    print("Test 3: Full optimization")
    alpha_opt, W_opt = compute_alpha_ejr_pb_full_optimization(M, costs, budget, verbose=True)
    print(f"Optimal alpha: {alpha_opt:.4f}, committee: {W_opt}")
