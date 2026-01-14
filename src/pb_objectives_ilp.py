"""
ILP formulations for maximizing PB objectives: AV, CC, PAIRS, CONS.

Implements exact ILP/MILP formulations from:
- reference/AV_CC_PB_ILP_Formulations.md (AV, CC)
- reference/PAIRS_CONS_PB_ILP.md (PAIRS, CONS)

Usage:
    from pb_objectives_ilp import maximize_av_ilp, maximize_cc_ilp, maximize_pairs_ilp, maximize_cons_ilp
    
    committee, score = maximize_av_ilp(M, costs, budget)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import gurobipy as gp
from gurobipy import GRB


def maximize_av_ilp(
    M: np.ndarray,
    costs: List[int],
    budget: int,
    verbose: bool = False,
    time_limit: int = 300
) -> Tuple[List[int], int]:
    """
    Maximize Approval Voting (AV) score using ILP.
    
    AV(W) = Σ_{v∈V} |A_v ∩ W| = Σ_{v,p} a_{v,p} · x_p
    
    This is essentially a 0/1 knapsack problem where the "value" of each
    project is its total approval count.
    
    Args:
        M: Approval matrix (n_voters, n_projects), boolean
        costs: List of project costs
        budget: Total budget B
        verbose: Whether to print Gurobi output
        time_limit: Time limit in seconds
        
    Returns:
        (committee, score): List of project indices and optimal AV score
    """
    n_voters, n_projects = M.shape
    
    try:
        model = gp.Model("maximize_av")
        model.setParam('OutputFlag', 1 if verbose else 0)
        model.setParam('TimeLimit', time_limit)
        
        # Decision variables: x_p = 1 if project p is selected
        x = model.addVars(n_projects, vtype=GRB.BINARY, name="x")
        
        # Objective: maximize total approvals
        # AV = Σ_{v,p} a_{v,p} · x_p = Σ_p (Σ_v a_{v,p}) · x_p
        # The coefficient for each x_p is the number of voters who approve p
        approval_counts = M.sum(axis=0)  # Sum over voters for each project
        
        model.setObjective(
            gp.quicksum(int(approval_counts[p]) * x[p] for p in range(n_projects)),
            GRB.MAXIMIZE
        )
        
        # Budget constraint: Σ_p c(p) · x_p ≤ B
        model.addConstr(
            gp.quicksum(costs[p] * x[p] for p in range(n_projects)) <= budget,
            name="budget"
        )
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            committee = [p for p in range(n_projects) if x[p].X > 0.5]
            score = int(round(model.ObjVal))
            return committee, score
        else:
            if verbose:
                print(f"AV ILP failed with status {model.status}")
            return [], 0
            
    except gp.GurobiError as e:
        print(f"Gurobi error in AV ILP: {e}")
        return [], 0


def maximize_cc_ilp(
    M: np.ndarray,
    costs: List[int],
    budget: int,
    verbose: bool = False,
    time_limit: int = 300
) -> Tuple[List[int], int]:
    """
    Maximize Chamberlin-Courant (CC) score using ILP.
    
    CC(W) = |{ v ∈ V : A_v ∩ W ≠ ∅ }|
    
    This is the Budgeted Maximum Coverage problem.
    
    Args:
        M: Approval matrix (n_voters, n_projects), boolean
        costs: List of project costs
        budget: Total budget B
        verbose: Whether to print Gurobi output
        time_limit: Time limit in seconds
        
    Returns:
        (committee, score): List of project indices and optimal CC score
    """
    n_voters, n_projects = M.shape
    
    try:
        model = gp.Model("maximize_cc")
        model.setParam('OutputFlag', 1 if verbose else 0)
        model.setParam('TimeLimit', time_limit)
        
        # Decision variables
        x = model.addVars(n_projects, vtype=GRB.BINARY, name="x")  # Project selection
        y = model.addVars(n_voters, vtype=GRB.BINARY, name="y")    # Voter representation
        
        # Objective: maximize number of represented voters
        model.setObjective(
            gp.quicksum(y[v] for v in range(n_voters)),
            GRB.MAXIMIZE
        )
        
        # Budget constraint
        model.addConstr(
            gp.quicksum(costs[p] * x[p] for p in range(n_projects)) <= budget,
            name="budget"
        )
        
        # Coverage constraint: y_v ≤ Σ_p a_{v,p} · x_p
        # Voter v can only be represented if at least one of their approved projects is selected
        for v in range(n_voters):
            approved_projects = np.where(M[v, :])[0]
            if len(approved_projects) > 0:
                model.addConstr(
                    y[v] <= gp.quicksum(x[p] for p in approved_projects),
                    name=f"cover_{v}"
                )
            else:
                # Voter approves nothing, cannot be represented
                model.addConstr(y[v] == 0, name=f"cover_{v}")
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            committee = [p for p in range(n_projects) if x[p].X > 0.5]
            score = int(round(model.ObjVal))
            return committee, score
        else:
            if verbose:
                print(f"CC ILP failed with status {model.status}")
            return [], 0
            
    except gp.GurobiError as e:
        print(f"Gurobi error in CC ILP: {e}")
        return [], 0


def maximize_pairs_ilp(
    M: np.ndarray,
    costs: List[int],
    budget: int,
    verbose: bool = False,
    time_limit: int = 300
) -> Tuple[List[int], int]:
    """
    Maximize PAIRS score using ILP.
    
    PAIRS(W) = |{ {i,j} ⊆ V : A_i ∩ A_j ∩ W ≠ ∅ }|
    
    Args:
        M: Approval matrix (n_voters, n_projects), boolean
        costs: List of project costs
        budget: Total budget B
        verbose: Whether to print Gurobi output
        time_limit: Time limit in seconds
        
    Returns:
        (committee, score): List of project indices and optimal PAIRS score
    """
    n_voters, n_projects = M.shape
    
    # Precompute I_{ij} = A_i ∩ A_j for all voter pairs
    # Only consider pairs where I_{ij} ≠ ∅ (optimization)
    if verbose:
        print(f"Precomputing voter pair intersections for {n_voters} voters...")
    
    # Compute shared approval matrix: shared[i,j] = list of projects both approve
    pair_intersections: Dict[Tuple[int, int], List[int]] = {}
    
    for i in range(n_voters):
        approved_i = set(np.where(M[i, :])[0])
        if not approved_i:
            continue
        for j in range(i + 1, n_voters):
            approved_j = set(np.where(M[j, :])[0])
            intersection = approved_i & approved_j
            if intersection:
                pair_intersections[(i, j)] = list(intersection)
    
    if verbose:
        print(f"Found {len(pair_intersections)} voter pairs with shared approvals")
    
    if len(pair_intersections) == 0:
        # No pairs share any approvals, PAIRS score is always 0
        return [], 0
    
    try:
        model = gp.Model("maximize_pairs")
        model.setParam('OutputFlag', 1 if verbose else 0)
        model.setParam('TimeLimit', time_limit)
        
        # Decision variables
        x = model.addVars(n_projects, vtype=GRB.BINARY, name="x")  # Project selection
        
        # y_{ij} for each pair with non-empty intersection
        y = model.addVars(pair_intersections.keys(), vtype=GRB.BINARY, name="y")
        
        # Objective: maximize number of connected pairs
        model.setObjective(
            gp.quicksum(y[pair] for pair in pair_intersections.keys()),
            GRB.MAXIMIZE
        )
        
        # Budget constraint
        model.addConstr(
            gp.quicksum(costs[p] * x[p] for p in range(n_projects)) <= budget,
            name="budget"
        )
        
        # Pair coverage constraint: y_{ij} ≤ Σ_{p ∈ I_{ij}} x_p
        for pair, intersection in pair_intersections.items():
            model.addConstr(
                y[pair] <= gp.quicksum(x[p] for p in intersection),
                name=f"pair_{pair[0]}_{pair[1]}"
            )
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            committee = [p for p in range(n_projects) if x[p].X > 0.5]
            score = int(round(model.ObjVal))
            return committee, score
        else:
            if verbose:
                print(f"PAIRS ILP failed with status {model.status}")
            return [], 0
            
    except gp.GurobiError as e:
        print(f"Gurobi error in PAIRS ILP: {e}")
        return [], 0


def maximize_cons_ilp(
    M: np.ndarray,
    costs: List[int],
    budget: int,
    verbose: bool = False,
    time_limit: int = 600
) -> Tuple[List[int], int]:
    """
    Maximize CONS score using ILP with flow-based connectivity.
    
    CONS(W) = |{ {u,v} ⊆ V : u and v are connected by W }|
    
    Two voters are connected if there's a path where each step shares
    an approved project in W.
    
    Uses multi-commodity flow formulation for connectivity.
    
    WARNING: This formulation has O(|V|^4) flow variables in the worst case
    and may be slow for large instances.
    
    Args:
        M: Approval matrix (n_voters, n_projects), boolean
        costs: List of project costs
        budget: Total budget B
        verbose: Whether to print Gurobi output
        time_limit: Time limit in seconds
        
    Returns:
        (committee, score): List of project indices and optimal CONS score
    """
    n_voters, n_projects = M.shape
    
    # Precompute edges: voter pairs with shared approvals
    if verbose:
        print(f"Precomputing potential edges for {n_voters} voters...")
    
    # edge_projects[(i,j)] = list of projects in A_i ∩ A_j
    edge_projects: Dict[Tuple[int, int], List[int]] = {}
    
    for i in range(n_voters):
        approved_i = set(np.where(M[i, :])[0])
        if not approved_i:
            continue
        for j in range(i + 1, n_voters):
            approved_j = set(np.where(M[j, :])[0])
            intersection = approved_i & approved_j
            if intersection:
                edge_projects[(i, j)] = list(intersection)
    
    if verbose:
        print(f"Found {len(edge_projects)} potential edges")
    
    if len(edge_projects) == 0:
        # No edges possible, CONS score is always 0
        return [], 0
    
    # Build list of unordered edges
    edges = list(edge_projects.keys())
    
    # For flow, we need directed arcs from each undirected edge
    # Arc (i, j) means flow from i to j
    arcs = []
    for (i, j) in edges:
        arcs.append((i, j))
        arcs.append((j, i))
    arc_set = set(arcs)
    
    # Build adjacency list for flow conservation
    neighbors: Dict[int, List[int]] = {v: [] for v in range(n_voters)}
    for (i, j) in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    
    # Determine which voter pairs to consider for connectivity
    # Only pairs that could potentially be connected (i.e., both have some approvals)
    voter_pairs = []
    voters_with_approvals = [v for v in range(n_voters) if M[v, :].any()]
    for i, vi in enumerate(voters_with_approvals):
        for vj in voters_with_approvals[i + 1:]:
            voter_pairs.append((vi, vj))
    
    if verbose:
        print(f"Considering {len(voter_pairs)} voter pairs for connectivity")
        print(f"Number of arcs: {len(arcs)}")
    
    try:
        model = gp.Model("maximize_cons")
        model.setParam('OutputFlag', 1 if verbose else 0)
        model.setParam('TimeLimit', time_limit)
        
        # Decision variables
        # x_p: project selection
        x = model.addVars(n_projects, vtype=GRB.BINARY, name="x")
        
        # e_{ij}: edge activation (1 if edge {i,j} is active)
        e = model.addVars(edges, vtype=GRB.BINARY, name="e")
        
        # y_{st}: connectivity indicator (1 if s and t are connected)
        y = model.addVars(voter_pairs, vtype=GRB.BINARY, name="y")
        
        # f^{s,t}_{i→j}: flow from i to j for commodity (s,t)
        # Only create flow variables for (s,t) pairs and arcs
        flow = {}
        for (s, t) in voter_pairs:
            for arc in arcs:
                flow[(s, t, arc[0], arc[1])] = model.addVar(
                    lb=0, ub=1, vtype=GRB.CONTINUOUS,
                    name=f"f_{s}_{t}_{arc[0]}_{arc[1]}"
                )
        
        # Objective: maximize connected pairs
        model.setObjective(
            gp.quicksum(y[pair] for pair in voter_pairs),
            GRB.MAXIMIZE
        )
        
        # Budget constraint
        model.addConstr(
            gp.quicksum(costs[p] * x[p] for p in range(n_projects)) <= budget,
            name="budget"
        )
        
        # Edge activation: e_{ij} ≤ Σ_{p ∈ I_{ij}} x_p
        for edge in edges:
            projects = edge_projects[edge]
            model.addConstr(
                e[edge] <= gp.quicksum(x[p] for p in projects),
                name=f"edge_{edge[0]}_{edge[1]}"
            )
        
        # Flow capacity: f^{s,t}_{i→j} ≤ e_{ij}
        # Need to map directed arc to undirected edge
        def get_edge(i, j):
            return (i, j) if (i, j) in edge_projects else (j, i)
        
        for (s, t) in voter_pairs:
            for arc in arcs:
                i, j = arc
                edge = get_edge(i, j)
                model.addConstr(
                    flow[(s, t, i, j)] <= e[edge],
                    name=f"cap_{s}_{t}_{i}_{j}"
                )
        
        # Flow conservation constraints
        for (s, t) in voter_pairs:
            # At source s: outflow - inflow = y_{st}
            outflow_s = gp.quicksum(flow[(s, t, s, j)] for j in neighbors[s])
            inflow_s = gp.quicksum(flow[(s, t, j, s)] for j in neighbors[s])
            model.addConstr(outflow_s - inflow_s == y[(s, t)], name=f"source_{s}_{t}")
            
            # At sink t: inflow - outflow = y_{st}
            inflow_t = gp.quicksum(flow[(s, t, j, t)] for j in neighbors[t])
            outflow_t = gp.quicksum(flow[(s, t, t, j)] for j in neighbors[t])
            model.addConstr(inflow_t - outflow_t == y[(s, t)], name=f"sink_{s}_{t}")
            
            # At intermediate nodes v ∉ {s,t}: outflow - inflow = 0
            for v in range(n_voters):
                if v == s or v == t or v not in neighbors or not neighbors[v]:
                    continue
                outflow_v = gp.quicksum(flow[(s, t, v, j)] for j in neighbors[v])
                inflow_v = gp.quicksum(flow[(s, t, j, v)] for j in neighbors[v])
                model.addConstr(outflow_v - inflow_v == 0, name=f"cons_{s}_{t}_{v}")
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            committee = [p for p in range(n_projects) if x[p].X > 0.5]
            score = int(round(model.ObjVal))
            return committee, score
        else:
            if verbose:
                print(f"CONS ILP failed with status {model.status}")
            return [], 0
            
    except gp.GurobiError as e:
        print(f"Gurobi error in CONS ILP: {e}")
        return [], 0


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
    
    print("=" * 60)
    print("TEST: 4 voters, 4 projects, budget=200")
    print("=" * 60)
    
    print("\n1. Testing AV ILP...")
    committee, score = maximize_av_ilp(M, costs, budget, verbose=False)
    print(f"   Committee: {committee}, AV score: {score}")
    
    print("\n2. Testing CC ILP...")
    committee, score = maximize_cc_ilp(M, costs, budget, verbose=False)
    print(f"   Committee: {committee}, CC score: {score}")
    
    print("\n3. Testing PAIRS ILP...")
    committee, score = maximize_pairs_ilp(M, costs, budget, verbose=False)
    print(f"   Committee: {committee}, PAIRS score: {score}")
    
    print("\n4. Testing CONS ILP...")
    committee, score = maximize_cons_ilp(M, costs, budget, verbose=False)
    print(f"   Committee: {committee}, CONS score: {score}")
