"""Load and parse pabulib participatory budgeting data files (.pb format)."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm


def parse_pb_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a pabulib .pb file into structured data.
    
    The .pb format has three sections:
    - META: key;value pairs with metadata (budget, num_projects, etc.)
    - PROJECTS: project_id;cost;votes;name
    - VOTES: voter_id;vote (comma-separated project IDs)
    
    Args:
        filepath: Path to the .pb file
        
    Returns:
        Dictionary with keys:
            - 'meta': dict of metadata
            - 'projects': dict mapping project_id to {cost, votes, name}
            - 'votes': dict mapping voter_id to list of approved project_ids
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into sections
    lines = content.strip().split('\n')
    
    result = {
        'meta': {},
        'projects': {},
        'votes': {}
    }
    
    current_section = None
    header_line = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line == 'META':
            current_section = 'meta'
            continue
        elif line == 'PROJECTS':
            current_section = 'projects'
            continue
        elif line == 'VOTES':
            current_section = 'votes'
            continue
        
        # Parse based on current section
        if current_section == 'meta':
            if line == 'key;value':
                continue  # Skip header
            parts = line.split(';', 1)
            if len(parts) == 2:
                key, value = parts
                # Try to convert to number if possible
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                result['meta'][key] = value
                
        elif current_section == 'projects':
            if line.startswith('project_id;'):
                continue  # Skip header
            parts = line.split(';', 3)
            if len(parts) >= 3:
                project_id = parts[0]
                cost = int(parts[1])
                votes = int(parts[2])
                name = parts[3] if len(parts) > 3 else ''
                result['projects'][project_id] = {
                    'cost': cost,
                    'votes': votes,
                    'name': name
                }
                
        elif current_section == 'votes':
            if line.startswith('voter_id;'):
                continue  # Skip header
            parts = line.split(';', 1)
            if len(parts) == 2:
                voter_id = parts[0]
                approved_projects = parts[1].split(',') if parts[1] else []
                # Clean up project IDs
                approved_projects = [p.strip() for p in approved_projects if p.strip()]
                result['votes'][voter_id] = approved_projects
    
    return result


def load_pb_file(filepath: str) -> Tuple[np.ndarray, List[str], List[int], int]:
    """
    Load a pabulib .pb file and convert to approval matrix format.
    
    Args:
        filepath: Path to the .pb file
        
    Returns:
        tuple: (M, project_ids, project_costs, budget) where:
            - M: numpy array of shape (n_voters, n_projects), boolean matrix
                 M[v][p] = 1 if voter v approves project p
            - project_ids: list of project IDs (strings)
            - project_costs: list of project costs (integers)
            - budget: total budget (integer)
    """
    print(f"Loading {filepath}...")
    
    data = parse_pb_file(filepath)
    
    # Extract metadata
    budget = data['meta'].get('budget', 0)
    num_projects = data['meta'].get('num_projects', len(data['projects']))
    num_votes = data['meta'].get('num_votes', len(data['votes']))
    
    print(f"  Budget: {budget:,}")
    print(f"  Projects: {num_projects}")
    print(f"  Voters: {num_votes}")
    
    # Create ordered list of projects (maintain consistent ordering)
    project_ids = list(data['projects'].keys())
    project_costs = [data['projects'][pid]['cost'] for pid in project_ids]
    
    # Create project_id to index mapping
    project_to_idx = {pid: idx for idx, pid in enumerate(project_ids)}
    
    # Build approval matrix
    n_voters = len(data['votes'])
    n_projects = len(project_ids)
    
    M = np.zeros((n_voters, n_projects), dtype=bool)
    
    for voter_idx, (voter_id, approved_projects) in enumerate(data['votes'].items()):
        for project_id in approved_projects:
            if project_id in project_to_idx:
                M[voter_idx, project_to_idx[project_id]] = True
    
    print(f"  Approval matrix shape: {M.shape}")
    print(f"  Total approvals: {M.sum():,}")
    print(f"  Avg approvals per voter: {M.sum(axis=1).mean():.2f}")
    print(f"  Project costs range: {min(project_costs):,} - {max(project_costs):,}")
    
    return M, project_ids, project_costs, budget


def get_pb_files(data_dir: str) -> List[str]:
    """
    Get all .pb files in a directory.
    
    Args:
        data_dir: Path to directory containing .pb files
        
    Returns:
        List of .pb file paths
    """
    data_path = Path(data_dir)
    pb_files = sorted(data_path.glob('*.pb'))
    return [str(f) for f in pb_files]


def enumerate_valid_committees(costs: List[int], budget: int, show_progress: bool = True) -> List[List[int]]:
    """
    Enumerate all budget-feasible committees using DFS with pruning.
    
    Uses DFS to find all subsets of projects whose total cost <= budget.
    Projects are sorted by cost (ascending) for efficient pruning:
    - Once a project is too expensive, all remaining projects (with higher cost) are also skipped.
    
    Args:
        costs: List of project costs
        budget: Total budget constraint
        show_progress: Whether to show progress bar
        
    Returns:
        List of valid committees, where each committee is a list of project indices
    """
    n = len(costs)
    
    # Sort projects by cost (ascending) for efficient pruning
    sorted_indices = sorted(range(n), key=lambda i: costs[i])
    sorted_costs = [costs[i] for i in sorted_indices]
    
    # First count to know the total (for progress bar)
    if show_progress:
        total_count = count_valid_committees(costs, budget)
        pbar = tqdm(total=total_count, desc="Enumerating committees", unit="committee")
    
    results = []
    
    def dfs(idx: int, current_cost: int, selected: List[int]):
        # Add current selection (including empty set)
        # Map back to original indices
        results.append([sorted_indices[i] for i in selected])
        if show_progress:
            pbar.update(1)
        
        # Try adding more projects
        for i in range(idx, n):
            if current_cost + sorted_costs[i] <= budget:
                dfs(i + 1, current_cost + sorted_costs[i], selected + [i])
            else:
                # Pruning: remaining projects cost more (sorted), so skip all
                break
    
    dfs(0, 0, [])
    
    if show_progress:
        pbar.close()
    
    return results


def count_valid_committees(costs: List[int], budget: int) -> int:
    """
    Count budget-feasible committees without storing them (for estimation).
    
    Args:
        costs: List of project costs
        budget: Total budget constraint
        
    Returns:
        Number of valid committees
    """
    n = len(costs)
    sorted_costs = sorted(costs)
    
    count = 0
    
    def dfs(idx: int, current_cost: int):
        nonlocal count
        count += 1
        
        for i in range(idx, n):
            if current_cost + sorted_costs[i] <= budget:
                dfs(i + 1, current_cost + sorted_costs[i])
            else:
                break
    
    dfs(0, 0)
    return count


if __name__ == "__main__":
    # Test the data loader
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "data/pb_selected_10_20251201_223154/netherlands_amsterdam_524_.pb"
    
    M, project_ids, project_costs, budget = load_pb_file(filepath)
    
    print(f"\nLoaded successfully:")
    print(f"  Matrix shape: {M.shape}")
    print(f"  Project IDs: {project_ids[:5]}..." if len(project_ids) > 5 else f"  Project IDs: {project_ids}")
    print(f"  Project costs: {project_costs[:5]}..." if len(project_costs) > 5 else f"  Project costs: {project_costs}")
    print(f"  Budget: {budget}")
    
    # Test committee enumeration
    print("\n--- Testing valid committee enumeration ---")
    num_valid = count_valid_committees(project_costs, budget)
    print(f"Number of valid committees: {num_valid:,}")
    print(f"Total possible committees (2^n): {2**len(project_costs):,}")
    print(f"Reduction factor: {2**len(project_costs) / num_valid:.1f}x")
    
    # Show a few example committees
    if num_valid <= 20:
        committees = enumerate_valid_committees(project_costs, budget, show_progress=False)
        print(f"\nAll {len(committees)} valid committees:")
        for i, c in enumerate(committees):
            total_cost = sum(project_costs[j] for j in c)
            print(f"  {i}: {c} (cost: {total_cost:,})")

