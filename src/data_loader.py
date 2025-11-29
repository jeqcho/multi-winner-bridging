"""Load and combine PrefLib approval voting data files."""

import numpy as np
from preflibtools.instances import CategoricalInstance


def load_preflib_file(url):
    """
    Load a single PrefLib categorical approval voting file from a URL.
    
    Args:
        url: URL to the .cat file (e.g., from PrefLib-Data GitHub)
    
    Returns:
        tuple: (M, candidate_names) where:
            - M: numpy array of shape (n_voters, n_candidates), boolean matrix
                 M[v][c] = 1 if voter v approves candidate c
            - candidate_names: list of candidate names/IDs
    """
    print(f"Loading {url}...")
    instance = CategoricalInstance()
    instance.parse_url(url)
    
    # Get candidate information
    candidate_names = list(instance.alternatives_name.values())
    n_candidates = len(candidate_names)
    print(f"Loaded {n_candidates} candidates: {candidate_names}")
    
    # Parse approval ballots
    all_votes = []
    for pref_tuple in instance.preferences:
        approved_candidates = pref_tuple[0]  # Tuple of approved candidate IDs
        count = instance.multiplicity[pref_tuple]
        
        # Convert to boolean vector (candidate IDs are 1-indexed)
        ballot = np.zeros(n_candidates, dtype=bool)
        for candidate_id in approved_candidates:
            ballot[candidate_id - 1] = True  # Convert to 0-indexed
        
        # Add this ballot 'count' times
        for _ in range(count):
            all_votes.append(ballot)
    
    # Convert to numpy array
    M = np.array(all_votes, dtype=bool)
    
    print(f"Total: {M.shape[0]} voters with {M.shape[1]} candidates")
    
    return M, candidate_names


def load_and_combine_data():
    """
    Load and combine all 6 approval voting files from PrefLib dataset 00071
    (2007 French Presidential Election).
    
    Returns:
        tuple: (M, candidate_names) where:
            - M: numpy array of shape (n_voters, n_candidates), boolean matrix
                 M[v][c] = 1 if voter v approves candidate c
            - candidate_names: list of candidate names/IDs (length 12)
    """
    # Dataset 00071 approval voting files - base URL
    base_url = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/00071%20-%20french-approval-2007/"
    file_names = [
        "00071-00000001.cat",  # Cigne-approval
        "00071-00000002.cat",  # Illkirch10-approval  
        "00071-00000003.cat",  # Illkirch3-approval
        "00071-00000004.cat",  # Illkirch8-approval
        "00071-00000005.cat",  # Louvigny1-approval
        "00071-00000006.cat",  # Louvigny2-approval
    ]
    
    all_votes = []
    candidate_names = None
    n_candidates = None
    
    for file_name in file_names:
        # Load the instance from PrefLib
        url = base_url + file_name
        print(f"Loading {file_name}...")
        instance = CategoricalInstance()
        instance.parse_url(url)
        
        # Get candidate information (only need to do this once)
        if candidate_names is None:
            candidate_names = list(instance.alternatives_name.values())
            n_candidates = len(candidate_names)
            print(f"Loaded {n_candidates} candidates: {candidate_names}")
        
        # Parse approval ballots
        # In categorical instances, preferences is a list of tuples (approved, not_approved)
        # and multiplicity is a dict mapping each tuple to its count
        for pref_tuple in instance.preferences:
            approved_candidates = pref_tuple[0]  # Tuple of approved candidate IDs
            count = instance.multiplicity[pref_tuple]
            
            # Convert to boolean vector (candidate IDs are 1-indexed)
            ballot = np.zeros(n_candidates, dtype=bool)
            for candidate_id in approved_candidates:
                ballot[candidate_id - 1] = True  # Convert to 0-indexed
            
            # Add this ballot 'count' times
            for _ in range(count):
                all_votes.append(ballot)
        
        print(f"  Added {instance.num_voters} voters from {file_name}")
    
    # Convert to numpy array
    M = np.array(all_votes, dtype=bool)
    
    print(f"\nTotal: {M.shape[0]} voters with {M.shape[1]} candidates")
    
    return M, candidate_names


if __name__ == "__main__":
    # Test the data loader
    M, candidates = load_and_combine_data()
    print(f"\nDataset shape: {M.shape}")
    print(f"Total approvals: {M.sum()}")
    print(f"Average approvals per voter: {M.sum(axis=1).mean():.2f}")
    print(f"Approvals per candidate: {M.sum(axis=0)}")

