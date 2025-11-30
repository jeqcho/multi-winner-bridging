# Multi-Winner Bridging

A comprehensive Python system to calculate and analyze approval voting scores (AV, CC, PAIRS, CONS, EJR) for all possible committee subsets from PrefLib voting datasets.

## Overview

This project:
1. Loads approval voting data from PrefLib datasets
2. Calculates 5 different scoring metrics for all possible committee subsets
3. Computes alpha-approximations for each metric
4. Runs the Method of Equal Shares (MES) algorithm
5. Visualizes the relationships between different metrics

## Supported Datasets

- **French Election (2007)**: Dataset 00071 - 12 candidates, 2836 voters from 6 polling stations
- **Camp Songs**: Dataset 00059
  - `file_02`: 8 candidates, 39 voters
  - `file_04`: 10 candidates, 56 voters

## Installation

```bash
# Initialize and install dependencies
uv sync
```

## Project Structure

```
├── pyproject.toml              # Project configuration
├── reference.md                # Score definitions and formulas
├── src/
│   ├── data_loader.py          # Load PrefLib data
│   ├── scoring.py              # Scoring functions (AV, CC, PAIRS, CONS, EJR)
│   ├── alpha_approx.py         # Alpha-approximation calculations (global)
│   ├── alpha_approx_by_size.py # Alpha-approximation calculations (by size)
│   ├── run_mes.py              # Method of Equal Shares
│   ├── plot_results.py         # Visualization (global)
│   ├── plot_results_by_size.py # Visualization (by size)
│   ├── plot_individual_sizes.py# Individual size plots
│   └── timer.py                # Time estimation
├── tests/
│   ├── test_scoring.py         # Unit tests for scoring functions
│   └── test_data_loader.py     # Unit tests for data loader
├── main.py                     # Main runner script
└── output/                     # Generated results and plots
```

## Usage

### Run Tests

```bash
uv run python tests/test_scoring.py
uv run python tests/test_data_loader.py
```

### Estimate Computation Time

```bash
uv run python src/timer.py
```

### Run Analysis

The main script supports both datasets via command line arguments:

```bash
# Process French Election dataset (2007)
uv run python main.py french_election

# Process all Camp Songs files
uv run python main.py camp_songs

# Process a specific Camp Songs file
uv run python main.py camp_songs --file file_02
uv run python main.py camp_songs --file file_04
```

### Run with Timestamped Logs

To save output to a timestamped log file (recommended for long-running analyses):

```bash
# French Election with logging
LOG_FILE="logs/french_election_$(date +%Y%m%d_%H%M%S).log" && \
uv run python -u main.py french_election 2>&1 | tee "$LOG_FILE"

# Camp Songs with logging
LOG_FILE="logs/camp_songs_$(date +%Y%m%d_%H%M%S).log" && \
uv run python -u main.py camp_songs 2>&1 | tee "$LOG_FILE"
```

The `-u` flag ensures unbuffered output for real-time logging. Logs are saved to the `logs/` directory with format `{dataset}_{YYYYMMDD_HHMMSS}.log`.

#### Output

For each dataset, the pipeline produces:

**CSV files:**
- `raw_scores.csv` - Raw scores for all subsets
- `alpha_scores_by_size.csv` - Alpha-approximations by committee size
- `max_scores_by_size.csv` - Maximum scores per size
- `voting_results.csv` - Voting method results (MES, AV, CC, PAV)

**Plots:**
- `alpha_plots_by_size.png` - Alpha plots by committee size
- `by_size/size_XX.png` - Individual plots for each committee size

Output is saved to:
- `output/french_election/` for French Election
- `output/camp_songs/file_02/` and `output/camp_songs/file_04/` for Camp Songs

## Scoring Metrics

### AV (Approval Voting)
Total number of approvals for committee members.

### CC (Chamberlin-Courant Coverage)
Number of voters who approve at least one committee member.

### PAIRS (Direct Pair Coverage)
Number of unordered voter pairs that share at least one approved committee member.

### CONS (Connectivity)
Number of voter pairs in the same connected component (connected via shared approved candidates).

### EJR (Extended Justified Representation)
Boolean property: whether the committee satisfies proportional representation.

### β-EJR
Maximum β ∈ [0,1] such that the committee satisfies β-EJR (relaxed version of EJR).

See `reference.md` for detailed mathematical definitions.

## Performance

### Notation

| Variable | Description |
|----------|-------------|
| **n** | Number of voters |
| **m** | Number of candidates |
| **k** | Committee size |
| **α(n)** | Inverse Ackermann function (effectively constant, ≤4 for practical n) |

### Algorithm Time Complexity

| Algorithm | Description | Time Complexity | Typical Runtime Share |
|-----------|-------------|-----------------|----------------------|
| **AV** | Sum of approvals for committee members | O(n × k) | ~0.1-6% |
| **CC** | Count voters with ≥1 approved member | O(n × k) | ~0.1-6% |
| **PAIRS** | Count voter pairs sharing ≥1 approved member | O(n² × k) | **~20-94%** (bottleneck) |
| **CONS** | Count voter pairs in same connected component | O(n × k × α(n)) | ~5-56% |

### Implementation Details

- **AV**: Simple matrix sum over committee columns: `M[:, W].sum()`
- **CC**: Row-wise OR check: `(M[:, W].sum(axis=1) > 0).sum()`
- **PAIRS**: Matrix multiplication `M_W @ M_W.T`, count upper triangle > 0
- **CONS**: Union-Find data structure; union all supporters per candidate, sum C(|component|, 2)

### Runtime Benchmarks

**French Election** (n=2836 voters, m=12 candidates, 2^m=4096 subsets):
| Algorithm | Time | Share |
|-----------|------|-------|
| PAIRS | 96.67s | **94.2%** |
| CONS | 5.55s | 5.4% |
| AV | 0.30s | 0.3% |
| CC | 0.09s | 0.1% |
| **Total** | **102.66s** (~1.7 min) | |

**Camp Songs file_02** (n=39 voters, m=8 candidates, 256 subsets):
- Total: 0.02s
- CONS: 34.4%, PAIRS: 19.9%, CC: 6.5%, AV: 6.2%

**Camp Songs file_04** (n=56 voters, m=10 candidates, 1024 subsets):
- Total: 0.07s
- CONS: 56.4%, PAIRS: 26.1%, CC: 6.1%, AV: 4.5%

### Key Insight

**PAIRS is the bottleneck** for large voter counts (n) due to its O(n²) matrix multiplication. For small datasets, CONS takes a relatively larger share since its O(n) term becomes comparable to other algorithms.

## Key Findings

Run the analysis to discover:
- Which committees maximize each metric
- Trade-offs between different objectives
- How connectivity (PAIRS/CONS) relates to representation (AV/CC/EJR)
- The Pareto frontier for multi-objective optimization

## References

- Dong et al., "Selecting Interlacing Committees" (2024)
- PrefLib: A Library for Preferences - https://preflib.github.io/PrefLib-Jekyll/
- Dataset 00071: 2007 French Presidential Election Approval Voting
- Dataset 00059: Camp Songs

## Citation

If you use this code or the datasets, please cite:
- Nicholas Mattei and Toby Walsh. PrefLib: A Library of Preference Data. Proceedings of Third International Conference on Algorithmic Decision Theory (ADT 2013)

## License

MIT License - see project for details.
