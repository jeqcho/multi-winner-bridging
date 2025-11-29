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

#### Output

For each dataset, the pipeline produces:

**CSV files:**
- `raw_scores.csv` - Raw scores for all subsets
- `alpha_scores.csv` - Alpha-approximations (global)
- `alpha_scores_by_size.csv` - Alpha-approximations by committee size
- `max_scores_by_size.csv` - Maximum scores per size
- `mes_results.csv` - MES algorithm results

**Plots:**
- `alpha_plots.png` - Global alpha approximation plots
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

- **PAIRS** is the bottleneck (~91% of computation time)
- French Election (12 candidates, 4096 subsets): ~34 minutes
- Camp Songs file_02 (8 candidates, 256 subsets): ~2 minutes
- Camp Songs file_04 (10 candidates, 1024 subsets): ~10 minutes

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
