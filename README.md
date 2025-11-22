# PrefLib Score Calculator

A comprehensive Python system to calculate and analyze approval voting scores (AV, CC, PAIRS, CONS, EJR) for all possible committee subsets of the 2007 French Presidential Election data from PrefLib.

## Overview

This project:
1. Loads and combines 6 approval voting files from PrefLib dataset 00071 (2,836 voters, 12 candidates)
2. Calculates 5 different scoring metrics for all 2^12 = 4,096 possible committee subsets
3. Computes alpha-approximations for each metric
4. Visualizes the relationships between different metrics

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
│   ├── data_loader.py          # Load and combine PrefLib data
│   ├── scoring.py              # Scoring functions (AV, CC, PAIRS, CONS, EJR)
│   ├── alpha_approx.py         # Alpha-approximation calculations
│   ├── plot_results.py         # Visualization
│   └── timer.py                # Time estimation
├── tests/
│   ├── test_scoring.py         # Unit tests for scoring functions
│   └── test_data_loader.py     # Unit tests for data loader
└── main.py                     # Main computation script
```

## Usage

### 1. Run Tests

```bash
uv run python tests/test_scoring.py
uv run python tests/test_data_loader.py
```

### 2. Estimate Computation Time

```bash
uv run python src/timer.py
```

Expected output: ~34 minutes for all 4,096 subsets

### 3. Calculate Scores (Main Computation)

⚠️ **WARNING: This will take approximately 34 minutes**

```bash
uv run python main.py
```

This produces `raw_scores.csv` with columns:
- `subset_size`: Committee size (0-12)
- `subset_indices`: JSON array of candidate indices
- `AV`, `CC`, `PAIRS`, `CONS`: Raw scores
- `EJR`: Boolean (True/False)
- `beta_EJR`: Maximum β for β-EJR satisfaction (0-1)

### 4. Calculate Alpha-Approximations

```bash
uv run python src/alpha_approx.py
```

This reads `raw_scores.csv` and produces `alpha_scores.csv` with additional columns:
- `alpha_AV`, `alpha_CC`, `alpha_PAIRS`, `alpha_CONS`: Normalized scores (0-1)
- `alpha_EJR`: Same as `beta_EJR`

### 5. Create Visualizations

```bash
uv run python src/plot_results.py
```

This produces `alpha_plots.png` with 6 scatter plots (2×3 grid):

**Row 1: alpha_PAIRS (x-axis)**
- alpha_PAIRS vs beta_AV (average alpha_AV for each PAIRS value)
- alpha_PAIRS vs beta_CC
- alpha_PAIRS vs beta_EJR
- Reference line: `beta = 1 - alpha` (a + b = 1)

**Row 2: alpha_CONS (x-axis)**
- alpha_CONS vs beta_AV
- alpha_CONS vs beta_CC
- alpha_CONS vs beta_EJR
- Reference line: `beta = 1 - alpha²` (a² + b = 1)

Points are colored by committee size (0-12) using the viridis colormap with transparency to handle overlapping points.

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
- Average time per subset: ~500ms
- Total time for 4,096 subsets: ~34 minutes
- Output file sizes: ~500KB (CSV)

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

## Citation

If you use this code or the 2007 French Presidential Election data, please cite:
- Nicholas Mattei and Toby Walsh. PrefLib: A Library of Preference Data. Proceedings of Third International Conference on Algorithmic Decision Theory (ADT 2013)

## License

MIT License - see project for details.

