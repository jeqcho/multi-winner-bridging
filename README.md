# Multi-Winner Bridging

A comprehensive Python system to calculate and analyze approval voting scores (AV, CC, PAIRS, CONS, EJR) for committee selection from PrefLib voting datasets and Pabulib participatory budgeting data.

## Overview

This project supports two types of committee selection problems:

**Fixed-size committee selection** (PrefLib data):
- Select k candidates from m total candidates
- Exhaustively evaluate all possible committees

**Budget-constrained selection** (Pabulib PB data):
- Select projects that fit within a budget constraint
- Projects have varying costs
- Exhaustively evaluate all budget-feasible committees

The pipeline:
1. Loads approval voting data from PrefLib or Pabulib datasets
2. Calculates 5 scoring metrics (AV, CC, PAIRS, CONS, EJR) for all valid committees
3. Computes alpha-approximations for each metric
4. Runs multiple voting methods (MES, greedy-AV, greedy-CC, greedy-PAV, and more)
5. Visualizes trade-offs between different metrics

## Supported Datasets

### PrefLib Datasets (Fixed-size committees)

- **French Election (2007)**: Dataset 00071 - 12 candidates, 2836 voters from 6 polling stations
- **Camp Songs**: Dataset 00059
  - `file_02`: 8 candidates, 39 voters
  - `file_04`: 10 candidates, 56 voters

### Pabulib Datasets (Budget-constrained)

Participatory budgeting data from [Pabulib](https://pabulib.org/). The `data/` directory contains PB instances from Poland, France, and the US.

**Curated selection** (`data/pb_selected_10_20251202_023743/`): Top 10 PB instances filtered by:
- At most 13 projects
- At most 10k voters
- Approval voting format
- Excludes experimental runs
- Ranked by Pabulib quality score

See the [filtered Pabulib query](https://pabulib.org/?votes_max=10000&projects_max=13&type=approval&exclude_experimental=true) for details.

## Installation

```bash
# Initialize and install dependencies
uv sync
```

### Gurobi Optimizer

This project requires [Gurobi Optimizer](https://www.gurobi.com/) for verifying EJR. We use the implementation of EJR checking from abcvoting, which uses Gurobi.

**Academic licenses are free!** If you're a student, faculty, or staff at a degree-granting academic institution:

1. Register for a free academic license at [Gurobi Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/)
2. Download your license key
3. Follow Gurobi's instructions to install the license (typically placing the `gurobi.lic` file in your home directory or setting the `GRB_LICENSE_FILE` environment variable)

## Project Structure

```
├── pyproject.toml              # Project configuration
├── reference.md                # Score definitions and formulas
│
├── main.py                     # Runner for PrefLib data (fixed-size committees)
├── main_pb.py                  # Runner for Pabulib PB data (full pipeline with plots)
├── main_pb_batch.py            # Batch runner for PB data (no plots, faster)
│
├── src/
│   ├── data_loader.py          # Load PrefLib data
│   ├── pb_data_loader.py       # Load Pabulib PB data (.pb files)
│   ├── scoring.py              # Scoring functions (AV, CC, PAIRS, CONS, EJR)
│   ├── alpha_approx.py         # Alpha-approximation calculations (global)
│   ├── alpha_approx_by_size.py # Alpha-approximation calculations (by size)
│   ├── mes.py                  # Method of Equal Shares (budget-aware)
│   ├── run_mes.py              # MES runner for fixed-size committees
│   ├── voting_methods.py       # Voting methods (AV, CC, PAV, budget variants)
│   ├── plot_results.py         # Visualization (global)
│   ├── plot_results_by_size.py # Visualization (by size)
│   ├── plot_individual_sizes.py# Individual size plots
│   ├── plot_ejr.py             # EJR-specific plots
│   └── timer.py                # Time estimation
│
├── scripts/                    # Analysis and plotting scripts (26 scripts)
│   ├── plot_alpha_*.py         # Alpha distribution plots
│   ├── plot_cons_vs_cc_*.py    # CONS vs CC analysis
│   ├── plot_metrics_bar.py     # Metrics bar charts
│   ├── plot_pareto_front.py    # Pareto frontier visualization
│   └── ...                     # Various analysis scripts
│
├── tests/
│   ├── test_scoring.py         # Unit tests for scoring functions
│   └── test_data_loader.py     # Unit tests for data loader
│
├── data/                       # Input datasets
│   ├── pb_selected_10_*/       # Curated PB datasets
│   └── *.pb                    # Individual PB files
│
├── output/                     # Generated results (CSV files, plots)
├── analysis/                   # Analysis outputs (plots, JSON results)
└── presentation/               # Presentation materials and plots
```

## Usage

### Run Tests

```bash
uv run pytest tests/
```

### Estimate Computation Time

```bash
uv run python src/timer.py
```

### PrefLib Data Analysis (Fixed-size committees)

The main script supports PrefLib datasets:

```bash
# Process French Election dataset (2007)
uv run python main.py french_election

# Process all Camp Songs files
uv run python main.py camp_songs

# Process a specific Camp Songs file
uv run python main.py camp_songs --file file_02
uv run python main.py camp_songs --file file_04
```

### Pabulib PB Data Analysis (Budget-constrained)

For participatory budgeting data with budget constraints:

```bash
# Process a single PB file (full pipeline with plots)
uv run python main_pb.py data/poland_warszawa_2018_wola.pb

# Process all PB files in a directory
uv run python main_pb.py data/pb_selected_10_20251202_023743/

# Test mode (runs on a small sample file)
uv run python main_pb.py --test
```

For batch processing without plots (faster):

```bash
# Batch process all PB files in directory
uv run python main_pb_batch.py data/pb_selected_10_20251202_023743/

# Process single file in batch mode
uv run python main_pb_batch.py data/poland_warszawa_2018_wola.pb
```

### Run with Timestamped Logs

To save output to a timestamped log file (recommended for long-running analyses):

```bash
# French Election with logging
LOG_FILE="logs/french_election_$(date +%Y%m%d_%H%M%S).log" && \
uv run python -u main.py french_election 2>&1 | tee "$LOG_FILE"

# PB data with logging
LOG_FILE="logs/pb_$(date +%Y%m%d_%H%M%S).log" && \
uv run python -u main_pb.py data/pb_selected_10_20251202_023743/ 2>&1 | tee "$LOG_FILE"
```

The `-u` flag ensures unbuffered output for real-time logging.

### Output

#### PrefLib Output (fixed-size committees)

For each PrefLib dataset, the pipeline produces:

**CSV files:**
- `raw_scores.csv` - Raw scores (AV, CC, PAIRS, CONS) for all subsets
- `alpha_scores_by_size.csv` - Alpha-approximations by committee size
- `max_scores_by_size.csv` - Maximum scores per size
- `voting_results.csv` - Voting method results (MES, AV, greedy-CC, greedy-PAV)

**Plots:**
- `alpha_plots_by_size.png` - Alpha plots by committee size
- `ejr_plots.png` - EJR analysis with voting methods
- `by_size/size_XX.png` - Individual plots for each committee size

Output locations:
- `output/french_election/` for French Election
- `output/camp_songs/file_02/` and `output/camp_songs/file_04/` for Camp Songs

#### Pabulib PB Output (budget-constrained)

For each PB dataset, `main_pb.py` produces:

**CSV files:**
- `raw_scores.csv` - Raw scores for all budget-feasible committees
- `alpha_scores.csv` - Alpha-approximations (normalized by global max)
- `max_scores.csv` - Maximum scores across all valid committees
- `voting_results.csv` - Results for 10 voting methods
- `ejr_data.csv` - EJR analysis for voting methods

**Plots:**
- `alpha_plots.png` - Alpha scatter plots (PAIRS/CONS vs AV/CC)
- `ejr_plots.png` - Voting methods vs EJR
- `ejr_plots_zoomed.png` - Zoomed EJR plots (0.8-1.0 range)

Output location: `output/pb/{dataset_name}/`

#### Batch Mode Output

`main_pb_batch.py` produces only CSV files (no plots):
- `raw_scores.csv`
- `voting_results.csv` (includes EJR boolean)

#### Analysis Outputs

The `analysis/` directory contains cross-dataset analysis:
- `alpha_histograms*.png` - Distribution of alpha values
- `cons_vs_cc_*.png` - CONS vs CC relationship analysis
- `pareto_front_scatter.png` - Pareto frontier visualization
- `metrics_bar.png` - Comparative metrics bar chart
- `ejr_*.json` - EJR analysis results

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

### α-EJR
Maximum α ∈ [0,1] such that the committee satisfies α-EJR (relaxed version of EJR).

See `reference.md` for detailed mathematical definitions.

## Voting Methods

### Greedy Methods (Fixed-size)

| Method | Description |
|--------|-------------|
| **AV** | Select top-k candidates by approval count |
| **greedy-CC** | Greedily maximize coverage (CC score) |
| **greedy-PAV** | Greedy Proportional Approval Voting with harmonic weights |
| **MES** | Method of Equal Shares |

### Budget-Aware Methods (PB data)

| Method | Description |
|--------|-------------|
| **MES** | Method of Equal Shares (budget-aware) |
| **greedy-AV** | Select projects by approval count within budget |
| **greedy-AV/cost** | Select by approval/cost ratio (cost-effectiveness) |
| **greedy-AV/cost²** | Select by approval/cost² ratio (penalizes expensive projects) |
| **greedy-CC** | Greedy coverage maximization within budget |
| **greedy-PAV** | Greedy PAV within budget |

### Max-Score Methods (from exhaustive search)

| Method | Description |
|--------|-------------|
| **PAIRS-AV** | Max PAIRS score, tiebreak by AV |
| **PAIRS-CC** | Max PAIRS score, tiebreak by CC |
| **CONS-AV** | Max CONS score, tiebreak by AV |
| **CONS-CC** | Max CONS score, tiebreak by CC |

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
- PrefLib - https://preflib.github.io/PrefLib-Jekyll/
- Pabulib - https://pabulib.org/
- Dataset 00071: 2007 French Presidential Election Approval Voting
- Dataset 00059: Camp Songs

## Citation

If you use this code or the datasets, please cite:
- Nicholas Mattei and Toby Walsh. PrefLib: A Library for Preference Data. Proceedings of Third International Conference on Algorithmic Decision Theory (ADT 2013)
- Pabulib: Stolicki, Szufa, Talmon. "Pabulib: A Participatory Budgeting Library" (2020)

## License

MIT License - see project for details.
