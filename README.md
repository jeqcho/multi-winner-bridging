# Multi-Winner Bridging

A Python system to calculate and analyze approval voting scores (AV, CC, PAIRS, CONS, EJR) for participatory budgeting committee selection using Pabulib datasets.

## Overview

This project analyzes **budget-constrained committee selection** for participatory budgeting:
- Select projects that fit within a budget constraint
- Projects have varying costs
- Exhaustively evaluate all budget-feasible committees

The pipeline:
1. Loads approval voting data from Pabulib PB datasets
2. Calculates 5 scoring metrics (AV, CC, PAIRS, CONS, EJR) for all valid committees
3. Computes alpha-approximations for each metric
4. Runs multiple voting methods (MES, greedy-AV, greedy-CC, greedy-PAV, and more)
5. Visualizes trade-offs between different metrics

## Supported Datasets

Participatory budgeting data from [Pabulib](https://pabulib.org/). See [Data Setup](#data-setup) to download the dataset.

The curated selection includes 338 PB instances filtered by:
- At most 13 projects
- At most 10k voters
- Approval voting format
- Excludes experimental runs

See the [filtered Pabulib query](https://pabulib.org/?votes_max=10000&projects_max=13&type=approval&exclude_experimental=true) for details.

## Installation

```bash
# Initialize and install dependencies
uv sync
```

## Data Setup

The participatory budgeting datasets are not included in the repository. Download them from Pabulib:

1. Download the dataset:
   ```bash
   curl -L -o data.zip "https://pabulib.org/download/snapshot/98659fc686f297cc"
   ```

2. Extract to the data/ directory:
   ```bash
   mkdir -p data && unzip data.zip -d data/
   ```

This downloads 338 curated PB instances from Poland, France, Netherlands, and the US.

### Gurobi Optimizer

This project requires [Gurobi Optimizer](https://www.gurobi.com/) for verifying EJR. We use the implementation of EJR checking from abcvoting, which uses Gurobi.

**Academic licenses are free!** If you're a student, faculty, or staff at a degree-granting academic institution:

1. Register for a free academic license at [Gurobi Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/)
2. Download your license key
3. Follow Gurobi's instructions to install the license (typically placing the `gurobi.lic` file in your home directory or setting the `GRB_LICENSE_FILE` environment variable)

## Project Structure

```
├── pyproject.toml              # Project configuration
├── main_pb.py                  # Runner for Pabulib PB data
│
├── src/
│   ├── pb_data_loader.py       # Load Pabulib PB data (.pb files)
│   ├── scoring.py              # Scoring functions (AV, CC, PAIRS, CONS, EJR)
│   ├── mes.py                  # Method of Equal Shares (budget-aware)
│   ├── voting_methods.py       # Voting methods (AV, CC, PAV, budget variants)
│   ├── alpha_ejr_pb_ilp.py     # ILP for optimal alpha-EJR
│   ├── pb_objectives_ilp.py    # ILP formulations for PB objectives
│   └── plot_ejr.py             # EJR-specific plots
│
├── scripts/                    # Analysis and plotting scripts (35 scripts)
│   ├── plot_alpha_*.py         # Alpha distribution plots
│   ├── plot_cons_vs_cc_*.py    # CONS vs CC analysis
│   ├── plot_metrics_bar.py     # Metrics bar charts
│   ├── plot_pareto_front.py    # Pareto frontier visualization
│   └── ...                     # Various analysis scripts
│
├── tests/
│   └── test_scoring.py         # Unit tests for scoring functions
│
├── reference/                  # Documentation and formulas
│
├── data/                       # Input datasets (download separately, see Data Setup)
│   └── *.pb                    # PB files from Pabulib
│
├── output/                     # Generated results (CSV files, plots)
└── analysis/                   # Analysis outputs (plots, JSON results)
```

## Usage

### Quick Start: Generate Visualizations

Pre-computed results for all 338 PB instances are already in `output/pb/`. To generate the summary visualizations:

```bash
# Generate the 3 main cross-dataset plots
uv run python scripts/plot_main_tradeoff_bar.py      # → analysis/main_tradeoff_bar.png
uv run python scripts/plot_metrics_bar.py           # → analysis/metrics_bar.png
uv run python scripts/plot_cons_vs_cc_scatter.py    # → analysis/cons_vs_cc_scatter.png
```

These scripts read from `output/pb/*/raw_scores.csv` and `output/pb/*/voting_results.csv` to produce aggregate analysis across all elections.

### Run Tests

```bash
uv run pytest tests/
```

### Step 1: Process PB Data with `main_pb.py`

`main_pb.py` is the core computation engine. For each PB election, it:
1. **Enumerates all budget-feasible committees** (exhaustive search)
2. **Scores each committee** on 4 metrics: AV, CC, PAIRS, CONS
3. **Runs 10 voting methods** (MES, greedy-AV, greedy-CC, etc.)
4. **Computes α-EJR** for each voting method's output
5. **Generates per-election plots** (alpha scatter, EJR analysis)

```bash
# Process a single PB file
uv run python main_pb.py data/poland_warszawa_2018_wola.pb

# Process all PB files in a directory
uv run python main_pb.py data/

# Test mode (runs on a small sample file)
uv run python main_pb.py --test
```

**Output per election** (saved to `output/pb/{dataset_name}/`):

| File | Description |
|------|-------------|
| `raw_scores.csv` | Scores for ALL budget-feasible committees |
| `voting_results.csv` | Results for 10 voting methods with alpha values |
| `alpha_scores.csv` | Alpha-approximations (normalized by global max) |
| `ejr_data.csv` | EJR analysis for voting methods |
| `alpha_plots.png` | Scatter plots (PAIRS/CONS vs AV/CC) |
| `ejr_plots.png` | Voting methods vs EJR |

### Step 2: Generate Cross-Dataset Visualizations

After `main_pb.py` has processed elections, run analysis scripts to aggregate results:

```bash
# Main summary plots
uv run python scripts/plot_main_tradeoff_bar.py      # Trade-off (1,1) achievement
uv run python scripts/plot_metrics_bar.py           # Per-metric optimality
uv run python scripts/plot_cons_vs_cc_scatter.py    # CONS vs CC relationship

# Additional analysis
uv run python scripts/plot_pareto_front.py          # Pareto frontier
uv run python scripts/plot_alpha_histograms.py      # Alpha distributions
```

**What each plot shows:**

| Plot | Description |
|------|-------------|
| `main_tradeoff_bar.png` | How often each voting method achieves (1,1) on trade-off pairs (e.g., PAIRS-AV, CONS-CC) |
| `metrics_bar.png` | How often each method achieves optimal (α=1) for individual metrics |
| `cons_vs_cc_scatter.png` | Relationship between CONS and CC across all committees in all elections |

### Run with Timestamped Logs

For long-running batch processing:

```bash
LOG_FILE="logs/pb_$(date +%Y%m%d_%H%M%S).log" && \
uv run python -u main_pb.py data/ 2>&1 | tee "$LOG_FILE"
```

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

See `reference/` for detailed mathematical definitions.

## Voting Methods

### Budget-Aware Methods

| Method | Description |
|--------|-------------|
| **MES** | Method of Equal Shares (budget-aware) |
| **greedy-AV** | Select projects by approval count within budget |
| **greedy-AV/cost** | Select by approval/cost ratio (cost-effectiveness) |
| **greedy-AV/cost²** | Select by approval/cost² ratio (penalizes expensive projects) |
| **greedy-CC** | Greedy coverage maximization within budget |
| **greedy-PAV** | Greedy PAV within budget |

## References

- Dong et al., "Selecting Interlacing Committees" (2024)
- Pabulib - https://pabulib.org/
- Stolicki, Szufa, Talmon. "Pabulib: A Participatory Budgeting Library" (2020)

## License

MIT License - see project for details.
