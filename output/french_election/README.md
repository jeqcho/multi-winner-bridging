# Output Files Summary

This directory contains all generated data and visualizations from the PrefLib Score Calculator.

## Directory Structure

```
output/
├── README.md                      (this file)
├── raw_scores.csv                 (raw scores for all 4,096 subsets)
├── alpha_scores_by_size.csv       (alpha scores normalized by size)
├── max_scores_by_size.csv         (max scores for each committee size)
├── voting_results.csv             (voting method results: MES, AV, CC, PAV)
├── alpha_plots_by_size.png        (size normalization plots)
└── by_size/                       (individual plots for each k)
    ├── size_00.png                (k=0: 1 subset)
    ├── size_01.png                (k=1: 12 subsets)
    ├── size_02.png                (k=2: 66 subsets)
    ├── size_03.png                (k=3: 220 subsets)
    ├── size_04.png                (k=4: 495 subsets)
    ├── size_05.png                (k=5: 792 subsets)
    ├── size_06.png                (k=6: 924 subsets)
    ├── size_07.png                (k=7: 792 subsets)
    ├── size_08.png                (k=8: 495 subsets)
    ├── size_09.png                (k=9: 220 subsets)
    ├── size_10.png                (k=10: 66 subsets)
    ├── size_11.png                (k=11: 12 subsets)
    └── size_12.png                (k=12: 1 subset)
```

## File Descriptions

### Data Files

#### `raw_scores.csv`
Raw scores for all 4,096 committee subsets.

**Columns:**
- `subset_size`: Committee size (0-12)
- `subset_indices`: JSON array of candidate indices
- `AV`: Approval Voting score
- `CC`: Chamberlin-Courant Coverage score
- `PAIRS`: Direct pair coverage score
- `CONS`: Connectivity score
- `EJR`: Boolean (True/False) - Extended Justified Representation
- `alpha_EJR`: Maximum α for α-EJR satisfaction (0-1)

#### `alpha_scores_by_size.csv`
Alpha-approximations normalized by **size-specific maximum** (within each committee size).

**Additional Columns:**
- `alpha_AV`: AV / max_AV_for_this_size
- `alpha_CC`: CC / max_CC_for_this_size
- `alpha_PAIRS`: PAIRS / max_PAIRS_for_this_size
- `alpha_CONS`: CONS / max_CONS_for_this_size
- `alpha_EJR`: Maximum α for α-EJR satisfaction

**Use case:** Compare committees within the same size class; understand trade-offs with size constraints.

#### `max_scores_by_size.csv`
Maximum achievable scores for each committee size.

**Columns:**
- `subset_size`: Committee size (k)
- `max_AV`: Maximum AV score for this size
- `max_CC`: Maximum CC score for this size
- `max_PAIRS`: Maximum PAIRS score for this size
- `max_CONS`: Maximum CONS score for this size

**Key insight:** Shows diminishing returns - e.g., CONS plateaus at k=10.

### Visualization Files

#### `alpha_plots_by_size.png`
Combined plot (2×3 grid) showing all committee sizes together, with alpha scores normalized by size.

**Color coding:** Committee size (viridis colormap)

**Use case:** Compare relative performance within each size class.

#### `by_size/size_XX.png` (13 files)
Individual plots for each committee size (k=0 to k=12).

**Color coding:** alpha_AV value (shows which committees have high AV scores)

**Use case:** Deep dive into specific committee sizes; see detailed trade-offs without clutter from other sizes.

## Plot Layout (All Plots)

Each visualization is a 2×3 grid:

### Row 1: alpha_PAIRS as x-axis
1. **PAIRS vs AV**: Trade-off between pair coverage and approval voting
2. **PAIRS vs CC**: Trade-off between pair coverage and voter coverage
3. **PAIRS vs EJR**: Pair coverage vs proportional representation

**Reference line:** `beta = 1 - alpha` (a + b = 1)

### Row 2: alpha_CONS as x-axis
4. **CONS vs AV**: Trade-off between connectivity and approval voting
5. **CONS vs CC**: Trade-off between connectivity and voter coverage (strong correlation!)
6. **CONS vs EJR**: Connectivity vs proportional representation

**Reference line:** `beta = 1 - alpha²` (a² + b = 1)

## Key Findings

### From `max_scores_by_size.csv`:
- **AV** grows linearly with committee size (more candidates = more approvals)
- **CC** plateaus quickly at k≥10 (2,694 voters, almost all voters covered)
- **PAIRS** grows with size, plateaus around k=10-12
- **CONS** plateaus at k=10 (3,627,471 pairs - adding more candidates doesn't improve connectivity)

### From Visualizations:
- **100% EJR satisfaction** - Every committee satisfies proportional representation!
- **CONS vs CC correlation** - Strong positive relationship between connectivity and coverage
- **Trade-offs** - High PAIRS scores often correlate with high AV/CC scores (not strictly opposing)
- **Size matters** - Larger committees generally dominate on all metrics

## Usage Examples

### Find best k=6 committee for PAIRS:
```python
import pandas as pd
df = pd.read_csv('alpha_scores_by_size.csv')
k6 = df[df['subset_size'] == 6]
best = k6.loc[k6['alpha_PAIRS'].idxmax()]
print(f"Best committee: {best['subset_indices']}")
```

### Compare trade-offs at k=5:
```python
k5 = df[df['subset_size'] == 5]
# Committees with high PAIRS but low AV
interesting = k5[(k5['alpha_PAIRS'] > 0.8) & (k5['alpha_AV'] < 0.6)]
```

### Find committees good at everything (Pareto optimal):
```python
df = pd.read_csv('alpha_scores_by_size.csv')
threshold = 0.8
pareto = df[(df['alpha_AV'] > threshold) & 
            (df['alpha_CC'] > threshold) & 
            (df['alpha_PAIRS'] > threshold) & 
            (df['alpha_CONS'] > threshold)]
```

## Dataset Information

**Source:** PrefLib Dataset 00071 - 2007 French Presidential Election (Approval Voting)

**Voters:** 2,836 (combined from 6 files)

**Candidates:** 12
1. Olivier Besancenot
2. Marie-George Buffet
3. Gérard Schivardi
4. François Bayrou
5. José Bové
6. Dominique Voynet
7. Philippe de Villiers
8. Ségolène Royal
9. Frédéric Nihous
10. Jean-Marie Le Pen
11. Arlette Laguiller
12. Nicolas Sarkozy

**Total committees analyzed:** 2^12 = 4,096

**Computation time:** ~31 minutes

## Citation

If you use this data or analysis, please cite:
- Nicholas Mattei and Toby Walsh. "PrefLib: A Library of Preference Data." ADT 2013.
- Dong et al. "Selecting Interlacing Committees." (2024)





