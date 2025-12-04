# CONS vs CC Quadratic Relationship Analysis

This document analyzes the empirical relationship between α_CONS and α_CC across all committees from all participatory budgeting elections.

## Key Finding

The data strongly supports the relationship:

$$\alpha_{CONS} = \alpha_{CC}^2$$

This means that the normalized CONS score is approximately the square of the normalized CC score for virtually all committees.

## Dataset Summary

- **Total elections analyzed:** 338
- **Total committees analyzed:** 216,585

## Tolerance Analysis

We measure how many points/elections fall within a given tolerance of the theoretical curve α_CONS = α_CC².

### Points within tolerance

| Tolerance | Points within | Percentage |
|-----------|---------------|------------|
| 0.0001 | 78,627 / 216,585 | 36.30% |
| 0.001 | 209,456 / 216,585 | 96.71% |
| 0.005 | 215,099 / 216,585 | 99.31% |
| 0.01 | 215,276 / 216,585 | 99.40% |
| 0.02 | 215,697 / 216,585 | 99.59% |
| 0.05 | 216,119 / 216,585 | 99.78% |
| 0.1 | 216,383 / 216,585 | 99.91% |
| 0.15 | 216,491 / 216,585 | 99.96% |
| 0.2 | 216,540 / 216,585 | 99.98% |

### Elections with ALL committees within tolerance

| Tolerance | Elections | Percentage |
|-----------|-----------|------------|
| 0.0001 | 18 / 338 | 5.33% |
| 0.001 | 266 / 338 | 78.70% |
| 0.005 | 306 / 338 | 90.53% |
| 0.01 | 310 / 338 | 91.72% |
| 0.02 | 314 / 338 | 92.90% |
| 0.05 | 321 / 338 | 94.97% |
| 0.1 | 333 / 338 | 98.52% |
| 0.15 | 334 / 338 | 98.82% |
| 0.2 | 335 / 338 | 99.11% |

## Error Statistics

### Per-point error (|α_CONS - α_CC²|)

- **Mean error:** 0.000551
- **Median error:** 0.000138
- **Std deviation:** 0.006046
- **Min error:** 0.000000
- **Max error:** 0.366859

### Per-election max error

- **Mean max error:** 0.007759
- **Median max error:** 0.000422
- **Std deviation:** 0.035465
- **Max max error:** 0.366859

## Interpretation

The extremely low error rates demonstrate that:

1. **99.4% of all committees** fall within 0.01 of the theoretical curve
2. **91.7% of elections** have ALL their committees within 0.01 of the curve
3. The mean error is only **0.0006**, indicating near-perfect adherence to the relationship

This suggests that α_CONS = α_CC² is not just a theoretical bound but an empirical law that holds across real-world participatory budgeting elections.

## Related Plots

- `cons_vs_cc_scatter.png` - All committees from all elections
- `cons_vs_cc_single_election.png` - Single election example (poland_warszawa_2019_rejon-poludniowy)