# PAIRS: ILP vs Enumeration Comparison (Larger Elections)

**Generated:** 2026-01-16 04:48:58

## Overview

This experiment compares ILP vs enumeration for the **PAIRS objective only** on elections with 10 projects and 150-1000 voters.

## Elections Tested

| Election | Projects | Voters | Valid Committees |
|----------|----------|--------|------------------|
| France_Toulouse_2022_7_-_Sept_Deniers_Ginestous-Se... | 10 | 154 | 510 |
| France_Toulouse_2022_13_-_Rangueil_Sauzelong_Jules... | 10 | 304 | 443 |
| France_Toulouse_2022_8_-_Minimes_Barriere_de_Paris... | 10 | 512 | 535 |
| France_Toulouse_2022_12_-_Pont_des_Demoiselles_Orm... | 10 | 659 | 323 |
| France_Toulouse_2022_1_-_Capitole_Arnaud_Bernard_C... | 10 | 972 | 611 |

## PAIRS Results

| Voters | Committees | Optimal PAIRS | Enum Time (s) | ILP Time (s) | Speedup | Match |
|--------|------------|---------------|---------------|--------------|---------|-------|
| 154 | 510 | 3,733 | 0.0655 | 0.2307 | 0.28x (Enum) | Yes |
| 304 | 443 | 19,117 | 0.1875 | 0.3973 | 0.47x (Enum) | Yes |
| 512 | 535 | 46,528 | 0.6779 | 1.0041 | 0.68x (Enum) | Yes |
| 659 | 323 | 144,474 | 0.6233 | 2.7770 | 0.22x (Enum) | Yes |
| 972 | 611 | 185,090 | 2.7591 | 3.9039 | 0.71x (Enum) | Yes |

## Summary

- **Total Enumeration Time:** 4.3132s
- **Total ILP Time:** 8.3130s
- **Overall:** Enum is 1.9x faster overall
- **All Results Match:** Yes

## Analysis

- Enumeration was faster for all tested elections
- PAIRS ILP scales with O(|V|^2) variables, while enumeration scales with number of valid committees
- For 10 projects with typical PB budgets, there are ~500-1000 valid committees regardless of voter count
- ILP time increases with voters (more pair variables), enumeration time increases with committees

Both methods produce identical optimal scores, confirming correctness.
