# Warsaw 2017 Wawrzyszew Committee Analysis

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | `poland_warszawa_2017_wawrzyszew.pb` |
| Number of Voters | 2,238 |
| Number of Candidates | 13 |
| Total Budget | 700,000 |
| Total Approvals | 7,652 |
| Avg Approvals per Voter | 3.42 |
| Max Possible Voter Pairs | 2,503,203 |
| Max CONS in dataset | 1,792,671 |
| Max AV in dataset | 6,352 |
| Single-approval voters | 139 (6.2%) |

## Project Statistics

| Project | Cost | Approvals | % of Voters | Single-Only Voters | % of Total |
|---------|------|-----------|-------------|-------------------|------------|
| 0 | 49,800 | 1,133 | 50.6% | 15 | 0.7% |
| 1 | 15,400 | 1,013 | 45.3% | 45 | 2.0% |
| 2 | 260,000 | 475 | 21.2% | 10 | 0.4% |
| 3 | 198,000 | 687 | 30.7% | 6 | 0.3% |
| 4 | 25,000 | 852 | 38.1% | 2 | 0.1% |
| 5 | 80,480 | 805 | 36.0% | 3 | 0.1% |
| 6 | 6,000 | 447 | 20.0% | 2 | 0.1% |
| 7 | 340,000 | 682 | 30.5% | 8 | 0.4% |
| 8 | 6,900 | 477 | 21.3% | 3 | 0.1% |
| 9 | 395,000 | 430 | 19.2% | 40 | 1.8% |
| 10 | 14,400 | 463 | 20.7% | 0 | 0.0% |
| 11 | 200,000 | 96 | 4.3% | 4 | 0.2% |
| 12 | 200,000 | 92 | 4.1% | 1 | 0.0% |

## Project Appearances in Filtered Sets

| Project | In A | % of A | In B | % of B | In A2 | % of A2 | In B2 | % of B2 | In C | % of C | In D | % of D |
|---------|------|--------|------|--------|-------|---------|-------|---------|------|--------|------|--------|
| 0 | 7 | 3.0% | 473 | 90.1% | 0 | 0.0% | 473 | 100.0% | 0 | 0.0% | 1,108 | 100.0% |
| 1 | 199 | 85.4% | 408 | 77.7% | 199 | 88.1% | 362 | 76.5% | 636 | 48.4% | 538 | 48.6% |
| 2 | 93 | 39.9% | 148 | 28.2% | 89 | 39.4% | 144 | 30.4% | 378 | 28.8% | 282 | 25.5% |
| 3 | 119 | 51.1% | 218 | 41.5% | 119 | 52.7% | 198 | 41.9% | 446 | 33.9% | 358 | 32.3% |
| 4 | 174 | 74.7% | 364 | 69.3% | 174 | 77.0% | 319 | 67.4% | 616 | 46.9% | 527 | 47.6% |
| 5 | 151 | 64.8% | 318 | 60.6% | 151 | 66.8% | 277 | 58.6% | 558 | 42.5% | 457 | 41.2% |
| 6 | 155 | 66.5% | 320 | 61.0% | 149 | 65.9% | 285 | 60.3% | 647 | 49.2% | 549 | 49.5% |
| 7 | 7 | 3.0% | 158 | 30.1% | 0 | 0.0% | 106 | 22.4% | 294 | 22.4% | 227 | 20.5% |
| 8 | 156 | 67.0% | 325 | 61.9% | 150 | 66.4% | 289 | 61.1% | 646 | 49.2% | 548 | 49.5% |
| 9 | 34 | 14.6% | 52 | 9.9% | 34 | 15.0% | 52 | 11.0% | 213 | 16.2% | 151 | 13.6% |
| 10 | 155 | 66.5% | 315 | 60.0% | 149 | 65.9% | 279 | 59.0% | 638 | 48.6% | 538 | 48.6% |
| 11 | 73 | 31.3% | 143 | 27.2% | 72 | 31.9% | 133 | 28.1% | 444 | 33.8% | 356 | 32.1% |
| 12 | 73 | 31.3% | 143 | 27.2% | 72 | 31.9% | 133 | 28.1% | 444 | 33.8% | 356 | 32.1% |

## Filtered Sets

| Set | Description | α_CONS Range | α_AV Range | Committees |
|-----|-------------|--------------|------------|------------|
| A  | Lower CONS | 0.4 - 0.6  | 0.5 - 0.9 | 233 |
| B  | Higher CONS | 0.8 - 1.0  | 0.6 - 1.0 | 525 |
| A2 | A without project 0 | 0.4 - 0.6  | 0.5 - 0.9 | 226 |
| B2 | B with project 0 only | 0.8 - 1.0  | 0.6 - 1.0 | 473 |
| C  | All without project 0 | any | any | 1314 |
| D  | All with project 0 | any | any | 1108 |

### Alpha Threshold Conversions

| Threshold | α Value | Raw Score |
|-----------|---------|-----------|
| Set A CONS min | 0.4 | 717,068 |
| Set A CONS max | 0.6 | 1,075,603 |
| Set A AV min | 0.5 | 3,176 |
| Set A AV max | 0.9 | 5,717 |
| Set B CONS min | 0.8 | 1,434,137 |
| Set B CONS max | 1.0 | 1,792,671 |
| Set B AV min | 0.6 | 3,811 |
| Set B AV max | 1.0 | 6,352 |

## Score Statistics

### Set A (Lower CONS)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | 785,631 | 1,031,766 | 935,440 |
| AV (raw) | 3,177 | 5,219 | 3,665 |
| α_CONS | 0.438 | 0.576 | - |
| α_AV | 0.500 | 0.822 | - |

### Set B (Higher CONS)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | 1,473,186 | 1,792,671 | 1,660,019 |
| AV (raw) | 3,815 | 6,352 | 4,450 |
| α_CONS | 0.822 | 1.000 | - |
| α_AV | 0.601 | 1.000 | - |

### Set A2 (A without project 0)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | 785,631 | 1,031,766 | 933,713 |
| AV (raw) | 3,177 | 5,219 | 3,676 |
| α_CONS | 0.438 | 0.576 | - |
| α_AV | 0.500 | 0.822 | - |

### Set B2 (B with project 0)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | 1,473,186 | 1,792,671 | 1,664,834 |
| AV (raw) | 3,817 | 6,352 | 4,473 |
| α_CONS | 0.822 | 1.000 | - |
| α_AV | 0.601 | 1.000 | - |

### Set C (All without project 0)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | 0 | 1,679,028 | 898,029 |
| AV (raw) | 0 | 5,426 | 2,567 |
| α_CONS | 0.000 | 0.937 | - |
| α_AV | 0.000 | 0.854 | - |

### Set D (All with project 0)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| CONS (raw) | 641,278 | 1,792,671 | 1,513,248 |
| AV (raw) | 1,133 | 6,352 | 3,646 |
| α_CONS | 0.358 | 1.000 | - |
| α_AV | 0.178 | 1.000 | - |

## Candidate 0 Presence

| Set | With Candidate 0 | Total | Percentage |
|-----|------------------|-------|------------|
| A   | 7 | 233 | 3.0% |
| B   | 473 | 525 | 90.1% |

## Match Analysis

When removing candidate 0 from Set B committees:

- **38.7%** of Set B committees (203/525) match a Set A committee
- **87.1%** of unique Set A committees (203/233) are covered by this matching

## Interpretation

Candidate 0 appears to be a "consensus-building" candidate that significantly increases CONS scores. It is present in 90.1% of Set B (high CONS) but only 3.0% of Set A (moderate CONS). Removing candidate 0 from Set B reveals that most of these committees share their remaining structure with Set A committees.
