# Implementation Guide: AV, CC, and PAV
**Source:** Optimized Democracy – Committee Elections (Ariel Procaccia, Fall 2025)  [oai_citation:0‡slides11.pdf](sediment://file_0000000053a8723089f08b5e7323f0fa)

This document defines the data structures and algorithms required to implement **Approval Voting (AV)**, **Chamberlin–Courant (CC)**, and **Proportional Approval Voting (PAV)** for committee selection.

---

## 1. Problem Setup

We have:

- **Voters**: $begin:math:text$ N \= \\\{1\, \\dots\, n\\\} $end:math:text$  
- **Candidates**: $begin:math:text$ A $end:math:text$  
- **Committee size**: $begin:math:text$ k $end:math:text$  
- **Approval ballots**: each voter $begin:math:text$ i $end:math:text$ submits a set  
  $begin:math:display$
  \\alpha\_i \\subseteq A
  $end:math:display$
  representing the candidates they approve.

- **Committee**:  
  $begin:math:display$
  W \\subseteq A\, \\quad \|W\| \= k
  $end:math:display$

- **Utility function**:  
  $begin:math:display$
  u\_i\(W\) \= \|\\alpha\_i \\cap W\|
  $end:math:display$

---

## 2. Approval Voting (AV)

### Purpose
Select the committee whose members receive the most approvals individually.

### Scoring rule
AV corresponds to Thiele weights:
$begin:math:display$
\(1\, 1\, 1\, \\ldots\)
$end:math:display$

For each candidate $begin:math:text$x$end:math:text$:

$begin:math:display$
\\text\{score\}\(x\) \= \|\\\{ i \\in N \: x \\in \\alpha\_i \\\}\|
$end:math:display$

### Committee selection
1. Compute the approval score for each candidate.
2. Sort candidates by score.
3. Select the top $begin:math:text$k$end:math:text$.

### Complexity
- Counting approvals: $begin:math:text$ O\(n\|A\|\) $end:math:text$  
- Sorting: $begin:math:text$ O\(\|A\| \\log \|A\|\) $end:math:text$

---

## 3. Chamberlin–Courant (CC)

### Purpose
Maximize **coverage** — the number of voters who approve at least one committee member.

### Score function
Thiele sequence:
$begin:math:display$
\(1\, 0\, 0\, \\ldots\)
$end:math:display$

A voter contributes 1 iff:
$begin:math:display$
u\_i\(W\) \\ge 1
$end:math:display$

Thus:
$begin:math:display$
\\text\{CC\-score\}\(W\) \= \|\\\{ i \\in N \: u\_i\(W\) \\ge 1 \\\}\|
$end:math:display$

### Committee selection

#### Exact (brute force)
1. Enumerate all $begin:math:text$ \\binom\{\|A\|\}\{k\} $end:math:text$ committees.
2. Compute CC-score.
3. Return the maximum.

#### Greedy approximation
1. Start with $begin:math:text$W\=\\varnothing$end:math:text$.
2. While $begin:math:text$\|W\| \< k$end:math:text$:
   - For each candidate $begin:math:text$x \\notin W$end:math:text$, compute marginal CC-score increase.
   - Add the candidate with the largest marginal increase.

---

## 4. Proportional Approval Voting (PAV)

### Purpose
Select a committee that is proportionally representative using **harmonic numbers**.

### Thiele weights
$begin:math:display$
\\left\(1\,\\\; \\frac12\,\\\; \\frac13\,\\\; \\frac14\,\\\; \\ldots\\right\)
$end:math:display$

### PAV-score
For each voter $begin:math:text$i$end:math:text$ with $begin:math:text$ t\_i \= u\_i\(W\) $end:math:text$:
$begin:math:display$
s\(t\_i\) \= 1 \+ \\frac12 \+ \\cdots \+ \\frac1\{t\_i\}
$end:math:display$

Total:
$begin:math:display$
\\text\{PAV\-score\}\(W\) \= \\sum\_\{i \\in N\} s\(t\_i\)
$end:math:display$

### Committee selection

#### Exact (brute force)
Enumerate all committees and maximize PAV-score.

#### Greedy PAV (widely used; satisfies EJR)
1. Start with $begin:math:text$W \= \\varnothing$end:math:text$.
2. While $begin:math:text$\|W\| \< k$end:math:text$:
   - For each candidate $begin:math:text$ x \\notin W $end:math:text$, compute the marginal PAV gain.
   - Add the one with the highest gain.

---

## 5. Required Input/Output Formats

### Input
- `num_voters: int`
- `num_candidates: int`
- `approvals: List[Set[int]]`
- `committee_size: int`

### Output
- `committee: Set[int]`

---

## 6. Recommended API

```python
def approval_voting(approvals, k):
    """Return AV committee."""

def chamberlin_courant(approvals, k, method="greedy"):
    """Return CC committee (exact or greedy)."""

def pav(approvals, k, method="greedy"):
    """Return PAV committee (exact or greedy)."""
```

---

## 7. Test Cases to Provide to Programmer

### Case 1 — Disjoint approvals (slides 9–10)
- 6 voters approve A  
- 4 approve B  
- 10 approve C  
- 2 approve D

Useful for verifying proportionality under PAV.

### Case 2 — JR violation for AV (slide 13)
- 1 voter approves {x}  
- 4 voters approve {y}  
With $begin:math:text$k\=1$end:math:text$, AV elects y → JR violated.

### Case 3 — CC fails EJR (slide 15)
Use the four-group example from the slide.

---

## 8. Implementation Notes

- Exact CC and PAV are NP-hard; brute force is only feasible for small candidate sets (<20).
- Incrementally update utilities $begin:math:text$u\_i\(W\)$end:math:text$ for efficiency.
- Greedy PAV satisfies **JR** and **EJR**.
- CC satisfies **JR**, fails **EJR**.
- AV satisfies neither JR nor EJR.