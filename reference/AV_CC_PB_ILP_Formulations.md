
# Maximizing AV and CC in Participatory Budgeting via ILP

This document describes standard **Integer Linear Programming (ILP)** formulations for computing optimal outcomes
under **Approval Voting (AV)** and **Chamberlin–Courant (CC)** objectives in the **participatory budgeting (PB)** setting.

---

## 1. Participatory Budgeting Model

- Set of projects: `P`
- Cost of project `p ∈ P`: `c(p)`
- Total budget: `B`
- Set of voters: `V`
- Each voter `v ∈ V` submits an approval ballot `A_v ⊆ P`

An outcome `W ⊆ P` must satisfy:

```
∑_{p∈W} c(p) ≤ B
```

---

## 2. Decision Variables

- For each project `p ∈ P`:

```
x_p ∈ {0,1}    (1 if project p is selected)
```

- For CC only, for each voter `v ∈ V`:

```
y_v ∈ {0,1}    (1 if voter v is represented)
```

Let:

```
a_{v,p} = 1 if p ∈ A_v, and 0 otherwise
```

---

## 3. Budget Constraint (PB)

```
∑_{p∈P} c(p) · x_p ≤ B
```

---

## 4. Maximizing Approval Voting (AV)

### Objective

```
AV(W) = ∑_{v∈V} |A_v ∩ W|
```

### ILP Formulation

Maximize:

```
∑_{v∈V} ∑_{p∈P} a_{v,p} · x_p
```

Subject to:

```
∑_{p∈P} c(p) · x_p ≤ B
x_p ∈ {0,1}   for all p ∈ P
```

### Remarks
- This is exactly a **0/1 knapsack** problem.
- Easily extensible with additional linear constraints.

---

## 5. Maximizing Chamberlin–Courant (CC)

### Objective

```
CC(W) = |{ v ∈ V : A_v ∩ W ≠ ∅ }|
```

### ILP Formulation

Maximize:

```
∑_{v∈V} y_v
```

Subject to:

```
∑_{p∈P} c(p) · x_p ≤ B
```

For each voter `v ∈ V`:

```
y_v ≤ ∑_{p∈P} a_{v,p} · x_p
```

And:

```
x_p ∈ {0,1}   for all p ∈ P
y_v ∈ {0,1}   for all v ∈ V
```

### Optional Strengthening

```
∑_{p∈P} a_{v,p} · x_p ≤ |A_v| · y_v
```

This tightens the LP relaxation but is not required for correctness.

---

## 6. α-Approximate AV and CC (Optional)

Given a scoring function `S ∈ {AV, CC}` and `α ∈ [0,1]`:

1. First solve the corresponding ILP to obtain `OPT_S`.
2. Add the constraint:

**AV:**
```
∑_{v,p} a_{v,p} · x_p ≥ α · OPT_AV
```

**CC:**
```
∑_{v} y_v ≥ α · OPT_CC
```

Then solve for feasibility or optimize secondary objectives.

Use rational arithmetic for `α` to avoid floating-point issues.

---

## 7. Practical Remarks

- Both formulations are standard in computational social choice and PB.
- CC corresponds to **Budgeted Maximum Coverage**.
- Modern MIP solvers handle realistic PB instances well.

---
