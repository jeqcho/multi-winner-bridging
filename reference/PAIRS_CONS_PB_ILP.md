
# PAIRS and CONS Objectives in Participatory Budgeting (ILP Formulations)

This document gives **exact ILP/MILP formulations** for computing the **PAIRS** and **CONS** objectives in the
**participatory budgeting (PB)** setting with approval ballots.

---

## 1. PB Model and Notation

- Candidates / projects: `C`
- Costs: `c(p)` for each `p ∈ C`
- Budget: `B`
- Voters: `V`
- Approval ballot of voter `i`: `A_i ⊆ C`
- Selected set (committee): `W ⊆ C`

Budget feasibility:
```
∑_{p∈C} c(p) · x_p ≤ B
```

Binary constants:
```
a_{i,p} = 1  if p ∈ A_i,  else 0
```

For each unordered voter pair `{i,j}`, define:
```
I_{ij} = A_i ∩ A_j
```

---

## 2. Decision Variables

- `x_p ∈ {0,1}` — project `p` is selected
- `y_{ij} ∈ {0,1}` — voter pair `{i,j}` is counted
- `e_{ij} ∈ {0,1}` — edge exists between voters `i` and `j`
- `f^{s,t}_{i→j} ≥ 0` — flow from voter `i` to `j` for commodity `(s,t)`

---

## 3. PAIRS Objective

### Definition

```
PAIRS(W) = |{ {i,j} ⊆ V : A_i ∩ A_j ∩ W ≠ ∅ }|
```

### ILP Formulation

Maximize:
```
∑_{ {i,j} ⊆ V } y_{ij}
```

Subject to:
```
∑_{p∈C} c(p) · x_p ≤ B
```

For each unordered voter pair `{i,j}`:
```
y_{ij} ≤ ∑_{p∈I_{ij}} x_p
```

Domains:
```
x_p ∈ {0,1}    for all p ∈ C
y_{ij} ∈ {0,1} for all {i,j} ⊆ V
```

### Remarks
- This is a **coverage ILP over voter pairs**
- Complexity: `O(|V|^2)` binary variables

---

## 4. CONS Objective

### Definition

Voters `u` and `v` are connected under `W` if there exists a path
`u = v₁, v₂, …, v_ℓ = v` such that:

```
A_{v_k} ∩ A_{v_{k+1}} ∩ W ≠ ∅   for all k
```

The objective counts connected voter pairs:
```
CONS(W) = |{ {u,v} ⊆ V : u ∼_W v }|
```

---

## 5. CONS MILP Formulation

### Step 1: Edge activation

Introduce edge variables:
```
e_{ij} ∈ {0,1}   for each unordered pair {i,j}
```

Link edges to selected projects:
```
e_{ij} ≤ ∑_{p∈I_{ij}} x_p
```

---

### Step 2: Connectivity variables

For each unordered pair `{s,t}`:
```
y_{st} ∈ {0,1}
```

Objective:
```
Maximize  ∑_{ {s,t} ⊆ V } y_{st}
```

---

### Step 3: Flow-based connectivity enforcement

Convert each unordered edge `{i,j}` into arcs `(i→j)` and `(j→i)`.

For each ordered pair `(s,t)` with `s ≠ t`, send **one unit of flow**
if `y_{st} = 1`.

Capacity constraints:
```
f^{s,t}_{i→j} ≤ e_{ij}
```

Flow conservation:

At source `s`:
```
∑_{j≠s} f^{s,t}_{s→j} − ∑_{j≠s} f^{s,t}_{j→s} = y_{st}
```

At sink `t`:
```
∑_{j≠t} f^{s,t}_{j→t} − ∑_{j≠t} f^{s,t}_{t→j} = y_{st}
```

At intermediate voters `v ∉ {s,t}`:
```
∑_{j≠v} f^{s,t}_{v→j} − ∑_{j≠v} f^{s,t}_{j→v} = 0
```

---

## 6. Full CONS MILP

Budget constraint:
```
∑_{p∈C} c(p) · x_p ≤ B
```

Edge constraints:
```
e_{ij} ≤ ∑_{p∈I_{ij}} x_p
```

Flow constraints:
```
f^{s,t}_{i→j} ≤ e_{ij}
```

Domains:
```
x_p ∈ {0,1}
e_{ij} ∈ {0,1}
y_{st} ∈ {0,1}
f^{s,t}_{i→j} ≥ 0
```

---

## 7. Practical Notes

- **PAIRS** is lightweight and scales well.
- **CONS** is exact but large (`O(|V|^4)` flow vars in worst case).
- Strongly recommended:
  - restrict `{i,j}` to pairs with `I_{ij} ≠ ∅`
  - solve small/medium instances exactly
- Standard solvers: Gurobi, CPLEX, CBC.

---

This formulation is **exact**, general, and directly implementable.
