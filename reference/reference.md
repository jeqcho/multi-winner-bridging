Grounded in standard definitions from Dong et al., “Selecting Interlacing Committees.”  ￼

# Scoring Definitions for PAIRS, CONS, AV, CC, and EJR

This file defines five quantities used to evaluate a **given committee** `W` on a dataset of voter approvals.

## 0) Notation & Inputs

- `V` — set of voters, |V| = n.
- `C` — set of candidates, |C| = m.
- For each voter `v ∈ V`, `A_v ⊆ C` is the set of candidates that `v` approves.
  - In code, this is usually an `n × m` Boolean matrix `M`, where `M[v][c] = 1 ⇔ c ∈ A_v`.
- `W ⊆ C` — the **committee** under evaluation; let `k = |W|`.

Throughout, `{u, v}` denotes an **unordered** pair of distinct voters.

---

## 1) Approval Voting (AV)

**Definition.**  
`AV(W) = Σ_{v ∈ V} |A_v ∩ W|`.

**Interpretation.**  
Total number of approvals received by the committee members (sum of each voter’s approved members in `W`).

**Implementation notes.**  
This is just a matrix–vector sum over the columns of `M` indexed by `W`.

---

## 2) Chamberlin–Courant Coverage (CC)

**Definition.**  
`CC(W) = |{ v ∈ V : (A_v ∩ W) ≠ ∅ }|`.

**Interpretation.**  
Number of voters who approve **at least one** member of `W` (i.e., represented voters).

**Implementation notes.**  
Row-wise: for each voter, check if any of the committee columns in that row are `1`.

---

## 3) PAIRS (Direct Pair Coverage)

**Definition.**  
`PAIRS(W) = |{ {u, v} ⊆ V : (A_u ∩ A_v ∩ W) ≠ ∅ }|`.

**Interpretation.**  
Counts unordered voter pairs that **share** at least one selected candidate. Equivalently, it is the CC score on the “pair instance,” where each “voter” is a pair `{u, v}` and its approval set is `A_u ∩ A_v`.

**Implementation notes.**
- Naïve: for every pair `{u, v}`, test whether any candidate in `W` is approved by **both** `u` and `v`.
- Faster: precompute, for each candidate `c ∈ W`, the supporter list `V_c = { v : c ∈ A_v }`; add `C(|V_c|, 2)` to a running total **but** be careful to **de-duplicate** pairs covered by multiple candidates. A robust approach:
  - Maintain a `visited` hash set of pairs; for each `c ∈ W`, iterate over `V_c` and add previously unseen pairs.

---

## 4) CONS (Connectivity via Selected Candidates)

**Definition.**  
Two voters `u, v` are **connected by W** if there exists a sequence of voters  
`u = v₀, v₁, …, v_s = v` such that for every step `i` there is **some** candidate in `W` approved by **both** `v_{i-1}` and `v_i`.  
Then  
`CONS(W) = |{ {u, v} ⊆ V : u and v are connected by W }|`.

**Equivalent graph view (recommended for implementation).**
1. Build an undirected graph `G_W = (V, E_W)` with an edge `{u, v}` iff there exists `c ∈ W` such that `c ∈ A_u ∩ A_v`.
2. Let the connected components of `G_W` be `S_1, …, S_t`.  
   Then  
   `CONS(W) = Σ_{i=1..t} C(|S_i|, 2) = Σ_i |S_i| · (|S_i| - 1) / 2`.

**Implementation notes.**
- Create supporter lists `V_c` for `c ∈ W` and add edges between all pairs in each `V_c`.
- Use Union–Find (DSU) or BFS/DFS to find component sizes; sum `C(size, 2)`.

---

## 5) EJR (Extended Justified Representation)

EJR is a **representation property** (boolean) parameterized by `k = |W|`. The canonical definition below is for the exact (α = 1) version.

**Definition (EJR).**  
For every integer `ℓ ∈ {1, …, k}` and every voter subset `S ⊆ V` such that:
- `|S| ≥ (ℓ / k) · n`  (i.e., the group deserves `ℓ` seats by size), and
- `| ⋂_{i ∈ S} A_i | ≥ ℓ`  (i.e., the group is ℓ-cohesive: they share at least `ℓ` commonly approved candidates),

there must exist **some** voter `i ∈ S` with `|A_i ∩ W| ≥ ℓ`.  
If this holds for all `ℓ` and all such `S`, then `W` **satisfies EJR**.

**Practical check (sufficient).**  
For each `ℓ = 1..k`, for each candidate-ℓ–subset `T ⊆ C`:
- Let `S_T = { v ∈ V : T ⊆ A_v }`.  
  If `|S_T| ≥ (ℓ / k) · n` **and** every `v ∈ S_T` has `|A_v ∩ W| < ℓ`, then `W` **violates** EJR.
- If no such `T` triggers a violation for every `ℓ`, `W` satisfies EJR.

**Output options.**
- `EJR_satisfied(W) ∈ {true, false}`.
- (Optional) Report the **violations** as tuples `(ℓ, T, S_T)` to aid debugging.

---

## 6) Edge Cases & Conventions

- If `W = ∅`: `AV = 0`, `CC = 0`, `PAIRS = 0`, `CONS = 0`; EJR is **not** satisfied unless `k = 0`.
- If a candidate is approved by **all** voters and is in `W`, then `CONS(W) = C(n, 2)` and `PAIRS(W) = C(n, 2)`.
- All pair counts use **unordered** pairs (each `{u, v}` counted at most once).
- Use 64-bit integers if `n` can be large, since `C(n, 2)` may overflow 32-bit.

---

## 7) Reference Implementations (sketch)

### AV

sum( count_nonzero(M[v, W]) for v in V )

### CC

sum( 1 if any(M[v, c] for c in W) else 0 for v in V )

### PAIRS

covered_pairs = set()
for c in W:
S = [v for v in V if M[v, c]]
for each unordered pair {u, v} in S:
covered_pairs.add((min(u,v), max(u,v)))
PAIRS = len(covered_pairs)

### CONS

init DSU over V
for c in W:
S = [v for v in V if M[v, c]]
union all vertices in S (e.g., chain unions)
sizes = multiset of component sizes from DSU
CONS = sum( s * (s - 1) // 2 for s in sizes )

### α-Extended Justified Representation (α-EJR)

Given an election \((V, C, k)\) and a parameter \(\alpha \in (0,1]\),  
a committee \(W \subseteq C\) is said to satisfy **α-EJR** if the following holds:

For every integer \(\ell \in [k]\) and every subset of voters \(S \subseteq V\) such that

- \(\alpha \cdot |S| \;\ge\; \frac{\ell}{k} \cdot |V|\), and  
- \(\left| \bigcap_{i \in S} A_i \right| \ge \ell\),

there exists at least one voter \(i \in S\) such that

\[
|W \cap A_i| \;\ge\; \ell .
\]

In words: any sufficiently large cohesive group of voters deserving \(\ell\) seats must have at least one member who receives \(\ell\) approved representatives in the committee.

We say that a voting rule \(f\) **satisfies α-EJR** if every committee \(W = f(V, C, k)\) produced by the rule satisfies α-EJR.