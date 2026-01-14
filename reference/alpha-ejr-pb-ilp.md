# ILP / MIP formulation for finding an optimal **α‑EJR** outcome  
## (Participatory Budgeting, cost‑aware version)

This document gives a **budget-aware** formulation of **α‑EJR** for **approval-based participatory budgeting (PB)**, using a **constraint-generation (cutting‑plane)** approach.

It is a direct rewrite of the committee version, replacing cardinality by **costs** and representation levels by **budget shares**.

---

## 1. Setting and definition (PB α‑EJR)

### Input
- Voters: \(V = \{1,\dots,n\}\)
- Projects: \(C = \{1,\dots,m\}\)
- Budget limit: \(B > 0\)
- Project costs: \(\text{cost}(c) > 0\) for all \(c \in C\)
- Approval ballots: \(A_{i,c} \in \{0,1\}\)

### Outcome
A funded set of projects \(W \subseteq C\) such that:
\[
\sum_{c\in W} \text{cost}(c) \le B
\]

### Voter utility
\[
u_i(W) \;=\; \sum_{c\in C} \text{cost}(c)\, A_{i,c}\, x_c
\]

---

### Definition (α‑EJR for PB)

Given \(\alpha \in (0,1]\), a funded set \(W\) satisfies **α‑EJR** if for every subset \(S \subseteq V\) and every \(q \in [0,B]\):

If
\[
\alpha \cdot |S| \;\ge\; \frac{q}{B}\cdot |V|
\quad\text{and}\quad
\exists T \subseteq \bigcap_{i\in S} A_i
\text{ with }
\sum_{c\in T} \text{cost}(c) \ge q,
\]
then
\[
\exists i\in S \text{ such that } u_i(W) \ge q.
\]

Setting \(\alpha=1\) recovers full (budget‑aware) EJR.

---

## 2. High‑level approach

Because α‑EJR quantifies over **all voter groups \(S\)** and **all budget shares \(q\)**, a full formulation is exponential.

We use a **cutting‑plane algorithm**:

1. **Master MIP**  
   Chooses a feasible budget allocation and maximizes α, enforcing α‑EJR only for a growing set of discovered violations.
2. **Separation MIP**  
   Given a candidate solution \((x,\alpha)\), finds a violating pair \((S,q)\) if one exists.
3. Iterate until no violation is found.

---

## 3. Master problem (with cuts)

### Variables
- \(x_c \in \{0,1\}\) — project \(c\) is funded
- \(\alpha \in [0,1]\)

For each added cut \(t\):
- a discovered voter set \(S_t \subseteq V\)
- a budget threshold \(q_t > 0\)
- auxiliary variables \(y_{i,t} \in \{0,1\}\) for all \(i \in S_t\)

### Objective
\[
\max\ \alpha
\]

### Base constraints

#### (i) Budget feasibility
\[
\sum_{c\in C} \text{cost}(c)\, x_c \;\le\; B
\]

#### (ii) Bounds
\[
0 \le \alpha \le 1
\]

---

### Cut constraints (α‑EJR enforcement)

Each cut \(t\) encodes:

> “Within \(S_t\), at least one voter receives approved projects of total cost at least \(q_t\).”

For each cut \(t\):
\[
\sum_{i\in S_t} y_{i,t} \ge 1
\]

For each \(i \in S_t\):
\[
\sum_{c\in C} \text{cost}(c)\, A_{i,c}\, x_c
\;\ge\;
q_t \cdot y_{i,t}
\]

If \(y_{i,t}=1\), voter \(i\) must receive utility at least \(q_t\).

---

## 4. Separation oracle (find violated α‑EJR constraints)

Given a candidate solution \((x^\*,\alpha^\*)\), define:
\[
u_i^\* = \sum_{c\in C} \text{cost}(c)\, A_{i,c}\, x_c^\*
\]

We search for a **violating pair** \((S,q)\) such that:

1. **Scaled size condition**
\[
\alpha^\* |S| \;\ge\; \frac{q}{B} |V|
\]

2. **Cohesiveness**
\[
\exists T \subseteq \bigcap_{i\in S} A_i
\quad\text{with}\quad
\sum_{c\in T} \text{cost}(c) \ge q
\]

3. **Representation failure**
\[
\forall i\in S:\quad u_i^\* < q
\]

If such \((S,q)\) exists, α‑EJR is violated.

---

## 5. Separation MIP (fixed q)

In practice, discretize \(q\) to:
\[
q \in \left\{ \text{cost}(c_1)+\dots+\text{cost}(c_t) : t=1,\dots,m \right\}
\]
or simply loop over candidate‑induced thresholds.

### Constants
- \(x^\*\), \(\alpha^\*\), \(u_i^\*\)
- chosen \(q\)

### Variables
- \(s_i \in \{0,1\}\): voter \(i \in S\)
- \(z_c \in \{0,1\}\): project \(c\) approved by **all** voters in \(S\)

### Constraints

#### (i) Size condition
\[
\alpha^\* \sum_i s_i \;\ge\; \frac{q}{B} |V|
\]

#### (ii) Common approval
For all \(i,c\):
\[
z_c \le A_{i,c} + (1 - s_i)
\]

#### (iii) Cohesiveness
\[
\sum_{c\in C} \text{cost}(c)\, z_c \ge q
\]

#### (iv) Representation failure
For all \(i\):
\[
u_i^\* \le q - \varepsilon + M(1 - s_i)
\]

where:
- \(M = B\) is sufficient
- \(\varepsilon > 0\) is a small tolerance

### Objective
Feasibility only (or maximize \(\sum_i s_i\) to obtain stronger cuts).

---

## 6. Full cutting‑plane algorithm

1. Initialize master with **no cuts**.
2. Solve master → obtain \((x^\*,\alpha^\*)\).
3. Compute \(u_i^\*\) for all voters.
4. For each candidate threshold \(q\):
   - Solve separation MIP.
   - If feasible:
     - extract \(S\)
     - add cut \((S_t,q_t)=(S,q)\) to master
     - return to step 2.
5. If no violations are found, the solution is **optimal α‑EJR**.

---

## 7. Notes and implementation tips

- The structure mirrors the committee α‑EJR formulation exactly.
- Only **cost accounting** and **budget shares** differ.
- This formulation works even when **full EJR (α=1) does not exist**.
- In practice, limit the set of tested \(q\) values for efficiency.

---

## 8. Summary

| Aspect | Committee α‑EJR | PB α‑EJR |
|------|-----------------|----------|
| Feasibility | \(|W|=k\) | \(\sum \text{cost} \le B\) |
| Representation | number of winners | approved budget |
| Cohesion | common candidates | common approved cost |
| Algorithm | cutting‑plane MIP | cutting‑plane MIP |

This is the **correct budgeting-aware version** of α‑EJR suitable for implementation.

