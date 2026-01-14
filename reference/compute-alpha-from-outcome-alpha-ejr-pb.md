# Computing **α(W)** for **α-EJR** given a fixed PB outcome \(W\)

This note explains how to compute the **largest** \(\alpha \in [0,1]\) such that a **given** funded outcome \(W\) (a set of projects) satisfies **α-EJR** in approval-based participatory budgeting (PB).

It also shows how to implement the computation using a **separation MIP** (or a family of MIPs over candidate thresholds).

---

## 1) PB model and α-EJR (quota-scaling)

- Voters \(V=\{1,\dots,n\}\)
- Projects \(C=\{1,\dots,m\}\)
- Budget limit \(B\)
- Costs \(\text{cost}(c) > 0\)
- Approvals \(A_{i,c}\in\{0,1\}\)
- Outcome \(W \subseteq C\) is feasible if \(\sum_{c\in W}\text{cost}(c)\le B\)

Define the (approval-cost) utility of voter \(i\) under \(W\):
\[
u_i(W)=\sum_{c\in W}\text{cost}(c)\,A_{i,c}.
\]

### α-EJR (budget-share form)
For any group \(S\subseteq V\) and any budget threshold \(q\in[0,B]\), if

1. **Scaled size condition**
\[
\alpha\,|S|\ \ge\ \frac{q}{B}\,n
\]
and

2. **Cohesiveness**
\[
\exists T\subseteq \bigcap_{i\in S}A_i \text{ such that }\sum_{c\in T}\text{cost}(c)\ge q,
\]

then α-EJR requires

3. **Representation**
\[
\exists i\in S:\ u_i(W)\ge q.
\]

---

## 2) What “computing α(W)” means

Given a fixed outcome \(W\), define:
\[
\alpha(W) \;=\; \max\{\alpha \in [0,1] : W \text{ satisfies α-EJR}\}.
\]

Equivalently, α-EJR can only be violated by a **witness** \((S,q)\) such that:

- \(S\) is cohesive up to \(q\), and
- **every** voter in \(S\) has \(u_i(W) < q\).

For such a witness, α-EJR would be triggered whenever:
\[
\alpha \ge \frac{(q/B)\,n}{|S|}.
\]

So the **maximum α that remains safe** is:
\[
\boxed{
\alpha(W)=\min\left(1,\ \min_{(S,q)\ \text{is a violation witness}} \frac{(q/B)\,n}{|S|}
\right)
}
\]
(If there is no violation witness, then \(\alpha(W)=1\).)

---

## 3) Discretizing \(q\): which thresholds matter?

The definition quantifies over a continuum of \(q\), but you only need to check **finitely many** \(q\)-values.

Two common safe discretizations:

### Option A (simple, solver-friendly)
Check:
\[
\mathcal{Q} = \{u_i(W) + \varepsilon : i\in V \} \cup \{\text{cost}(c): c\in C\} \cup \{B\}.
\]
Rationale: a violation needs \(u_i(W) < q\) for all \(i\in S\), so tight violations occur when \(q\) is just above some achieved utility.

### Option B (oracle-driven)
Let the separation MIP implicitly pick cohesive bundles; then you only loop over a modest set of candidate \(q\) values (e.g., \(\mathcal{Q}\) from Option A).

---

## 4) Computing α(W) via separation MIP (recommended)

### Precompute from \(W\)
Compute each voter’s achieved utility:
\[
u_i^\* := u_i(W).
\]

### Separation MIP for a fixed \(q\)

Given a candidate threshold \(q\), we try to find a **largest** violating group \(S\).  
If we find such an \(S\), it implies an upper bound:
\[
\alpha \le \frac{(q/B)\,n}{|S|}.
\]

#### Variables
- \(s_i \in \{0,1\}\): whether voter \(i\) is in \(S\)
- \(z_c \in \{0,1\}\): whether project \(c\) is approved by **all** voters in \(S\)

#### Constraints

1) **Common-approval definition**  
For all \(i\in V, c\in C\):
\[
z_c \le A_{i,c} + (1-s_i)
\]
(If \(s_i=1\), then \(z_c \le A_{i,c}\).)

2) **Cohesiveness at level \(q\)**
\[
\sum_{c\in C} \text{cost}(c)\, z_c \ge q
\]

3) **Representation failure (everyone in S below q)**
\[
u_i^\* \le q - \varepsilon + B\,(1-s_i)
\quad \forall i\in V
\]
(If \(s_i=1\), enforces \(u_i^\* < q\).)

4) **Nonempty group**
\[
\sum_i s_i \ge 1
\]

#### Objective
Maximize group size (to get the tightest α bound):
\[
\max\ \sum_{i\in V} s_i
\]

#### Output
- If infeasible: no violation at this \(q\).
- If feasible: let \(|S|=\sum_i s_i\). This yields a candidate bound:
\[
\alpha_q := \frac{(q/B)\,n}{|S|}.
\]

### Final computation of α(W)
Run the separation MIP for all \(q\in \mathcal{Q}\), and take:
\[
\boxed{\alpha(W)=\min\bigl(1,\ \min_{q\in\mathcal{Q}\ \text{with violation}} \alpha_q \bigr)}
\]

---

## 5) Alternative: one-shot formulation (no q-loop)

You can also make \(q\) a variable and search for the “most dangerous” violation directly, but the natural objective becomes **fractional**:
\[
\min\ \frac{(q/B)\,n}{\sum_i s_i}.
\]
Handle via:
- Charnes–Cooper transform, or
- loop over \(|S|\) (outer loop) and maximize \(q\) inside, or
- binary search on α with feasibility checks.

In practice, the **q-loop separation** is usually simpler.

---

## 6) Complexity notes

- Finding a violating witness \((S,q)\) is NP-hard in general (already hard in the unit-cost committee setting).
- Therefore computing \(\alpha(W)\) exactly is NP-hard in general.
- MIP-based separation is the standard exact approach for moderate instance sizes.

---

## 7) Committee (unit-cost) analogue (for reference)

If all costs are 1 and feasibility is \(|W|=k\), then use:
- utility \(u_i(W)=|W\cap A_i|\),
- cohesion \(\sum_c z_c \ge \ell\),
- thresholds indexed by \(\ell\in\{1,\dots,k\}\),

and compute:
\[
\alpha(W)=\min\left(1,\ \min_{(S,\ell)\ \text{violations}} \frac{(\ell/k)\,n}{|S|}\right).
\]

---

## Implementation checklist

Given: approvals \(A\), costs, budget \(B\), and outcome \(W\).

1. Compute \(u_i^\*\) for all voters.
2. Choose a threshold set \(\mathcal{Q}\) (start with \(u_i^\*+\varepsilon\) values).
3. For each \(q\in\mathcal{Q}\):
   - solve the separation MIP,
   - if feasible, compute \(\alpha_q\).
4. Return \(\alpha(W)=\min(1,\min_q \alpha_q)\).

