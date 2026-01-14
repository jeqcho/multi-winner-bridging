Given an election \((V, A, k)\) over \(C\) and \(\alpha \in (0,1]\), a committee \(W \subseteq C\) is said to **satisfy \(\alpha\)-EJR** if for every \(\ell \in [k]\) and every subset \(S \subseteq V\) such that  
\[
\alpha \cdot |S| \ge \frac{\ell}{k} \cdot |V|
\quad \text{and} \quad
\left| \bigcap_{i \in S} A_i \right| \ge \ell,
\]
there exists at least one voter \(i \in S\) such that  
\[
|W \cap A_i| \ge \ell.
\]

We say that a rule \(f\) **satisfies \(\alpha\)-EJR** if for every election \(\mathcal{E}\) it holds that \(f(\mathcal{E})\) satisfies \(\alpha\)-EJR. By setting \(\alpha\) to \(1\), we obtain the standard EJR axiom.