# DAV Mathematical Model (working specification)

**Version:** v0.1  
**Assumptions:** probabilistic repair; separate verification and recovery budgets.

## 1. Policy and random outcomes

Let \(\pi\) be an adaptive policy mapping the current state and history to the next action. Let \(\omega\) represent stochastic verifier, repair, recomputation, and intrinsic-error outcomes.

Actions are VERIFY(v), REPAIR(v), and RECOMPUTE(u). In the first version, repair and recomputation are triggered by verification outcomes rather than freely chosen by the agent.

## 2. Cost variables

- \(c_v^{ver}\): verification cost for node \(v\).
- \(c_v^{repair}\): repair cost for node \(v\).
- \(c_u^{rec}\): recomputation cost for node \(u\).
- \(B_{ver}\): verification budget.
- \(B_{rec}\): recovery budget.

For a policy and trajectory:

\[
C_{ver}(\pi,\omega)=\sum_v c_v^{ver}A_v(\pi,\omega),
\]

\[
C_{recov}(\pi,\omega)=
\sum_v c_v^{repair}B_v(\pi,\omega)
+\sum_u c_u^{rec}D_u(\pi,\omega),
\]

where \(A_v,B_v,D_u\) are action indicators.

## 3. Residual task loss

Let \(V_T\) be critical outputs. Define:

\[
L_{res}(\pi)=\sum_{u\in V_T}w_uP_\pi(Y_u^{final}=1).
\]

Intermediate nodes are used for lineage and propagation accounting; they should not all be added as independent losses unless they cause separate task/operational harm. This avoids counting the same propagated failure repeatedly.

## 4. Recommended constrained objective

\[
\boxed{
\min_\pi\;
\mathbb E_\pi[C_{ver}+C_{recov}]
+\lambda\sum_{u\in V_T}w_uP_\pi(Y_u^{final}=1)
}
\]

subject to separate expected-budget constraints:

\[
\mathbb E_\pi[C_{ver}]\leq B_{ver},
\]

\[
\mathbb E_\pi[C_{recov}]\leq B_{rec}.
\]

Here \(\lambda\) converts residual task loss to the cost/utility scale. If all terms are calibrated in the same monetary or utility unit, set \(\lambda=1\). Otherwise calibrate and report sensitivity to \(\lambda\).

### Hard budget safety

Expected constraints do not guarantee every realized trajectory stays within budget. The runtime must additionally check remaining resources before every action. For strict pathwise guarantees, impose:

\[
C_{ver}(\pi,\omega)\leq B_{ver},\qquad
C_{recov}(\pi,\omega)\leq B_{rec}
\]

for every admissible trajectory.

If recovery budget is insufficient, fail closed: block affected downstream tasks rather than consuming stale/untrusted data.

## 5. Expected-cost action value

At state \(s_t\), define:

\[
\Delta(v\mid s_t)=
J_{no\text{-}verify}(s_t)-J_{verify(v)}(s_t).
\]

A candidate score is:

\[
Score(v\mid s_t)=\frac{\Delta(v\mid s_t)}{c_v^{ver}}.
\]

The value estimator must integrate over PASS/FAIL observations, repair success/failure, recomputation success/failure, resource use, and terminal residual loss. It cannot be replaced by a static risk score without changing the method.

## 6. Theory status

Do not claim monotonicity, submodularity, adaptive submodularity, or a \(1-1/e\) approximation guarantee for the full stochastic model without proof. Analyze a simplified core model separately if useful. The full model includes stochastic observations, repair failures, overlapping descendants, and two resource constraints.
