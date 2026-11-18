# Problem Formulation

## 1. Dependency graph

Let the execution lineage be a directed acyclic graph:

\[
G=(V,E),
\]

where \((v,u)\in E\) means the computation of node \(u\) consumed a value from node \(v\). Each node stores its value, parents, children, version, status, cost parameters, and current error belief.

Let \(V_T\subseteq V\) be task-critical output nodes and \(w_u\geq 0\) their loss weights.

## 2. Intrinsic errors and propagation

For each node:

\[
X_v\sim\operatorname{Bernoulli}(p_v)
\]

denotes an intrinsic error introduced at node \(v\). For each edge:

\[
Z_{vu}\sim\operatorname{Bernoulli}(q_{vu})
\]

denotes propagation of an upstream error along edge \((v,u)\).

The node error indicator is:

\[
Y_u=X_u\lor\bigvee_{v\in Pa(u)}(Y_v\land Z_{vu}).
\]

Under mutual independence of relevant upstream events:

\[
P(Y_u=1)=1-(1-p_u)\prod_{v\in Pa(u)}(1-p_vq_{vu}).
\]

This product formula is not generally valid with correlated parents or shared latent causes. The implementation must make the independence assumption explicit.

## 3. Verification

For node \(v\), verifier sensitivity and specificity are:

\[
s_v=P(O_v=Fail\mid Y_v=1),\qquad
t_v=P(O_v=Pass\mid Y_v=0).
\]

Bayesian posterior after PASS:

\[
P(Y_v=1\mid Pass)=
\frac{p_v(1-s_v)}{p_v(1-s_v)+(1-p_v)t_v}.
\]

Posterior after FAIL:

\[
P(Y_v=1\mid Fail)=
\frac{p_vs_v}{p_vs_v+(1-p_v)(1-t_v)}.
\]

A verification result updates belief; it does not certify absolute truth. For a full DAG, downstream beliefs should be updated consistently with the dependency model.

## 4. Probabilistic repair and recomputation

Repair success:

\[
R_v\sim\operatorname{Bernoulli}(r_v).
\]

On success, create a new version of \(v\); on failure, mark it BLOCKED. A successful repair does not automatically validate descendants.

Recomputation success:

\[
K_u\sim\operatorname{Bernoulli}(\rho_u).
\]

Only descendants whose lineage consumed the invalidated old version are affected. For a repaired set \(S\):

\[
Affected(S)=\bigcup_{v\in S}Affected(v).
\]

Use a set union so shared descendants are recomputed once. Recompute in topological order and only when all required parent versions are current and valid.

## 5. State safety invariant

Suggested states: UNVERIFIED, VERIFIED, INVALID, REPAIRING, STALE, BLOCKED.

\[
Status(u)\in\{INVALID,STALE,BLOCKED\}
\Rightarrow u\text{ cannot be consumed as a trusted input.}
\]

Version checks are mandatory: a result verified against an old parent version becomes stale when that parent is replaced.

## 6. Scope

First prototype: static DAG per episode, observable lineage, replayable/idempotent computation, at most one repair and recomputation attempt per node. Side-effecting actions require idempotency or compensation and are excluded initially.
