# DAV Research Overview

**Full name:** Dual-Budget Adaptive Verification and Dependency-Guided Re-computation  
**Status:** Research formulation v0.1

## Research question

For long-horizon tool-using LLM agents, how should a system allocate a limited verification budget across intermediate results while accounting for probabilistic detection, dependency-mediated error propagation, and a separate limited budget for repair and downstream recomputation?

The central decision is **which result to verify next**, considering expected downstream loss and the recovery resources that may be required if an error is detected.

## Working hypothesis

A dependency-aware adaptive policy that estimates the expected marginal reduction in total execution cost and residual task loss can outperform static uncertainty/risk ranking under the same verification and recovery budgets. This is a hypothesis to test, not an established result.

## Components

1. DAG representation of intermediate data/results and versioned lineage.
2. Error model separating intrinsic node errors from propagated upstream errors.
3. Probabilistic verification using sensitivity and specificity.
4. Probabilistic repair; successful repair creates a new version.
5. Dependency-guided invalidation and topological recomputation.
6. Separate verification and recovery budgets.
7. Adaptive selection based on expected marginal objective reduction.

## Initial scope

- Dependencies are observable from execution logs.
- First experiments use synthetic DAGs with controlled error injection.
- Error probabilities, edge propagation probabilities, verifier quality, repair success and execution costs are configurable.
- At most one repair and one recomputation attempt per node per episode.
- Invalid, stale, or blocked nodes cannot be consumed as trusted inputs.
- Irreversible side-effecting tools are out of scope until idempotency/compensation is modeled.

## Explicit non-claims

- No submodularity or adaptive-submodularity claim.
- No approximation ratio for the proposed greedy policy.
- Closed-form propagation products require independence assumptions.
- AMCR is a candidate heuristic, not a proven optimal policy.

## Milestones

- [ ] Formalize conditional-independence assumptions.
- [ ] Implement exact policy enumeration for tiny DAGs.
- [ ] Implement Monte Carlo expected-cost estimator.
- [ ] Implement adaptive marginal-cost-reduction policy.
- [ ] Compare against static and random baselines.
- [ ] Run ablations.
- [ ] Evaluate on instrumented agent tasks.
