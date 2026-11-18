# DAV Experimental Plan

## Stage A — synthetic DAG validation

Generate controllable DAGs with configurable node count, edge density, depth, shared-descendant overlap, intrinsic error probabilities, edge propagation probabilities, verifier sensitivity/specificity, repair/recomputation success rates, costs, output weights, and separate budgets.

Start with tiny DAGs where exact policy enumeration is possible.

## Stage B — baselines

All methods must use the same verifier, repair model, recomputation semantics, and budget accounting.

1. No Verification.
2. Random Budgeted Verification.
3. Always Verify (subject to budget).
4. Uncertainty Ranking.
5. Impact Ranking.
6. Static Risk Ranking.
7. DAV / AMCR.

## Stage C — ablations

- w/o propagation model;
- w/o overlap-aware affected-set union;
- w/o recovery-cost awareness;
- deterministic repair instead of probabilistic repair;
- w/o adaptive belief/state update;
- unified budget vs separate budgets.

Change one component at a time and keep other conditions fixed.

## Stage D — instrumented agent tasks

Only after synthetic validation, integrate with agent environments whose tool traces permit reliable lineage extraction. ALFWorld, WebShop, and GAIA do not automatically provide field-level lineage, repair labels, or recomputation costs; instrumented wrappers and controlled fault injection will be required.

## Metrics

- Final Task Success Rate.
- Residual Error Loss.
- Error Propagation Rate.
- Verification Cost.
- Repair Cost.
- Re-computation Cost.
- Total Execution Cost.
- Recovery Success Rate.
- Budget Violation Rate.
- Gap to exact optimal policy on tiny DAGs.

## Controls

- Same task instances and random seeds across methods.
- Same verifier quality and repair/recompute success parameters.
- Report means and confidence intervals across seeds.
- Separate oracle-parameter from noisy-estimation experiments.
- Sweep both budgets independently.
- Report cost components, not only weighted objective.
- Sensitivity analysis for lambda and probability-estimation errors.
