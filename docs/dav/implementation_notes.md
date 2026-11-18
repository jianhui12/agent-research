# DAV Implementation Notes

## Package layers

- `graph.py`: DAG, parent/child relations, topological order, version lineage.
- `models.py`: node state, metadata, action/outcome records.
- `error_model.py`: intrinsic error and edge propagation simulation.
- `verifier.py`: probabilistic verifier and Bayesian posterior updates.
- `recovery.py`: probabilistic repair, invalidation, topological recomputation.
- `budget_manager.py`: independent budget ledgers and hard guards.
- `cost_estimator.py`: exact/Monte Carlo expected objective estimator.
- `policy.py`: AMCR candidate selection.
- `simulator.py`: episode execution and metrics.
- `baselines.py`: random, uncertainty, impact, static risk.

## Contracts

1. A node version increases whenever repair/recomputation creates a new value.
2. Each derived node records exact parent versions consumed.
3. A parent version change invalidates only derived results that consumed the old version.
4. Shared affected descendants are recomputed once.
5. Recompute order is topological.
6. INVALID, STALE, BLOCKED nodes cannot be trusted inputs.
7. Every action checks its relevant remaining budget before execution.
8. Record stochastic outcomes and random seeds.
9. Separate simulator ground truth from policy-estimated beliefs.
10. Never treat verifier PASS as ground-truth correctness.

## First milestone

Implement a 4–6 node deterministic DAG and test shared-descendant invalidation, version mismatch, topological recomputation, blocked propagation after repair failure, and hard budget enforcement. Then add probabilistic outcomes and Monte Carlo evaluation.
