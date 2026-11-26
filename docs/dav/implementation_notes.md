# DAV Implementation Notes

## Graph architecture: version-aware provenance DAG

The graph's vertices are **artifact versions**, not mutable logical data names.

- `LogicalNode`: stable identity and current-version pointer.
- `ArtifactVersion`: one immutable-in-identity data/result version with status and risk estimates.
- `DependencyEdge`: exact source-version to target-version consumption, with edge-level propagation probability and optional input role.
- `ExecutionRecord`: exact input/output version IDs, operation/tool, parameters, cost and timestamp.

A new repair/recomputation result creates a new artifact version. Never overwrite an old value in place.

## Graph API

- `add_logical_node()` / `add_node()`: register a logical data object.
- `add_version()`: create a version under a logical node.
- `get_current_version()`: retrieve the current version.
- `record_execution()`: register exact inputs/outputs and create provenance edges atomically.
- `get_direct_consumers()`: return versions directly consuming a version.
- `get_affected_nodes()`: return the deduplicated transitive descendant set.
- `invalidate_versions()`: mark affected descendants STALE.
- `supersede_version()`: advance the logical current-version pointer and invalidate old-version descendants.
- `topological_sort()`: order a full graph or selected affected subgraph.
- `check_lineage()` / `validate_graph()`: check version, execution and edge consistency.

## Recovery contract

1. Record the repair/recompute execution and create its output version first.
2. Mark the replaced source version INVALID.
3. Supersede it with the new version.
4. Invalidate only descendants reachable from the exact old version.
5. Recompute each affected version once, in topological order.
6. A recomputed output is a new version whose execution record points to the exact current input versions.
7. INVALID, STALE and BLOCKED versions must never be consumed as trusted inputs.

`recovery.py` currently builds a deterministic recovery plan; stochastic execution and budget accounting belong in the recovery runner/simulator.

## Other package layers

- `error_model.py`: intrinsic error and edge-level propagation probabilities.
- `verifier.py`: imperfect verifier and Bayesian posterior updates.
- `budget_manager.py`: independent verification/recovery ledgers with hard guards.
- `policy.py`: AMCR candidate selection.
- `cost_estimator.py`: future exact/Monte Carlo expected-objective estimator.
- `simulator.py`: episode execution and metrics.
- `baselines.py`: random, uncertainty, impact, static risk.

## First graph tests

Test shared-descendant deduplication, old-version-only invalidation, cycle rejection, topological order, execution lineage consistency, and preservation of old versions.
