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


## Recovery Engine v1 (implemented)

`src/dav/recovery_engine.py` provides `RecoveryEngine.recover(source_version_id, repair, recompute)`.

- `repair(artifact_version) -> value` is an injectable replayable repair callback.
- `recompute(logical_node_id, tuple[input_artifact_versions]) -> value` is a replayable recomputation callback.
- Bernoulli samples use the configurable `repair_success_prob` and `recompute_success_prob` on the old artifact version. A seeded `random.Random` enables reproducible experiments.
- Both callbacks can raise; exceptions become recorded failures rather than silently producing a trusted result.
- Every attempted repair/recompute spends from **recovery budget**, checked before attempting the operation. Verification budget is untouched.
- On successful execution, a **new UNVERIFIED version** is created with exact input-version lineage and a recorded execution cost. A successful computation is **not** a verifier PASS.
- Descendants are found from the old artifact version, deduplicated by version ID, and processed topologically. Shared descendants run at most once.
- If a required parent is blocked, stale, invalid, or not its logical node's current version, recomputation is blocked; no stale inputs are silently used.
- A failed repair blocks affected descendants. A failed recomputation or exhausted budget blocks the node and prevents dependent recomputation.

### Minimal usage

```python
import random
from dav.budget_manager import BudgetManager
from dav.recovery_engine import RecoveryEngine

engine = RecoveryEngine(graph, BudgetManager(verification_budget=3, recovery_budget=10),
                        rng=random.Random(42))

outcome = engine.recover(
    "D1@v1",
    repair=lambda old: repaired_value(old),
    recompute=lambda node_id, parents: recompute_value(node_id, parents),
)
print(outcome.success, outcome.replacements, outcome.recovery_spent)
```

### Current limitations

- Only a **single repair episode** is supported; no retry scheduling or joint candidate selection.
- Repair/recompute outcomes are treated as **oracle-observable** by the simulation harness. In a realistic environment, outcome uncertainty must be represented in beliefs and a validator must be consulted separately.
- Callbacks must be deterministic/replayable or idempotent. Side-effecting tool calls require stronger transaction/compensation design.
- Blocked tasks need a **non-completion penalty** in the future objective; blocked status cannot be treated as zero risk/loss.
- The Recovery Engine is a first executable recovery path, **not yet AMCR**; it does not decide which node to verify or estimate future costs with Monte Carlo rollouts.
- The initial implementation does not offer atomic rollback across exceptions thrown during graph mutation and assumes single-threaded execution with a stable DAG.
- The existing graph model assumes one producer execution per artifact version and does not yet support dynamic changes to dependency shape.

### Test command

```bash
pytest -v tests/test_recovery_engine.py
pytest -v
```
