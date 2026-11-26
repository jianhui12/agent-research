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
- Blocked tasks need a **non-completion penalty**; the terminal objective implemented below now supplies it.
- The Recovery Engine is a first executable recovery path, **not yet AMCR**; it does not decide which node to verify or estimate future costs with Monte Carlo rollouts.
- The initial implementation does not offer atomic rollback across exceptions thrown during graph mutation and assumes single-threaded execution with a stable DAG.
- The existing graph model assumes one producer execution per artifact version and does not yet support dynamic changes to dependency shape.

### Test command

```bash
pytest -v tests/test_recovery_engine.py
pytest -v
```


## Error Model v1: correlation-aware exact/Monte Carlo inference

`src/dav/error_model.py` implements a noisy-OR **generative** error model on exact artifact-version lineage:

```text
E[v] = intrinsic_error[v] OR OR_{u -> v}(E[u] AND transmission[u,v])
```

Intrinsic errors at each artifact and edge transmission events are independent
Bernoulli primitives. Derived error indicators can be **correlated** through
shared ancestor error variables. This distinction matters for a diamond
D1->D2, D1->D3, D2->D4, D3->D4: with P(E[D1])=0.5 and deterministic edges,
the correct P(E[D4])=0.5, whereas the legacy independence-product baseline
gives 0.75.

The public APIs:

- `sample_ground_truth(graph, seed=...)`: draw a joint error realization for a synthetic episode. This is an *oracle* sample, not an observed error label.
- `infer_error_risks(graph, method="exact", observations={...})`: enumerate primitive independent Bernoulli events. Exact inference is capped by `max_exact_events` (default 20 uncertain primitives) to avoid exponential blow-up.
- `infer_error_risks(graph, method="mc", samples=10000, seed=..., observations={...})`: Monte Carlo likelihood weighting; supports evidence from imperfect binary verifiers and reproducible seeds. Importance-weight degeneracy is possible for rare evidence, requiring diagnostics in subsequent work.
- `propagate_error_beliefs(graph)`: old independent-parent marginal approximation retained **only as a baseline**.

`observations` maps artifact version IDs to `"pass"`/`"fail"`. A verifier has
sensitivity P(fail | error) and specificity P(pass | correct).
Evidence on a downstream artifact can update upstream error beliefs.
Inference **does not** set NodeStatus or overwrite `posterior_error_prob`;
the returned `RiskEstimate.probabilities` is a conditional belief snapshot.

The default scope is all current versions plus their ancestor versions.
Unusable (INVALID/STALE/BLOCKED/REPAIRING) versions within that scope
raise instead of being interpreted as clean. Explicit
`target_version_ids` limits the required ancestry and identifies versions
charged for expected loss. By default, expected loss is the sum of
`loss_weight * P(error)` for **current** versions, not all historical
versions. This weighted risk does **not** include noncompletion
penalties for blocked tasks or intervention costs; use objective.py for those.

### Example

```python
from dav.error_model import infer_error_risks, sample_ground_truth

truth = sample_ground_truth(graph, seed=123)
exact = infer_error_risks(graph, method="exact",
                          target_version_ids=["D4@v1"])
posterior = infer_error_risks(
    graph, method="mc", samples=50000, seed=123,
    target_version_ids=["D4@v1"],
    observations={"D4@v1": "pass"},
)
print(truth, exact.expected_loss, posterior.probabilities)
```

### Model assumptions and remaining research work

- This v1 noisy-OR model assumes primitive Bernoulli independence and no
  hidden correlated tool failures, contextual dependence, or confounding.
- Source `intrinsic_error_prob` is a prior for new synthetic error draws; it
  is distinct from `posterior_error_prob`. Do not reuse the posterior as
  intrinsic probability without a formally defined state update.
- Evidence likelihood assumes verifier observations are conditionally
  independent given per-version errors; repeated checks by the same
  correlated model violate that assumption.
- `RiskEstimate.expected_loss` is observational risk, **not** an estimate
  of expected value of verifying or repairing an artifact. Counterfactual
  interventions and dynamic decision policies remain future work.
- Exact inference is exponential in uncertain primitive count; MC may have
  high variance when conditioning on low-probability evidence.
- This implementation has no graph-level calibration procedure for
  propagation probabilities or verifier quality parameters.

### Error Model tests

```bash
pytest -v tests/test_error_model.py
pytest -v
```


## Observable state and terminal objective v1

`DAVState` in `src/dav/state.py` owns a version DAG, independent budget
ledgers, and an observation map keyed by exact version ID. It does not expose
latent ground truth. `state.fork()` deep-copies nested graph data, execution
records, budget balances and evidence; no live callbacks, external services,
or RNG streams are copied. Payloads must support deepcopy. Candidate rollouts
must use isolated, side-effect-free simulation callbacks.

- `record_verification(version_id, observation)` validates the current usable
  lineage, charges verification cost, and records a single binary observation
  per version. Repeated verification of the same version is unsupported.
  PASS/FAIL is evidence, not a correctness label or an automatic repair command.
- `checkpoint()` captures remaining balances; `costs_since(checkpoint)`
  measures subsequent consumption. Without a checkpoint, costs are cumulative
  from state construction. Initial supplied observations are already paid.
- `infer_risks(target_version_ids, method="exact"|"mc", ...)` conditions on
  all stored evidence, including historical observations. Historical vertices
  are admitted only as evidence ancestry, not as usable targets. Old evidence
  may inform shared unchanged ancestors, but never transfers to new versions.
  The status of a historical vertex is not treated as error ground truth.
- Current output availability requires every ancestor to be current and
  operationally usable. Blocked, invalid, stale, repairing, and missing
  outputs are incomplete for terminal evaluation.

### Recovery risk initialization

A completed recovery operation does not certify the generated artifact.
`ArtifactVersion.repair_residual_error_prob` is the conditional semantic
error prior after a completed repair (synthetic default 0.05). It is
**not** 1 minus `repair_success_prob`; the latter is an operational success
probability. A repair is modeled as fresh independent reacquisition with
no data-dependency inputs. Its old source is linked through execution
metadata, not a noisy-OR propagation edge.

`recompute_intrinsic_error_prob` optionally overrides intrinsic risk of a
fresh replay; None inherits the previous intrinsic rate. New versions copy
verification/repair/recompute probabilities, costs, loss weight and metadata,
but remain UNVERIFIED and do not copy the old posterior or evidence.
The `posterior_error_prob` field is initialized to the new intrinsic rate
only; it is not a full marginal. Use state.infer_risks for actual beliefs.
Recomputation reuses version-edge propagation configuration, so residual
upstream risk can still propagate. Only current descendants are replayed.
Historical versions remain available for provenance.

### Objective

`TaskTarget(logical_node_id, error_loss, blocked_loss)` identifies a required
task output with two independent finite nonnegative penalties.
`ObjectiveConfig(targets, lambda_v=1.0, lambda_r=1.0)` uses explicit stable
logical IDs, not historical versions. A missing output for a registered
logical node is incomplete; an unregistered target is a configuration error.

For each terminal logical target j, let A_j be 1 if its current output
and all ancestors are usable and 0 otherwise. The per-rollout objective is

```text
J = sum_j [ A_j * w_j * E_j + (1-A_j) * b_j ]
    + lambda_v * C_v + lambda_r * C_r
```

Here E_j is a latent binary error on the current output, w_j is its
error-loss weight, and b_j is its noncompletion penalty. A blocked
output is charged once through b_j, not simultaneously through w_j.
C_v/C_r are actual amounts debited from the separate budgets, not planned
costs; failed attempted operations are included. Each budget remains a
hard constraint. Choose b_j relative to task failure severity; zero or small
noncompletion penalties may favor aborting, and require explicit justification.

`evaluate_expected` computes conditional expected terminal error loss for
usable outputs via the joint inference model, plus deterministic blocked
loss and spent costs. `evaluate_realized` instead requires oracle truth for
usable outputs, and is for synthetic evaluation only. Both return a
decomposed `ObjectiveResult` with a `total` property.

```python
from dav.state import DAVState
from dav.objective import ObjectiveConfig, TaskTarget, evaluate_expected

state = DAVState(graph, budgets)
config = ObjectiveConfig((TaskTarget("D4", error_loss=10, blocked_loss=15),),
                         lambda_v=0.2, lambda_r=0.1)
checkpoint = state.checkpoint()
branch = state.fork()
branch.record_verification("D1@v1", "fail")
# Apply a recovery policy to branch.graph and branch.budgets if appropriate.
result = evaluate_expected(branch, config, checkpoint=checkpoint)
print(result.total, result.error_loss, result.blocked_loss)
```

### Limitations before AMCR

- The 0.05 repair residual prior is a configurable experimental assumption,
  not measured real-world performance. Reacquisition independence and fresh
  replay errors are model assumptions, not established properties of tools.
- Historical evidence increases inference scope and can make exact inference
  infeasible. Likelihood weighting may degenerate with rare evidence.
- State contains observations, not a persistent posterior particle population.
  Cost Estimator still needs conditioned joint samples and consistent latent
  state transitions: unchanged artifacts must not be resampled per action.
  It must also preserve action-independent randomness for fair comparisons.
- Objective is additive over explicit outputs; correlated task-level loss or
  joint success requirements need an alternate objective.
- This is fixed-state expected/realized evaluation, not an implemented
  counterfactual action-value estimator or AMCR policy.

### Tests

```bash
pytest -v tests/test_state.py tests/test_objective.py tests/test_recovery_engine.py
pytest -v
```
