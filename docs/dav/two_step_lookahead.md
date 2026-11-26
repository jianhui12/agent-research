# DAV-AMCR-2Step v0.2 (experimental)

This branch is based on base commit c64595a12f269708c4ff2ea4c88ee1e8acdd0761.
The original AMCRPolicy and its tests remain unchanged. No merge into the
base branch is required to run this experiment.

## Integrated A/B/C design

A — tie breaking: among eligible actions within tie_tolerance of the best
two-step absolute net gain, prefer the smaller immediate verification cost,
then the lexicographically smaller version ID. Default tolerance: 1e-9
objective units, not a statistical test.

B — budget-aware combination: each first action's successor has its own
remaining verification and recovery budgets and updated version DAG.
The continuation may be a second feasible verification or stop. Do not sum
fixed single-action gains or use a static knapsack approximation.

C — adaptive lookahead: evaluate each root action on shared posterior worlds.
For every reachable observable successor, select a second action using
a separate one-step sample bank conditioned on its updated evidence.
Evaluate the selected second action on the original outer world's updated
latent state. Cache continuations by observable successor, never by hidden
error labels. Independent inner planning samples prevent direct per-world
oracle action selection.

## Objective and policy

Let V0(S) be expected terminal error and noncompletion loss.
The ideal depth-two Bellman recursion under the fixed verify/repair-on-FAIL
transition is:

    Q_h(S,a) = E[weighted action costs + V_(h-1)(S') | S,a]
    V_h(S) = min(V0(S), min_feasible_a Q_h(S,a))

TwoStepCostEstimator approximates Q_2 using nested Monte Carlo. The inner
continuation chooses the best positive absolute net gain, not the best
gain-to-cost ratio. The first action also maximizes absolute two-step
net gain; dividing two-step gain by only the first verification cost would
ignore the second action's resource usage. A and B remain hard budget
constraints through actual transitions. Stop is available at both depths.
The outer policy executes only the selected first action and replans.

By default evaluate ALL feasible first actions, including actions with
negative immediate value. Optional top_k screening ranks by single-step
gain / cost (ties by lower verification cost, then version ID). It can
miss pure information actions and is disabled by default.

## API

    from dav.lookahead import TwoStepAMCRPolicy, TwoStepCostEstimator

    estimator = TwoStepCostEstimator(
        config, outer_rollouts=128, inner_rollouts=128, seed=42,
        top_k=None, tie_tolerance=1e-9,
    )
    policy = TwoStepAMCRPolicy(estimator)
    decision = policy.decide(state)
    result = simulator.run(state, policy, seed=123)

The decision extends PolicyDecision with lookahead diagnostics. Each
candidate includes its two-step ActionEstimate, paired one-step total,
continuation rate and number of distinct observable successors.
ActionEstimate costs include BOTH steps; fail_probability refers to the
first check. Ranking score is absolute two-step net gain, not density.

## Deterministic reference cases

* Budget 3: gains [10,15,4], costs [2,3,1]. Base greedy first selects A
  and achieves gain 14. Two-step first selects B and achieves gain 15.
* Budget 4: gains [10,15,9], costs [2,3,2]. Two-step selects A then C
  (gain 19), rather than B (gain 15).
* Information value: an inexpensive perfect check on a sibling output B
  reveals a shared ancestor error affecting target C. B has negative
  one-step net gain -1 but two-step expected total 1.6 versus 3 for
  directly verifying C and 4 for stop.

These are analytical test cases, not claims of general benchmark superiority.

## Limitations

Nested MC is costly, especially with many observable recovery outcomes,
rare evidence, or many feasible candidates. Caching is valid only for this
synthetic transition with fixed version parameters and deterministic
placeholder values. Finite outer and inner banks cause estimation/selection
error; outer SE is conditional on the sampled posterior and chosen
continuation rule and does not include inner policy estimation error.
The optional confidence_z gate is heuristic. Two-step planning is not
globally optimal, nor an exact planning oracle. Repair-on-FAIL and
model-matched synthetic errors remain assumptions shared with the base.
Test and profile before running large experiments.

## Tests

    pytest -v tests/test_lookahead.py
    pytest -q

Expected collection: 9 new tests, 109 total relative to the branch point.
