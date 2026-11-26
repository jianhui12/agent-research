# Expanded DAV synthetic benchmark (base v0.1)

The original dav.pilot entry point and its eight default policies are retained.
Default AMCRPolicy settings preserve gain/cost ordering, positive-gain stopping
and the original version-ID tie rule. This experiment extension does not merge
the two-step branch or change main.

## Comparisons

All policies execute against the same actual graph, verifier, recovery transition,
objective and hard budgets. Each paired episode has one graph seed, environment
seed and planner seed; fresh policy instances prevent cross-episode carryover.

| Policy ID | Definition |
|---|---|
| no_verification | Stop immediately |
| random | Random feasible check |
| risk | Maximum posterior error marginal |
| uncertainty | Maximum p(1-p) |
| impact | Current usable descendant count including self |
| risk_impact | Risk times that count |
| cost_aware_risk_impact | Risk times count divided by check cost |
| amcr | Original one-step full AMCR |
| terminal_once | Risk-first on initial required output versions only |
| terminal_only | Risk-first on current required output versions only |
| terminal_amcr | Original AMCR, candidate set restricted to required outputs |
| amcr_no_stop | Same AMCR score; select top feasible even if gain <= 0 |
| amcr_absolute_gain | Same AMCR estimates and stop rule; rank by absolute gain |
| risk_positive_stop | Risk ranking within the same AMCR positive-gain eligible set |
| target_risk_impact | Posterior risk times unique reachable required-output error loss / check cost |
| amcr_independent_worlds | Same posterior marginals; independently sample rollout error labels |
| amcr_optimistic_recovery | Planner ignores recovery failures, binding recovery budget and weighted recovery costs |

The terminal-AMCR comparison isolates the candidate-set restriction while retaining
the estimator and stopping rule. Risk-positive-stop isolates ranking within the
same eligibility gate, not a cheap risk-only baseline; it recomputes the posterior
bank with the estimator's seed and pays extra planning time. No-stop disables
only the positive-gain filter, never feasibility. The independent-world ablation
retains correlation-aware marginal inference but destroys joint sampling
correlations (including deterministic relationships); it is NOT an ablation of
the entire graph model. The optimistic-recovery ablation changes only hypothetical
planning: operational success=1, sufficient hypothetical recovery budget and
lambda_r=0. Actual environment and actual reported objective are unchanged.
It retains residual/replay errors, and bundles three assumptions rather than
identifying their separate effects. None of these policies see environment truth.

## Scenarios

Fixed graphs: chain (6 nodes), diamond (4), layered (7), fanout (7 nodes,
3 required outputs), multi_output (8 nodes, 2 required outputs). The
random_dag scenario uses a seeded ordered backbone plus random additional
incoming edges, with bounded fan-in and a designated output suffix. Every
node reaches an output. This is not a uniform sample of all DAGs.

Profiles:
- legacy: old parameters, all verified artifacts can be independently reacquired.
- derived: identical parameters, except non-root data must replay actual inputs.
  RecoveryEngine uses the existing recomputation cost/success/intrinsic risk for
  the selected derived artifact, preserves incoming propagation edges and
  records exact input versions. Successful execution need not correct erroneous
  input data. Root acquisition still uses repair cost/success/residual risk.
  Both the estimator and simulator share this opt-in transition.
- shared_source: derived recovery, propagation 0.95, target verification cost
  doubled, V=5 and R=10. A combined stress scenario, NOT a single-factor
  causal comparison.
- high_error: derived with intrinsic risk 0.30.
- weak_verifier: derived with sensitivity 0.70, specificity 0.75.
- low_propagation: derived with propagation 0.25.

Every ScenarioConfig field can be overridden using --set KEY=VALUE.
--verification-budgets and --recovery-budgets form a Cartesian product
within each profile. Use one-factor changes for sensitivity analysis.
By default only the legacy profile is run. All default profiles are
synthetic, not calibrated or model-mismatch experiments.

## Run in stages

First run tests (expected 137 collected if the previous 116 remain present):

    pytest -v tests/test_benchmark.py
    pytest -q

Smoke check (2 episodes x 2 topologies x 2 profiles x 4 policies = 32 rows):

    python -m dav.benchmark --profiles legacy derived --topologies diamond fanout --policies no_verification risk terminal_amcr amcr --episodes 2 --rollouts 16 --seed 43 --trace --output results/benchmark_smoke.csv

First priority: compare full and terminal-constrained AMCR on the old setup,
with the old 100 seed schedule (200 x 5 = 1000 rows):

    python -m dav.benchmark --profiles legacy --topologies chain diamond --policies no_verification risk terminal_once terminal_amcr amcr --episodes 100 --rollouts 64 --seed 43 --trace --output results/terminal_legacy_seed43.csv

Then use a matched recovery-mode and multi-output experiment with fresh seeds.
Keep budgets equal across profiles; 2 x 2 x 6 x 100 = 2400 rows:

    python -m dav.benchmark --profiles legacy derived --topologies fanout multi_output --policies no_verification risk terminal_only terminal_amcr target_risk_impact amcr --verification-budgets 5 --recovery-budgets 10 --episodes 100 --rollouts 64 --seed 44 --output results/multi_output_seed44.csv

Example targeted ablation (2 x 7 x 100 = 1400 rows):

    python -m dav.benchmark --profiles derived --topologies chain fanout --policies risk risk_positive_stop amcr amcr_no_stop amcr_absolute_gain amcr_independent_worlds amcr_optimistic_recovery --verification-budgets 5 --recovery-budgets 10 --episodes 100 --rollouts 64 --seed 45 --output results/ablation_seed45.csv

Small budget grid (3 x 3 x 2 x 3 x 30 = 1620 rows):

    python -m dav.benchmark --profiles derived --topologies diamond fanout --policies risk terminal_amcr amcr --verification-budgets 2 4 6 --recovery-budgets 2 4 8 --episodes 30 --rollouts 64 --seed 46 --output results/budget_grid_seed46.csv

Optional scalability smoke:
    
    python -m dav.benchmark --profiles derived --topologies random_dag --set random_nodes=12 --set random_targets=3 --set random_fan_in=3 --policies risk terminal_amcr amcr --episodes 5 --rollouts 32 --seed 47 --output results/random_smoke_seed47.csv

## Outputs

For --output results/example.csv:
- example.csv: one row per scenario/topology/episode/policy, including all
  configuration fields, separate configured target loss weights and realized
  losses, seeds, costs, accuracy/completion, first action, target checks,
  verification confusion counts, recovery event counts, initial success and
  corrected/degraded indicators.
- example.summary.csv: grouped means and conditional correction/degradation rates
  (blank if denominator zero), initial-correct/wrong counts, summed events.
- example.manifest.json: exact scenario settings and seed/sampling protocol.
- example.trace.jsonl: optional full offline trace including initial DAG edges,
  initial and final latent labels, observations, candidate estimates/rankings
  where supplied, budget checkpoints, recovery events and stop diagnostics.
  Trace truth must never be fed to the policy. For risk_positive_stop, stored
  ranking is the shared AMCR eligibility diagnostic, not a risk-sorted list.

The program refuses existing output paths; use a new filename for each run.
Rows/traces accumulate in memory and files are written after the run; do not
enable full traces for huge sweeps without profiling. Failures (including
posterior ESS failures) abort with scenario and seed context, rather than
silently discarding failed runs. Do not confuse the number of policy rows
with independent trials; common seeds across scenarios and topologies require
paired or seed-clustered analysis. Save git commit separately (git rev-parse HEAD).
No claim of significance follows from a single small parameter sweep.

## Remaining research boundaries

The benchmark is still an already-constructed synthetic version DAG, with
model-matched probabilities and forced repair/replay on FAIL. Derived replay
improves lineage realism but is not real tool execution or independently
adaptive root-cause recovery. More heterogeneous tool priors, parameter
mismatch between planner and environment, multiple validators, selective
recovery, independent test seeds, exact planning comparison and real Agent
benchmarks remain future work. These files do not establish benchmark
superiority; execute tests and collect results before conclusions.
