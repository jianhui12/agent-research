# Base AMCR pilot experiment

This pilot deliberately uses the original one-step AMCRPolicy, not the
experimental two-step branch. The purpose is to check the evaluation pipeline
and identify preliminary patterns before benchmark-scale runs.

## Controls

All eight policies share feasible_candidates, verification/recovery
transition, graph, budgets, target definition, and paired environment seed.
Each episode-policy run gets a fresh policy and its own copy of the state.
The planner seed is separate from the environment seed. The program checks
that initial latent error worlds match across policies.

| Policy | Rule |
|---|---|
| no_verification | Stop immediately |
| random | Uniform random feasible candidate |
| risk | Highest posterior error probability |
| uncertainty | Highest p(1-p) |
| impact | Largest count of usable current affected versions, including self |
| risk_impact | Posterior error probability times impact |
| cost_aware_risk_impact | Risk times impact divided by verification cost |
| amcr | Original paired Monte Carlo gain/cost ranking with positive-gain stopping |

Heuristic baselines continue until no feasible action remains, while AMCR
can stop with budget left. This is a comparison of complete policies, not
an isolated comparison of rankings; matched stopping rules and AMCR
no-stop ablations are needed for the latter. Risk-based baselines use the
same joint-posterior model rather than an artificially weaker independent
marginal approximation.

## Configurations and outputs

Fixed DAGs: chain (6 nodes), diamond (4), layered (7). Defaults: chain and
diamond. Each node has synthetic intrinsic error 0.12, sensitivity and
specificity 0.90, repair completion 0.90, recomputation completion 0.95,
repair residual error 0.05. Edge propagation probability is 0.75.
Verification costs alternate 1 and 2; repair/recompute each cost 1.
Only terminal nodes are required targets. Error loss is 20, noncompletion
loss 30; lambda_v=lambda_r=0.25. Default verification budget 3 and recovery
budget 4. These are initial synthetic assumptions, not empirically fitted.

The CSV has one row per topology, episode, and policy, including both seeds,
budget and sampling settings, total objective, error/noncompletion losses,
actual spending, all-target success, completion and correct-output rates,
action count, stop reason and runtime. Runtime is not deterministic.
Failure (including posterior ESS failure) raises with the episode/policy
context rather than silently dropping results. CSV is written on completion.

## Run

First run the original and new tests:

    pytest -q

Expected collection: 116 tests if the base branch point contains the
previous 100 tests. The 16 new test cases cover baseline decisions,
feasibility, fixed DAG construction and paired pilot reproducibility.

Smoke run (4 rows):

    python -m dav.pilot --topologies diamond --policies no_verification amcr --episodes 2 --rollouts 16 --output results/smoke.csv

Initial pilot (160 rows):

    python -m dav.pilot --topologies chain diamond --episodes 10 --rollouts 64 --seed 42 --output results/pilot_base_seed42.csv

A second budget setting may be run separately:

    python -m dav.pilot --topologies chain diamond --episodes 10 --rollouts 64 --verification-budget 5 --recovery-budget 8 --seed 42 --output results/pilot_base_v5_r8_seed42.csv

Do not overwrite results from a different configuration with the same filename.
Do not interpret 10-episode means as statistically significant. Subsequent
experiments should use preregistered larger samples, paired confidence intervals,
multiple DAG sizes and topologies, parameter sensitivity, runtime profiling,
and model mismatch. Do not tune on held-out evaluation seeds.

The two-step branch research/dav-amcr-2step-v0.2 was created from base
commit c64595a, before these pilot files. It remains isolated until the
base experiment is validated. Port the same pilot protocol to the
experimental branch when ready; do not modify base AMCR to make that comparison.
