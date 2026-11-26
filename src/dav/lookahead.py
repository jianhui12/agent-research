"""Budget-aware two-step receding-horizon AMCR (experimental branch).

Each first action is evaluated in paired posterior worlds. A second action
is selected using ONLY the resulting observable state and an independent
inner sample bank, then evaluated on the outer world. Stop is available at
both depths. No fixed additive assumption about verification gains is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Iterable

from .cost_estimator import (
    ActionEstimate, CostComparison, MonteCarloCostEstimator, RolloutResult,
    _summarize,
)
from .joint_sampler import sample_joint_posterior
from .latent_state import ScenarioNoise
from .objective import ObjectiveConfig, evaluate_realized
from .policy import PolicyDecision, RankedAction, feasible_candidates
from .state import DAVState
from .transition import VerificationTransition, execute_verification, validate_candidate


def _nonnegative(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _choose(state: DAVState, estimates: dict[str, ActionEstimate], *,
            min_gain: float, tie_tolerance: float, confidence_z: float) -> str | None:
    """Maximize absolute net gain, then prefer lower first-action cost.

    Do not divide the multi-step value by the first action's cost: future
    actions also consume budget and their costs already enter the objective.
    """
    eligible = []
    for v, estimate in estimates.items():
        gain = estimate.marginal_reduction
        se = estimate.paired_gain_standard_error
        if (estimate.version_id != v or not math.isfinite(gain)
                or not math.isfinite(se) or se < 0):
            raise ValueError(f"Invalid action estimate: {v}")
        if gain - confidence_z * se > min_gain:
            eligible.append(v)
    if not eligible:
        return None
    best = max(estimates[v].marginal_reduction for v in eligible)
    tied = [v for v in eligible
            if best - estimates[v].marginal_reduction <= tie_tolerance]
    return min(tied, key=lambda v: (state.graph.versions[v].verify_cost, v))


def _observable_key(state: DAVState, transition: VerificationTransition) -> tuple:
    """Cache key for reachable observable successor states, not latent worlds.

    Within one estimate_many call the root is fixed and callbacks return None.
    First action, label, observable operation events, current graph version
    statuses/edges, and budgets uniquely determine the successor in this model.
    No hidden truth, environment seed or sampled error is included.
    """
    events = () if transition.recovery is None else tuple(
        (e.action, e.version_id, e.outcome, e.cost, e.new_version_id, e.reason)
        for e in transition.recovery.events
    )
    return (
        transition.version_id, transition.observation, events,
        tuple(sorted(state.observations.items())),
        tuple(sorted((v, a.status.value) for v, a in state.graph.versions.items())),
        tuple(sorted((n, a.current_version_id)
                     for n, a in state.graph.logical_nodes.items())),
        tuple(sorted(state.graph.graph.edges())),
        state.checkpoint(),
    )


@dataclass(frozen=True)
class TwoStepAction:
    estimate: ActionEstimate
    one_step_expected_total: float
    continuation_rate: float
    observable_successors: int


@dataclass(frozen=True)
class TwoStepComparison:
    baseline: ActionEstimate
    candidates: dict[str, TwoStepAction]
    screened_out: tuple[str, ...]
    outer_rollouts: int
    inner_rollouts: int
    posterior_method: str
    posterior_effective_samples: float | None
    log_evidence_probability: float


@dataclass(frozen=True)
class TwoStepDecision(PolicyDecision):
    lookahead: TwoStepComparison | None = None


class TwoStepCostEstimator:
    """Nested Monte Carlo for an adaptive depth-two policy.

    Outer samples score complete trajectories. Inner samples select a
    continuation from an independent posterior conditioned on observable
    evidence. Continuations are cached only within a single decision.
    The default evaluates all first actions, including negative one-step
    gains; optional top_k screening is approximate and can miss information
    value. Both budget ledgers are copied and enforced by the shared transition.
    """

    def __init__(self, config: ObjectiveConfig, *, outer_rollouts: int = 128,
                 inner_rollouts: int = 128, posterior_method: str = "auto",
                 posterior_samples: int = 5000, max_exact_events: int = 14,
                 min_effective_samples: float = 20, seed: int = 0,
                 top_k: int | None = None, min_gain: float = 0.0,
                 tie_tolerance: float = 1e-9, confidence_z: float = 0.0):
        for name, count in (("outer_rollouts", outer_rollouts),
                            ("inner_rollouts", inner_rollouts)):
            if type(count) is not int or count < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        for name, value in (("min_gain", min_gain), ("tie_tolerance", tie_tolerance),
                            ("confidence_z", confidence_z)):
            _nonnegative(value, name)
        self.config = config
        self.outer_rollouts = outer_rollouts
        self.inner_rollouts = inner_rollouts
        self.seed = seed
        self.top_k = top_k
        self.min_gain = min_gain
        self.tie_tolerance = tie_tolerance
        self.confidence_z = confidence_z
        self.posterior_method = posterior_method
        self.posterior_samples = posterior_samples
        self.max_exact_events = max_exact_events
        self.min_effective_samples = min_effective_samples

    def _one_step(self, seed: int, rollouts: int) -> MonteCarloCostEstimator:
        return MonteCarloCostEstimator(
            self.config, rollouts=rollouts, seed=seed,
            posterior_method=self.posterior_method,
            posterior_samples=self.posterior_samples,
            max_exact_events=self.max_exact_events,
            min_effective_samples=self.min_effective_samples,
        )

    def _continuation(self, state: DAVState, estimator: MonteCarloCostEstimator) -> str | None:
        candidates = feasible_candidates(state)
        if not candidates:
            return None
        comparison = estimator.estimate_many(state, candidates)
        return _choose(state, comparison.candidates, min_gain=self.min_gain,
                       tie_tolerance=self.tie_tolerance, confidence_z=self.confidence_z)

    def estimate_many(self, state: DAVState,
                      candidate_version_ids: Iterable[str]) -> TwoStepComparison:
        candidates = sorted(set(candidate_version_ids))
        for v in candidates:
            validate_candidate(state, v)
        for target in self.config.targets:
            state.current_version_id(target.logical_node_id)
        # Fixed independent streams. Inner planning never uses an outer seed,
        # latent error label or outer realized objective to choose an action.
        rng = random.Random(self.seed)
        posterior_seed, draw_seed, inner_seed, screening_seed = (
            rng.getrandbits(64) for _ in range(4)
        )
        screened_out = ()
        if self.top_k is not None and len(candidates) > self.top_k:
            screening = self._one_step(screening_seed, self.inner_rollouts).estimate_many(
                state, candidates
            )
            ranked = sorted(candidates, key=lambda v: (
                -screening.candidates[v].marginal_reduction / state.graph.versions[v].verify_cost,
                state.graph.versions[v].verify_cost, v))
            screened_out = tuple(sorted(ranked[self.top_k:]))
            candidates = sorted(ranked[:self.top_k])
        posterior = sample_joint_posterior(
            state.graph, state.observations, method=self.posterior_method,
            samples=self.posterior_samples, seed=posterior_seed,
            max_exact_events=self.max_exact_events,
            min_effective_samples=self.min_effective_samples,
        )
        worlds = posterior.draw(self.outer_rollouts, seed=draw_seed)
        noises = tuple(ScenarioNoise(rng.getrandbits(64)) for _ in worlds)
        checkpoint = state.checkpoint()
        baseline_runs = [
            RolloutResult(evaluate_realized(state, self.config, world,
                                            checkpoint=checkpoint), None)
            for world in worlds
        ]
        baseline_totals = [run.objective.total for run in baseline_runs]
        inner = self._one_step(inner_seed, self.inner_rollouts)
        estimates = {}
        for v in candidates:
            continuations: dict[tuple, str | None] = {}
            runs, first_totals = [], []
            continued = 0
            for world, noise in zip(worlds, noises):
                branch = state.fork()
                first = execute_verification(branch, v, world, noise)
                first_totals.append(evaluate_realized(
                    branch, self.config, first.truth, checkpoint=checkpoint).total)
                key = _observable_key(branch, first)
                if key not in continuations:
                    continuations[key] = self._continuation(branch, inner)
                second_id = continuations[key]
                final_truth = first.truth
                if second_id is not None:
                    second = execute_verification(branch, second_id, first.truth, noise)
                    final_truth = second.truth
                    continued += 1
                runs.append(RolloutResult(evaluate_realized(
                    branch, self.config, final_truth, checkpoint=checkpoint),
                    first.observation))
            estimates[v] = TwoStepAction(
                _summarize(v, runs, baseline_totals), fmean(first_totals),
                continued / self.outer_rollouts, len(continuations))
        return TwoStepComparison(
            _summarize(None, baseline_runs, baseline_totals), estimates,
            screened_out, self.outer_rollouts, self.inner_rollouts,
            posterior.method, posterior.effective_sample_size,
            posterior.log_evidence_probability)


class TwoStepAMCRPolicy:
    """Execute only the first chosen action, then replan at the next step.

    Scores in the inherited ranking are absolute two-step net gains, not
    the one-step AMCR density. Full diagnostics are in decision.lookahead.
    """

    def __init__(self, estimator: TwoStepCostEstimator):
        self.estimator = estimator

    def decide(self, state: DAVState) -> TwoStepDecision:
        candidates = feasible_candidates(state)
        if not candidates:
            return TwoStepDecision(None, "no_feasible_candidate")
        comparison = self.estimator.estimate_many(state, candidates)
        estimates = {v: result.estimate for v, result in comparison.candidates.items()}
        ranking = tuple(sorted((
            RankedAction(v, state.graph.versions[v].verify_cost, e,
                         e.marginal_reduction,
                         e.marginal_reduction - self.estimator.confidence_z *
                         e.paired_gain_standard_error)
            for v, e in estimates.items()
        ), key=lambda a: (-a.score, a.verify_cost, a.version_id)))
        selected = _choose(state, estimates, min_gain=self.estimator.min_gain,
                           tie_tolerance=self.estimator.tie_tolerance,
                           confidence_z=self.estimator.confidence_z)
        # Preserve common comparison API for analysis; lookahead adds diagnostics.
        common = CostComparison(comparison.baseline, estimates,
                                comparison.outer_rollouts, comparison.posterior_method,
                                comparison.posterior_effective_samples,
                                comparison.log_evidence_probability)
        return TwoStepDecision(selected, "verify" if selected is not None else "no_positive_gain",
                               ranking, common, comparison)
