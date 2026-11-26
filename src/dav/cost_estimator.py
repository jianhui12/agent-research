"""One-step paired Monte Carlo action-value estimator for DAV.

Baseline: stop and evaluate terminal outputs. Candidate: verify one exact
version, on FAIL attempt recovery, on PASS stop; then evaluate. This is
a fixed lookahead policy, not a globally optimal planning algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean, stdev
from typing import Iterable, Mapping

from .joint_sampler import JointPosterior, sample_joint_posterior
from .latent_state import ScenarioNoise
from .objective import ObjectiveConfig, ObjectiveResult, evaluate_realized
from .transition import execute_verification, validate_candidate
from .state import DAVState


@dataclass(frozen=True)
class RolloutResult:
    objective: ObjectiveResult
    observation: str | None


@dataclass(frozen=True)
class ActionEstimate:
    version_id: str | None
    expected_total: float
    standard_error: float
    expected_error_loss: float
    expected_blocked_loss: float
    expected_verification_cost: float
    expected_recovery_cost: float
    probability_any_blocked: float
    fail_probability: float
    marginal_reduction: float
    paired_gain_standard_error: float


@dataclass(frozen=True)
class CostComparison:
    baseline: ActionEstimate
    candidates: dict[str, ActionEstimate]
    rollouts: int
    posterior_method: str
    posterior_effective_samples: float | None
    log_evidence_probability: float


def _rollout(state: DAVState, config: ObjectiveConfig,
             candidate: str | None, truth: Mapping[str, bool],
             noise: ScenarioNoise) -> RolloutResult:
    checkpoint = state.checkpoint()
    if candidate is None:
        return RolloutResult(evaluate_realized(state, config, truth,
                                               checkpoint=checkpoint), None)
    branch = state.fork()
    step = execute_verification(branch, candidate, truth, noise)
    result = evaluate_realized(branch, config, step.truth, checkpoint=checkpoint)
    return RolloutResult(result, step.observation)


def _se(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _summarize(candidate: str | None, runs: list[RolloutResult],
               baseline_totals: list[float]) -> ActionEstimate:
    totals = [run.objective.total for run in runs]
    gains = [base - total for base, total in zip(baseline_totals, totals)]
    return ActionEstimate(
        version_id=candidate,
        expected_total=fmean(totals),
        standard_error=_se(totals),
        expected_error_loss=fmean(run.objective.error_loss for run in runs),
        expected_blocked_loss=fmean(run.objective.blocked_loss for run in runs),
        expected_verification_cost=fmean(run.objective.verification_cost for run in runs),
        expected_recovery_cost=fmean(run.objective.recovery_cost for run in runs),
        probability_any_blocked=fmean(bool(run.objective.blocked_targets) for run in runs),
        fail_probability=fmean(run.observation == "fail" for run in runs),
        marginal_reduction=fmean(gains),
        paired_gain_standard_error=_se(gains),
    )


class MonteCarloCostEstimator:
    """Implements policy.ExpectedCostEstimator with stable version-ID actions.

    estimate_many shares one conditional posterior sample bank and event-keyed
    random streams across the baseline and all candidates. estimate() uses the
    same deterministic seed schedule on each call for interface compatibility.
    No mutable state cache is retained.
    """

    def __init__(
        self, config: ObjectiveConfig, *, rollouts: int = 1000,
        posterior_method: str = "auto", posterior_samples: int = 5000,
        max_exact_events: int = 14, min_effective_samples: float = 20,
        seed: int = 0,
    ) -> None:
        if type(rollouts) is not int or rollouts < 2:
            raise ValueError("rollouts must be an integer >= 2 for uncertainty estimation")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.config = config
        self.rollouts = rollouts
        self.posterior_method = posterior_method
        self.posterior_samples = posterior_samples
        self.max_exact_events = max_exact_events
        self.min_effective_samples = min_effective_samples
        self.seed = seed

    def _posterior(self, state: DAVState, seed: int) -> JointPosterior:
        return sample_joint_posterior(
            state.graph, state.observations, method=self.posterior_method,
            samples=self.posterior_samples, seed=seed,
            max_exact_events=self.max_exact_events,
            min_effective_samples=self.min_effective_samples,
        )

    def estimate_many(self, state: DAVState,
                      candidate_version_ids: Iterable[str]) -> CostComparison:
        candidates = tuple(dict.fromkeys(candidate_version_ids))
        for v in candidates:
            validate_candidate(state, v)
        # Detect malformed task configuration before expensive sampling.
        for target in self.config.targets:
            state.current_version_id(target.logical_node_id)
        rng = random.Random(self.seed)
        posterior = self._posterior(state, rng.getrandbits(64))
        worlds = posterior.draw(self.rollouts, seed=rng.getrandbits(64))
        noises = tuple(ScenarioNoise(rng.getrandbits(64)) for _ in worlds)
        baseline_runs = [
            _rollout(state, self.config, None, truth, noise)
            for truth, noise in zip(worlds, noises)
        ]
        baseline_totals = [run.objective.total for run in baseline_runs]
        estimates = {}
        for v in candidates:
            runs = [_rollout(state, self.config, v, truth, noise)
                    for truth, noise in zip(worlds, noises)]
            estimates[v] = _summarize(v, runs, baseline_totals)
        return CostComparison(
            baseline=_summarize(None, baseline_runs, baseline_totals),
            candidates=estimates,
            rollouts=self.rollouts,
            posterior_method=posterior.method,
            posterior_effective_samples=posterior.effective_sample_size,
            log_evidence_probability=posterior.log_evidence_probability,
        )

    def estimate(self, state: DAVState, candidate_node: str | None) -> float:
        """Return Q(S,a) for the existing policy protocol; action is a version ID."""
        if candidate_node is None:
            return self.estimate_many(state, ()).baseline.expected_total
        return self.estimate_many(state, (candidate_node,)).candidates[candidate_node].expected_total
