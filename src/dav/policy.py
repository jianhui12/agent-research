"""One-step adaptive marginal-cost-reduction (AMCR) verification policy."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .cost_estimator import ActionEstimate, CostComparison, validate_candidate
from .state import DAVState


class ExpectedCostEstimator(Protocol):
    def estimate(self, state, candidate_node: str | None) -> float:
        ...


@dataclass
class CandidateScore:
    node_id: str
    marginal_reduction: float
    score_per_verification_cost: float


def rank_candidates(state, candidate_nodes: list[str],
                    estimator: ExpectedCostEstimator,
                    verification_costs: dict[str, float]) -> list[CandidateScore]:
    """Legacy scalar ranking helper, retained for compatibility."""
    baseline = estimator.estimate(state, candidate_node=None)
    scores = []
    for node_id in candidate_nodes:
        cost = verification_costs[node_id]
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("Verification costs must be finite and positive")
        with_verification = estimator.estimate(state, candidate_node=node_id)
        gain = baseline - with_verification
        scores.append(CandidateScore(node_id, gain, gain / cost))
    return sorted(scores, key=lambda item: item.score_per_verification_cost,
                  reverse=True)


class PairedEstimator(Protocol):
    def estimate_many(self, state: DAVState,
                      candidate_version_ids: list[str]) -> CostComparison:
        ...


@dataclass(frozen=True)
class RankedAction:
    version_id: str
    verify_cost: float
    estimate: ActionEstimate
    score: float
    conservative_gain: float


@dataclass(frozen=True)
class PolicyDecision:
    version_id: str | None
    reason: str
    ranking: tuple[RankedAction, ...] = ()
    comparison: CostComparison | None = None


class VerificationPolicy(Protocol):
    def decide(self, state: DAVState) -> PolicyDecision:
        ...


def feasible_candidates(state: DAVState) -> list[str]:
    """All usable current, unobserved, affordable positive-cost versions.

    Recovery affordability is NOT a hard candidate filter: the recovery
    outcome and possible blocking are already priced into the estimator.
    """
    observations = state.observations
    remaining = state.checkpoint().verification_remaining
    candidates = []
    for node in state.graph.logical_nodes.values():
        v = node.current_version_id
        if v is None or v in observations or not state.is_usable(v):
            continue
        cost = state.graph.versions[v].verify_cost
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            raise ValueError(f"Candidate verification cost must be finite and positive: {v}")
        if cost <= remaining:
            validate_candidate(state, v)
            candidates.append(v)
    return sorted(candidates)


class AMCRPolicy:
    """Choose the highest mean gain per verification cost among positive gains.

    confidence_z optionally gates actions by gain - z * paired SE; this is a
    heuristic uncertainty guard, not a simultaneous confidence guarantee.
    Scores still use mean gain. Defaults implement the original AMCR rule.
    """

    def __init__(self, estimator: PairedEstimator, *, min_gain: float = 0.0,
                 confidence_z: float = 0.0, adaptive_stop: bool = True,
                 normalize_cost: bool = True,
                 target_ids: tuple[str, ...] | None = None) -> None:
        for name, value in (("min_gain", min_gain), ("confidence_z", confidence_z)):
            if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        self.estimator = estimator
        self.min_gain = min_gain
        self.confidence_z = confidence_z
        self.adaptive_stop = adaptive_stop
        self.normalize_cost = normalize_cost
        self.target_ids = None if target_ids is None else frozenset(target_ids)

    def decide(self, state: DAVState) -> PolicyDecision:
        candidates = feasible_candidates(state)
        if self.target_ids is not None:
            candidates = [v for v in candidates if state.graph.versions[v].node_id in self.target_ids]
        if not candidates:
            return PolicyDecision(None, "no_feasible_candidate")
        comparison = self.estimator.estimate_many(state, candidates)
        if set(comparison.candidates) != set(candidates):
            raise ValueError("Estimator did not return exactly the feasible candidates")
        ranking = []
        for version_id in candidates:
            estimate = comparison.candidates[version_id]
            gain, se = estimate.marginal_reduction, estimate.paired_gain_standard_error
            if (estimate.version_id != version_id or not math.isfinite(gain)
                    or not math.isfinite(se) or se < 0):
                raise ValueError(f"Invalid action estimate: {version_id}")
            cost = state.graph.versions[version_id].verify_cost
            ranking.append(RankedAction(version_id, cost, estimate,
                                        gain / cost if self.normalize_cost else gain,
                                        gain - self.confidence_z * se))
        ranking.sort(key=lambda a: (-a.score, a.version_id))
        eligible = ([a for a in ranking if a.conservative_gain > self.min_gain]
                    if self.adaptive_stop else ranking)
        if not eligible:
            return PolicyDecision(None, "no_positive_gain", tuple(ranking), comparison)
        return PolicyDecision(eligible[0].version_id, "verify", tuple(ranking), comparison)


class StopPolicy:
    """No-verification control; immediately evaluates existing task outputs."""

    def decide(self, state: DAVState) -> PolicyDecision:
        return PolicyDecision(None, "stop_policy")
