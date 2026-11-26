"""AMCR ranking interface; use MonteCarloCostEstimator for one-step values.

Candidate IDs passed to that estimator are exact artifact version IDs.
This legacy ranking helper does not filter budget or choose a stop action.
For a shared sample bank and diagnostics, use estimate_many directly.
"""
from dataclasses import dataclass
from typing import Protocol


class ExpectedCostEstimator(Protocol):
    def estimate(self, state, candidate_node: str | None) -> float:
        """Expected future objective under the candidate action."""
        ...


@dataclass
class CandidateScore:
    node_id: str
    marginal_reduction: float
    score_per_verification_cost: float


def rank_candidates(state, candidate_nodes: list[str],
                    estimator: ExpectedCostEstimator,
                    verification_costs: dict[str, float]) -> list[CandidateScore]:
    baseline = estimator.estimate(state, candidate_node=None)
    scores = []
    for node_id in candidate_nodes:
        cost = verification_costs[node_id]
        if cost <= 0:
            raise ValueError("Verification costs must be positive")
        with_verification = estimator.estimate(state, candidate_node=node_id)
        gain = baseline - with_verification
        scores.append(CandidateScore(node_id, gain, gain / cost))
    return sorted(scores, key=lambda item: item.score_per_verification_cost,
                  reverse=True)
