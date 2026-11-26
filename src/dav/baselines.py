"""Simple verification policies for paired synthetic DAV experiments.

All baselines share the simulator's candidate feasibility and recovery
transition. Heuristic baselines spend until no feasible action remains;
StopPolicy is the no-verification control. They never see environment truth.
"""
from __future__ import annotations

import random

from .joint_sampler import sample_joint_posterior
from .policy import PolicyDecision, feasible_candidates
from .state import DAVState


HEURISTICS = ("random", "risk", "uncertainty", "impact", "risk_impact",
              "cost_aware_risk_impact")


class HeuristicPolicy:
    """Adaptive heuristic rankings on current observable state.

    Impact counts the candidate and its usable current descendants once.
    Risk is the correlation-aware marginal posterior error probability.
    Uncertainty uses p*(1-p), with the same maximizer as binary entropy.
    """

    def __init__(self, name: str, *, seed: int = 0,
                 posterior_method: str = "auto", posterior_samples: int = 5000,
                 max_exact_events: int = 14, min_effective_samples: float = 20):
        if name not in HEURISTICS:
            raise ValueError(f"Unknown heuristic: {name}")
        self.name = name
        self.rng = random.Random(seed)
        self.posterior_method = posterior_method
        self.posterior_samples = posterior_samples
        self.max_exact_events = max_exact_events
        self.min_effective_samples = min_effective_samples

    def decide(self, state: DAVState) -> PolicyDecision:
        candidates = feasible_candidates(state)
        if not candidates:
            return PolicyDecision(None, "no_feasible_candidate")
        if self.name == "random":
            return PolicyDecision(self.rng.choice(candidates), "verify")

        risks = {}
        if self.name != "impact":
            posterior = sample_joint_posterior(
                state.graph, state.observations, method=self.posterior_method,
                samples=self.posterior_samples, seed=self.rng.getrandbits(64),
                max_exact_events=self.max_exact_events,
                min_effective_samples=self.min_effective_samples,
            )
            risks = {v: posterior.marginal(v) for v in candidates}

        def score(v: str) -> float:
            if self.name == "risk":
                return risks[v]
            if self.name == "uncertainty":
                return risks[v] * (1 - risks[v])
            affected = state.graph.get_affected_nodes(v)
            impact = 1 + sum(state.is_usable(d) for d in affected)
            if self.name == "impact":
                return float(impact)
            value = risks[v] * impact
            if self.name == "cost_aware_risk_impact":
                value /= state.graph.versions[v].verify_cost
            return value

        best = min(candidates, key=lambda v: (-score(v), v))
        return PolicyDecision(best, "verify")
