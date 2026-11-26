"""Experiment-only policies; legacy pilot policy names remain unchanged."""
from __future__ import annotations

import random

from .cost_estimator import MonteCarloCostEstimator
from .joint_sampler import sample_joint_posterior
from .objective import ObjectiveConfig
from .pilot import make_policy as make_legacy_policy, POLICIES as LEGACY_POLICIES
from .policy import AMCRPolicy, PolicyDecision, feasible_candidates


EXTRA_POLICIES = (
    "terminal_once", "terminal_only", "terminal_amcr",
    "amcr_no_stop", "amcr_absolute_gain", "risk_positive_stop",
    "target_risk_impact", "amcr_independent_worlds",
)
BENCHMARK_POLICIES = (*LEGACY_POLICIES, *EXTRA_POLICIES)


class TerminalPolicy:
    """Risk-first restricted to required outputs; once restricts to v1.

    All benchmark episodes start from a fresh v1 graph. No oracle input.
    """
    def __init__(self, config: ObjectiveConfig, *, once: bool, seed: int,
                 posterior_samples: int):
        self.targets = frozenset(t.logical_node_id for t in config.targets)
        self.once = once
        self.seed = seed
        self.samples = posterior_samples

    def decide(self, state):
        candidates = [
            v for v in feasible_candidates(state)
            if state.graph.versions[v].node_id in self.targets
            and (not self.once or state.graph.versions[v].version_index == 1)
        ]
        if not candidates:
            return PolicyDecision(None, "no_feasible_candidate")
        posterior = sample_joint_posterior(
            state.graph, state.observations, samples=self.samples, seed=self.seed)
        best = min(candidates, key=lambda v: (-posterior.marginal(v), v))
        return PolicyDecision(best, "verify")


class RiskPositiveStop:
    """Rank by posterior risk among the SAME positive-gain set as AMCR."""
    def __init__(self, estimator):
        self.amcr = AMCRPolicy(estimator)
        self.estimator = estimator

    def decide(self, state):
        d = self.amcr.decide(state)
        if d.version_id is None:
            return d
        # Reconstruct the exact same posterior bank seed used in estimate_many.
        posterior = self.estimator._posterior(
            state, random.Random(self.estimator.seed).getrandbits(64))
        eligible = [a for a in d.ranking if a.conservative_gain > self.amcr.min_gain]
        best = min(eligible, key=lambda a: (-posterior.marginal(a.version_id), a.version_id))
        return PolicyDecision(best.version_id, "verify", d.ranking, d.comparison)


class TargetRiskImpact:
    """Loss-weighted reachable required outputs, counted once; no recovery model."""
    def __init__(self, config, *, seed, posterior_samples):
        self.config = config
        self.seed = seed
        self.samples = posterior_samples

    def decide(self, state):
        candidates = feasible_candidates(state)
        if not candidates:
            return PolicyDecision(None, "no_feasible_candidate")
        posterior = sample_joint_posterior(
            state.graph, state.observations, samples=self.samples, seed=self.seed)
        targets = {state.current_version_id(t.logical_node_id): t.error_loss
                   for t in self.config.targets}
        def score(v):
            affected = set(state.graph.get_affected_nodes(v)) | {v}
            impact = sum(w for d, w in targets.items()
                         if d in affected and state.is_usable(d))
            return posterior.marginal(v) * impact / state.graph.versions[v].verify_cost
        return PolicyDecision(min(candidates, key=lambda v: (-score(v), v)), "verify")


class _IndependentWorlds:
    """Retain exact/MC marginal beliefs, intentionally discard joint correlation."""
    def __init__(self, posterior):
        self.method = posterior.method + "_independent_marginals"
        self.effective_sample_size = posterior.effective_sample_size
        self.log_evidence_probability = posterior.log_evidence_probability
        ids = sorted(posterior.particles[0].errors)
        self.risks = {v: posterior.marginal(v) for v in ids}

    def draw(self, count, *, seed=0):
        rng = random.Random(seed)
        return tuple({v: rng.random() < p for v, p in self.risks.items()}
                     for _ in range(count))


class IndependentWorldEstimator(MonteCarloCostEstimator):
    """Ablates joint rollout sampling, NOT the correlation-aware inference step.

    Independent assignments may violate deterministic DAG relationships.
    Only the planner is changed; the actual environment remains joint.
    """
    def _posterior(self, state, seed):
        return _IndependentWorlds(super()._posterior(state, seed))


def make_benchmark_policy(name, config, *, seed, rollouts, posterior_samples):
    kwargs = dict(seed=seed, rollouts=rollouts, posterior_samples=posterior_samples)
    if name in LEGACY_POLICIES:
        return make_legacy_policy(name, config, **kwargs)
    if name in ("terminal_once", "terminal_only"):
        return TerminalPolicy(config, once=name == "terminal_once",
                              seed=seed, posterior_samples=posterior_samples)
    if name == "target_risk_impact":
        return TargetRiskImpact(config, seed=seed, posterior_samples=posterior_samples)
    if name not in EXTRA_POLICIES:
        raise ValueError(f"Unknown benchmark policy: {name}")
    estimator_cls = (IndependentWorldEstimator if name == "amcr_independent_worlds"
                     else MonteCarloCostEstimator)
    estimator = estimator_cls(config, **kwargs)
    if name == "risk_positive_stop":
        return RiskPositiveStop(estimator)
    return AMCRPolicy(
        estimator,
        adaptive_stop=name != "amcr_no_stop",
        normalize_cost=name != "amcr_absolute_gain",
        target_ids=tuple(t.logical_node_id for t in config.targets)
                   if name == "terminal_amcr" else None,
    )
