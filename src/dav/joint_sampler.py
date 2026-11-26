"""Joint posterior over immutable artifact errors under noisy-OR semantics.

All existing versions (including history) have fixed latent errors within
one sampled world. Only verifier evidence, not operational status, is used
as a likelihood. Exact enumeration integrates out primitive edge events.
MC is prior importance sampling with log-space likelihood weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from types import MappingProxyType
from typing import Mapping

from .error_model import _prob
from .graph import DependencyGraph


class PosteriorSamplingError(RuntimeError):
    """Impossible exact evidence, or insufficient weighted MC support."""


@dataclass(frozen=True)
class JointParticle:
    errors: Mapping[str, bool]
    weight: float


@dataclass(frozen=True)
class JointPosterior:
    particles: tuple[JointParticle, ...]
    method: str
    log_evidence_probability: float
    effective_sample_size: float | None
    proposal_count: int | None

    def marginal(self, version_id: str) -> float:
        return math.fsum(p.weight for p in self.particles if p.errors[version_id])

    def draw(self, count: int, *, seed: int = 0) -> tuple[dict[str, bool], ...]:
        """Resample entire joint assignments, never independent marginals."""
        _positive_integer(count, "count")
        rng = random.Random(seed)
        chosen = rng.choices(self.particles, weights=[p.weight for p in self.particles],
                             k=count)
        return tuple(dict(p.errors) for p in chosen)


@dataclass(frozen=True)
class _Factor:
    version_id: str
    intrinsic: float
    parents: tuple[tuple[str, float], ...]
    # likelihood of the observed verifier label, given correct/error
    likelihood_correct: float
    likelihood_error: float

    def probability(self, errors: Mapping[str, bool]) -> float:
        # Accumulate noisy-OR probability directly to avoid cancellation
        # of very small intrinsic probabilities in 1 - (1 - p).
        probability = self.intrinsic
        for parent, transmission in self.parents:
            if errors[parent]:
                probability += (1 - probability) * transmission
        return min(1.0, max(0.0, probability))

    def log_likelihood(self, error: bool) -> float:
        likelihood = self.likelihood_error if error else self.likelihood_correct
        return math.log(likelihood) if likelihood > 0 else -math.inf


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _factors(graph: DependencyGraph, observations: Mapping[str, str]):
    for v, observation in observations.items():
        if v not in graph.versions:
            raise KeyError(f"Unknown observation version: {v}")
        if not isinstance(observation, str) or observation.lower() not in ("pass", "fail"):
            raise ValueError("Verification observation must be 'pass' or 'fail'")
    factors = []
    uncertain = 0
    for v in graph.topological_sort():
        artifact = graph.versions[v]
        intrinsic = _prob(artifact.intrinsic_error_prob, f"{v}.intrinsic_error_prob")
        parents = tuple(
            (u, _prob(graph.graph.edges[u, v]["propagation_prob"], f"{u}->{v}"))
            for u in graph.graph.predecessors(v)
        )
        uncertain += int(0 < intrinsic < 1) + sum(0 < p < 1 for _, p in parents)
        correct = error = 1.0
        if v in observations:
            sensitivity = _prob(artifact.verifier_sensitivity, f"{v}.sensitivity")
            specificity = _prob(artifact.verifier_specificity, f"{v}.specificity")
            if observations[v].lower() == "fail":
                correct, error = 1 - specificity, sensitivity
            else:
                correct, error = specificity, 1 - sensitivity
        factors.append(_Factor(v, intrinsic, parents, correct, error))
    return factors, uncertain


def _exact(factors):
    # Iterative enumeration avoids recursion depth failures on long DAGs.
    # Conditional noisy-OR probabilities integrate independent edge primitives.
    stack = [(0, {}, 0.0)]
    while stack:
        i, errors, log_mass = stack.pop()
        if i == len(factors):
            yield errors, log_mass
            continue
        factor = factors[i]
        p = factor.probability(errors)
        for error, mass in ((False, 1 - p), (True, p)):
            if mass <= 0:
                continue
            log_likelihood = factor.log_likelihood(error)
            if log_likelihood == -math.inf:
                continue
            assignment = dict(errors)
            assignment[factor.version_id] = error
            stack.append((i + 1, assignment, log_mass + math.log(mass) + log_likelihood))


def _mc(factors, count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        errors: dict[str, bool] = {}
        log_weight = 0.0
        for factor in factors:
            error = rng.random() < factor.probability(errors)
            errors[factor.version_id] = error
            log_weight += factor.log_likelihood(error)
            if log_weight == -math.inf:
                break  # This proposal contributes zero mass.
        if log_weight != -math.inf:
            yield errors, log_weight


def sample_joint_posterior(
    graph: DependencyGraph, observations: Mapping[str, str] | None = None, *,
    method: str = "auto", samples: int = 5000, seed: int = 0,
    max_exact_events: int = 14, min_effective_samples: float = 20.0,
) -> JointPosterior:
    """Build a normalized distribution/particle approximation to P(E | evidence).

    All registered version IDs are sampled, including superseded versions.
    Exact mode conservatively bounds enumeration by uncertain primitive count.
    MC ESS is computed from proposal weights, not from resampled duplicates.
    An insufficient ESS raises instead of returning a silently unreliable cloud.
    """
    if method not in ("auto", "exact", "mc"):
        raise ValueError("method must be 'auto', 'exact' or 'mc'")
    _positive_integer(samples, "samples")
    if type(max_exact_events) is not int or max_exact_events < 0:
        raise ValueError("max_exact_events must be a non-negative integer")
    if (not isinstance(min_effective_samples, (int, float))
            or not math.isfinite(min_effective_samples) or min_effective_samples < 1):
        raise ValueError("min_effective_samples must be finite and at least one")
    factors, uncertain = _factors(graph, observations or {})
    if method == "auto":
        method = "exact" if uncertain <= max_exact_events else "mc"
    if method == "exact" and uncertain > max_exact_events:
        raise ValueError(f"Exact posterior exceeds limit: {uncertain} uncertain events; use mc")
    assignments = list(_exact(factors) if method == "exact" else _mc(factors, samples, seed))
    if not assignments:
        if method == "exact":
            raise PosteriorSamplingError("Evidence has zero probability under the model")
        raise PosteriorSamplingError("MC found no nonzero-weight proposals; increase samples or use exact")
    peak = max(log_mass for _, log_mass in assignments)
    relative = [math.exp(log_mass - peak) for _, log_mass in assignments]
    total = math.fsum(relative)
    weights = [w / total for w in relative]
    ess = None
    log_evidence = peak + math.log(total)
    if method == "mc":
        ess = min(float(samples), 1.0 / math.fsum(w * w for w in weights))
        log_evidence -= math.log(samples)
        if ess + 1e-9 < min_effective_samples:
            raise PosteriorSamplingError(
                f"MC effective sample size {ess:.2f} below {min_effective_samples:g}; "
                "increase samples or use exact"
            )
    return JointPosterior(
        particles=tuple(JointParticle(MappingProxyType(errors), weight)
                        for (errors, _), weight in zip(assignments, weights) if weight > 0),
        method=method, log_evidence_probability=min(0.0, log_evidence),
        effective_sample_size=ess, proposal_count=samples if method == "mc" else None,
    )
