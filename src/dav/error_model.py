"""Correlated error propagation and evidence-conditioned risk on version DAGs.

Generative semantics:
  E[v] = Intrinsic[v] OR any(E[u] AND Transmission[u,v]).
Every intrinsic-error and transmission Bernoulli variable is independent,
but *derived node errors are not*: common ancestors induce correlation.

Exact inference enumerates primitive Bernoulli variables (exponential).
Monte Carlo uses likelihood weighting for imperfect verifier observations.
No graph beliefs/statuses are mutated by inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import random
from typing import Mapping

import networkx as nx

from .graph import DependencyGraph
from .models import NodeStatus


UNUSABLE = {NodeStatus.INVALID, NodeStatus.STALE, NodeStatus.BLOCKED,
            NodeStatus.REPAIRING}


@dataclass(frozen=True)
class RiskEstimate:
    probabilities: dict[str, float]
    expected_loss: float
    evidence_probability: float | None
    method: str
    samples: int | None = None


def _prob(value: float, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return float(value)


def _scope(graph: DependencyGraph, target_version_ids: list[str] | None,
           *, current_only: bool,
           evidence_version_ids: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    if target_version_ids is not None:
        targets = list(dict.fromkeys(target_version_ids))
    elif current_only:
        targets = [node.current_version_id for node in graph.logical_nodes.values()
                   if node.current_version_id is not None]
    else:
        targets = list(graph.versions)
    for v in targets:
        if v not in graph.versions:
            raise KeyError(f"Unknown artifact version: {v}")
    selected = set(targets)
    for v in targets:
        selected.update(nx.ancestors(graph.graph, v))
    bad = sorted(v for v in selected if graph.versions[v].status in UNUSABLE)
    if bad:
        raise ValueError("Cannot infer risk from unusable version(s): " + ", ".join(bad))
    # Historical observations describe past artifact errors, even when those
    # artifacts are no longer consumable. They may inform unchanged ancestors.
    # Only the explicitly selected target lineage must be operationally usable.
    for v in evidence_version_ids:
        if v not in graph.versions:
            raise KeyError(f"Unknown observation version: {v}")
        selected.add(v)
        selected.update(nx.ancestors(graph.graph, v))
    return graph.topological_sort(selected), targets


def _primitive_variables(graph: DependencyGraph, order: list[str]):
    variables: list[tuple[str, str, str | None, float]] = []
    for v in order:
        variables.append(("intrinsic", v, None,
                          _prob(graph.versions[v].intrinsic_error_prob, f"{v}.intrinsic_error_prob")))
        for u in graph.graph.predecessors(v):
            variables.append(("edge", u, v,
                              _prob(graph.graph.edges[u, v]["propagation_prob"],
                                    f"edge {u}->{v}.propagation_prob")))
    return variables


def _evaluate(order: list[str], variables, bits: tuple[bool, ...]) -> dict[str, bool]:
    inherent: dict[str, bool] = {}
    enabled: dict[tuple[str, str], bool] = {}
    for (kind, u, v, _), bit in zip(variables, bits):
        if kind == "intrinsic":
            inherent[u] = bit
        else:
            enabled[u, v] = bit
    errors = {}
    for v in order:
        errors[v] = inherent[v] or any(
            errors[u] and enabled[u, v] for (kind, u, target, _), flag in zip(variables, bits)
            if kind == "edge" and target == v
        )
    return errors


def _observations(graph: DependencyGraph, observations: Mapping[str, str] | None,
                  order: list[str]) -> dict[str, str]:
    result = {}
    for version_id, observation in (observations or {}).items():
        if version_id not in order:
            raise ValueError(f"Observation for version outside inference scope: {version_id}")
        if not isinstance(observation, str) or observation.lower() not in ("pass", "fail"):
            raise ValueError("Verifier observation must be 'pass' or 'fail'")
        result[version_id] = observation.lower()
        ver = graph.versions[version_id]
        _prob(ver.verifier_sensitivity, f"{version_id}.verifier_sensitivity")
        _prob(ver.verifier_specificity, f"{version_id}.verifier_specificity")
    return result


def _likelihood(graph: DependencyGraph, truth: dict[str, bool],
                observations: dict[str, str]) -> float:
    likelihood = 1.0
    for version_id, obs in observations.items():
        node = graph.versions[version_id]
        if obs == "fail":
            likelihood *= node.verifier_sensitivity if truth[version_id] else 1 - node.verifier_specificity
        else:
            likelihood *= 1 - node.verifier_sensitivity if truth[version_id] else node.verifier_specificity
    return likelihood


def _nonnegative_finite(value: float, label: str) -> float:
    """Validate a cost/loss weight, not a Bernoulli probability."""
    if (not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _loss(graph: DependencyGraph, probabilities: dict[str, float], targets: list[str]) -> float:
    return sum(_nonnegative_finite(graph.versions[v].loss_weight, f"{v}.loss_weight") *
               probabilities[v] for v in targets)


def propagate_error_beliefs(graph: DependencyGraph) -> dict[str, float]:
    """Legacy independence approximation; WRONG for correlated parents.

    Retained for explicit baseline comparison only. Does not interpret verifier
    observations or status. Use infer_error_risks for correlation-aware inference.
    """
    beliefs: dict[str, float] = {}
    for version_id in graph.topological_sort():
        version = graph.versions[version_id]
        no_error = 1.0 - version.intrinsic_error_prob
        for parent_id in graph.graph.predecessors(version_id):
            edge_prob = graph.graph.edges[parent_id, version_id]["propagation_prob"]
            no_error *= 1.0 - beliefs[parent_id] * edge_prob
        beliefs[version_id] = 1.0 - no_error
    return beliefs


def infer_error_risks(
    graph: DependencyGraph, *, method: str = "exact",
    target_version_ids: list[str] | None = None,
    observations: Mapping[str, str] | None = None,
    samples: int = 10000, seed: int | None = None,
    max_exact_events: int = 20, current_only: bool = True,
    include_evidence_history: bool = False,
) -> RiskEstimate:
    """Infer joint-model marginals conditional on noisy verifier evidence.

    target_version_ids limits the modeled ancestry and weighted loss terms; all
    ancestors are still included in returned probabilities. By default infer
    all current version outputs (and required ancestors) and weight current
    versions. Invalid/stale/blocked target lineage raises instead of counting as safe.
    Opt in to include_evidence_history to condition on historical observations
    without making those versions operationally usable or charging their loss.
    """
    evidence_ids = tuple(observations or {}) if include_evidence_history else ()
    order, targets = _scope(graph, target_version_ids, current_only=current_only,
                            evidence_version_ids=evidence_ids)
    variables = _primitive_variables(graph, order)
    obs = _observations(graph, observations, order)
    if method not in ("exact", "mc"):
        raise ValueError("method must be 'exact' or 'mc'")
    if method == "mc" and (not isinstance(samples, int) or samples <= 0):
        raise ValueError("samples must be a positive integer")
    if method == "exact" and (not isinstance(max_exact_events, int) or max_exact_events < 0):
        raise ValueError("max_exact_events must be a non-negative integer")

    # Remove constants: 0 and 1 probability Bernoulli variables need no branching.
    varying = [i for i, (_, _, _, p) in enumerate(variables) if 0 < p < 1]
    if method == "exact" and len(varying) > max_exact_events:
        raise ValueError(f"Exact inference needs {len(varying)} uncertain events, "
                         f"limit is {max_exact_events}; use method='mc'")

    sums = dict.fromkeys(order, 0.0)
    denominator = 0.0
    if method == "exact":
        # Enumerate primitive events, not derived (correlated) node errors.
        for assignment in product((False, True), repeat=len(varying)):
            bits = [p == 1 for _, _, _, p in variables]
            mass = 1.0
            for idx, bit in zip(varying, assignment):
                p = variables[idx][3]
                bits[idx] = bit
                mass *= p if bit else 1.0 - p
            truth = _evaluate(order, variables, tuple(bits))
            weight = mass * _likelihood(graph, truth, obs)
            denominator += weight
            for v in order:
                if truth[v]:
                    sums[v] += weight
    else:
        rng = random.Random(seed)
        for _ in range(samples):
            bits = tuple(p == 1 if p in (0, 1) else rng.random() < p
                         for _, _, _, p in variables)
            truth = _evaluate(order, variables, bits)
            weight = _likelihood(graph, truth, obs)
            denominator += weight
            for v in order:
                if truth[v]:
                    sums[v] += weight
        denominator /= samples
        for v in order:
            sums[v] /= samples
    if denominator <= 0:
        raise ValueError("Evidence has zero probability under the error model")
    beliefs = {v: sums[v] / denominator for v in order}
    return RiskEstimate(
        probabilities=beliefs,
        expected_loss=_loss(graph, beliefs, targets),
        evidence_probability=denominator if obs else 1.0,
        method=method,
        samples=samples if method == "mc" else None,
    )


def sample_ground_truth(
    graph: DependencyGraph, *, seed: int | None = None,
    target_version_ids: list[str] | None = None,
    current_only: bool = True,
) -> dict[str, bool]:
    """Draw one joint error assignment with shared ancestor effects."""
    order, _ = _scope(graph, target_version_ids, current_only=current_only)
    variables = _primitive_variables(graph, order)
    rng = random.Random(seed)
    bits = tuple(p == 1 if p in (0, 1) else rng.random() < p
                 for _, _, _, p in variables)
    return _evaluate(order, variables, bits)
