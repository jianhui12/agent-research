"""Latent-world transitions for synthetic DAV rollouts (not observable state).

Existing artifact truth is immutable inside a world. A completed repair is
a fresh acquisition; a completed replay samples a new intrinsic error and
edge transmissions using the *actual* new input-version truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from .error_model import _prob
from .graph import DependencyGraph
from .recovery_engine import RecoveryOutcome


@dataclass(frozen=True)
class ScenarioNoise:
    """Independent pseudorandom event keys, shared across counterfactuals.

    Keys are separated by purpose; no dependence on the order of calls.
    Use a new scenario seed per independent rollout.
    """
    seed: int

    def uniform(self, *key: str) -> float:
        payload = json.dumps([self.seed, *key], ensure_ascii=True,
                             separators=(",", ":")).encode("ascii")
        digest = hashlib.blake2b(payload, digest_size=8,
                                 person=b"DAV-noise-v1").digest()
        # Use 53 bits so rounding cannot yield 1.0.
        return (int.from_bytes(digest, "big") >> 11) / (1 << 53)

    def recovery_uniform(self, action: str, version_id: str) -> float:
        return self.uniform("operation-success", action, version_id)


def simulate_verification(graph: DependencyGraph, version_id: str,
                          truth: Mapping[str, bool], noise: ScenarioNoise) -> str:
    """Sample a verifier label with sensitivity/specificity; charge elsewhere."""
    artifact = graph.versions[version_id]
    sensitivity = _prob(artifact.verifier_sensitivity, f"{version_id}.sensitivity")
    specificity = _prob(artifact.verifier_specificity, f"{version_id}.specificity")
    error = truth[version_id]
    if type(error) is not bool:
        raise ValueError("Latent error must be boolean")
    fail_probability = sensitivity if error else 1 - specificity
    return "fail" if noise.uniform("verification", version_id) < fail_probability else "pass"


def advance_after_recovery(
    graph: DependencyGraph, previous_truth: Mapping[str, bool],
    outcome: RecoveryOutcome, noise: ScenarioNoise,
) -> dict[str, bool]:
    """Extend a complete pre-action world with *only* successful new outputs.

    Failed operations produce no new latent variables; blocked old outputs
    retain their original error labels for historical evidence/audit.
    """
    truth = dict(previous_truth)
    new_ids = set(outcome.replacements.values())
    if (new_ids & truth.keys()
            or set(graph.versions) != set(truth) | new_ids
            or len(new_ids) != len(outcome.replacements)
            or not set(outcome.replacements).issubset(truth)):
        raise ValueError("Recovery outcome must identify exactly the newly created versions")
    if any(type(value) is not bool for value in truth.values()):
        raise ValueError("Latent errors must be boolean")
    for version_id in graph.topological_sort(new_ids):
        artifact = graph.versions[version_id]
        execution = graph.executions.get(artifact.execution_id)
        if execution is None or execution.operation not in ("repair", "recompute"):
            raise ValueError(f"Missing recovery execution for {version_id}")
        intrinsic = _prob(artifact.intrinsic_error_prob, f"{version_id}.intrinsic")
        error = noise.uniform("new-intrinsic", execution.operation, version_id) < intrinsic
        for parent in graph.graph.predecessors(version_id):
            propagation = _prob(graph.graph.edges[parent, version_id]["propagation_prob"],
                                f"{parent}->{version_id}")
            transmitted = noise.uniform("new-transmission", parent, version_id) < propagation
            if truth[parent] and transmitted:
                error = True
        truth[version_id] = error
    return truth
