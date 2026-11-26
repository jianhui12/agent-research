"""Shared synthetic verification/recovery transition for planning and execution.

This is an environment operation, not a policy. The latent error assignment
is only used to generate verifier labels and new artifact errors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .latent_state import ScenarioNoise, advance_after_recovery, simulate_verification
from .recovery_engine import RecoveryEngine, RecoveryOutcome
from .state import DAVState


def validate_candidate(state: DAVState, version_id: str) -> None:
    """Only usable, not-yet-observed versions with positive affordable cost."""
    if not state.is_usable(version_id):
        raise ValueError(f"Candidate is unusable: {version_id}")
    if version_id in state.observations:
        raise ValueError(f"Candidate already has verification evidence: {version_id}")
    cost = state.graph.versions[version_id].verify_cost
    if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
        raise ValueError(f"Candidate verification cost must be finite and positive: {version_id}")
    if cost > state.budgets.verification_remaining:
        raise ValueError(f"Candidate exceeds verification budget: {version_id}")


@dataclass(frozen=True)
class VerificationTransition:
    version_id: str
    observation: str
    recovery: RecoveryOutcome | None
    truth: dict[str, bool]


def execute_verification(
    state: DAVState, version_id: str, truth: Mapping[str, bool],
    noise: ScenarioNoise,
) -> VerificationTransition:
    """Mutate an owned state by verifying once, and attempting repair on FAIL.

    Returns a new truth mapping; never mutates the previous world. Both real
    synthetic episodes and hypothetical rollouts use this exact transition.
    The callbacks are placeholders: no external tool is executed here.
    """
    validate_candidate(state, version_id)
    if set(truth) != set(state.graph.versions) or any(type(v) is not bool for v in truth.values()):
        raise ValueError("Latent truth must cover all existing versions with boolean errors")
    observation = simulate_verification(state.graph, version_id, truth, noise)
    state.record_verification(version_id, observation)
    recovery = None
    updated_truth = dict(truth)
    if observation == "fail":
        recovery = RecoveryEngine(
            state.graph, state.budgets, success_uniform=noise.recovery_uniform,
        ).recover(version_id, repair=lambda old: None,
                  recompute=lambda node_id, parents: None)
        updated_truth = advance_after_recovery(state.graph, truth, recovery, noise)
    return VerificationTransition(version_id, observation, recovery, updated_truth)
