"""Task-output objective for DAV, with distinct error and noncompletion losses.

Terminal targets are stable logical node IDs, not historical artifact versions.
Unavailable targets incur noncompletion loss only (not both error and block).
Both expected and realized evaluation share exactly the same cost accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .state import DAVState, BudgetSnapshot


def _nonnegative(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class TaskTarget:
    logical_node_id: str
    error_loss: float
    blocked_loss: float

    def __post_init__(self) -> None:
        if not isinstance(self.logical_node_id, str) or not self.logical_node_id:
            raise ValueError("Target logical node ID must not be empty")
        _nonnegative(self.error_loss, "error_loss")
        _nonnegative(self.blocked_loss, "blocked_loss")


@dataclass(frozen=True)
class ObjectiveConfig:
    targets: tuple[TaskTarget, ...]
    lambda_v: float = 1.0
    lambda_r: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        if not self.targets:
            raise ValueError("At least one task target is required")
        ids = [target.logical_node_id for target in self.targets]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate logical task targets")
        _nonnegative(self.lambda_v, "lambda_v")
        _nonnegative(self.lambda_r, "lambda_r")


@dataclass(frozen=True)
class ObjectiveResult:
    error_loss: float
    blocked_loss: float
    verification_cost: float
    recovery_cost: float
    weighted_verification_cost: float
    weighted_recovery_cost: float
    blocked_targets: tuple[str, ...]

    @property
    def total(self) -> float:
        return (self.error_loss + self.blocked_loss
                + self.weighted_verification_cost + self.weighted_recovery_cost)


def _partition(state: DAVState, config: ObjectiveConfig
               ) -> tuple[list[tuple[TaskTarget, str]], list[TaskTarget]]:
    usable: list[tuple[TaskTarget, str]] = []
    blocked: list[TaskTarget] = []
    for target in config.targets:
        # A registered logical target with no output yet is incomplete.
        # Unknown logical IDs are configuration errors, not free omissions.
        version_id = state.current_version_id(target.logical_node_id)
        if version_id is None or not state.is_usable(version_id):
            blocked.append(target)
        else:
            usable.append((target, version_id))
    return usable, blocked


def _result(state: DAVState, config: ObjectiveConfig, error_loss: float,
            blocked: list[TaskTarget], checkpoint: BudgetSnapshot | None
            ) -> ObjectiveResult:
    costs = state.costs_since(checkpoint)
    return ObjectiveResult(
        error_loss=error_loss,
        blocked_loss=sum(target.blocked_loss for target in blocked),
        verification_cost=costs.verification,
        recovery_cost=costs.recovery,
        weighted_verification_cost=config.lambda_v * costs.verification,
        weighted_recovery_cost=config.lambda_r * costs.recovery,
        blocked_targets=tuple(target.logical_node_id for target in blocked),
    )


def evaluate_expected(
    state: DAVState, config: ObjectiveConfig, *,
    checkpoint: BudgetSnapshot | None = None,
    method: str = "exact", samples: int = 10000, seed: int | None = None,
    max_exact_events: int = 20,
) -> ObjectiveResult:
    """Expected terminal loss conditional on observed evidence plus paid costs.

    This evaluates a fixed state, not the counterfactual value of verification.
    checkpoint=None counts costs from state construction; a rollout checkpoint
    counts only subsequent spending.
    """
    usable, blocked = _partition(state, config)
    error_loss = 0.0
    if usable:
        estimate = state.infer_risks(
            [v for _, v in usable], method=method, samples=samples,
            seed=seed, max_exact_events=max_exact_events,
        )
        error_loss = sum(target.error_loss * estimate.probabilities[v]
                         for target, v in usable)
    return _result(state, config, error_loss, blocked, checkpoint)


def evaluate_realized(
    state: DAVState, config: ObjectiveConfig, truth: Mapping[str, bool], *,
    checkpoint: BudgetSnapshot | None = None,
) -> ObjectiveResult:
    """Realized terminal loss using simulator-oracle truth, never verifier labels.

    Missing truth for any usable output is an error. Blocked outputs are charged
    their noncompletion penalty regardless of any old truth entry.
    """
    usable, blocked = _partition(state, config)
    error_loss = 0.0
    for target, version_id in usable:
        if version_id not in truth:
            raise KeyError(f"Missing ground truth for usable output: {version_id}")
        if not isinstance(truth[version_id], bool):
            raise ValueError(f"Ground truth must be boolean: {version_id}")
        if truth[version_id]:
            error_loss += target.error_loss
    return _result(state, config, error_loss, blocked, checkpoint)
