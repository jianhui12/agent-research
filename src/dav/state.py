"""Observable DAV state and isolated candidate-rollout snapshots.

Ground truth is deliberately not stored in DAVState. Historical evidence is
keyed by immutable artifact version IDs, never by a mutable logical node ID.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import networkx as nx

from .budget_manager import BudgetManager
from .error_model import RiskEstimate, infer_error_risks
from .graph import DependencyGraph
from .models import NodeStatus


UNUSABLE = {NodeStatus.INVALID, NodeStatus.STALE, NodeStatus.BLOCKED,
            NodeStatus.REPAIRING}


@dataclass(frozen=True)
class BudgetSnapshot:
    verification_remaining: float
    recovery_remaining: float

    def __post_init__(self) -> None:
        for value in (self.verification_remaining, self.recovery_remaining):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Budget snapshot must contain finite non-negative balances")


@dataclass(frozen=True)
class ActionCosts:
    verification: float
    recovery: float


class DAVState:
    """Observable state; incoming graph and budgets are owned by this state.

    Initial observations are pre-existing evidence, already accounted for
    in the supplied remaining budgets. Record new checks via record_verification.
    A fork preserves the accounting origin but owns independent mutable objects.
    """

    def __init__(self, graph: DependencyGraph, budgets: BudgetManager, *,
                 observations: Mapping[str, str] | None = None) -> None:
        self.graph = graph
        self.budgets = budgets
        self._observations: dict[str, str] = {}
        for version_id, observation in (observations or {}).items():
            if version_id not in graph.versions:
                raise KeyError(f"Unknown observation version: {version_id}")
            self._observations[version_id] = self._normalize(observation)
        self._origin = self.checkpoint()

    @property
    def observations(self) -> dict[str, str]:
        """Return a copy; evidence cannot be silently overwritten."""
        return dict(self._observations)

    def fork(self) -> DAVState:
        """Deep-copy graph (including values/metadata), budgets, and evidence.

        External resources and callbacks are not part of this observable state.
        Artifact payloads must be deepcopy-compatible.
        """
        return deepcopy(self)

    def checkpoint(self) -> BudgetSnapshot:
        return BudgetSnapshot(self.budgets.verification_remaining,
                              self.budgets.recovery_remaining)

    def costs_since(self, checkpoint: BudgetSnapshot | None = None) -> ActionCosts:
        origin = self._origin if checkpoint is None else checkpoint
        now = self.checkpoint()
        verification = origin.verification_remaining - now.verification_remaining
        recovery = origin.recovery_remaining - now.recovery_remaining
        if verification < 0 or recovery < 0:
            raise ValueError("Budget balance increased since accounting checkpoint")
        return ActionCosts(verification, recovery)

    def current_version_id(self, logical_node_id: str) -> str | None:
        return self.graph.logical_nodes[logical_node_id].current_version_id

    def is_usable(self, version_id: str) -> bool:
        """Check operational availability, not correctness or verifier PASS.

        A current output with any stale/noncurrent/blocked ancestor is unavailable.
        """
        if version_id not in self.graph.versions:
            raise KeyError(f"Unknown artifact version: {version_id}")
        lineage = nx.ancestors(self.graph.graph, version_id) | {version_id}
        return all(
            self.graph.versions[v].status not in UNUSABLE
            and self.current_version_id(self.graph.versions[v].node_id) == v
            for v in lineage
        )

    def record_verification(self, version_id: str, observation: str) -> None:
        """Charge exactly once and store one binary observation for this version.

        A FAIL is evidence, not an oracle label. This method does not invalidate
        outputs or imply repair; that is a separate policy action.
        """
        normalized = self._normalize(observation)
        if not self.is_usable(version_id):
            raise ValueError(f"Cannot verify unusable version: {version_id}")
        if version_id in self._observations:
            raise ValueError(f"Version already has verification evidence: {version_id}")
        version = self.graph.versions[version_id]
        for p in (version.verifier_sensitivity, version.verifier_specificity):
            if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Verifier parameters must be finite probabilities")
        self.budgets.spend_verification(version.verify_cost)
        self._observations[version_id] = normalized

    def infer_risks(self, target_version_ids: Iterable[str], *,
                    method: str = "exact", samples: int = 10000,
                    seed: int | None = None, max_exact_events: int = 20) -> RiskEstimate:
        targets = list(dict.fromkeys(target_version_ids))
        for version_id in targets:
            if not self.is_usable(version_id):
                raise ValueError(f"Cannot evaluate unusable target: {version_id}")
        # Evidence on old outputs can still inform unchanged ancestors.
        # New versions do not inherit observations from replaced versions.
        return infer_error_risks(
            self.graph, target_version_ids=targets, observations=self._observations,
            method=method, samples=samples, seed=seed,
            max_exact_events=max_exact_events, include_evidence_history=True,
        )

    @staticmethod
    def _normalize(observation: str) -> str:
        if not isinstance(observation, str) or observation.lower() not in ("pass", "fail"):
            raise ValueError("Verification observation must be 'pass' or 'fail'")
        return observation.lower()
