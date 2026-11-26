"""Budget-aware probabilistic recovery on a versioned provenance DAG.

V1: one repair and at most one recomputation attempt per affected artifact;
outcomes are oracle-observed in the simulator. Successful execution yields
UNVERIFIED (not VERIFIED) versions. Callbacks must be replayable/idempotent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Callable

from .budget_manager import BudgetManager
from .graph import DependencyGraph
from .models import ArtifactVersion, NodeStatus


RepairFunction = Callable[[ArtifactVersion], Any]
RecomputeFunction = Callable[[str, tuple[ArtifactVersion, ...]], Any]
UNUSABLE = {NodeStatus.INVALID, NodeStatus.STALE, NodeStatus.BLOCKED, NodeStatus.REPAIRING}


@dataclass(frozen=True)
class RecoveryEvent:
    action: str
    version_id: str
    outcome: str
    cost: float
    new_version_id: str | None = None
    reason: str = ""


@dataclass
class RecoveryOutcome:
    source_version_id: str
    repaired_version_id: str | None = None
    replacements: dict[str, str] = field(default_factory=dict)
    affected: set[str] = field(default_factory=set)
    blocked: set[str] = field(default_factory=set)
    events: list[RecoveryEvent] = field(default_factory=list)
    recovery_spent: float = 0.0

    @property
    def success(self) -> bool:
        return self.repaired_version_id is not None and not self.blocked


class RecoveryEngine:
    """Run an observable recovery episode under a hard recovery budget."""

    def __init__(
        self, graph: DependencyGraph, budgets: BudgetManager,
        *, rng: random.Random | None = None,
    ) -> None:
        self.graph = graph
        self.budgets = budgets
        self.rng = rng if rng is not None else random.Random()
        self._sequence = 0

    def recover(
        self, source_version_id: str, repair: RepairFunction,
        recompute: RecomputeFunction,
    ) -> RecoveryOutcome:
        graph = self.graph
        source = graph.versions[source_version_id]
        if graph.get_current_version(source.node_id).version_id != source_version_id:
            raise ValueError("Can only repair the current version")
        if source.status in UNUSABLE:
            raise ValueError("Cannot repair an unusable version")
        self._prob(source.repair_success_prob)
        affected = graph.get_affected_nodes(source_version_id)
        result = RecoveryOutcome(source_version_id, affected=affected)
        initial = self.budgets.recovery_remaining

        if source.repair_cost > self.budgets.recovery_remaining:
            source.status = NodeStatus.BLOCKED
            self._block(result, affected)
            self._event(result, "repair", source_version_id, "budget_exhausted", 0)
            return result

        self.budgets.spend_recovery(source.repair_cost)
        source.status = NodeStatus.REPAIRING
        try:
            value = repair(source)
            repaired = self.rng.random() < source.repair_success_prob
        except Exception as exc:
            repaired = False
            reason = type(exc).__name__
        else:
            reason = ""

        if not repaired:
            source.status = NodeStatus.BLOCKED
            self._block(result, affected)
            self._event(result, "repair", source_version_id, "failed",
                        source.repair_cost, reason=reason)
            result.recovery_spent = initial - self.budgets.recovery_remaining
            return result

        new = graph.add_version(source.node_id, value=value,
                                status=NodeStatus.UNVERIFIED, make_current=False)
        graph.record_execution(
            self._execution_id(), "repair", [], [new.version_id],
            execution_cost=source.repair_cost,
            metadata={"repaired_from_version": source_version_id},
        )
        graph.supersede_version(source_version_id, new.version_id)
        result.repaired_version_id = new.version_id
        result.replacements[source_version_id] = new.version_id
        self._event(result, "repair", source_version_id, "succeeded",
                    source.repair_cost, new_version_id=new.version_id)

        # Exact old-version lineage; shared descendants are processed once.
        for old_id in graph.topological_sort(affected):
            old = graph.versions[old_id]
            parents = tuple(graph.graph.predecessors(old_id))
            inputs = tuple(result.replacements.get(p, p) for p in parents)
            valid = all(
                graph.versions[p].status not in UNUSABLE
                and graph.get_current_version(graph.versions[p].node_id).version_id == p
                for p in inputs
            )
            if not valid:
                old.status = NodeStatus.BLOCKED
                result.blocked.add(old_id)
                self._event(result, "recompute", old_id, "blocked_input", 0)
                continue

            self._prob(old.recompute_success_prob)
            if old.recompute_cost > self.budgets.recovery_remaining:
                old.status = NodeStatus.BLOCKED
                result.blocked.add(old_id)
                self._event(result, "recompute", old_id, "budget_exhausted", 0)
                continue

            self.budgets.spend_recovery(old.recompute_cost)
            try:
                value = recompute(old.node_id, tuple(graph.versions[p] for p in inputs))
                succeeded = self.rng.random() < old.recompute_success_prob
            except Exception as exc:
                succeeded = False
                reason = type(exc).__name__
            else:
                reason = ""
            if not succeeded:
                old.status = NodeStatus.BLOCKED
                result.blocked.add(old_id)
                self._event(result, "recompute", old_id, "failed",
                            old.recompute_cost, reason=reason)
                continue

            new = graph.add_version(old.node_id, value=value,
                                    status=NodeStatus.UNVERIFIED, make_current=False)
            metadata = {
                (new_parent, new.version_id): {
                    "propagation_prob": graph.graph.edges[original, old_id].get(
                        "propagation_prob", 1.0),
                    "input_role": graph.graph.edges[original, old_id].get("input_role"),
                }
                for original, new_parent in zip(parents, inputs)
            }
            graph.record_execution(
                self._execution_id(), "recompute", inputs, [new.version_id],
                execution_cost=old.recompute_cost,
                metadata={"recomputed_from_version": old_id},
                edge_metadata=metadata,
            )
            graph.supersede_version(old_id, new.version_id)
            result.replacements[old_id] = new.version_id
            self._event(result, "recompute", old_id, "succeeded",
                        old.recompute_cost, new_version_id=new.version_id)

        result.recovery_spent = initial - self.budgets.recovery_remaining
        return result

    @staticmethod
    def _prob(value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("Success probability must be in [0, 1]")

    def _execution_id(self) -> str:
        while True:
            self._sequence += 1
            execution_id = f"recovery-{self._sequence}"
            if execution_id not in self.graph.executions:
                return execution_id

    @staticmethod
    def _event(
        outcome: RecoveryOutcome, action: str, version: str, status: str,
        cost: float, *, new_version_id: str | None = None, reason: str = "",
    ) -> None:
        outcome.events.append(
            RecoveryEvent(action, version, status, cost, new_version_id, reason)
        )

    def _block(self, outcome: RecoveryOutcome, affected: set[str]) -> None:
        for version_id in affected:
            self.graph.versions[version_id].status = NodeStatus.BLOCKED
            outcome.blocked.add(version_id)
