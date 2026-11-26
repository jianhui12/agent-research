"""Recovery planning utilities built on version-aware provenance.

Actual repair/recomputation execution is orchestrated by the recovery runner.
This module only derives a deterministic, deduplicated plan from graph lineage.
"""
from dataclasses import dataclass

from .graph import DependencyGraph
from .models import NodeStatus


@dataclass(frozen=True)
class RecoveryPlan:
    invalidated_source_version: str
    affected_version_ids: frozenset[str]
    recomputation_order: tuple[str, ...]


def build_recovery_plan(
    graph: DependencyGraph, old_version_id: str, new_version_id: str
) -> RecoveryPlan:
    """Supersede an old artifact and return affected versions in DAG order.

    The new version must already exist and should have been produced by a
    recorded repair execution. The old version remains in the graph for audit.
    """
    graph.mark_status(old_version_id, NodeStatus.INVALID)
    affected = graph.supersede_version(old_version_id, new_version_id)
    order = graph.topological_sort(affected)
    return RecoveryPlan(
        invalidated_source_version=old_version_id,
        affected_version_ids=frozenset(affected),
        recomputation_order=tuple(order),
    )
