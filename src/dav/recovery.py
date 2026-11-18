"""Probabilistic repair/recompute skeleton; lineage filtering is TODO."""
from dataclasses import dataclass
import random
from .budget_manager import BudgetManager
from .graph import DependencyGraph
from .models import NodeStatus


@dataclass
class RecoveryResult:
    success: bool
    affected: set[str]


def repair_and_recompute(graph: DependencyGraph, node_id: str,
                         budgets: BudgetManager, rng: random.Random) -> RecoveryResult:
    node = graph.nodes[node_id]
    budgets.spend_recovery(node.repair_cost)
    node.status = NodeStatus.REPAIRING

    if rng.random() >= node.repair_success_prob:
        node.status = NodeStatus.BLOCKED
        return RecoveryResult(False, set())

    node.version += 1
    node.status = NodeStatus.VERIFIED
    affected = graph.invalidate_descendants(node_id)

    for child_id in graph.topological_order(affected):
        child = graph.nodes[child_id]
        if child.status == NodeStatus.BLOCKED:
            continue
        if any(graph.nodes[parent].status in {
            NodeStatus.INVALID, NodeStatus.STALE, NodeStatus.BLOCKED
        } for parent in graph.graph.predecessors(child_id)):
            child.status = NodeStatus.BLOCKED
            continue
        budgets.spend_recovery(child.recompute_cost)
        if rng.random() < child.recompute_success_prob:
            child.version += 1
            child.status = NodeStatus.VERIFIED
        else:
            child.status = NodeStatus.BLOCKED

    return RecoveryResult(True, affected)
