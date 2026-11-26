"""Recovery Engine v1 integration tests: topology, budgets, and failures."""
import random

import pytest

from dav.budget_manager import BudgetManager
from dav.graph import DependencyGraph
from dav.models import NodeStatus
from dav.recovery_engine import RecoveryEngine


def diamond():
    graph = DependencyGraph()
    for node_id in ("D1", "D2", "D3", "D4", "D5"):
        graph.add_logical_node(node_id)
        graph.add_version(node_id, value=node_id)
    graph.record_execution("d2", "compute", ["D1@v1"], ["D2@v1"])
    graph.record_execution("d3", "compute", ["D1@v1"], ["D3@v1"])
    graph.record_execution(
        "d4", "compute", ["D2@v1", "D3@v1", "D5@v1"], ["D4@v1"]
    )
    return graph


def compute(node_id, parents):
    return (node_id, tuple(p.version_id for p in parents))


def test_successful_repair_rebuilds_diamond_once_with_correct_lineage():
    graph = diamond()
    budgets = BudgetManager(verification_budget=2, recovery_budget=5)
    visited = []

    def recompute(node_id, parents):
        visited.append(node_id)
        return compute(node_id, parents)

    outcome = RecoveryEngine(graph, budgets, rng=random.Random(4)).recover(
        "D1@v1", lambda source: "fixed", recompute
    )
    assert outcome.success
    assert sorted(visited) == ["D2", "D3", "D4"]
    assert visited.index("D2") < visited.index("D4")
    assert visited.index("D3") < visited.index("D4")
    assert set(graph.graph.predecessors("D4@v2")) == {
        "D2@v2", "D3@v2", "D5@v1"
    }
    assert graph.versions["D1@v1"].status == NodeStatus.INVALID
    assert graph.get_current_version("D4").status == NodeStatus.UNVERIFIED
    assert graph.versions["D5@v1"].status == NodeStatus.UNVERIFIED
    assert outcome.recovery_spent == 4
    assert budgets.verification_remaining == 2
    assert budgets.recovery_remaining == 1
    assert graph.check_lineage() == []


def test_repair_failure_blocks_all_descendants_and_charges_once():
    graph = diamond()
    graph.versions["D1@v1"].repair_success_prob = 0
    budgets = BudgetManager(10, 10)
    outcome = RecoveryEngine(graph, budgets).recover(
        "D1@v1", lambda source: "fixed", compute
    )
    assert not outcome.success
    assert outcome.repaired_version_id is None
    assert outcome.blocked == {"D2@v1", "D3@v1", "D4@v1"}
    assert all(graph.versions[x].status == NodeStatus.BLOCKED for x in outcome.blocked)
    assert graph.versions["D1@v1"].status == NodeStatus.BLOCKED
    assert budgets.recovery_remaining == 9
    assert graph.check_lineage() == []


def test_exhausted_budget_blocks_unfinished_descendants():
    graph = diamond()
    budgets = BudgetManager(0, 2)
    outcome = RecoveryEngine(graph, budgets).recover(
        "D1@v1", lambda source: "fixed", compute
    )
    assert not outcome.success
    assert outcome.recovery_spent == 2
    assert budgets.recovery_remaining == 0
    assert graph.versions["D4@v1"].status == NodeStatus.BLOCKED
    assert len(outcome.replacements) == 2
    assert graph.check_lineage() == []


def test_recomputation_failure_blocks_shared_descendant():
    graph = diamond()
    graph.versions["D2@v1"].recompute_success_prob = 0
    outcome = RecoveryEngine(graph, BudgetManager(0, 10)).recover(
        "D1@v1", lambda source: "fixed", compute
    )
    assert not outcome.success
    assert "D2@v1" in outcome.blocked
    assert "D4@v1" in outcome.blocked
    assert "D4@v1" not in outcome.replacements
    assert graph.check_lineage() == []


def test_zero_budget_fails_closed_without_creating_versions():
    graph = diamond()
    outcome = RecoveryEngine(graph, BudgetManager(0, 0)).recover(
        "D1@v1", lambda source: "fixed", compute
    )
    assert outcome.recovery_spent == 0
    assert not outcome.replacements
    assert graph.get_current_version("D1").version_id == "D1@v1"
    assert graph.versions["D1@v1"].status == NodeStatus.BLOCKED


def test_noncurrent_version_cannot_be_repaired():
    graph = diamond()
    graph.add_version("D1", value="new")
    with pytest.raises(ValueError):
        RecoveryEngine(graph, BudgetManager(0, 3)).recover(
            "D1@v1", lambda source: "fixed", compute
        )


def test_callback_exception_is_safe_failure():
    graph = diamond()
    budgets = BudgetManager(0, 4)

    def bad_repair(source):
        raise RuntimeError("tool unavailable")

    outcome = RecoveryEngine(graph, budgets).recover(
        "D1@v1", bad_repair, compute
    )
    assert not outcome.success
    assert outcome.events[0].reason == "RuntimeError"
    assert outcome.recovery_spent == 1
    assert graph.check_lineage() == []


def test_recovery_preserves_config_but_initializes_fresh_risk():
    from dav.error_model import infer_error_risks

    graph = diamond()
    root = graph.versions["D1@v1"]
    root.intrinsic_error_prob = 0.6
    root.posterior_error_prob = 0.99
    root.repair_residual_error_prob = 0.12
    root.verifier_sensitivity = 0.85
    root.verifier_specificity = 0.9
    root.verify_cost = 3
    root.loss_weight = 10
    root.metadata = {"settings": ["tool"]}
    d2 = graph.versions["D2@v1"]
    d2.intrinsic_error_prob = 0.2
    d2.posterior_error_prob = 0.95
    graph.versions["D3@v1"].recompute_intrinsic_error_prob = 0.08

    outcome = RecoveryEngine(graph, BudgetManager(5, 10), rng=random.Random(2)).recover(
        "D1@v1", lambda old: "fixed", compute)
    assert outcome.success
    new = graph.get_current_version("D1")
    assert new.intrinsic_error_prob == pytest.approx(0.12)
    assert new.posterior_error_prob == pytest.approx(0.12)
    assert new.verify_cost == 3
    assert new.loss_weight == 10
    assert new.verifier_sensitivity == 0.85
    assert new.verifier_specificity == 0.9
    assert new.status == NodeStatus.UNVERIFIED
    assert new.metadata == root.metadata
    new.metadata["settings"].append("changed")
    assert root.metadata["settings"] == ["tool"]
    assert graph.get_current_version("D2").intrinsic_error_prob == pytest.approx(0.2)
    assert graph.get_current_version("D3").intrinsic_error_prob == pytest.approx(0.08)
    risks = infer_error_risks(graph).probabilities
    assert risks["D1@v2"] == pytest.approx(0.12)
    assert risks["D2@v2"] == pytest.approx(0.296)
    assert graph.check_lineage() == []


def test_default_repair_residual_risk_is_not_zero():
    graph = diamond()
    result = RecoveryEngine(graph, BudgetManager(0, 10)).recover(
        "D1@v1", lambda old: "fixed", compute)
    assert result.success
    assert graph.get_current_version("D1").intrinsic_error_prob == pytest.approx(0.05)


def test_historical_descendant_is_not_reactivated_by_recovery():
    graph = DependencyGraph()
    graph.add_logical_node("A")
    graph.add_logical_node("B")
    graph.add_version("A")
    graph.add_version("B")
    graph.record_execution("b-old", "compute", ["A@v1"], ["B@v1"])
    graph.add_version("B", value="independent newer B")
    graph.record_execution("b-new", "fresh", [], ["B@v2"])
    graph.supersede_version("B@v1", "B@v2")
    result = RecoveryEngine(graph, BudgetManager(0, 10)).recover(
        "A@v1", lambda old: "fixed", compute)
    assert result.success
    assert not result.affected
    assert graph.get_current_version("B").version_id == "B@v2"
    assert len(graph.logical_nodes["B"].version_ids) == 2
    assert result.recovery_spent == 1


def test_invalid_recovery_risk_fails_before_mutation_or_spending():
    graph = diamond()
    graph.versions["D2@v1"].recompute_intrinsic_error_prob = float("nan")
    budgets = BudgetManager(0, 10)
    with pytest.raises(ValueError, match="Probability"):
        RecoveryEngine(graph, budgets).recover("D1@v1", lambda old: "fixed", compute)
    assert budgets.recovery_remaining == 10
    assert graph.get_current_version("D1").version_id == "D1@v1"
    assert graph.versions["D1@v1"].status == NodeStatus.UNVERIFIED
