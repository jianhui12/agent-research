"""Observable state and historical verification evidence regression tests."""
import random

import pytest

from dav.budget_manager import BudgetExceeded, BudgetManager
from dav.graph import DependencyGraph
from dav.models import NodeStatus
from dav.recovery_engine import RecoveryEngine
from dav.state import DAVState


def branching_state(verification=5, recovery=6):
    graph = DependencyGraph()
    for v in ("A", "B", "C"):
        graph.add_logical_node(v)
        graph.add_version(v, value={"items": [v]}, metadata={"tags": [v]})
    graph.versions["A@v1"].intrinsic_error_prob = 0.3
    graph.record_execution("b", "compute", ["A@v1"], ["B@v1"])
    graph.record_execution("c", "compute", ["A@v1"], ["C@v1"])
    return DAVState(graph, BudgetManager(verification, recovery))


def test_fork_isolates_nested_graph_values_edges_budgets_and_evidence():
    state = branching_state()
    fork = state.fork()
    fork.graph.versions["A@v1"].value["items"].append("new")
    fork.graph.versions["A@v1"].metadata["tags"].append("new")
    fork.graph.graph.edges["A@v1", "B@v1"]["propagation_prob"] = 0.2
    fork.record_verification("A@v1", "pass")
    fork.budgets.spend_recovery(2)
    assert state.graph.versions["A@v1"].value == {"items": ["A"]}
    assert state.graph.versions["A@v1"].metadata == {"tags": ["A"]}
    assert state.graph.graph.edges["A@v1", "B@v1"]["propagation_prob"] == 1
    assert state.budgets.verification_remaining == 5
    assert state.budgets.recovery_remaining == 6
    assert state.observations == {}
    assert fork.graph.nodes is fork.graph.versions
    assert fork.observations == {"A@v1": "pass"}


def test_verification_records_evidence_without_claiming_truth_and_charges_once():
    state = branching_state()
    state.record_verification("A@v1", "FAIL")
    assert state.graph.versions["A@v1"].status == NodeStatus.UNVERIFIED
    assert state.observations == {"A@v1": "fail"}
    assert state.costs_since().verification == 1
    with pytest.raises(ValueError, match="already"):
        state.record_verification("A@v1", "pass")
    assert state.budgets.verification_remaining == 4
    leaked = state.observations
    leaked.clear()
    assert state.observations == {"A@v1": "fail"}


def test_budget_failure_does_not_add_observation():
    state = branching_state(verification=0)
    with pytest.raises(BudgetExceeded):
        state.record_verification("A@v1", "pass")
    assert state.observations == {}
    assert state.budgets.verification_remaining == 0


def test_checkpoint_counts_only_future_costs_in_fork():
    state = branching_state()
    state.record_verification("A@v1", "pass")
    checkpoint = state.checkpoint()
    fork = state.fork()
    fork.record_verification("B@v1", "pass")
    fork.budgets.spend_recovery(2)
    assert fork.costs_since().verification == 2
    assert fork.costs_since(checkpoint).verification == 1
    assert fork.costs_since(checkpoint).recovery == 2
    assert state.costs_since(checkpoint).verification == 0


def test_historical_evidence_informs_unchanged_ancestor_not_new_repair():
    state = branching_state()
    state.record_verification("B@v1", "fail")
    state.graph.versions["B@v1"].repair_residual_error_prob = 0.15
    result = RecoveryEngine(state.graph, state.budgets, rng=random.Random(1)).recover(
        "B@v1", lambda old: "fresh", lambda node, parents: "replayed")
    assert result.success
    assert state.observations == {"B@v1": "fail"}
    assert state.graph.versions["B@v1"].status == NodeStatus.INVALID
    risks = state.infer_risks(["B@v2", "C@v1"]).probabilities
    assert risks["A@v1"] == pytest.approx(1.0)
    assert risks["C@v1"] == pytest.approx(1.0)
    assert risks["B@v2"] == pytest.approx(0.15)
    assert state.graph.check_lineage() == []


def test_noncurrent_ancestor_makes_output_unusable():
    state = branching_state()
    state.graph.add_version("A")  # B still references A@v1
    assert not state.is_usable("B@v1")
    with pytest.raises(ValueError, match="unusable"):
        state.record_verification("B@v1", "pass")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_nonfinite_or_negative_budget_and_cost_rejected(bad):
    with pytest.raises(ValueError):
        BudgetManager(bad, 1)
    manager = BudgetManager(1, 1)
    with pytest.raises(ValueError):
        manager.spend_recovery(bad)
    assert manager.recovery_remaining == 1
