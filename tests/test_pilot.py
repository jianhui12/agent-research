"""Tests for shared baseline feasibility and paired pilot experiment setup."""
import pytest

from dav.baselines import HEURISTICS, HeuristicPolicy
from dav.budget_manager import BudgetManager
from dav.graph import DependencyGraph
from dav.pilot import POLICIES, make_pilot_state, run_pilot, summarize
from dav.state import DAVState


def chain():
    graph = DependencyGraph()
    for node, p, cost in (("A", 0.1, 2), ("B", 0.8, 1), ("C", 0.0, 1)):
        graph.add_logical_node(node)
        graph.add_version(node, intrinsic_error_prob=p, verify_cost=cost)
    graph.record_execution("b", "compute", ["A@v1"], ["B@v1"],
                           edge_metadata={("A@v1", "B@v1"): {"propagation_prob": 0}})
    graph.record_execution("c", "compute", ["B@v1"], ["C@v1"],
                           edge_metadata={("B@v1", "C@v1"): {"propagation_prob": 0}})
    return DAVState(graph, BudgetManager(3, 3))


@pytest.mark.parametrize("name,expected", [
    ("risk", "B@v1"), ("uncertainty", "B@v1"), ("impact", "A@v1"),
    ("risk_impact", "B@v1"), ("cost_aware_risk_impact", "B@v1"),
])
def test_heuristic_scores(name, expected):
    assert HeuristicPolicy(name).decide(chain()).version_id == expected


@pytest.mark.parametrize("name", HEURISTICS)
def test_heuristics_share_budget_and_observation_feasibility(name):
    state = chain()
    state.record_verification("B@v1", "pass")
    state.budgets.spend_verification(1)
    assert HeuristicPolicy(name).decide(state).version_id == "C@v1"
    state.budgets.spend_verification(1)
    assert HeuristicPolicy(name).decide(state).reason == "no_feasible_candidate"


def test_random_policy_has_reproducible_independent_seed():
    a, b = HeuristicPolicy("random", seed=7), HeuristicPolicy("random", seed=7)
    state = chain()
    assert [a.decide(state) for _ in range(8)] == [b.decide(state) for _ in range(8)]


@pytest.mark.parametrize("topology,count", [("chain", 6), ("diamond", 4), ("layered", 7)])
def test_pilot_graphs_have_explicit_targets_and_lineage(topology, count):
    state, config = make_pilot_state(topology, 3, 4)
    assert len(state.graph.versions) == count
    assert state.graph.check_lineage() == []
    assert config.targets
    for target in config.targets:
        assert not state.graph.get_direct_consumers(state.current_version_id(target.logical_node_id))


def test_paired_pilot_produces_complete_rows_and_budget_safe_metrics():
    kwargs = dict(topologies=("diamond",), policies=("no_verification", "random", "amcr"),
                  episodes=2, rollouts=4, seed=10)
    rows = run_pilot(**kwargs)
    assert len(rows) == 6
    for episode in range(2):
        group = [r for r in rows if r["episode"] == episode]
        assert len({r["environment_seed"] for r in group}) == 1
        assert len({r["planner_seed"] for r in group}) == 1
    for row in rows:
        assert row["verification_cost"] <= row["verification_budget"]
        assert row["recovery_cost"] <= row["recovery_budget"]
        assert row["correct_output_rate"] <= row["completion_rate"]
        assert row["total_cost"] == pytest.approx(
            row["error_loss"] + row["blocked_loss"] + 0.25 *
            (row["verification_cost"] + row["recovery_cost"]))
    assert len(summarize(rows)) == 3
    assert len(POLICIES) == 8
    repeated = run_pilot(**kwargs)
    stable = lambda rs: [{k: v for k, v in r.items() if k != "seconds"} for r in rs]
    assert stable(rows) == stable(repeated)
