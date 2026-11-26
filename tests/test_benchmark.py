"""Regression tests for the expanded synthetic experimental protocol."""
import json
from dataclasses import replace

import pytest

from dav.benchmark import _overrides, run_benchmark, summarize_benchmark
from dav.budget_manager import BudgetManager
from dav.cost_estimator import MonteCarloCostEstimator
from dav.experiment_policies import (
    BENCHMARK_POLICIES, IndependentWorldEstimator, make_benchmark_policy,
)
from dav.graph import DependencyGraph
from dav.latent_state import ScenarioNoise
from dav.objective import ObjectiveConfig, TaskTarget
from dav.pilot import make_pilot_state, run_pilot
from dav.scenarios import ScenarioConfig, make_scenario, random_parents
from dav.state import DAVState
from dav.transition import execute_verification


def policy(name, config):
    return make_benchmark_policy(name, config, seed=5, rollouts=8, posterior_samples=5000)


def independent(gains=(8, 12), costs=(1, 2), p=1):
    graph = DependencyGraph()
    for name, cost in zip(("A", "B"), costs):
        graph.add_logical_node(name)
        graph.add_version(name, verify_cost=cost, intrinsic_error_prob=p,
                          repair_residual_error_prob=0)
    return DAVState(graph, BudgetManager(4, 4)), ObjectiveConfig(
        tuple(TaskTarget(n, g, 2 * g) for n, g in zip(("A", "B"), gains)), 0, 0)


@pytest.mark.parametrize("topology", ["chain", "diamond", "layered"])
def test_legacy_scenario_matches_original_pilot(topology):
    old, old_obj = make_pilot_state(topology, 3, 4)
    new, new_obj = make_scenario(topology, ScenarioConfig())
    assert new_obj == old_obj
    assert new.checkpoint() == old.checkpoint()
    assert new.graph.versions == old.graph.versions
    assert new.graph.logical_nodes == old.graph.logical_nodes
    assert new.graph.executions == old.graph.executions
    assert list(new.graph.graph.edges(data=True)) == list(old.graph.graph.edges(data=True))


def test_legacy_benchmark_reproduces_pilot_episode_metrics():
    kwargs = dict(topologies=("diamond",), policies=("no_verification", "random", "risk", "amcr"),
                  episodes=2, rollouts=8, seed=43)
    original = run_pilot(**kwargs)
    rows, _ = run_benchmark(scenarios={"legacy": ScenarioConfig()}, **kwargs)
    for a, b in zip(original, rows):
        for key in a:
            if key != "seconds":
                assert a[key] == b[key], key


def test_absolute_gain_versus_density_ablation():
    state, config = independent()
    assert policy("amcr", config).decide(state).version_id == "A@v1"
    assert policy("amcr_absolute_gain", config).decide(state).version_id == "B@v1"


def test_no_stop_ablation_retains_ranking_and_budget_guard():
    state, config = independent(p=0)
    config = ObjectiveConfig(config.targets, 1, 1)
    standard = policy("amcr", config).decide(state)
    forced = policy("amcr_no_stop", config).decide(state)
    assert standard.version_id is None
    assert forced.ranking == standard.ranking
    assert forced.version_id == forced.ranking[0].version_id
    assert forced.ranking[0].estimate.marginal_reduction < 0
    state.budgets.spend_verification(4)
    assert policy("amcr_no_stop", config).decide(state).version_id is None


@pytest.mark.parametrize("name", ["terminal_once", "terminal_only", "terminal_amcr"])
def test_terminal_controls_restrict_to_required_outputs(name):
    state, config = independent(gains=(100, 1))
    config = ObjectiveConfig((config.targets[1],), 0, 0)
    decision = policy(name, config).decide(state)
    assert decision.version_id == "B@v1"


def test_terminal_once_excludes_replacement_but_terminal_only_can_recheck():
    state, config = independent()
    config = ObjectiveConfig((config.targets[1],), 0, 0)
    state.graph.add_version("B", intrinsic_error_prob=1)
    assert policy("terminal_once", config).decide(state).version_id is None
    assert policy("terminal_only", config).decide(state).version_id == "B@v2"


def test_risk_matched_gate_uses_same_amcr_eligible_set():
    state, config = independent(gains=(1, 20), costs=(1, 1))
    regular = policy("amcr", config).decide(state)
    risk = policy("risk_positive_stop", config).decide(state)
    assert regular.version_id == "B@v1"
    assert risk.version_id == "A@v1"  # Equal risk, deterministic ID tie.
    assert risk.ranking == regular.ranking
    assert risk.comparison == regular.comparison


def test_target_weighted_impact_counts_only_required_outputs():
    state, config = independent(gains=(1, 20), costs=(1, 1))
    assert policy("target_risk_impact", config).decide(state).version_id == "B@v1"


def test_independent_world_ablation_preserves_marginals_but_breaks_dependence():
    cfg = ScenarioConfig(intrinsic_error=0, propagation=1)
    state, objective = make_scenario("diamond", cfg)
    state.graph.versions["D0@v1"].intrinsic_error_prob = 0.5
    joint = MonteCarloCostEstimator(objective)._posterior(state, 4)
    independent = IndependentWorldEstimator(objective)._posterior(state, 4)
    assert all(p == pytest.approx(0.5) for p in independent.risks.values())
    assert all(len(set(w.values())) == 1 for w in joint.draw(50, seed=7))
    assert any(len(set(w.values())) > 1 for w in independent.draw(50, seed=7))
    assert not state.observations


@pytest.mark.parametrize("mode,expected_error,expected_operation,expected_cost", [
    ("acquire", False, "repair", 1),
    ("derived", True, "recompute", 3),
])
def test_terminal_recovery_mode_changes_lineage_and_residual_truth(
    mode, expected_error, expected_operation, expected_cost
):
    cfg = ScenarioConfig(recovery_mode=mode, intrinsic_error=1, propagation=1,
                         sensitivity=1, specificity=1, repair_success=1, recompute_success=1,
                         residual_error=0, replay_intrinsic=0, recompute_cost=3,
                         verification_budget=4, recovery_budget=20)
    state, _ = make_scenario("chain", cfg)
    truth = dict.fromkeys(state.graph.versions, True)
    outcome = execute_verification(state, "D5@v1", truth, ScenarioNoise(7))
    assert outcome.observation == "fail"
    assert outcome.truth["D5@v2"] is expected_error
    assert outcome.recovery.events[0].action == expected_operation
    assert outcome.recovery.recovery_spent == expected_cost
    assert list(state.graph.graph.predecessors("D5@v2")) == (
        ["D4@v1"] if mode == "derived" else [])
    assert state.graph.check_lineage() == []
    if mode == "derived":
        later = execute_verification(state, "D0@v1", outcome.truth, ScenarioNoise(7))
        assert later.recovery.replacements["D5@v2"] == "D5@v3"
        assert all(not later.truth[state.current_version_id(n)]
                   for n in state.graph.logical_nodes)
        assert state.graph.check_lineage() == []


@pytest.mark.parametrize("topology,n_targets", [("fanout", 3), ("multi_output", 2)])
def test_multi_target_topologies(topology, n_targets):
    state, config = make_scenario(topology, ScenarioConfig())
    assert len(config.targets) == n_targets
    assert state.graph.check_lineage() == []
    assert all(not state.graph.get_direct_consumers(state.current_version_id(t.logical_node_id))
               for t in config.targets)


def test_random_dag_is_seeded_and_all_nodes_reach_a_target():
    cfg = ScenarioConfig(random_nodes=15, random_targets=4, random_fan_in=3)
    a = random_parents(cfg, 12)
    assert a == random_parents(cfg, 12)
    assert a != random_parents(cfg, 13)
    assert all(len(ps) <= 3 and all(p < i for p in ps) for i, ps in enumerate(a))
    state, config = make_scenario("random_dag", cfg, graph_seed=12)
    targets = {state.current_version_id(t.logical_node_id) for t in config.targets}
    assert len(targets) == 4
    assert state.graph.check_lineage() == []
    for v in state.graph.versions:
        assert ({v} | set(state.graph.get_affected_nodes(v))) & targets


def test_benchmark_trace_and_paired_metrics_are_complete():
    cfg = ScenarioConfig(intrinsic_error=0.3, recovery_mode="derived",
                         verification_budget=3, recovery_budget=4)
    rows, traces = run_benchmark(
        scenarios={"derived": cfg}, topologies=("fanout",),
        policies=("no_verification", "terminal_only", "amcr", "amcr_no_stop"),
        episodes=2, rollouts=4, seed=9, trace=True)
    assert len(rows) == len(traces) == 8
    for row, trace in zip(rows, traces):
        assert row["steps"] == len(trace["steps"])
        assert row["steps"] == sum(row[k] for k in
            ("true_positive", "false_positive", "true_negative", "false_negative"))
        assert row["verification_cost"] <= cfg.verification_budget
        assert row["recovery_cost"] <= cfg.recovery_budget
        assert row["total_cost"] == pytest.approx(row["error_loss"] + row["blocked_loss"] +
            cfg.lambda_v * row["verification_cost"] + cfg.lambda_r * row["recovery_cost"])
        json.dumps(trace, allow_nan=False)
    for ep in range(2):
        group = [t for t in traces if t["episode"] == ep]
        assert all(t["initial_truth"] == group[0]["initial_truth"] for t in group)
        assert all(t["initial_edges"] == group[0]["initial_edges"] for t in group)
    assert len(summarize_benchmark(rows)) == 4
    assert len(BENCHMARK_POLICIES) == 16


def test_overrides_and_invalid_configuration():
    assert _overrides(["intrinsic_error=0.25", "recovery_mode=derived"]) == {
        "intrinsic_error": 0.25, "recovery_mode": "derived"}
    with pytest.raises(ValueError):
        _overrides(["unknown=2"])
    with pytest.raises(ValueError):
        replace(ScenarioConfig(), sensitivity=1.5)
    with pytest.raises(ValueError):
        replace(ScenarioConfig(), random_targets=12)
