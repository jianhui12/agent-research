"""Recovery transitions preserve old errors and sample new artifacts only."""
import pytest

from dav.budget_manager import BudgetManager
from dav.graph import DependencyGraph
from dav.latent_state import ScenarioNoise, advance_after_recovery, simulate_verification
from dav.recovery_engine import RecoveryEngine


def chain():
    graph = DependencyGraph()
    for node in ("A", "B", "C", "D"):
        graph.add_logical_node(node)
        graph.add_version(node, intrinsic_error_prob=0.0)
    graph.record_execution("b", "compute", ["A@v1"], ["B@v1"])
    graph.record_execution("d", "compute", ["B@v1", "C@v1"], ["D@v1"])
    return graph


def recover(graph, source, budget, noise):
    return RecoveryEngine(graph, BudgetManager(0, budget),
                          success_uniform=noise.recovery_uniform).recover(
        source, lambda old: None, lambda node, parents: None)


def test_transition_preserves_history_and_propagates_unchanged_parent_error():
    graph = chain()
    graph.versions["A@v1"].repair_residual_error_prob = 0
    truth = {"A@v1": True, "B@v1": True, "C@v1": True, "D@v1": True}
    original = dict(truth)
    noise = ScenarioNoise(42)
    outcome = recover(graph, "A@v1", 10, noise)
    assert outcome.success
    updated = advance_after_recovery(graph, truth, outcome, noise)
    assert truth == original
    assert all(updated[v] == error for v, error in original.items())
    assert updated["A@v2"] is False
    assert updated["B@v2"] is False
    assert updated["D@v2"] is True  # C@v1 was not repaired or resampled.
    assert len(updated) == 7


def test_failed_recovery_preserves_existing_world_without_new_errors():
    graph = chain()
    graph.versions["A@v1"].repair_success_prob = 0
    truth = dict.fromkeys(graph.versions, True)
    noise = ScenarioNoise(2)
    outcome = recover(graph, "A@v1", 5, noise)
    assert not outcome.success
    assert advance_after_recovery(graph, truth, outcome, noise) == truth


def test_partial_recovery_creates_only_successful_output_errors():
    graph = chain()
    graph.versions["A@v1"].repair_residual_error_prob = 1
    truth = dict.fromkeys(graph.versions, False)
    noise = ScenarioNoise(2)
    outcome = recover(graph, "A@v1", 1, noise)
    assert not outcome.success
    assert set(outcome.replacements.values()) == {"A@v2"}
    updated = advance_after_recovery(graph, truth, outcome, noise)
    assert updated["A@v2"] is True
    assert set(updated) == set(truth) | {"A@v2"}
    with pytest.raises(ValueError, match="exactly"):
        advance_after_recovery(graph, {"A@v1": False}, outcome, noise)


def test_keyed_noise_is_reproducible_and_independent_of_call_order():
    noise = ScenarioNoise(13)
    a = noise.uniform("verification", "A@v1")
    b = noise.recovery_uniform("repair", "A@v1")
    assert 0 <= a < 1 and 0 <= b < 1
    assert a != b
    assert ScenarioNoise(13).recovery_uniform("repair", "A@v1") == b
    assert noise.uniform("verification", "B@v1") != a
    assert noise.uniform("verification", "A@v1") == a


def test_verifier_simulation_distinguishes_false_positive_and_false_negative():
    graph = chain()
    a = graph.versions["A@v1"]
    a.verifier_specificity = 0
    a.verifier_sensitivity = 0
    noise = ScenarioNoise(1)
    assert simulate_verification(graph, "A@v1", {"A@v1": False}, noise) == "fail"
    assert simulate_verification(graph, "A@v1", {"A@v1": True}, noise) == "pass"
