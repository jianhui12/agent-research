"""Exact/Monte Carlo correlated-error propagation and evidence tests."""
import pytest

from dav.error_model import (
    infer_error_risks, propagate_error_beliefs, sample_ground_truth,
)
from dav.graph import DependencyGraph
from dav.models import NodeStatus


def diamond(root_probability=0.5):
    graph = DependencyGraph()
    for node_id in ("D1", "D2", "D3", "D4"):
        graph.add_logical_node(node_id)
        graph.add_version(node_id)
    graph.versions["D1@v1"].intrinsic_error_prob = root_probability
    graph.record_execution("e2", "compute", ["D1@v1"], ["D2@v1"])
    graph.record_execution("e3", "compute", ["D1@v1"], ["D3@v1"])
    graph.record_execution("e4", "compute", ["D2@v1", "D3@v1"], ["D4@v1"])
    return graph


def test_exact_diamond_preserves_shared_ancestor_correlation():
    graph = diamond()
    exact = infer_error_risks(graph)
    naive = propagate_error_beliefs(graph)
    assert exact.probabilities["D4@v1"] == pytest.approx(0.5)
    assert naive["D4@v1"] == pytest.approx(0.75)
    assert exact.expected_loss == pytest.approx(2.0)  # four current artifacts


def test_monte_carlo_matches_exact_with_fixed_seed():
    graph = diamond(0.37)
    exact = infer_error_risks(graph)
    mc = infer_error_risks(graph, method="mc", samples=30000, seed=123)
    for v in exact.probabilities:
        assert mc.probabilities[v] == pytest.approx(exact.probabilities[v], abs=0.02)
    assert mc.expected_loss == pytest.approx(exact.expected_loss, abs=0.08)


def test_evidence_on_descendant_updates_ancestor_risk():
    graph = diamond()
    graph.versions["D4@v1"].verifier_sensitivity = 1
    graph.versions["D4@v1"].verifier_specificity = 1
    passed = infer_error_risks(graph, observations={"D4@v1": "pass"})
    failed = infer_error_risks(graph, observations={"D4@v1": "fail"})
    assert passed.probabilities["D1@v1"] == pytest.approx(0)
    assert failed.probabilities["D1@v1"] == pytest.approx(1)
    assert passed.evidence_probability == pytest.approx(0.5)


def test_imperfect_verifier_bayesian_update_and_mc():
    graph = DependencyGraph()
    graph.add_logical_node("A")
    graph.add_version("A")
    version = graph.versions["A@v1"]
    version.intrinsic_error_prob = 0.2
    version.verifier_sensitivity = 0.9
    version.verifier_specificity = 0.8
    exact = infer_error_risks(graph, observations={"A@v1": "fail"})
    # P(error | fail) = .2*.9 / (.2*.9 + .8*.2)
    assert exact.probabilities["A@v1"] == pytest.approx(0.18 / 0.34)
    assert exact.evidence_probability == pytest.approx(0.34)
    mc = infer_error_risks(
        graph, method="mc", samples=20000, seed=9,
        observations={"A@v1": "fail"},
    )
    assert mc.probabilities["A@v1"] == pytest.approx(0.18 / 0.34, abs=0.03)


def test_unrelated_observation_does_not_change_independent_source():
    graph = DependencyGraph()
    for v in ("A", "B"):
        graph.add_logical_node(v)
        graph.add_version(v)
        graph.versions[v + "@v1"].intrinsic_error_prob = 0.3
    result = infer_error_risks(graph, observations={"A@v1": "pass"})
    assert result.probabilities["B@v1"] == pytest.approx(0.3)


def test_exact_limit_rejects_exponential_growth_and_mc_works():
    graph = diamond()
    with pytest.raises(ValueError, match="limit"):
        infer_error_risks(graph, max_exact_events=0)
    result = infer_error_risks(graph, method="mc", samples=100, seed=2)
    assert 0 <= result.probabilities["D4@v1"] <= 1


def test_zero_probability_evidence_raises():
    graph = diamond(root_probability=1)
    with pytest.raises(ValueError, match="zero probability"):
        infer_error_risks(graph, observations={"D4@v1": "pass"})


def test_stale_inputs_fail_closed_not_counted_as_safe():
    graph = diamond()
    graph.versions["D2@v1"].status = NodeStatus.STALE
    with pytest.raises(ValueError, match="unusable"):
        infer_error_risks(graph)


def test_current_versions_exclude_unrelated_historical_versions():
    graph = DependencyGraph()
    graph.add_logical_node("A")
    graph.add_version("A", intrinsic_error_prob=1.0)
    graph.add_version("A", intrinsic_error_prob=0.0)
    result = infer_error_risks(graph)
    assert list(result.probabilities) == ["A@v2"]
    assert result.expected_loss == pytest.approx(0)
    with pytest.raises(ValueError, match="unusable"):
        graph.versions["A@v1"].status = NodeStatus.INVALID
        infer_error_risks(graph, current_only=False)


def test_expected_loss_uses_selected_targets_weights_only():
    graph = diamond()
    graph.versions["D4@v1"].loss_weight = 10
    result = infer_error_risks(graph, target_version_ids=["D4@v1"])
    assert result.expected_loss == pytest.approx(5)
    assert "D1@v1" in result.probabilities


def test_seeded_ground_truth_is_reproducible_and_correlated():
    graph = diamond()
    a = sample_ground_truth(graph, seed=31)
    b = sample_ground_truth(graph, seed=31)
    assert a == b
    assert len({a[v] for v in ("D1@v1", "D2@v1", "D3@v1", "D4@v1")}) == 1


def test_invalid_params_and_observations_raise():
    graph = diamond()
    graph.versions["D1@v1"].intrinsic_error_prob = 2
    with pytest.raises(ValueError, match="intrinsic_error_prob"):
        infer_error_risks(graph)
    graph.versions["D1@v1"].intrinsic_error_prob = 0.5
    with pytest.raises(ValueError, match="observation"):
        infer_error_risks(graph, observations={"D1@v1": "maybe"})
    with pytest.raises(ValueError, match="scope"):
        infer_error_risks(graph, target_version_ids=["D1@v1"],
                          observations={"D4@v1": "fail"})


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_loss_weight_rejects_negative_or_nonfinite_values(weight):
    graph = diamond()
    graph.versions["D4@v1"].loss_weight = weight
    with pytest.raises(ValueError, match="loss_weight.*finite and non-negative"):
        infer_error_risks(graph, target_version_ids=["D4@v1"])
