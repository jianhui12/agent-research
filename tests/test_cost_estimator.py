"""One-step cost estimator against analytical and deterministic references."""
import pytest

from dav.budget_manager import BudgetManager
from dav.cost_estimator import MonteCarloCostEstimator
from dav.graph import DependencyGraph
from dav.objective import ObjectiveConfig, TaskTarget
from dav.policy import rank_candidates
from dav.state import DAVState


def single_state(p=0.4, verification=2, recovery=3):
    graph = DependencyGraph()
    graph.add_logical_node("A")
    graph.add_version("A", intrinsic_error_prob=p, repair_residual_error_prob=0,
                      repair_cost=2)
    return DAVState(graph, BudgetManager(verification, recovery))


def config(*, lambda_v=0.2, lambda_r=0.1):
    return ObjectiveConfig((TaskTarget("A", 10, 15),), lambda_v, lambda_r)


def test_deterministic_repair_costs_and_state_isolation():
    state = single_state(p=1)
    comparison = MonteCarloCostEstimator(config(), rollouts=20).estimate_many(
        state, ["A@v1"])
    assert comparison.baseline.expected_total == 10
    candidate = comparison.candidates["A@v1"]
    assert candidate.expected_total == pytest.approx(0.4)
    assert candidate.expected_verification_cost == 1
    assert candidate.expected_recovery_cost == 2
    assert candidate.expected_error_loss == candidate.expected_blocked_loss == 0
    assert candidate.marginal_reduction == pytest.approx(9.6)
    assert candidate.paired_gain_standard_error == 0
    assert state.graph.get_current_version("A").version_id == "A@v1"
    assert len(state.graph.versions) == 1
    assert state.observations == {}
    assert state.budgets.verification_remaining == 2
    assert state.budgets.recovery_remaining == 3


def test_exact_analytical_reference_for_imperfect_verifier_and_stochastic_repair():
    state = single_state()
    version = state.graph.versions["A@v1"]
    version.verifier_sensitivity = 0.75
    version.verifier_specificity = 0.8
    version.repair_success_prob = 0.8
    version.repair_residual_error_prob = 0.1
    comparison = MonteCarloCostEstimator(config(), rollouts=4000,
                                         seed=37).estimate_many(state, ["A@v1"])
    candidate = comparison.candidates["A@v1"]
    # P(fail) = .4*.75 + .6*.2 = .42, including false positives.
    # Expected terminal error loss: .4*.25*10 + .42*.8*.1*10 = 1.336.
    # Expected block loss: .42*.2*15 = 1.26.
    # Verification and expected recovery costs: .2 + .42*2*.1 = .284.
    assert comparison.baseline.expected_total == pytest.approx(4, abs=0.25)
    assert candidate.expected_total == pytest.approx(2.88, abs=0.3)
    assert candidate.fail_probability == pytest.approx(0.42, abs=0.025)
    assert candidate.expected_error_loss == pytest.approx(1.336, abs=0.25)
    assert candidate.expected_blocked_loss == pytest.approx(1.26, abs=0.25)
    assert candidate.expected_recovery_cost == pytest.approx(0.84, abs=0.05)
    assert candidate.marginal_reduction == pytest.approx(
        comparison.baseline.expected_total - candidate.expected_total)
    assert candidate.paired_gain_standard_error > 0


def test_no_recovery_budget_charges_noncompletion_not_free_success():
    state = single_state(p=1, recovery=0)
    c = MonteCarloCostEstimator(config(), rollouts=10).estimate_many(
        state, ["A@v1"]).candidates["A@v1"]
    assert c.expected_total == pytest.approx(15.2)
    assert c.expected_blocked_loss == 15
    assert c.expected_recovery_cost == 0
    assert c.probability_any_blocked == 1
    assert c.marginal_reduction < 0


def test_false_positive_repairs_correct_version_without_oracle_leakage():
    state = single_state(p=0)
    state.graph.versions["A@v1"].verifier_specificity = 0
    state.graph.versions["A@v1"].repair_residual_error_prob = 1
    comparison = MonteCarloCostEstimator(config(), rollouts=10).estimate_many(
        state, ["A@v1"])
    candidate = comparison.candidates["A@v1"]
    assert comparison.baseline.expected_total == 0
    assert candidate.fail_probability == 1
    assert candidate.expected_recovery_cost == 2
    assert candidate.expected_error_loss == 10
    assert candidate.expected_total == pytest.approx(10.4)


def test_always_pass_keeps_truth_and_paired_gain_has_zero_variance():
    state = single_state(p=0.4)
    state.graph.versions["A@v1"].verifier_sensitivity = 0
    estimator = MonteCarloCostEstimator(config(lambda_v=1), rollouts=200, seed=51)
    comparison = estimator.estimate_many(state, ["A@v1"])
    candidate = comparison.candidates["A@v1"]
    assert comparison.baseline.standard_error > 0
    assert candidate.expected_error_loss == comparison.baseline.expected_error_loss
    assert candidate.marginal_reduction == -1
    assert candidate.paired_gain_standard_error == 0
    assert candidate.expected_recovery_cost == 0


def test_candidate_order_and_single_action_interface_are_consistent():
    state = single_state()
    graph = state.graph
    graph.add_logical_node("B")
    graph.add_version("B", intrinsic_error_prob=0.2)
    graph.record_execution("b", "compute", ["A@v1"], ["B@v1"])
    obj = ObjectiveConfig((TaskTarget("B", 10, 15),), 0.2, 0.1)
    estimator = MonteCarloCostEstimator(obj, rollouts=70, seed=7)
    a = estimator.estimate_many(state, ["A@v1", "B@v1"])
    b = estimator.estimate_many(state, ["B@v1", "A@v1"])
    assert a == b
    assert estimator.estimate(state, None) == a.baseline.expected_total
    assert estimator.estimate(state, "A@v1") == a.candidates["A@v1"].expected_total
    ranked = rank_candidates(state, ["A@v1", "B@v1"], estimator,
                             {"A@v1": 1, "B@v1": 1})
    assert ranked[0].marginal_reduction >= ranked[1].marginal_reduction


def test_costs_before_rollout_are_not_charged_again_and_evidence_conditions_worlds():
    state = single_state(p=0.4)
    graph = state.graph
    graph.add_logical_node("B")
    graph.add_version("B")
    graph.record_execution("b", "compute", ["A@v1"], ["B@v1"])
    state.record_verification("B@v1", "fail")  # perfect evidence A is wrong
    state.budgets.spend_recovery(1)
    comparison = MonteCarloCostEstimator(config(), rollouts=10).estimate_many(
        state, ["A@v1"])
    assert comparison.baseline.expected_total == 10
    assert comparison.baseline.expected_verification_cost == 0
    assert comparison.baseline.expected_recovery_cost == 0
    assert comparison.candidates["A@v1"].fail_probability == 1
    assert comparison.candidates["A@v1"].expected_recovery_cost == 2
    assert state.observations == {"B@v1": "fail"}


def test_infeasible_and_previously_observed_candidates_are_rejected():
    state = single_state(verification=0)
    estimator = MonteCarloCostEstimator(config(), rollouts=10)
    with pytest.raises(ValueError, match="budget"):
        estimator.estimate(state, "A@v1")
    state = single_state()
    state.record_verification("A@v1", "fail")
    with pytest.raises(ValueError, match="already"):
        estimator.estimate(state, "A@v1")
    state.graph.add_version("A")
    with pytest.raises(ValueError, match="unusable"):
        estimator.estimate(state, "A@v1")


def test_mc_posterior_diagnostics_are_reported():
    state = single_state()
    comparison = MonteCarloCostEstimator(
        config(), rollouts=50, posterior_method="mc", posterior_samples=200,
    ).estimate_many(state, ["A@v1"])
    assert comparison.posterior_method == "mc"
    assert comparison.posterior_effective_samples == pytest.approx(200)
    assert comparison.log_evidence_probability == pytest.approx(0)
