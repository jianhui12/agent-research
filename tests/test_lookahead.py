"""Depth-two planning counterexamples and observable continuation tests."""
import pytest

from dav.budget_manager import BudgetManager
from dav.cost_estimator import MonteCarloCostEstimator
from dav.graph import DependencyGraph
from dav.lookahead import TwoStepAMCRPolicy, TwoStepCostEstimator
from dav.objective import ObjectiveConfig, TaskTarget
from dav.policy import AMCRPolicy
from dav.simulator import SyntheticSimulator
from dav.state import DAVState


def independent(gains, costs, budget):
    graph = DependencyGraph()
    targets = []
    for name, gain, cost in zip(("A", "B", "C"), gains, costs):
        graph.add_logical_node(name)
        graph.add_version(name, intrinsic_error_prob=1,
                          repair_residual_error_prob=0, verify_cost=cost)
        targets.append(TaskTarget(name, gain, 2 * gain))
    return (DAVState(graph, BudgetManager(budget, 4)),
            ObjectiveConfig(tuple(targets), 0, 0))


def estimator(config, **kwargs):
    return TwoStepCostEstimator(config, outer_rollouts=12, inner_rollouts=12,
                                seed=9, **kwargs)


def test_budget_three_counterexample_prefers_larger_cost_action():
    state, config = independent([10, 15, 4], [2, 3, 1], 3)
    greedy = AMCRPolicy(MonteCarloCostEstimator(config, rollouts=8))
    improved = TwoStepAMCRPolicy(estimator(config))
    assert greedy.decide(state).version_id == "A@v1"
    decision = improved.decide(state)
    assert decision.version_id == "B@v1"
    assert decision.lookahead.candidates["A@v1"].estimate.marginal_reduction == 14
    assert decision.lookahead.candidates["B@v1"].estimate.marginal_reduction == 15
    assert decision.lookahead.candidates["A@v1"].continuation_rate == 1
    assert decision.lookahead.candidates["B@v1"].continuation_rate == 0
    sim = SyntheticSimulator(config)
    assert sim.run(state, greedy, seed=3).objective.total == 15
    assert sim.run(state, improved, seed=3).objective.total == 14


def test_budget_four_counterexample_prefers_low_cost_combination():
    state, config = independent([10, 15, 9], [2, 3, 2], 4)
    decision = TwoStepAMCRPolicy(estimator(config)).decide(state)
    assert decision.version_id == "A@v1"
    assert decision.lookahead.candidates["A@v1"].estimate.marginal_reduction == 19
    assert decision.lookahead.candidates["B@v1"].estimate.marginal_reduction == 15


def test_equal_depth_two_value_prefers_low_first_cost_not_id():
    state, config = independent([5, 5], [2, 1], 2)
    decision = TwoStepAMCRPolicy(estimator(config)).decide(state)
    assert decision.lookahead.candidates["A@v1"].estimate.marginal_reduction == 5
    assert decision.lookahead.candidates["B@v1"].estimate.marginal_reduction == 5
    assert decision.version_id == "B@v1"


def test_information_only_first_action_can_have_negative_one_step_gain():
    graph = DependencyGraph()
    for name, cost in (("A", 100), ("B", 1), ("C", 3)):
        graph.add_logical_node(name)
        graph.add_version(name, intrinsic_error_prob=0.2 if name == "A" else 0,
                          verify_cost=cost, repair_cost=0, repair_residual_error_prob=0)
    graph.record_execution("b", "compute", ["A@v1"], ["B@v1"])
    graph.record_execution("c", "compute", ["A@v1"], ["C@v1"])
    state = DAVState(graph, BudgetManager(4, 1))
    config = ObjectiveConfig((TaskTarget("C", 20, 30),), 1, 0)
    result = TwoStepAMCRPolicy(TwoStepCostEstimator(
        config, outer_rollouts=500, inner_rollouts=24, seed=20)).decide(state)
    assert result.version_id == "B@v1"
    b = result.lookahead.candidates["B@v1"]
    assert b.one_step_expected_total - result.lookahead.baseline.expected_total == pytest.approx(1)
    assert b.estimate.expected_total == pytest.approx(1.6, abs=0.2)
    assert b.estimate.marginal_reduction > 0
    assert b.continuation_rate == pytest.approx(0.2, abs=0.06)
    assert b.observable_successors == 2


def test_observable_successor_reused_across_distinct_latent_worlds():
    state, config = independent([10, 10], [1, 1], 2)
    a = state.graph.versions["A@v1"]
    a.intrinsic_error_prob = 0.5
    a.verifier_sensitivity = 0  # Always PASS regardless of truth.
    a.verifier_specificity = 1

    class CountingEstimator(TwoStepCostEstimator):
        calls = 0
        def _continuation(self, observable, inner):
            self.calls += 1
            assert not hasattr(observable, "truth")
            assert observable.observations == {"A@v1": "pass"}
            return super()._continuation(observable, inner)

    e = CountingEstimator(config, outer_rollouts=60, inner_rollouts=10, seed=1)
    a = e.estimate_many(state, ["A@v1"]).candidates["A@v1"]
    assert a.observable_successors == 1
    assert e.calls == 1
    assert a.continuation_rate == 1
    assert a.estimate.expected_verification_cost == 2
    assert state.observations == {}
    assert len(state.graph.versions) == 2
    assert state.budgets.verification_remaining == 2


def test_recovery_budget_enforced_across_both_steps():
    state, config = independent([10, 15], [1, 1], 2)
    state.budgets.spend_recovery(3)  # Exactly one repair affordable.
    decision = TwoStepAMCRPolicy(estimator(config)).decide(state)
    assert decision.version_id == "B@v1"
    assert decision.lookahead.candidates["B@v1"].continuation_rate == 0
    assert decision.lookahead.candidates["B@v1"].estimate.expected_recovery_cost == 1


def test_candidate_order_reproducibility_and_optional_screening():
    state, config = independent([10, 15, 4], [2, 3, 1], 3)
    e = estimator(config)
    assert e.estimate_many(state, ["A@v1", "B@v1"]) == e.estimate_many(
        state, ["B@v1", "A@v1"])
    result = estimator(config, top_k=1).estimate_many(state, ["A@v1", "B@v1", "C@v1"])
    assert set(result.candidates) == {"A@v1"}
    assert result.screened_out == ("B@v1", "C@v1")


def test_stop_remains_available_at_both_depths():
    state, config = independent([10, 15], [1, 1], 2)
    for version in state.graph.versions.values():
        version.intrinsic_error_prob = 0
    config = ObjectiveConfig(config.targets, 1, 1)
    decision = TwoStepAMCRPolicy(estimator(config)).decide(state)
    assert decision.version_id is None
    assert decision.reason == "no_positive_gain"
    assert all(a.continuation_rate == 0 for a in decision.lookahead.candidates.values())


def test_no_feasible_action_returns_without_sampling():
    state, config = independent([10, 15], [1, 1], 0)
    decision = TwoStepAMCRPolicy(estimator(config)).decide(state)
    assert decision.reason == "no_feasible_candidate"
    assert decision.lookahead is None
