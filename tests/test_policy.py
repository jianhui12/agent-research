"""AMCR action selection, feasibility and stopping."""
from dataclasses import replace

import pytest

from dav.budget_manager import BudgetManager
from dav.cost_estimator import MonteCarloCostEstimator
from dav.graph import DependencyGraph
from dav.models import NodeStatus
from dav.objective import ObjectiveConfig, TaskTarget
from dav.policy import AMCRPolicy, feasible_candidates
from dav.state import DAVState


def state_with_outputs():
    graph = DependencyGraph()
    for name in ("A", "B"):
        graph.add_logical_node(name)
        graph.add_version(name, intrinsic_error_prob=1,
                          repair_residual_error_prob=0)
    return DAVState(graph, BudgetManager(4, 4))


def test_amcr_selects_gain_per_cost_not_raw_gain():
    state = state_with_outputs()
    state.graph.versions["B@v1"].verify_cost = 2
    config = ObjectiveConfig((TaskTarget("A", 8, 12),
                              TaskTarget("B", 12, 18)), 0, 0)
    policy = AMCRPolicy(MonteCarloCostEstimator(config, rollouts=5))
    decision = policy.decide(state)
    assert decision.version_id == "A@v1"
    assert decision.ranking[0].score == 8
    assert decision.ranking[1].estimate.marginal_reduction == 12
    assert decision.ranking[1].score == 6
    assert state.observations == {}


def test_feasible_candidates_filter_historical_blocked_observed_and_expensive():
    state = state_with_outputs()
    graph = state.graph
    for name in ("C", "D", "E", "F"):
        graph.add_logical_node(name)
    graph.add_version("C", status=NodeStatus.BLOCKED)
    graph.add_version("D", verify_cost=10)
    graph.add_version("E")
    graph.add_version("E", make_current=True)
    state.record_verification("A@v1", "pass")
    assert feasible_candidates(state) == ["B@v1", "E@v2"]


def test_no_feasible_candidate_skips_estimator():
    state = state_with_outputs()
    state.budgets.spend_verification(4)

    class NeverCalled:
        def estimate_many(self, state, candidates):
            raise AssertionError("Estimator must not run")

    decision = AMCRPolicy(NeverCalled()).decide(state)
    assert decision.version_id is None
    assert decision.reason == "no_feasible_candidate"


def test_nonpositive_gain_stops_even_when_budget_remains():
    state = state_with_outputs()
    for v in state.graph.versions.values():
        v.intrinsic_error_prob = 0
    config = ObjectiveConfig((TaskTarget("A", 8, 12),), 1, 1)
    decision = AMCRPolicy(MonteCarloCostEstimator(config, rollouts=5)).decide(state)
    assert decision.version_id is None
    assert decision.reason == "no_positive_gain"
    assert all(action.score < 0 for action in decision.ranking)


def test_recovery_budget_is_priced_not_used_as_candidate_filter():
    state = state_with_outputs()
    state.budgets.spend_recovery(4)
    config = ObjectiveConfig((TaskTarget("A", 8, 12),
                              TaskTarget("B", 8, 12)), 1, 1)
    assert feasible_candidates(state) == ["A@v1", "B@v1"]
    decision = AMCRPolicy(MonteCarloCostEstimator(config, rollouts=5)).decide(state)
    assert decision.reason == "no_positive_gain"
    assert all(action.estimate.expected_blocked_loss == 12
               for action in decision.ranking)


def test_ties_are_deterministic_and_gain_threshold_is_strict():
    state = state_with_outputs()
    config = ObjectiveConfig((TaskTarget("A", 8, 12),
                              TaskTarget("B", 8, 12)), 0, 0)
    estimator = MonteCarloCostEstimator(config, rollouts=5)
    assert AMCRPolicy(estimator).decide(state).version_id == "A@v1"
    assert AMCRPolicy(estimator, min_gain=8).decide(state).version_id is None


def test_optional_uncertainty_gate_uses_paired_standard_error():
    state = state_with_outputs()
    config = ObjectiveConfig((TaskTarget("A", 8, 12),
                              TaskTarget("B", 8, 12)), 0, 0)
    real = MonteCarloCostEstimator(config, rollouts=5).estimate_many(
        state, ["A@v1", "B@v1"])

    class StubEstimator:
        def estimate_many(self, state, candidates):
            return replace(real, candidates={
                "A@v1": replace(real.candidates["A@v1"], marginal_reduction=3,
                                paired_gain_standard_error=2),
                "B@v1": replace(real.candidates["B@v1"], marginal_reduction=2,
                                paired_gain_standard_error=0.1),
            })

    assert AMCRPolicy(StubEstimator()).decide(state).version_id == "A@v1"
    assert AMCRPolicy(StubEstimator(), confidence_z=2).decide(state).version_id == "B@v1"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 0])
def test_invalid_verification_cost_is_rejected(bad):
    state = state_with_outputs()
    state.graph.versions["A@v1"].verify_cost = bad
    with pytest.raises(ValueError, match="positive"):
        feasible_candidates(state)
