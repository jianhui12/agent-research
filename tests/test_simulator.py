"""Closed-loop simulator tests with persistent truth and paired initial worlds."""
import pytest

from dav.budget_manager import BudgetManager
from dav.cost_estimator import MonteCarloCostEstimator
from dav.graph import DependencyGraph
from dav.objective import ObjectiveConfig, TaskTarget
from dav.policy import AMCRPolicy, PolicyDecision, StopPolicy
from dav.simulator import SyntheticSimulator
from dav.state import DAVState


def fixture_state(p=1, verification_budget=3, recovery_budget=3):
    graph = DependencyGraph()
    for name in ("A", "B"):
        graph.add_logical_node(name)
        graph.add_version(name, intrinsic_error_prob=p,
                          repair_residual_error_prob=0)
    config = ObjectiveConfig((TaskTarget("A", 10, 15),
                              TaskTarget("B", 20, 30)), 0.2, 0.1)
    return DAVState(graph, BudgetManager(verification_budget, recovery_budget)), config


def test_adaptive_episode_replans_after_recovery_and_stops_with_budget_left():
    state, config = fixture_state()
    policy = AMCRPolicy(MonteCarloCostEstimator(config, rollouts=8, seed=7))
    result = SyntheticSimulator(config).run(state, policy, seed=42)
    assert [s.decision.version_id for s in result.steps] == ["B@v1", "A@v1"]
    assert [s.observation for s in result.steps] == ["fail", "fail"]
    assert result.stop_reason == "no_positive_gain"
    assert result.final_state.budgets.verification_remaining == 1
    assert result.final_state.budgets.recovery_remaining == 1
    assert result.objective.total == pytest.approx(0.6)
    assert result.success
    assert result.completion_rate == 1
    assert result.correct_output_rate == 1
    assert result.initial_truth == {"A@v1": True, "B@v1": True}
    assert result.final_truth == {"A@v1": True, "B@v1": True,
                                  "A@v2": False, "B@v2": False}
    assert result.final_state.observations == {"B@v1": "fail", "A@v1": "fail"}
    assert state.graph.get_current_version("A").version_id == "A@v1"
    assert state.budgets.verification_remaining == 3
    assert state.observations == {}


def test_zero_budget_stops_without_spending_or_hiding_errors():
    state, config = fixture_state(verification_budget=0)
    result = SyntheticSimulator(config).run(
        state, AMCRPolicy(MonteCarloCostEstimator(config, rollouts=5)))
    assert result.stop_reason == "no_feasible_candidate"
    assert result.steps == ()
    assert result.objective.total == 30
    assert result.completion_rate == 1
    assert result.correct_output_rate == 0
    assert not result.success


def test_no_verification_control_and_amcr_share_initial_world():
    state, config = fixture_state(p=0.6)
    simulator = SyntheticSimulator(config)
    a = simulator.run(state, StopPolicy(), seed=123)
    b = simulator.run(state, AMCRPolicy(MonteCarloCostEstimator(config, rollouts=10)), seed=123)
    assert a.initial_truth == b.initial_truth
    assert all(b.final_truth[v] == error for v, error in a.initial_truth.items())
    assert a.objective.verification_cost == a.objective.recovery_cost == 0
    assert a.stop_reason == "stop_policy"


def test_seed_reproducibility_of_actions_truth_and_objective():
    state, config = fixture_state(p=0.8)
    policy = AMCRPolicy(MonteCarloCostEstimator(config, rollouts=12, seed=6))
    simulator = SyntheticSimulator(config)
    a = simulator.run(state, policy, seed=999)
    b = simulator.run(state, policy, seed=999)
    assert a.initial_truth == b.initial_truth
    assert a.final_truth == b.final_truth
    assert a.steps == b.steps
    assert a.objective == b.objective
    assert a.stop_reason == b.stop_reason


def test_policy_receives_observation_only_and_cannot_mutate_environment():
    state, config = fixture_state()

    class MutatingPolicy:
        def decide(self, observable):
            assert isinstance(observable, DAVState)
            assert not hasattr(observable, "truth")
            assert not hasattr(observable, "initial_truth")
            observable.graph.versions["A@v1"].intrinsic_error_prob = 0
            observable.budgets.spend_verification(1)
            return PolicyDecision(None, "custom_stop")

    result = SyntheticSimulator(config).run(state, MutatingPolicy())
    assert result.stop_reason == "custom_stop"
    assert result.final_state.graph.versions["A@v1"].intrinsic_error_prob == 1
    assert result.final_state.budgets.verification_remaining == 3
    assert result.objective.total == 30


def test_failed_recovery_is_noncompletion_not_correct_output():
    state, config = fixture_state(recovery_budget=0)

    class VerifyOnce:
        def decide(self, observable):
            return PolicyDecision("A@v1", "verify")

    result = SyntheticSimulator(config, max_steps=1).run(state, VerifyOnce())
    assert result.stop_reason == "max_steps"
    assert result.objective.blocked_loss == 15
    assert result.objective.error_loss == 20
    assert result.objective.total == pytest.approx(35.2)
    assert result.completion_rate == 0.5
    assert result.correct_output_rate == 0
    assert result.targets[0].correct is None
    assert result.steps[0].recovery is not None
    assert not result.success


def test_preexisting_evidence_conditions_initial_world_without_double_charging():
    state, config = fixture_state(p=0.4)
    state.record_verification("A@v1", "fail")
    result = SyntheticSimulator(config).run(state, StopPolicy(), seed=3)
    assert result.initial_truth["A@v1"] is True
    assert result.objective.verification_cost == 0
    assert result.objective.recovery_cost == 0
    assert result.final_state.observations == {"A@v1": "fail"}


def test_invalid_policy_action_is_rejected_by_environment():
    state, config = fixture_state(verification_budget=0)

    class InvalidPolicy:
        def decide(self, observable):
            return PolicyDecision("A@v1", "verify")

    with pytest.raises(ValueError, match="budget"):
        SyntheticSimulator(config).run(state, InvalidPolicy())
    assert state.budgets.verification_remaining == 0


def test_step_cap_zero_executes_no_actions():
    state, config = fixture_state()

    class NeverCalled:
        def decide(self, observable):
            raise AssertionError("Policy must not run")

    result = SyntheticSimulator(config, max_steps=0).run(state, NeverCalled())
    assert result.stop_reason == "max_steps"
    assert not result.steps
    assert result.objective.total == 30
