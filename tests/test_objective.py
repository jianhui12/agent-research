"""Terminal objective: stable task identities, explicit loss, independent costs."""
import pytest

from dav.budget_manager import BudgetManager
from dav.graph import DependencyGraph
from dav.models import NodeStatus
from dav.objective import (
    ObjectiveConfig, TaskTarget, evaluate_expected, evaluate_realized,
)
from dav.state import DAVState


def make_state():
    graph = DependencyGraph()
    for node in ("A", "B", "C", "D"):
        graph.add_logical_node(node)
    graph.add_version("A", intrinsic_error_prob=0.25)
    graph.add_version("B")
    graph.add_version("C", intrinsic_error_prob=0.5)
    graph.record_execution("b", "compute", ["A@v1"], ["B@v1"])
    return DAVState(graph, BudgetManager(5, 5))


def test_expected_objective_counts_only_explicit_targets_and_paid_costs():
    state = make_state()
    state.budgets.spend_verification(1)
    state.budgets.spend_recovery(2)
    config = ObjectiveConfig((TaskTarget("B", 8, 12), TaskTarget("C", 4, 7)),
                             lambda_v=0.5, lambda_r=0.25)
    result = evaluate_expected(state, config)
    assert result.error_loss == pytest.approx(4)
    assert result.blocked_loss == 0
    assert result.verification_cost == 1
    assert result.recovery_cost == 2
    assert result.total == pytest.approx(5)


def test_blocked_target_charged_once_instead_of_error_and_block_loss():
    state = make_state()
    state.graph.mark_status("A@v1", NodeStatus.BLOCKED)
    config = ObjectiveConfig((TaskTarget("B", 8, 12), TaskTarget("C", 4, 7)))
    expected = evaluate_expected(state, config)
    realized = evaluate_realized(state, config, {"B@v1": True, "C@v1": True})
    assert expected.blocked_targets == ("B",)
    assert expected.blocked_loss == realized.blocked_loss == 12
    assert expected.error_loss == pytest.approx(2)
    assert realized.error_loss == 4
    assert realized.total == 16


def test_unproduced_target_has_noncompletion_loss_and_needs_no_truth():
    state = make_state()
    config = ObjectiveConfig((TaskTarget("D", 2, 9),))
    assert evaluate_expected(state, config).total == 9
    assert evaluate_realized(state, config, {}).total == 9


def test_historical_version_does_not_count_twice():
    state = make_state()
    state.graph.add_version("C", intrinsic_error_prob=0.1)
    state.graph.supersede_version("C@v1", "C@v2")
    config = ObjectiveConfig((TaskTarget("C", 10, 20),))
    assert evaluate_expected(state, config).error_loss == pytest.approx(1)
    assert evaluate_realized(state, config, {"C@v1": True, "C@v2": False}).total == 0


def test_expected_objective_uses_evidence_and_rollout_checkpoint():
    state = make_state()
    state.record_verification("C@v1", "pass")
    checkpoint = state.checkpoint()
    fork = state.fork()
    fork.record_verification("A@v1", "fail")
    config = ObjectiveConfig((TaskTarget("B", 8, 12),))
    result = evaluate_expected(fork, config, checkpoint=checkpoint)
    assert result.error_loss == pytest.approx(8)
    assert result.verification_cost == 1
    assert result.total == pytest.approx(9)
    assert evaluate_expected(state, config).error_loss == pytest.approx(2)


def test_realized_requires_truth_for_usable_target():
    state = make_state()
    config = ObjectiveConfig((TaskTarget("B", 8, 12),))
    with pytest.raises(KeyError, match="Missing ground truth"):
        evaluate_realized(state, config, {})


def test_config_rejects_duplicates_and_invalid_weights():
    target = TaskTarget("B", 8, 12)
    with pytest.raises(ValueError, match="Duplicate"):
        ObjectiveConfig((target, target))
    with pytest.raises(ValueError):
        TaskTarget("B", 8, -1)
    with pytest.raises(ValueError):
        ObjectiveConfig((target,), lambda_r=float("nan"))


def test_zero_block_penalty_is_explicit_not_implicit():
    state = make_state()
    config = ObjectiveConfig((TaskTarget("D", 2, 0),))
    assert evaluate_expected(state, config).blocked_targets == ("D",)
    assert evaluate_expected(state, config).total == 0
