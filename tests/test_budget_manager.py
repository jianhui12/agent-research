import pytest
from dav.budget_manager import BudgetExceeded, BudgetManager


def test_budgets_are_separate():
    manager = BudgetManager(2, 5)
    manager.spend_verification(1)
    assert manager.verification_remaining == 1
    assert manager.recovery_remaining == 5


def test_hard_budget_guard():
    manager = BudgetManager(1, 1)
    with pytest.raises(BudgetExceeded):
        manager.spend_recovery(2)
