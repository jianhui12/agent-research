"""Independent hard-budget ledgers for verification and recovery."""
class BudgetExceeded(RuntimeError):
    pass


class BudgetManager:
    def __init__(self, verification_budget: float, recovery_budget: float):
        if verification_budget < 0 or recovery_budget < 0:
            raise ValueError("Budgets must be non-negative")
        self.verification_remaining = float(verification_budget)
        self.recovery_remaining = float(recovery_budget)

    def spend_verification(self, cost: float) -> None:
        self._spend("verification_remaining", cost)

    def spend_recovery(self, cost: float) -> None:
        self._spend("recovery_remaining", cost)

    def _spend(self, field: str, cost: float) -> None:
        if cost < 0:
            raise ValueError("Cost must be non-negative")
        remaining = getattr(self, field)
        if cost > remaining:
            raise BudgetExceeded(f"Need {cost}, only {remaining} remains")
        setattr(self, field, remaining - cost)
