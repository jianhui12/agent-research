"""Independent hard-budget ledgers for verification and recovery."""
import math


class BudgetExceeded(RuntimeError):
    pass


class BudgetManager:
    def __init__(self, verification_budget: float, recovery_budget: float):
        if any(not isinstance(b, (int, float)) or not math.isfinite(b) or b < 0
               for b in (verification_budget, recovery_budget)):
            raise ValueError("Budgets must be finite and non-negative")
        self.verification_remaining = float(verification_budget)
        self.recovery_remaining = float(recovery_budget)

    def spend_verification(self, cost: float) -> None:
        self._spend("verification_remaining", cost)

    def spend_recovery(self, cost: float) -> None:
        self._spend("recovery_remaining", cost)

    def _spend(self, field: str, cost: float) -> None:
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError("Cost must be finite and non-negative")
        remaining = getattr(self, field)
        if cost > remaining:
            raise BudgetExceeded(f"Need {cost}, only {remaining} remains")
        setattr(self, field, remaining - cost)
