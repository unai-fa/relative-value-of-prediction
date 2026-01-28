"""Constant Relative Risk Aversion (CRRA) utility function."""

import numpy as np
from .base import UtilityFunction


class CRRAUtility(UtilityFunction):
    """CRRA utility function for allocation problems.

    Implements the utility GAIN from allocation:
        u(y, a) = [(y + b * a)^{1-rho} - y^{1-rho}] / (1-rho)

    Where:
        - y: outcome value
        - a: action (0 or 1)
        - b: benefit parameter (effect of action on outcome)
        - rho: relative risk aversion coefficient (rho > 0, rho != 1)

    When a=0: u(y, 0) = 0 (no gain from no allocation)
    When a=1: u(y, 1) = [(y + b)^{1-rho} - y^{1-rho}] / (1-rho) (gain from allocation)
    """

    def __init__(self, b: float, rho: float):
        """Initialize CRRA utility function.

        Args:
            b: Benefit parameter - the additive effect of taking action
            rho: Relative risk aversion coefficient (must be > 0 and != 1)

        Raises:
            ValueError: If rho <= 0 or rho == 1
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        if rho == 1:
            raise ValueError(
                "rho cannot equal 1"
            )

        self.b = b
        self.rho = rho

    def compute(self, y: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute CRRA utility gains for each unit.

        Args:
            y: Ground truth outcomes of shape (n,)
            actions: Binary actions (0 or 1) of shape (n,)

        Returns:
            Array of utility gains of shape (n,)
        """
        exponent = 1 - self.rho

        # Utility with action: (y + b * a)^{1-rho} / (1-rho)
        u_with_action = np.power(y + self.b * actions, exponent) / exponent

        # Baseline utility (no action): y^{1-rho} / (1-rho)
        u_baseline = np.power(y, exponent) / exponent

        # Return the gain from allocation
        return u_with_action - u_baseline

    def __repr__(self) -> str:
        return f"CRRAUtility(b={self.b}, rho={self.rho})"
