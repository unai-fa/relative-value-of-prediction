"""CRRA benefit lever."""

from typing import TYPE_CHECKING
from .base import ParameterizedLever

if TYPE_CHECKING:
    from ..problem import AllocationProblem


class CRRABenefitLever(ParameterizedLever):
    """Lever that modifies the benefit parameter b in CRRAUtility.

    theta = the new benefit value (or increment in marginal mode)

    Two modes:
    - Setting mode (marginal=False): theta is the absolute benefit value
    - Marginal mode (marginal=True): theta is the INCREMENT in benefit above baseline
      The baseline is automatically extracted from problem.utility.b

    Cost model:
    - cost = theta * n_allocated (from problem.constraint.get_capacity())
    - In marginal mode, theta is the increment, so cost is incremental

    Can represent:
    - Changes in transfer size (e.g., cash transfer amount)
    - Uncertainty over treatment effect magnitude

    Example:
        # Setting mode: set benefit to 100
        lever = CRRABenefitLever(
            name="Transfer size",
            new_benefit=100,
        )

        # Marginal mode: theta is additional benefit above problem's current b
        lever = CRRABenefitLever(
            name="Transfer size",
            new_benefit=20,  # theta = 20 increment -> final = problem.utility.b + 20
            marginal=True,
        )
    """

    def __init__(
        self,
        name: str,
        new_benefit: float,
        marginal: bool = False,
    ):
        """Initialize CRRA benefit lever.

        Args:
            name: Identifier for this lever
            new_benefit: New benefit value (this is theta). In marginal mode,
                this is the INCREMENT above the problem's current benefit.
            marginal: If True, theta is interpreted as increment above baseline
                (baseline extracted from problem.utility.b)
        """
        if marginal and new_benefit < 0:
            raise ValueError(f"In marginal mode, new_benefit (increment) must be >= 0, got {new_benefit}")

        super().__init__(name, theta=new_benefit)
        self.marginal = marginal

    @property
    def new_benefit(self) -> float:
        """Alias for theta (in marginal mode, this is the increment)."""
        return self.theta

    def get_effective_benefit(self, problem: 'AllocationProblem') -> float:
        """The actual benefit value that will be applied.

        In setting mode: same as theta
        In marginal mode: problem.utility.b + theta
        """
        if self.marginal:
            from ..utilities import CRRAUtility
            if not isinstance(problem.utility, CRRAUtility):
                raise TypeError(
                    f"CRRABenefitLever in marginal mode requires CRRAUtility, "
                    f"got {type(problem.utility).__name__}"
                )
            return problem.utility.b + self.theta
        return self.theta

    def with_theta(self, theta: float) -> 'CRRABenefitLever':
        """Return new lever with different benefit (preserves mode)."""
        return CRRABenefitLever(
            name=self.name,
            new_benefit=theta,
            marginal=self.marginal,
        )

    def compute_cost(self, problem: 'AllocationProblem') -> float:
        """Compute benefit cost = theta * n_allocated.

        The cost of increasing benefit by theta is theta per beneficiary.
        n_allocated is read from the problem's constraint.
        """
        n_allocated = problem.constraint.get_capacity()
        return self.theta * n_allocated

    def for_budget(self, budget: float, problem: 'AllocationProblem') -> 'CRRABenefitLever':
        """Return lever with benefit adjusted to match budget.

        theta = budget / n_allocated (benefit per person given total budget).

        Args:
            budget: Target budget for benefit
            problem: Allocation problem for context

        Returns:
            New lever with adjusted theta
        """
        n_allocated = problem.constraint.get_capacity()
        theta = budget / n_allocated if n_allocated > 0 else 0.0
        return self.with_theta(theta)

    def apply(self, problem: 'AllocationProblem') -> 'AllocationProblem':
        """Return new problem with modified CRRA benefit.

        In marginal mode, extracts baseline from problem.utility.b and adds theta.

        Args:
            problem: Allocation problem with CRRAUtility

        Returns:
            New AllocationProblem with modified benefit
        """
        from ..problem import AllocationProblem
        from ..utilities import CRRAUtility

        if not isinstance(problem.utility, CRRAUtility):
            raise TypeError(
                f"CRRABenefitLever requires CRRAUtility, "
                f"got {type(problem.utility).__name__}"
            )

        effective_benefit = self.get_effective_benefit(problem)

        new_utility = CRRAUtility(
            b=effective_benefit,
            rho=problem.utility.rho,
        )

        return AllocationProblem(
            data=problem.data,
            utility=new_utility,
            constraint=problem.constraint,
            policy=problem.policy,
        )

    def __repr__(self):
        if self.marginal:
            return (
                f"CRRABenefitLever(name='{self.name}', "
                f"theta={self.theta}, marginal=True)"
            )
        return f"CRRABenefitLever(name='{self.name}', new_benefit={self.theta})"
