"""Budget expansion lever."""

from typing import TYPE_CHECKING
from .base import ParameterizedLever
from ..constraints.coverage import CoverageConstraint

if TYPE_CHECKING:
    from ..problem import AllocationProblem


class ExpandCoverageLever(ParameterizedLever):
    """Lever that increases budget/coverage capacity.

    theta = coverage_increase (fraction, e.g., 0.1 = +10% more people)
    cost = theta * n * marginal_cost_per_person
    """

    def __init__(
        self,
        name: str,
        coverage_increase: float,
        marginal_cost_per_person: float,
    ):
        """Initialize budget expansion lever.

        Args:
            name: Identifier for this lever
            coverage_increase: Additional coverage as fraction (e.g., 0.1 = +10%)
            marginal_cost_per_person: Cost per additional person served
        """
        if coverage_increase < 0:
            raise ValueError(f"coverage_increase must be non-negative, got {coverage_increase}")

        super().__init__(name, theta=coverage_increase)
        self.marginal_cost_per_person = marginal_cost_per_person

    @property
    def coverage_increase(self) -> float:
        """Alias for theta."""
        return self.theta

    def with_theta(self, theta: float) -> 'ExpandCoverageLever':
        """Return new lever with different coverage_increase."""
        return ExpandCoverageLever(
            name=self.name,
            coverage_increase=theta,
            marginal_cost_per_person=self.marginal_cost_per_person,
        )

    def with_marginal_cost(self, marginal_cost: float) -> 'ExpandCoverageLever':
        """Return new lever with different marginal_cost_per_person.

        Useful in lever linkage to update cost based on benefit level.

        Args:
            marginal_cost: New cost per additional person served

        Returns:
            New lever with updated marginal_cost_per_person
        """
        return ExpandCoverageLever(
            name=self.name,
            coverage_increase=self.theta,
            marginal_cost_per_person=marginal_cost,
        )

    def compute_cost(self, problem: 'AllocationProblem') -> float:
        n_additional = int(self.theta * problem.data.n)
        return n_additional * self.marginal_cost_per_person

    def for_budget(self, budget: float, problem: 'AllocationProblem') -> 'ExpandCoverageLever':
        """Return lever with coverage_increase adjusted to match budget.

        Inverts: budget = theta * n * marginal_cost
        So: theta = budget / (n * marginal_cost)

        At budget=0, returns lever with theta=0 (no additional coverage).
        """
        if budget < 0:
            raise ValueError(f"Budget must be non-negative, got {budget}")

        if budget == 0:
            return self.with_theta(0.0)

        n = problem.data.n
        theta = budget / (n * self.marginal_cost_per_person)

        new_lever = self.with_theta(theta)
        new_lever.name = self.name
        return new_lever

    def apply(self, problem: 'AllocationProblem') -> 'AllocationProblem':
        """Return new problem with expanded coverage constraint.

        Args:
            problem: Allocation problem with CoverageConstraint

        Returns:
            New AllocationProblem with expanded constraint
        """
        from ..problem import AllocationProblem

        if not isinstance(problem.constraint, CoverageConstraint):
            raise TypeError(
                f"ExpandCoverageLever requires CoverageConstraint, "
                f"got {type(problem.constraint).__name__}"
            )

        n_additional = int(self.theta * problem.data.n)
        new_capacity = problem.constraint.get_capacity() + n_additional

        new_constraint = CoverageConstraint(max_coverage=new_capacity)

        return AllocationProblem(
            data=problem.data,
            utility=problem.utility,
            constraint=new_constraint,
            policy=problem.policy,
        )

    def __repr__(self):
        return (
            f"ExpandCoverageLever(name='{self.name}', "
            f"coverage_increase={self.theta}, "
            f"marginal_cost={self.marginal_cost_per_person})"
        )
