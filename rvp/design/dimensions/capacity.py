"""Capacity (alpha) dimension."""

from typing import Tuple

from ...constraints import CoverageConstraint
from ...problem import AllocationProblem
from ..dimension import DesignDimension


class CapacityDimension(DesignDimension):
    """Allocation capacity as a fraction of the population (alpha).

    Stateless. ``at(problem, theta)`` returns a new AllocationProblem with
    a fresh ``CoverageConstraint(max_coverage=theta, population_size=N)``.
    Any prior constraint on ``problem`` is fully overwritten.

    Parameters
    ----------
    bounds : tuple[float, float]
        (low, high) coverage fractions used when constructing grids.
    name : str
        Human-readable name (default: "capacity").
    """

    target_component = "constraint"

    def __init__(
        self,
        bounds: Tuple[float, float] = (0.01, 0.50),
        name: str = "capacity",
    ):
        self.bounds_ = bounds
        self.name = name

    def at(self, problem: AllocationProblem, theta: float) -> AllocationProblem:
        alpha = float(theta)
        if not 0 < alpha <= 1:
            raise ValueError(f"CapacityDimension theta must be in (0, 1], got {theta}")

        return AllocationProblem(
            data=problem.data,
            utility=problem.utility,
            constraint=CoverageConstraint(
                max_coverage=alpha,
                population_size=problem.data.n,
            ),
            policy=problem.policy,
        )

    @property
    def bounds(self) -> Tuple[float, float]:
        return self.bounds_
