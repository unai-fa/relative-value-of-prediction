"""CRRA benefit (b) dimension."""

from typing import Tuple

from ...problem import AllocationProblem
from ...utilities import CRRAUtility
from ..dimension import DesignDimension


class BenefitDimension(DesignDimension):
    """Transfer benefit ``b`` for a CRRA utility.

    Stateless. ``at(problem, theta)`` returns a new AllocationProblem whose
    utility is ``CRRAUtility(b=theta, rho=problem.utility.rho)``. ``problem``
    must have a ``CRRAUtility``; the rho is preserved.

    Parameters
    ----------
    bounds : tuple[float, float]
        (low, high) benefit values used when constructing grids.
    name : str
    """

    target_component = "utility"

    def __init__(
        self,
        bounds: Tuple[float, float] = (1.0, 1000.0),
        name: str = "benefit",
    ):
        self.bounds_ = bounds
        self.name = name

    def at(self, problem: AllocationProblem, theta: float) -> AllocationProblem:
        if not isinstance(problem.utility, CRRAUtility):
            raise TypeError(
                f"BenefitDimension expects CRRAUtility, got {type(problem.utility).__name__}"
            )
        return AllocationProblem(
            data=problem.data,
            utility=CRRAUtility(b=float(theta), rho=problem.utility.rho),
            constraint=problem.constraint,
            policy=problem.policy,
        )

    @property
    def bounds(self) -> Tuple[float, float]:
        return self.bounds_
