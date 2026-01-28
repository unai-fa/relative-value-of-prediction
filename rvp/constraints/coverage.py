"""Coverage constraint for limiting number of units that can be targeted."""

import numpy as np
from typing import Optional
from .base import ResourceConstraint


class CoverageConstraint(ResourceConstraint):
    """Constraint on maximum number of units that can be targeted.

    All units have uniform cost of 1. Capacity is the max number of units.

    The max_coverage can be specified as:
        - Float in (0, 1): Fraction of population (requires population_size)
        - Int >= 1: Absolute number of units
    """

    def __init__(
        self,
        max_coverage: float,
        population_size: Optional[int] = None,
    ):
        """Initialize coverage constraint.

        Args:
            max_coverage: Maximum coverage, either:
                - Float in (0, 1) for fractional coverage
                - Int >= 1 for absolute coverage
            population_size: Population size, required if max_coverage is fractional
        """
        if isinstance(max_coverage, float) and 0 < max_coverage < 1:
            if population_size is None:
                raise ValueError(
                    f"population_size is required when max_coverage is fractional (got {max_coverage})"
                )
            self._capacity = int(max_coverage * population_size)
        elif isinstance(max_coverage, (int, float)) and max_coverage >= 1:
            self._capacity = int(max_coverage)
        else:
            raise ValueError(
                f"max_coverage must be either a float in (0, 1) or an int >= 1, got {max_coverage}"
            )

    def get_capacity(self) -> int:
        """Get maximum number of units that can be targeted."""
        return self._capacity

    def get_unit_costs(self, n: int) -> np.ndarray:
        """All units have uniform cost of 1."""
        return np.ones(n)
