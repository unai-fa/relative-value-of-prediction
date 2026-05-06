"""Design dimension abstract base class.

A design dimension is one axis of the design space. For each value of
its parameter (theta), it returns an `AllocationProblem` that has been
modified along this axis.
"""

from abc import ABC, abstractmethod
from typing import Literal, Tuple

import numpy as np

from ..problem import AllocationProblem


# A dimension declares which component of the AllocationProblem it modifies.
# The DesignSpace uses this to enforce that no two dimensions in the same
# space modify the same component (avoids ordering ambiguity).
TargetComponent = Literal["data", "utility", "constraint", "policy"]


class DesignDimension(ABC):
    """One axis of a design space.

    Subclasses implement `at(problem, theta)` to return a new
    AllocationProblem with this dimension's modification applied.

    Attributes
    ----------
    name : str
        Human-readable identifier.
    target_component : {'data', 'utility', 'constraint', 'policy'}
        Which part of the AllocationProblem this dimension modifies.
        Used by DesignSpace to enforce one-dim-per-component.
    """

    name: str
    target_component: TargetComponent

    @abstractmethod
    def at(self, problem: AllocationProblem, theta) -> AllocationProblem:
        """Return the AllocationProblem at theta along this dimension."""

    @property
    def is_discrete(self) -> bool:
        """Whether this dimension takes only a finite set of values."""
        return False

    @property
    def bounds(self) -> Tuple[float, float]:
        """Continuous domain for this dimension.

        Continuous dimensions should override this. Discrete dimensions should
        override ``values`` instead.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define continuous bounds")

    @property
    def values(self) -> np.ndarray:
        """Admissible theta values for a discrete dimension."""
        raise NotImplementedError(f"{type(self).__name__} does not define discrete values")

    def grid(self, n: int = 50) -> np.ndarray:
        """Return the theta values to use for a sweep or surface.

        Continuous dimensions are evaluated on ``n`` evenly spaced points over
        their bounds. Discrete dimensions ignore ``n`` and return their exact
        admissible values.
        """
        if self.is_discrete:
            return np.asarray(self.values)
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        low, high = self.bounds
        return np.linspace(float(low), float(high), n)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, target={self.target_component!r})"
