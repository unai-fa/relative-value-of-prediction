"""Policy lever base classes."""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..problem import AllocationProblem


class PolicyLever(ABC):
    """Abstract base class for policy levers.

    A lever modifies an allocation problem (data, utility, or constraint)
    to represent a policy intervention.
    """

    def __init__(self, name: str, cost: Optional[float] = None):
        """Initialize a policy lever.

        Args:
            name: Short identifier for the lever
            cost: Fixed cost of applying this lever (None if not specified)
        """
        self.name = name
        self.cost = cost

    @abstractmethod
    def apply(self, problem: 'AllocationProblem') -> 'AllocationProblem':
        """Apply lever to an allocation problem.

        Args:
            problem: The allocation problem to modify

        Returns:
            A new AllocationProblem with modified components
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class ParameterizedLever(PolicyLever):
    """A lever with an intensity parameter theta.

    The cost can be:
    - None (not available)
    - A fixed value (via self.cost)
    - Computed from theta (override compute_cost)
    """

    def __init__(self, name: str, theta: float, cost: Optional[float] = None):
        """Initialize a parameterized lever.

        Args:
            name: Short identifier for the lever
            theta: Intensity parameter
            cost: Fixed cost (for levers where cost doesn't depend on theta)
        """
        super().__init__(name, cost=cost)
        self.theta = theta

    @abstractmethod
    def with_theta(self, theta: float) -> 'ParameterizedLever':
        """Return a new lever with a different theta value.

        Args:
            theta: New intensity parameter

        Returns:
            New lever instance with updated theta
        """
        pass

    def compute_cost(self, problem: 'AllocationProblem') -> Optional[float]:
        """Compute cost for current theta.

        Override this if cost mapping is available.
        Default returns None.
        """
        return None

    def for_budget(self, budget: float, problem: 'AllocationProblem') -> 'ParameterizedLever':
        """Return lever with theta adjusted to match the given budget.

        Only available if cost can be computed from theta and inverted.

        Args:
            budget: Target budget
            problem: Allocation problem for context

        Returns:
            New lever with theta set to achieve the target budget

        Raises:
            NotImplementedError: If this lever doesn't support budget adjustment
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support budget adjustment"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', theta={self.theta})"
