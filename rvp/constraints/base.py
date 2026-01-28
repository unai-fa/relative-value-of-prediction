from abc import ABC, abstractmethod
import numpy as np


class ResourceConstraint(ABC):
    """Abstract base class for resource constraints.

    A resource constraint defines limits on what actions can be taken,
    typically modeling budget and capacity limitations.

    Subclasses must implement:
    - get_capacity(): Maximum total cost allowed
    - get_unit_costs(n): Per-unit costs (can depend on data/features)

    Default implementations of is_feasible() and get_cost() use these.
    """

    @abstractmethod
    def get_capacity(self) -> float:
        """Get the maximum allowed cost/resource capacity."""
        pass

    @abstractmethod
    def get_unit_costs(self, n: int) -> np.ndarray:
        """Get per-unit costs for all units.

        Args:
            n: Number of units

        Returns:
            Array of shape (n,) with cost for each unit
        """
        pass

    def get_cost(self, actions: np.ndarray) -> float:
        """Compute the total cost of actions.

        Args:
            actions: Binary actions array

        Returns:
            Total cost (sum of unit costs where action=1)
        """
        unit_costs = self.get_unit_costs(len(actions))
        return float(np.sum(actions * unit_costs))

    def is_feasible(self, actions: np.ndarray) -> bool:
        """Check if actions satisfy the constraint.

        Args:
            actions: Binary actions array

        Returns:
            True if total cost <= capacity
        """
        return self.get_cost(actions) <= self.get_capacity()
