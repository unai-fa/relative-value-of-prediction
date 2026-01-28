"""Utility modification lever."""

from typing import TYPE_CHECKING
from .base import ParameterizedLever

if TYPE_CHECKING:
    from ..problem import AllocationProblem


class UtilityValueLever(ParameterizedLever):
    """Lever that modifies a specific value in PartitionedUtility.

    theta = the new value for the specified index

    Can represent:
    - Uncertainty over utility structure (sensitivity analysis)
    - Policy changes that affect treatment effectiveness or harm

    Example:
        # Modify harm parameter (index 0) in PartitionedUtility([0.25, 0.75], [-1, 0, 1])
        lever = UtilityValueLever(
            name="Harm sensitivity",
            value_index=0,
            new_value=-2,  # theta = -2, changing harm from -1 to -2
        )
    """

    def __init__(
        self,
        name: str,
        value_index: int,
        new_value: float,
    ):
        """Initialize utility value lever.

        Args:
            name: Identifier for this lever
            value_index: Index of value to modify in utility.values
            new_value: New value to set (this is theta)
        """
        super().__init__(name, theta=new_value)
        self.value_index = value_index

    @property
    def new_value(self) -> float:
        """Alias for theta."""
        return self.theta

    def with_theta(self, theta: float) -> 'UtilityValueLever':
        """Return new lever with different value."""
        return UtilityValueLever(
            name=self.name,
            value_index=self.value_index,
            new_value=theta,
        )

    def apply(self, problem: 'AllocationProblem') -> 'AllocationProblem':
        """Return new problem with modified utility.

        Args:
            problem: Allocation problem with PartitionedUtility

        Returns:
            New AllocationProblem with modified utility values
        """
        from ..problem import AllocationProblem
        from ..utilities import PartitionedUtility

        if not isinstance(problem.utility, PartitionedUtility):
            raise TypeError(
                f"UtilityValueLever requires PartitionedUtility, "
                f"got {type(problem.utility).__name__}"
            )

        # Create new values array with modified value
        new_values = list(problem.utility.values)
        new_values[self.value_index] = self.theta

        new_utility = PartitionedUtility(
            thresholds=problem.utility.thresholds,
            values=new_values,
            threshold_type=problem.utility.threshold_type,
        )

        return AllocationProblem(
            data=problem.data,
            utility=new_utility,
            constraint=problem.constraint,
            policy=problem.policy,
        )

    def __repr__(self):
        return (
            f"UtilityValueLever(name='{self.name}', "
            f"value_index={self.value_index}, "
            f"new_value={self.theta})"
        )
