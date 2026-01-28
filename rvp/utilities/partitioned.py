"""Utility function based on partitioning the outcome space."""

import numpy as np
from typing import List
from .base import UtilityFunction


class PartitionedUtility(UtilityFunction):
    """Utility function that assigns fixed values based on outcome partitions.

    Partitions the outcome space into bins and assigns a fixed utility value
    when acting on units in each bin. No action gives utility 0.

    Examples:
        # Two bins (like old TopPercentileUtility): target top 10%
        utility = PartitionedUtility(
            thresholds=[0.9],        # 90th percentile
            values=[-0.5, 1.0],      # penalty below, reward above
        )

        # Three bins with absolute thresholds
        utility = PartitionedUtility(
            thresholds=[30, 70],
            values=[-5, 0, 10],
            threshold_type='absolute'
        )
    """

    def __init__(
        self,
        thresholds: List[float],
        values: List[float],
        threshold_type: str = 'percentile',
    ):
        """Initialize partitioned utility function.

        Args:
            thresholds: k threshold values defining k+1 bins (must be sorted ascending)
                       - If 'percentile': values in [0,1] (e.g., 0.9 = 90th percentile)
                       - If 'absolute': actual outcome values
            values: k+1 utility values, one for each bin
            threshold_type: 'percentile' or 'absolute'
        """
        if len(values) != len(thresholds) + 1:
            raise ValueError(
                f"Need {len(thresholds)+1} values for {len(thresholds)} thresholds, "
                f"got {len(values)} values"
            )

        if threshold_type not in ['percentile', 'absolute']:
            raise ValueError(f"threshold_type must be 'percentile' or 'absolute', got {threshold_type}")

        if threshold_type == 'percentile':
            if not all(0 <= t <= 1 for t in thresholds):
                raise ValueError("Percentile thresholds must be in [0, 1]")

        if thresholds != sorted(thresholds):
            raise ValueError("Thresholds must be sorted in ascending order")

        self.thresholds = thresholds
        self.values = np.array(values)
        self.threshold_type = threshold_type

    def compute(self, y: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute utilities for each unit.

        Args:
            y: Ground truth outcomes of shape (n,)
            actions: Binary actions (0 or 1) of shape (n,)

        Returns:
            Array of utilities of shape (n,)
        """
        # Get absolute threshold values
        if self.threshold_type == 'absolute':
            abs_thresholds = np.array(self.thresholds)
        else:
            abs_thresholds = np.percentile(y, [t * 100 for t in self.thresholds])

        # Find bin for each outcome
        bin_indices = np.searchsorted(abs_thresholds, y, side='right')

        # Utilities: values[bin] if action=1, else 0
        utilities = np.where(actions == 1, self.values[bin_indices], 0.0)

        return utilities

    def __repr__(self) -> str:
        return (
            f"PartitionedUtility(thresholds={self.thresholds}, "
            f"values={list(self.values)}, threshold_type='{self.threshold_type}')"
        )
