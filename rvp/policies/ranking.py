"""Ranking policy for allocation problems."""

from typing import Optional
import numpy as np
from .base import Policy
from ..utilities import UtilityFunction
from ..constraints import ResourceConstraint


class RankingPolicy(Policy):
    """Policy that ranks units and allocates to top k.

    Can rank by:
    - Predictions directly (default)
    - Expected utility (using utility function with predictions)

    Can filter by:
    - Minimum prediction threshold
    - Minimum expected utility threshold

    Examples:
        # Simple: rank by predictions, take top k (highest first)
        policy = RankingPolicy()

        # Poverty targeting: allocate to lowest predicted consumption
        policy = RankingPolicy(ascending=True)

        # Rank by expected utility
        policy = RankingPolicy(rank_by='utility')

        # Only allocate if prediction >= 0.5
        policy = RankingPolicy(min_prediction=0.5)

        # Only allocate if expected utility > 0
        policy = RankingPolicy(rank_by='utility', min_utility=0.0)
    """

    def __init__(
        self,
        rank_by: str = 'prediction',
        ascending: bool = False,
        min_prediction: float | None = None,
        min_utility: float | None = None,
    ):
        """Initialize ranking policy.

        Args:
            rank_by: 'prediction' or 'utility' - what to rank units by
            ascending: If True, rank lowest first (e.g., for poverty targeting where
                      lower consumption = higher priority). Default False (highest first).
            min_prediction: If set, only allocate to units with prediction >= this value
            min_utility: If set, only allocate to units with expected utility >= this value
        """
        if rank_by not in ['prediction', 'utility']:
            raise ValueError(f"rank_by must be 'prediction' or 'utility', got {rank_by}")

        self.rank_by = rank_by
        self.ascending = ascending
        self.min_prediction = min_prediction
        self.min_utility = min_utility

    def __call__(
        self,
        predictions: np.ndarray,
        constraint: ResourceConstraint,
        utility: Optional[UtilityFunction] = None,
    ) -> np.ndarray:
        """Compute allocation given predictions and constraints.

        Ranks eligible units by score, then greedily adds them until
        the constraint capacity is exhausted.

        Args:
            predictions: Predicted outcomes of shape (n,)
            constraint: Resource constraint to satisfy
            utility: Utility function (required if rank_by='utility' or min_utility is set)

        Returns:
            Binary actions array
        """
        n = len(predictions)

        # Compute scores to rank by
        if self.rank_by == 'utility':
            if utility is None:
                raise ValueError("utility is required when rank_by='utility'")
            scores = utility.compute(predictions, np.ones(n, dtype=int))
        else:
            scores = predictions

        # Build mask of eligible units
        eligible = np.ones(n, dtype=bool)

        if self.min_prediction is not None:
            eligible &= predictions >= self.min_prediction

        if self.min_utility is not None:
            if utility is None:
                raise ValueError("utility is required when min_utility is set")
            expected_utilities = utility.compute(predictions, np.ones(n, dtype=int))
            eligible &= expected_utilities >= self.min_utility

        # Initialize actions
        actions = np.zeros(n, dtype=int)

        if not np.any(eligible):
            return actions

        # Get per-unit costs and capacity
        unit_costs = constraint.get_unit_costs(n)
        capacity = constraint.get_capacity()

        # Get eligible indices sorted by score
        eligible_indices = np.where(eligible)[0]
        eligible_scores = scores[eligible_indices]
        sorted_order = np.argsort(eligible_scores)
        if not self.ascending:
            sorted_order = sorted_order[::-1]  # descending (highest first)
        ranked_indices = eligible_indices[sorted_order]

        # Check if uniform costs (can use fast path)
        if np.all(unit_costs == unit_costs[0]):
            # Uniform costs: just take top k
            unit_cost = unit_costs[0]
            k = int(capacity // unit_cost)
            k = min(k, len(ranked_indices))
            actions[ranked_indices[:k]] = 1
        else:
            # Heterogeneous costs: greedy selection
            current_cost = 0.0
            for idx in ranked_indices:
                if current_cost + unit_costs[idx] <= capacity:
                    actions[idx] = 1
                    current_cost += unit_costs[idx]

        return actions

    def __repr__(self) -> str:
        parts = [f"rank_by='{self.rank_by}'"]
        if self.ascending:
            parts.append("ascending=True")
        if self.min_prediction is not None:
            parts.append(f"min_prediction={self.min_prediction}")
        if self.min_utility is not None:
            parts.append(f"min_utility={self.min_utility}")
        return f"RankingPolicy({', '.join(parts)})"
