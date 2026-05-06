"""Allocation problem definition."""

from collections.abc import Callable
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np

from .data import AllocationData
from .utilities import UtilityFunction
from .constraints import ResourceConstraint

if TYPE_CHECKING:
    from .policies import Policy


class AllocationProblem:
    """Defines an allocation problem with data, utility, constraint, and policy.

    An allocation problem combines:
    - Data (features, outcomes, predictions)
    - A utility function (how to value outcomes)
    - Resource constraint (limits on actions)
    - Policy (how to allocate given predictions)

    When data contains multiple datasets, evaluate() averages results across them.
    """

    def __init__(
        self,
        data: AllocationData,
        utility: UtilityFunction,
        constraint: ResourceConstraint,
        policy: 'Policy',
    ):
        """Initialize an allocation problem.

        Args:
            data: The allocation data (features, outcomes, predictions)
            utility: The utility function to optimize
            constraint: Resource constraint to satisfy
            policy: Allocation policy
        """
        self.data = data
        self.utility = utility
        self.constraint = constraint
        self.policy = policy

    def evaluate(self, subgroup_mask: Optional[np.ndarray | Callable] = None) -> Dict[str, Any]:
        """Evaluate the problem's policy.

        If the data has multiple datasets, evaluates each and returns
        averaged results.

        Args:
            subgroup_mask: Optional boolean mask or callable to compute welfare only for
                          a subgroup. If callable, it is called as
                          ``subgroup_mask(df)`` for each dataset DataFrame and must
                          return a boolean mask. If provided, utility metrics are
                          computed only for masked individuals.

        Returns:
            Dictionary containing:
                - total_utility: Total utility achieved (averaged if multiple datasets)
                - mean_utility: Mean utility per unit
                - n_allocated: Number of units allocated to
                - actions: The actions taken (None if multiple datasets)
        """
        results = []
        for i in range(self.data.n_datasets):
            results.append(self._evaluate_single(i, subgroup_mask))

        if self.data.n_datasets == 1:
            return results[0]

        return self._average_results(results)

    def _evaluate_single(
        self,
        dataset_index: int = 0,
        subgroup_mask: Optional[np.ndarray | Callable] = None,
    ) -> Dict[str, Any]:
        """Evaluate policy for a single dataset.

        Args:
            dataset_index: Index of dataset to evaluate.
            subgroup_mask: Optional boolean mask or callable for subgroup evaluation.
        """
        dataset = self.data.get_dataset(dataset_index)
        resolved_subgroup_mask = self._resolve_subgroup_mask(subgroup_mask, dataset)

        # Get actions from policy
        actions = self.policy(dataset.predictions, self.constraint, self.utility)

        # Validate actions
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)

        if actions.shape[0] != dataset.n:
            raise ValueError(
                f"Policy returned {actions.shape[0]} actions but data has {dataset.n} samples"
            )

        if not np.all((actions == 0) | (actions == 1)):
            raise ValueError("Actions must be binary (0 or 1)")

        if not self.constraint.is_feasible(actions):
            raise ValueError(
                f"Actions are not feasible: cost={self.constraint.get_cost(actions):.2f}, "
                f"capacity={self.constraint.get_capacity():.2f}"
            )

        # Compute utilities using TRUE outcomes
        utilities = self.utility.compute(dataset.y, actions)

        # Filter to subgroup if mask provided
        if resolved_subgroup_mask is not None:
            utilities = utilities[resolved_subgroup_mask]
            actions_for_count = actions[resolved_subgroup_mask]
        else:
            actions_for_count = actions

        total_utility = float(np.sum(utilities))

        # Compute normalized utility (ratio to random allocation)
        # Shows how many times better than random: 1.0 = same as random, 1.5 = 50% better
        # Only available with constant unit costs
        try:
            self._check_unit_costs(dataset.n)
            random_result = self._evaluate_random_single(dataset_index, resolved_subgroup_mask)
            random_utility = random_result['total_utility']
            utility_ratio = total_utility / random_utility if random_utility != 0 else None
        except ValueError:
            utility_ratio = None

        return {
            "total_utility": total_utility,
            "mean_utility": float(np.mean(utilities)),
            "utility_ratio": utility_ratio,
            "n_allocated": int(np.sum(actions_for_count == 1)),
            "actions": actions,
        }

    def _average_results(self, results: list) -> Dict[str, Any]:
        """Average results from multiple dataset evaluations."""
        n = len(results)
        avg_total = sum(r["total_utility"] for r in results) / n
        avg_mean = sum(r["mean_utility"] for r in results) / n
        avg_allocated = sum(r["n_allocated"] for r in results) / n

        # Handle utility_ratio (may be None)
        norm_values = [r["utility_ratio"] for r in results if r["utility_ratio"] is not None]
        avg_normalized = sum(norm_values) / len(norm_values) if norm_values else None

        return {
            "total_utility": avg_total,
            "mean_utility": avg_mean,
            "utility_ratio": avg_normalized,
            "n_allocated": avg_allocated,
            "actions": None,  # Actions differ per dataset
        }

    def _check_unit_costs(self, n: int):
        """Check that all unit costs are equal (constant costs assumption)."""
        unit_costs = self.constraint.get_unit_costs(n)
        if not np.allclose(unit_costs, unit_costs[0]):
            raise ValueError(
                "evaluate_random only supports constant unit costs. "
                f"Got varying costs: min={unit_costs.min()}, max={unit_costs.max()}"
            )
        return unit_costs[0]

    def evaluate_random(self, subgroup_mask: Optional[np.ndarray | Callable] = None) -> Dict[str, Any]:
        """Evaluate expected welfare under random allocation.

        Computes expected utility analytically: each individual has
        probability p = n_to_allocate / n of being allocated.

        Only supports constant unit costs.

        If data has multiple datasets, averages results across them.

        Args:
            subgroup_mask: Optional boolean mask or callable to compute welfare only for
                          a subgroup. If callable, it is called as
                          ``subgroup_mask(df)`` for each dataset DataFrame and must
                          return a boolean mask.

        Returns:
            Dictionary with total_utility, mean_utility, n_allocated
        """
        results = []
        for i in range(self.data.n_datasets):
            results.append(self._evaluate_random_single(i, subgroup_mask))

        if self.data.n_datasets == 1:
            return results[0]

        return self._average_results(results)

    def _evaluate_random_single(
        self,
        dataset_index: int = 0,
        subgroup_mask: Optional[np.ndarray | Callable] = None,
    ) -> Dict[str, Any]:
        """Evaluate random allocation for a single dataset."""
        dataset = self.data.get_dataset(dataset_index)
        resolved_subgroup_mask = self._resolve_subgroup_mask(subgroup_mask, dataset)
        y = dataset.y
        n = len(y)

        unit_cost = self._check_unit_costs(n)

        n_to_allocate = int(self.constraint.get_capacity() / unit_cost)
        p = n_to_allocate / n  # probability of being allocated

        # Expected utility per individual
        u_if_allocated = self.utility.compute(y, np.ones(n))
        u_if_not_allocated = self.utility.compute(y, np.zeros(n))
        expected_utilities = p * u_if_allocated + (1 - p) * u_if_not_allocated

        if resolved_subgroup_mask is not None:
            expected_utilities = expected_utilities[resolved_subgroup_mask]

        total_utility = float(np.sum(expected_utilities))
        n_subgroup = np.sum(resolved_subgroup_mask) if resolved_subgroup_mask is not None else n

        return {
            "total_utility": total_utility,
            "mean_utility": total_utility / n_subgroup,
            "utility_ratio": None,
            "n_allocated": n_to_allocate,
            "actions": None,
        }

    def _resolve_subgroup_mask(
        self,
        subgroup_mask: Optional[np.ndarray | Callable],
        dataset: AllocationData,
    ) -> Optional[np.ndarray]:
        """Return a validated boolean subgroup mask for one dataset."""
        if subgroup_mask is None:
            return None

        if callable(subgroup_mask):
            subgroup_mask = subgroup_mask(dataset.df_single)

        mask = np.asarray(subgroup_mask)
        if mask.ndim != 1:
            raise ValueError(
                f"subgroup_mask must be one-dimensional, got shape {mask.shape}"
            )
        if mask.shape[0] != dataset.n:
            raise ValueError(
                f"subgroup_mask has length {mask.shape[0]} but data has {dataset.n} samples"
            )
        if mask.dtype != bool:
            raise ValueError("subgroup_mask must be boolean")
        if not np.any(mask):
            raise ValueError("subgroup_mask selects no individuals")

        return mask
