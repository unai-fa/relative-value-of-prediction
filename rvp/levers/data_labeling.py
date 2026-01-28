"""Data labeling lever for controlling share of labeled data."""

from typing import TYPE_CHECKING, Optional, Union, List
import numpy as np
from .base import ParameterizedLever

if TYPE_CHECKING:
    from ..problem import AllocationProblem


class DataLabelingLever(ParameterizedLever):
    """Lever that controls the share of test-time data that is labeled (usable for targeting).

    theta = label_share (fraction of individuals with usable predictions)

    Unlabeled individuals have their predictions set to max + offset + noise
    (or min - offset - noise for descending) to deprioritize them in ranking.
    This simulates the real-world constraint that you can only target
    individuals for whom you have data.

    Two modes:
    - Setting mode (marginal=False): theta is the absolute label share
    - Marginal mode (marginal=True): theta is the INCREMENT in label share above baseline

    Cost model:
    - Setting mode: cost = n * label_share * cost_per_label (total cost)
    - Marginal mode: cost = n * theta * cost_per_label (incremental cost only)

    IMPORTANT: If full_predictions is provided, this lever stores its own prediction
    source and will OVERRIDE the problem's predictions when apply() is called.
    This means it cannot be composed with other levers that modify predictions -
    it will always use its stored full_predictions as the source, masked by theta.

    Example:
        # Setting mode: 50% of individuals have labels, $10 per label
        lever = DataLabelingLever(
            name="Data collection",
            label_share=0.5,
            cost_per_label=10.0,
            ascending=True,
        )

        # Marginal mode: baseline is 70% labeled, theta is additional share
        lever = DataLabelingLever(
            name="Data collection",
            label_share=0.1,  # theta = 10% increment -> final = 80%
            cost_per_label=10.0,
            baseline_share=0.7,
            marginal=True,
            full_predictions=full_preds,
        )
    """

    def __init__(
        self,
        name: str,
        label_share: float,
        cost_per_label: float = 1.0,
        ascending: bool = True,
        seed: int = 42,
        full_predictions: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        baseline_share: float = 0.0,
        marginal: bool = False,
    ):
        """Initialize data labeling lever.

        Args:
            name: Identifier for this lever
            label_share: Fraction of individuals with labels (0 to 1), this is theta.
                In marginal mode, this is the INCREMENT above baseline_share.
            cost_per_label: Cost to label each individual
            ascending: If True, unlabeled get inf (deprioritized when ranking ascending,
                      i.e., poverty targeting). If False, unlabeled get -inf.
            seed: Random seed for reproducible label selection
            full_predictions: Optional predictions at 100% labeling. Can be:
                - None: Use problem's predictions as source (current behavior)
                - np.ndarray: Single dataset predictions
                - List[np.ndarray]: Multi-dataset predictions (one per dataset)
                When provided, apply() will use these as the source and override
                the problem's predictions. This enables bidirectional labeling changes.
            baseline_share: Baseline label share (used in marginal mode)
            marginal: If True, theta is interpreted as increment above baseline_share.
                If False (default), theta is the absolute label share.
        """
        # Validate based on mode
        if marginal:
            effective_share = baseline_share + label_share
            if not 0 <= effective_share <= 1:
                raise ValueError(
                    f"baseline_share + label_share must be in [0, 1], "
                    f"got {baseline_share} + {label_share} = {effective_share}"
                )
            if label_share < 0:
                raise ValueError(f"In marginal mode, label_share (increment) must be >= 0, got {label_share}")
        else:
            if not 0 <= label_share <= 1:
                raise ValueError(f"label_share must be in [0, 1], got {label_share}")

        # Normalize full_predictions to list format
        if full_predictions is not None:
            if isinstance(full_predictions, np.ndarray):
                self._full_predictions = [full_predictions]
            else:
                self._full_predictions = list(full_predictions)
        else:
            self._full_predictions = None

        super().__init__(name, theta=label_share)
        self.cost_per_label = cost_per_label
        self.ascending = ascending
        self.seed = seed
        self.baseline_share = baseline_share
        self.marginal = marginal

    @property
    def label_share(self) -> float:
        """Alias for theta (in marginal mode, this is the increment)."""
        return self.theta

    @property
    def effective_share(self) -> float:
        """The actual label share that will be applied.

        In setting mode: same as theta
        In marginal mode: baseline_share + theta
        """
        if self.marginal:
            return self.baseline_share + self.theta
        return self.theta

    @property
    def full_predictions(self) -> Optional[List[np.ndarray]]:
        """Full predictions at 100% labeling (if stored)."""
        return self._full_predictions

    def with_theta(self, theta: float) -> 'DataLabelingLever':
        """Return new lever with different label_share (preserves mode and baseline)."""
        return DataLabelingLever(
            name=self.name,
            label_share=theta,
            cost_per_label=self.cost_per_label,
            ascending=self.ascending,
            seed=self.seed,
            full_predictions=self._full_predictions,
            baseline_share=self.baseline_share,
            marginal=self.marginal,
        )

    def compute_cost(self, problem: 'AllocationProblem') -> float:
        """Compute labeling cost.

        In setting mode: total cost = n * label_share * cost_per_label
        In marginal mode: incremental cost = n * theta * cost_per_label
            (theta is the increment above baseline)
        """
        n = problem.data.n
        if self.marginal:
            # Incremental cost: only the additional labels
            n_incremental = int(n * self.theta)
            return n_incremental * self.cost_per_label
        else:
            # Total cost
            n_labeled = int(n * self.theta)
            return n_labeled * self.cost_per_label

    def for_budget(self, budget: float, problem: 'AllocationProblem') -> 'DataLabelingLever':
        """Return lever with label_share adjusted to match budget.

        In setting mode: theta = budget / (n * cost_per_label), capped at 1.0
        In marginal mode: theta = budget / (n * cost_per_label), capped so
            effective_share = baseline_share + theta <= 1.0

        Args:
            budget: Target budget for labeling
            problem: Allocation problem for context (to get n)

        Returns:
            New lever with adjusted theta
        """
        n = problem.data.n
        cost_per_share = n * self.cost_per_label  # cost to go from 0% to 100%

        if self.marginal:
            # How much additional share can we afford?
            affordable_increment = budget / cost_per_share
            # Cap so we don't exceed 100% total
            max_increment = 1.0 - self.baseline_share
            theta = min(affordable_increment, max_increment)
        else:
            # Setting mode: theta is absolute share
            if budget >= cost_per_share:
                theta = 1.0
            else:
                theta = budget / cost_per_share

        return self.with_theta(theta)

    def _create_label_mask(self, n: int) -> np.ndarray:
        """Create random mask for which individuals have labels.

        Uses a fixed random ordering based on seed, then selects the first
        n * effective_share individuals. This ensures monotonicity: individuals
        labeled at share=0.3 are a subset of those labeled at share=0.5.

        Args:
            n: Number of individuals

        Returns:
            Boolean array of shape (n,) where True = has label
        """
        # Create a fixed random ordering of individuals based on seed
        rng = np.random.default_rng(self.seed)
        ordering = rng.permutation(n)

        # Use effective_share (baseline + theta in marginal mode, theta in setting mode)
        n_labeled = int(n * self.effective_share)

        # Label the first n_labeled individuals in the random ordering
        mask = np.zeros(n, dtype=bool)
        mask[ordering[:n_labeled]] = True

        return mask

    def apply(self, problem: 'AllocationProblem') -> 'AllocationProblem':
        """Return new problem with masked predictions for unlabeled individuals.

        If full_predictions was provided at init, uses those as the source
        (overriding whatever is in problem.data). Otherwise uses problem's
        current predictions.

        Args:
            problem: Allocation problem

        Returns:
            New AllocationProblem with modified predictions
        """
        from ..problem import AllocationProblem
        from ..data import AllocationData

        # Validate full_predictions count matches datasets if provided
        if self._full_predictions is not None:
            if len(self._full_predictions) != problem.data.n_datasets:
                raise ValueError(
                    f"full_predictions has {len(self._full_predictions)} arrays but "
                    f"problem has {problem.data.n_datasets} datasets"
                )

        # Handle multi-dataset case
        new_dfs = []
        for i in range(problem.data.n_datasets):
            dataset = problem.data.get_dataset(i)
            df = dataset.df_single.copy()

            # Create label mask
            mask = self._create_label_mask(len(df))

            # Get source predictions: stored full_predictions or problem's predictions
            if self._full_predictions is not None:
                source_predictions = self._full_predictions[i].copy()
            else:
                source_predictions = df[problem.data.predictions_col].values.copy()

            predictions = source_predictions.copy()
            n_unlabeled = np.sum(~mask)

            if n_unlabeled > 0:
                # Use seeded RNG for reproducible noise
                rng = np.random.default_rng(self.seed + i)
                noise = rng.normal(0, 1e-6, n_unlabeled)

                if self.ascending:
                    # Unlabeled get max + 5 + noise (deprioritized when sorting ascending)
                    base_value = np.max(predictions[mask]) + 5 if np.any(mask) else 5
                    predictions[~mask] = base_value + noise
                else:
                    # Unlabeled get min - 5 - noise (deprioritized when sorting descending)
                    base_value = np.min(predictions[mask]) - 5 if np.any(mask) else -5
                    predictions[~mask] = base_value + noise

            df[problem.data.predictions_col] = predictions
            new_dfs.append(df)

        # Create new data with modified predictions
        new_data = AllocationData(
            df=new_dfs if len(new_dfs) > 1 else new_dfs[0],
            covariate_cols=problem.data.covariate_cols,
            ground_truth_col=problem.data.ground_truth_col,
            predictions_col=problem.data.predictions_col,
        )

        return AllocationProblem(
            data=new_data,
            utility=problem.utility,
            constraint=problem.constraint,
            policy=problem.policy,
        )

    @classmethod
    def from_data(
        cls,
        data,  # AllocationData
        name: str = "Data labeling",
        label_share: float = 1.0,
        cost_per_label: float = 1.0,
        ascending: bool = True,
        seed: int = 42,
        baseline_share: float = 0.0,
        marginal: bool = False,
    ) -> 'DataLabelingLever':
        """Create lever with full_predictions extracted from AllocationData.

        This extracts the current predictions from the data and stores them
        as the "full" (100% labeled) predictions source.

        Args:
            data: AllocationData to extract predictions from
            name: Lever name
            label_share: Initial label share (theta). In marginal mode, this is
                the increment above baseline_share.
            cost_per_label: Cost per label
            ascending: Ranking direction
            seed: Random seed
            baseline_share: Baseline label share (for marginal mode)
            marginal: If True, theta is interpreted as increment above baseline

        Returns:
            DataLabelingLever with stored full_predictions
        """
        # Extract predictions from each dataset
        full_predictions = []
        for i in range(data.n_datasets):
            dataset = data.get_dataset(i)
            full_predictions.append(dataset.predictions.copy())

        return cls(
            name=name,
            label_share=label_share,
            cost_per_label=cost_per_label,
            ascending=ascending,
            seed=seed,
            full_predictions=full_predictions,
            baseline_share=baseline_share,
            marginal=marginal,
        )

    def __repr__(self):
        has_full = self._full_predictions is not None
        if self.marginal:
            return (
                f"DataLabelingLever(name='{self.name}', "
                f"theta={self.theta}, baseline={self.baseline_share}, "
                f"effective_share={self.effective_share}, marginal=True, "
                f"cost_per_label={self.cost_per_label}, "
                f"has_full_predictions={has_full})"
            )
        return (
            f"DataLabelingLever(name='{self.name}', "
            f"label_share={self.theta}, "
            f"cost_per_label={self.cost_per_label}, "
            f"ascending={self.ascending}, "
            f"has_full_predictions={has_full})"
        )
