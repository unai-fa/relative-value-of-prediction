"""Prediction improvement lever."""

from typing import Optional, Callable, Tuple, TYPE_CHECKING
import numpy as np

from .base import ParameterizedLever

if TYPE_CHECKING:
    from ..problem import AllocationProblem


def _is_percentile_range(range_tuple: Tuple[float, float]) -> bool:
    """Infer if range is percentile (both values in (0,1)) or absolute."""
    low, high = range_tuple
    return 0 < low < 1 and 0 < high < 1


def _compute_range_mask(
    values: np.ndarray,
    range_tuple: Tuple[float, float],
) -> np.ndarray:
    """Compute boolean mask for values in range (percentile or absolute)."""
    low, high = range_tuple

    if _is_percentile_range(range_tuple):
        low_threshold = np.percentile(values, low * 100)
        high_threshold = np.percentile(values, high * 100)
    else:
        low_threshold, high_threshold = low, high

    return (values >= low_threshold) & (values <= high_threshold)


class PredictionImprovementLever(ParameterizedLever):
    """Lever that improves prediction quality via residual scaling.

    theta = error_reduction (fraction, e.g., 0.2 = 20% error reduction)

    Improvement is applied by moving predictions closer to ground truth:
        new_pred = pred + theta * (y - pred)

    Can target specific subgroups via masks (combined with AND):
    - covariate_mask: callable on dataframe
    - outcome_range: (low, high) tuple for outcomes
    - prediction_range: (low, high) tuple for predictions

    Ranges are inferred as percentile if both values in (0,1), else absolute.
    """

    def __init__(
        self,
        name: str,
        error_reduction: float,
        covariate_mask: Optional[Callable] = None,
        outcome_range: Optional[Tuple[float, float]] = None,
        prediction_range: Optional[Tuple[float, float]] = None,
        cost: Optional[float] = None,
    ):
        """Initialize prediction improvement lever.

        Args:
            name: Identifier for this lever
            error_reduction: Target error reduction (0 to 1), e.g., 0.2 = 20% reduction
            covariate_mask: Optional callable that takes df and returns boolean mask
            outcome_range: Optional (low, high) range for outcomes
            prediction_range: Optional (low, high) range for predictions
            cost: Fixed cost of this lever (None if not specified)

        Ranges are inferred as percentile if both values in (0,1), else absolute.
        Multiple masks are combined with AND.
        """
        super().__init__(name, theta=error_reduction, cost=cost)
        self.covariate_mask = covariate_mask
        self.outcome_range = outcome_range
        self.prediction_range = prediction_range

    @property
    def error_reduction(self) -> float:
        """Alias for theta."""
        return self.theta

    def with_theta(self, theta: float) -> 'PredictionImprovementLever':
        """Return new lever with different error_reduction.

        Note: cost is set to None since this lever has no cost mapping.
        """
        return PredictionImprovementLever(
            name=self.name,
            error_reduction=theta,
            covariate_mask=self.covariate_mask,
            outcome_range=self.outcome_range,
            prediction_range=self.prediction_range,
            cost=None,  # No cost mapping, can't infer cost for new theta
        )

    def _build_mask(self, data) -> np.ndarray:
        """Build combined mask from all mask specifications."""
        n = data.n
        mask = np.ones(n, dtype=bool)

        if self.covariate_mask is not None:
            cov_mask = self.covariate_mask(data.df_single)
            mask &= np.asarray(cov_mask, dtype=bool)

        if self.outcome_range is not None:
            outcome_mask = _compute_range_mask(data.y, self.outcome_range)
            mask &= outcome_mask

        if self.prediction_range is not None:
            pred_mask = _compute_range_mask(data.predictions, self.prediction_range)
            mask &= pred_mask

        return mask

    def apply(self, problem: 'AllocationProblem') -> 'AllocationProblem':
        """Return new problem with improved predictions.

        If data has multiple datasets, improves all of them.

        Args:
            problem: Allocation problem with data to improve

        Returns:
            New AllocationProblem with improved predictions
        """
        from ..problem import AllocationProblem
        from ..data import AllocationData

        data = problem.data

        # Improve each dataset
        new_dfs = []
        for i in range(data.n_datasets):
            dataset = data.get_dataset(i)
            new_df = dataset.df_single.copy()

            mask = self._build_mask(dataset)
            predictions = dataset.predictions
            residuals = dataset.y - predictions
            improved_predictions = predictions.copy()
            improved_predictions[mask] += self.theta * residuals[mask]

            new_df[data.predictions_col] = improved_predictions
            new_dfs.append(new_df)

        # Create new data (single df or list)
        if len(new_dfs) == 1:
            improved_data = AllocationData(
                df=new_dfs[0],
                covariate_cols=data.covariate_cols,
                ground_truth_col=data.ground_truth_col,
                predictions_col=data.predictions_col,
            )
        else:
            improved_data = AllocationData(
                df=new_dfs,
                covariate_cols=data.covariate_cols,
                ground_truth_col=data.ground_truth_col,
                predictions_col=data.predictions_col,
            )

        return AllocationProblem(
            data=improved_data,
            utility=problem.utility,
            constraint=problem.constraint,
            policy=problem.policy,
        )

    def __repr__(self):
        parts = [f"name='{self.name}'", f"error_reduction={self.theta}"]
        if self.covariate_mask is not None:
            parts.append("covariate_mask=...")
        if self.outcome_range is not None:
            parts.append(f"outcome_range={self.outcome_range}")
        if self.prediction_range is not None:
            parts.append(f"prediction_range={self.prediction_range}")
        if self.cost is not None:
            parts.append(f"cost={self.cost}")
        return f"PredictionImprovementLever({', '.join(parts)})"
