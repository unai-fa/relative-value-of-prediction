"""Data-labeling dimension — fraction of population with revealed predictions.

At theta=1 every individual sees their model prediction. At theta<1 a random
``(1-theta)`` fraction of the population is "unlabeled". By default, their
score is replaced with the population mean of predictions, which leaves them
tied and randomly ordered relative to each other. Optionally, unlabeled units
can instead be pushed to the tail of the ranking.

This is a *stateful* dimension: it stores the baseline predictions and reveal
order at construction time, so ``at(p, theta)`` is idempotent regardless of
what predictions ``p`` currently carries.
"""

from typing import Literal, Tuple

import numpy as np

from ...data import AllocationData
from ...problem import AllocationProblem
from ..dimension import DesignDimension


class DataLabelingDimension(DesignDimension):
    """Fraction of the population with revealed model predictions.

    Parameters
    ----------
    base_problem : AllocationProblem
        Problem whose predictions define the baseline. The dimension
        snapshots the predictions per dataset at construction time.
    bounds : tuple[float, float]
        (low, high) label fractions used when constructing grids.
    seed : int
        Base seed for the per-dataset reveal orderings.
    unlabeled_strategy : {'population_mean', 'ranking_tail'}
        How to score units without revealed predictions. ``'population_mean'``
        preserves the original behavior. ``'ranking_tail'`` assigns a finite
        extreme value that ranks unlabeled units last under the problem's
        ranking direction.
    name : str
    """

    target_component = "data"

    def __init__(
        self,
        base_problem: AllocationProblem,
        bounds: Tuple[float, float] = (0.0, 1.0),
        seed: int = 42,
        unlabeled_strategy: Literal["population_mean", "ranking_tail"] = "population_mean",
        name: str = "label_fraction",
    ):
        if unlabeled_strategy not in {"population_mean", "ranking_tail"}:
            raise ValueError(
                "unlabeled_strategy must be one of "
                "{'population_mean', 'ranking_tail'}"
            )

        self.bounds_ = bounds
        self.seed = seed
        self.unlabeled_strategy = unlabeled_strategy
        self.name = name

        # Snapshot baseline (predictions, ground truth, and the per-dataset
        # reveal ordering). All subsequent ``at()`` calls rebuild from these.
        base_data = base_problem.data
        self._gt_col = base_data.ground_truth_col
        self._pred_col = base_data.predictions_col
        self._covariate_cols = list(base_data.covariate_cols)

        self._datasets = []
        for i in range(base_data.n_datasets):
            ds = base_data.get_dataset(i)
            self._datasets.append(ds.df_single.copy())

        self._reveal_orders = [
            np.random.default_rng(seed + i).permutation(len(df))
            for i, df in enumerate(self._datasets)
        ]

    def at(self, problem: AllocationProblem, theta: float) -> AllocationProblem:
        """Build a new AllocationProblem at this label fraction.

        The original ``problem``'s utility / constraint / policy are kept;
        its data is fully replaced by the dimension's stored baseline with
        the (1-theta) fraction of predictions masked according to
        ``unlabeled_strategy``.
        """
        new_dfs = []
        for df, order in zip(self._datasets, self._reveal_orders):
            df = df.copy()
            preds = df[self._pred_col].to_numpy(copy=True)
            n_lab = int(float(theta) * len(preds))
            preds[order[n_lab:]] = self._unlabeled_prediction_value(
                preds=preds,
                problem=problem,
            )
            df[self._pred_col] = preds
            new_dfs.append(df)

        new_data = AllocationData(
            df=new_dfs if len(new_dfs) > 1 else new_dfs[0],
            covariate_cols=self._covariate_cols,
            ground_truth_col=self._gt_col,
            predictions_col=self._pred_col,
        )
        return AllocationProblem(
            data=new_data,
            utility=problem.utility,
            constraint=problem.constraint,
            policy=problem.policy,
        )

    @property
    def bounds(self) -> Tuple[float, float]:
        return self.bounds_

    def _unlabeled_prediction_value(
        self,
        preds: np.ndarray,
        problem: AllocationProblem,
    ) -> float:
        if self.unlabeled_strategy == "population_mean":
            return float(np.mean(preds))

        return _ranking_tail_prediction_value(preds=preds, problem=problem)


def _ranking_tail_prediction_value(
    preds: np.ndarray,
    problem: AllocationProblem,
) -> float:
    finite_preds = np.asarray(preds, dtype=float)
    finite_preds = finite_preds[np.isfinite(finite_preds)]
    if len(finite_preds) == 0:
        return 0.0

    pred_min = float(np.min(finite_preds))
    pred_max = float(np.max(finite_preds))
    scale = max(1.0, pred_max - pred_min)
    ascending = bool(getattr(problem.policy, "ascending", False))
    if ascending:
        return pred_max + scale
    return pred_min - scale
