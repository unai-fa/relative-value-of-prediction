import numpy as np
import pandas as pd
import pytest

from rvp import AllocationData, AllocationProblem, DataLabelingDimension
from rvp.constraints import CoverageConstraint
from rvp.policies import RankingPolicy
from rvp.utilities import PartitionedUtility


def _problem(*, ascending=True):
    df = pd.DataFrame(
        {
            "predictions": [1.0, 2.0, 3.0, 4.0],
            "ground_truth": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return AllocationProblem(
        data=AllocationData(df=df),
        utility=PartitionedUtility(thresholds=[0.5], values=[0.0, 1.0]),
        constraint=CoverageConstraint(max_coverage=2),
        policy=RankingPolicy(ascending=ascending),
    )


def test_data_labeling_default_masks_unlabeled_to_population_mean():
    problem = _problem(ascending=True)
    dim = DataLabelingDimension(base_problem=problem, seed=0)

    masked = dim.at(problem, 0.5).data.predictions

    assert np.count_nonzero(masked == np.mean([1.0, 2.0, 3.0, 4.0])) == 2


def test_data_labeling_ranking_tail_sends_unlabeled_last_when_ascending():
    problem = _problem(ascending=True)
    dim = DataLabelingDimension(
        base_problem=problem,
        seed=0,
        unlabeled_strategy="ranking_tail",
    )

    masked = dim.at(problem, 0.5).data.predictions

    assert np.count_nonzero(masked > 4.0) == 2


def test_data_labeling_ranking_tail_sends_unlabeled_last_when_descending():
    problem = _problem(ascending=False)
    dim = DataLabelingDimension(
        base_problem=problem,
        seed=0,
        unlabeled_strategy="ranking_tail",
    )

    masked = dim.at(problem, 0.5).data.predictions

    assert np.count_nonzero(masked < 1.0) == 2


def test_data_labeling_rejects_unknown_unlabeled_strategy():
    problem = _problem()

    with pytest.raises(ValueError, match="unlabeled_strategy"):
        DataLabelingDimension(base_problem=problem, unlabeled_strategy="unknown")


def test_data_labeling_extreme_label_shares_are_idempotent():
    problem = _problem(ascending=True)
    dim = DataLabelingDimension(base_problem=problem, seed=0)

    fully_labeled = dim.at(problem, 1.0).data.predictions
    fully_unlabeled = dim.at(problem, 0.0).data.predictions

    np.testing.assert_allclose(fully_labeled, problem.data.predictions)
    np.testing.assert_allclose(
        fully_unlabeled,
        np.full(problem.data.n, np.mean(problem.data.predictions)),
    )
