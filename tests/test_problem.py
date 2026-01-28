"""Tests for AllocationProblem."""
import numpy as np
import pandas as pd
import pytest

from rvp import AllocationData, AllocationProblem
from rvp.utilities import CRRAUtility
from rvp.constraints import CoverageConstraint
from rvp.policies import RankingPolicy


class TestAllocationData:
    """Tests for data container."""

    def test_instantiation(self):
        """Data can be created from DataFrame."""
        df = pd.DataFrame({
            'predictions': [0.1, 0.5, 0.9],
            'ground_truth': [0.2, 0.4, 0.8]
        })
        data = AllocationData(df=df)
        assert data.n == 3

    def test_requires_predictions_col(self):
        """Missing predictions column raises error."""
        df = pd.DataFrame({'ground_truth': [1, 2, 3]})
        with pytest.raises(ValueError):
            AllocationData(df=df)

    def test_requires_ground_truth_col(self):
        """Missing ground_truth column raises error."""
        df = pd.DataFrame({'predictions': [1, 2, 3]})
        with pytest.raises(ValueError):
            AllocationData(df=df)

    def test_properties(self):
        """Data exposes y, predictions, n properties."""
        df = pd.DataFrame({
            'predictions': [0.1, 0.5, 0.9],
            'ground_truth': [0.2, 0.4, 0.8]
        })
        data = AllocationData(df=df)
        assert len(data.y) == 3
        assert len(data.predictions) == 3
        np.testing.assert_array_equal(data.y, [0.2, 0.4, 0.8])

    def test_multiple_datasets(self):
        """Data can hold multiple DataFrames."""
        df1 = pd.DataFrame({'predictions': [0.1, 0.2], 'ground_truth': [0.3, 0.4]})
        df2 = pd.DataFrame({'predictions': [0.5, 0.6], 'ground_truth': [0.7, 0.8]})
        data = AllocationData(df=[df1, df2])
        assert data.n_datasets == 2


class TestAllocationProblem:
    """Tests for the core allocation problem."""

    def test_instantiation(self):
        """Problem can be created with all components."""
        df = pd.DataFrame({
            'predictions': np.random.rand(100),
            'ground_truth': np.random.rand(100)
        })
        data = AllocationData(df=df)
        utility = CRRAUtility(rho=2.0, b=100)
        constraint = CoverageConstraint(max_coverage=0.2, population_size=100)
        policy = RankingPolicy(ascending=True)

        problem = AllocationProblem(
            data=data,
            utility=utility,
            constraint=constraint,
            policy=policy
        )
        assert problem.data is data
        assert problem.utility is utility

    def test_evaluate_returns_dict(self):
        """Evaluation returns dictionary with expected keys."""
        df = pd.DataFrame({
            'predictions': np.random.rand(100),
            'ground_truth': np.random.rand(100) + 0.1  # Ensure positive for CRRA
        })
        data = AllocationData(df=df)
        problem = AllocationProblem(
            data=data,
            utility=CRRAUtility(rho=2.0, b=100),
            constraint=CoverageConstraint(max_coverage=0.2, population_size=100),
            policy=RankingPolicy(ascending=True)
        )

        result = problem.evaluate()
        assert isinstance(result, dict)
        assert 'total_utility' in result
        assert 'mean_utility' in result
        assert 'utility_ratio' in result
        assert 'n_allocated' in result
        assert 'actions' in result

    def test_allocates_correct_number(self):
        """Number allocated should match constraint capacity."""
        df = pd.DataFrame({
            'predictions': np.random.rand(100),
            'ground_truth': np.random.rand(100) + 0.1
        })
        data = AllocationData(df=df)
        problem = AllocationProblem(
            data=data,
            utility=CRRAUtility(rho=2.0, b=100),
            constraint=CoverageConstraint(max_coverage=0.2, population_size=100),
            policy=RankingPolicy(ascending=True)
        )

        result = problem.evaluate()
        assert result['n_allocated'] == 20  # 20% of 100

    def test_utility_ratio_measures_vs_random(self):
        """utility_ratio compares to random allocation."""
        # Create data where predictions perfectly predict ground truth
        # This should give utility_ratio > 1
        np.random.seed(42)
        ground_truth = np.random.rand(100) + 0.1
        predictions = ground_truth + np.random.randn(100) * 0.01  # Nearly perfect

        df = pd.DataFrame({
            'predictions': predictions,
            'ground_truth': ground_truth
        })
        data = AllocationData(df=df)
        problem = AllocationProblem(
            data=data,
            utility=CRRAUtility(rho=2.0, b=100),
            constraint=CoverageConstraint(max_coverage=0.2, population_size=100),
            policy=RankingPolicy(ascending=True)  # Target lowest values
        )

        result = problem.evaluate()
        # With good predictions targeting low y, should beat random
        assert result['utility_ratio'] is not None
        assert result['utility_ratio'] > 1.0
