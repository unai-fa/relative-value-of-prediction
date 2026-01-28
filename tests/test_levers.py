"""Tests for policy levers."""
import numpy as np
import pandas as pd
import pytest

from rvp import AllocationData, AllocationProblem
from rvp.utilities import CRRAUtility
from rvp.constraints import CoverageConstraint
from rvp.policies import RankingPolicy
from rvp.levers import (
    PredictionImprovementLever,
    ExpandCoverageLever,
    DataLabelingLever,
    CRRABenefitLever
)


@pytest.fixture
def basic_problem():
    """Create a basic allocation problem for testing levers."""
    np.random.seed(42)
    df = pd.DataFrame({
        'predictions': np.random.rand(100),
        'ground_truth': np.random.rand(100) + 0.1
    })
    data = AllocationData(df=df)
    return AllocationProblem(
        data=data,
        utility=CRRAUtility(rho=2.0, b=100),
        constraint=CoverageConstraint(max_coverage=0.2, population_size=100),
        policy=RankingPolicy(ascending=True)
    )


class TestPredictionImprovementLever:
    """Tests for prediction improvement lever.

    Improves predictions via: new_pred = pred + theta * (y - pred)
    theta=0: no change, theta=1: perfect predictions
    """

    def test_instantiation(self):
        """Lever can be created."""
        lever = PredictionImprovementLever(name="Test", error_reduction=0.5)
        assert lever.name == "Test"
        assert lever.theta == 0.5
        assert lever.error_reduction == 0.5

    def test_theta_zero_no_change(self, basic_problem):
        """theta=0 should not change predictions."""
        lever = PredictionImprovementLever(name="Test", error_reduction=0.0)
        modified = lever.apply(basic_problem)

        original_preds = basic_problem.data.predictions
        modified_preds = modified.data.predictions
        np.testing.assert_array_almost_equal(original_preds, modified_preds)

    def test_theta_one_perfect_predictions(self, basic_problem):
        """theta=1 should make predictions equal ground truth."""
        lever = PredictionImprovementLever(name="Test", error_reduction=1.0)
        modified = lever.apply(basic_problem)

        ground_truth = modified.data.y
        predictions = modified.data.predictions
        np.testing.assert_array_almost_equal(predictions, ground_truth)

    def test_with_theta_creates_new_lever(self):
        """with_theta should create a new lever with different theta."""
        lever = PredictionImprovementLever(name="Test", error_reduction=0.5)
        new_lever = lever.with_theta(0.8)
        assert new_lever.theta == 0.8
        assert lever.theta == 0.5  # Original unchanged

    def test_improves_utility_ratio(self, basic_problem):
        """Better predictions should improve utility ratio."""
        baseline = basic_problem.evaluate()

        lever = PredictionImprovementLever(name="Test", error_reduction=0.5)
        improved = lever.apply(basic_problem)
        improved_result = improved.evaluate()

        # Better predictions should give higher utility ratio
        assert improved_result['utility_ratio'] >= baseline['utility_ratio']


class TestExpandCoverageLever:
    """Tests for capacity expansion lever.

    Increases constraint capacity by theta * n people.
    cost = theta * n * marginal_cost_per_person
    """

    def test_instantiation(self):
        """Lever can be created."""
        lever = ExpandCoverageLever(
            name="Test",
            coverage_increase=0.1,
            marginal_cost_per_person=100
        )
        assert lever.name == "Test"
        assert lever.coverage_increase == 0.1
        assert lever.marginal_cost_per_person == 100

    def test_negative_coverage_raises(self):
        """Negative coverage increase should raise error."""
        with pytest.raises(ValueError):
            ExpandCoverageLever(
                name="Test",
                coverage_increase=-0.1,
                marginal_cost_per_person=100
            )

    def test_increases_capacity(self, basic_problem):
        """Applying lever should increase constraint capacity."""
        original_capacity = basic_problem.constraint.get_capacity()

        lever = ExpandCoverageLever(
            name="Test",
            coverage_increase=0.1,
            marginal_cost_per_person=100
        )
        modified = lever.apply(basic_problem)

        new_capacity = modified.constraint.get_capacity()
        assert new_capacity == original_capacity + 10  # 10% of 100

    def test_compute_cost(self, basic_problem):
        """Cost should be theta * n * cost_per_person."""
        lever = ExpandCoverageLever(
            name="Test",
            coverage_increase=0.1,
            marginal_cost_per_person=100
        )
        cost = lever.compute_cost(basic_problem)
        # 0.1 * 100 people * 100 cost = 1000
        assert cost == 1000

    def test_for_budget(self, basic_problem):
        """for_budget should return lever matching target budget."""
        lever = ExpandCoverageLever(
            name="Test",
            coverage_increase=0.1,
            marginal_cost_per_person=100
        )
        # Budget of 500 should give coverage_increase of 0.05
        # theta = 500 / (100 * 100) = 0.05
        new_lever = lever.for_budget(500, basic_problem)
        assert abs(new_lever.theta - 0.05) < 1e-10


class TestDataLabelingLever:
    """Tests for data labeling lever.

    Controls fraction of individuals with usable predictions.
    Unlabeled individuals get deprioritized in ranking.
    """

    def test_instantiation(self):
        """Lever can be created."""
        lever = DataLabelingLever(
            name="Test",
            label_share=0.5,
            cost_per_label=10.0,
            ascending=True,
            seed=42
        )
        assert lever.name == "Test"
        assert lever.label_share == 0.5
        assert lever.cost_per_label == 10.0

    def test_invalid_label_share(self):
        """label_share outside [0,1] should raise error."""
        with pytest.raises(ValueError):
            DataLabelingLever(name="Test", label_share=1.5)

    def test_from_data_factory(self, basic_problem):
        """from_data creates lever with stored predictions."""
        lever = DataLabelingLever.from_data(
            data=basic_problem.data,
            label_share=0.5,
            cost_per_label=10.0,
            ascending=True,
            seed=42
        )
        assert lever.full_predictions is not None
        assert len(lever.full_predictions) == 1

    def test_unlabeled_deprioritized(self, basic_problem):
        """Unlabeled individuals should get high prediction values (ascending)."""
        lever = DataLabelingLever(
            name="Test",
            label_share=0.5,
            ascending=True,
            seed=42
        )
        modified = lever.apply(basic_problem)

        # With ascending=True, unlabeled get max + offset
        # So max of modified predictions should be higher than original
        original_max = np.max(basic_problem.data.predictions)
        modified_max = np.max(modified.data.predictions)
        assert modified_max > original_max

    def test_compute_cost(self, basic_problem):
        """Cost should be n * label_share * cost_per_label."""
        lever = DataLabelingLever(
            name="Test",
            label_share=0.5,
            cost_per_label=10.0
        )
        cost = lever.compute_cost(basic_problem)
        # 100 * 0.5 * 10 = 500
        assert cost == 500

    def test_from_data_can_restore_predictions(self, basic_problem):
        """When initialized from data, can reduce and restore predictions."""
        original_preds = basic_problem.data.predictions.copy()

        # Create lever from data (stores full predictions)
        lever = DataLabelingLever.from_data(
            data=basic_problem.data,
            label_share=1.0,  # Start at 100%
            cost_per_label=10.0,
            ascending=True,
            seed=42
        )

        # Reduce to 50% labeled
        reduced_lever = lever.with_theta(0.5)
        reduced_problem = reduced_lever.apply(basic_problem)

        # Some predictions should now be masked (different from original)
        reduced_preds = reduced_problem.data.predictions
        assert not np.allclose(original_preds, reduced_preds)

        # Restore to 100% labeled
        restored_lever = lever.with_theta(1.0)
        restored_problem = restored_lever.apply(basic_problem)

        # Predictions should be restored to original
        restored_preds = restored_problem.data.predictions
        np.testing.assert_array_almost_equal(original_preds, restored_preds)


class TestCRRABenefitLever:
    """Tests for CRRA benefit lever.

    Modifies the benefit parameter b in CRRAUtility.
    """

    def test_instantiation(self):
        """Lever can be created."""
        lever = CRRABenefitLever(name="Test", new_benefit=200)
        assert lever.name == "Test"
        assert lever.new_benefit == 200

    def test_changes_benefit(self, basic_problem):
        """Applying lever should change utility benefit."""
        original_b = basic_problem.utility.b

        lever = CRRABenefitLever(name="Test", new_benefit=200)
        modified = lever.apply(basic_problem)

        assert modified.utility.b == 200
        assert modified.utility.b != original_b

    def test_with_theta(self):
        """with_theta creates new lever with different benefit."""
        lever = CRRABenefitLever(name="Test", new_benefit=100)
        new_lever = lever.with_theta(200)
        assert new_lever.new_benefit == 200
        assert lever.new_benefit == 100  # Original unchanged

    def test_marginal_mode(self, basic_problem):
        """Marginal mode adds to existing benefit."""
        lever = CRRABenefitLever(
            name="Test",
            new_benefit=50,  # Increment
            marginal=True
        )
        modified = lever.apply(basic_problem)

        # Original b=100, increment=50, so new b=150
        assert modified.utility.b == 150
