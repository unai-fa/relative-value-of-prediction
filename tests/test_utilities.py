"""Tests for utility functions."""
import numpy as np
import pytest

from rvp.utilities import CRRAUtility, PartitionedUtility


class TestCRRAUtility:
    """Tests for CRRA utility function.
    """

    def test_instantiation(self):
        """CRRA utility can be created with valid parameters."""
        u = CRRAUtility(rho=2.0, b=100)
        assert u.rho == 2.0
        assert u.b == 100

    def test_invalid_rho_zero(self):
        """rho must be > 0."""
        with pytest.raises(ValueError):
            CRRAUtility(rho=0, b=100)

    def test_invalid_rho_one(self):
        """rho cannot equal 1."""
        with pytest.raises(ValueError):
            CRRAUtility(rho=1.0, b=100)

    def test_no_action_zero_gain(self):
        """When action=0, utility gain should be 0."""
        u = CRRAUtility(rho=2.0, b=100)
        y = np.array([1.0, 2.0, 3.0])
        actions = np.array([0, 0, 0])
        result = u.compute(y, actions)
        np.testing.assert_array_almost_equal(result, [0, 0, 0])

    def test_action_gives_positive_gain(self):
        """When action=1, utility gain should be positive."""
        u = CRRAUtility(rho=2.0, b=100)
        y = np.array([1.0, 2.0, 3.0])
        actions = np.array([1, 1, 1])
        result = u.compute(y, actions)
        assert np.all(result > 0)

    def test_diminishing_returns(self):
        """Gain is higher for lower y."""
        u = CRRAUtility(rho=2.0, b=100)
        y_poor = np.array([1.0])
        y_rich = np.array([10.0])
        actions = np.array([1])
        gain_poor = u.compute(y_poor, actions)[0]
        gain_rich = u.compute(y_rich, actions)[0]

        assert gain_poor > gain_rich


class TestPartitionedUtility:
    """Tests for step-function utility.

    Assigns fixed utility values based on outcome bins.
    Action=1 gives the bin's value, action=0 gives 0.
    """

    def test_instantiation(self):
        """Partitioned utility can be created."""
        u = PartitionedUtility(
            thresholds=[0.5],
            values=[1, 0],
            threshold_type='percentile'
        )
        assert len(u.thresholds) == 1
        assert len(u.values) == 2

    def test_values_must_match_thresholds(self):
        """Need k+1 values for k thresholds."""
        with pytest.raises(ValueError):
            PartitionedUtility(
                thresholds=[0.5],
                values=[1, 0, 2],  # 3 values for 1 threshold - wrong
                threshold_type='percentile'
            )

    def test_binary_partition_percentile(self):
        """Bottom 50% gets value 1, top 50% gets value 0."""
        u = PartitionedUtility(
            thresholds=[0.5],
            values=[1, 0],
            threshold_type='percentile'
        )
        # 10 evenly spaced values
        ground_truth = np.linspace(0, 1, 10)
        actions = np.ones(10)
        result = u.compute(ground_truth, actions)
        # Bottom 50% should get value 1, top 50% should get value 0
        assert np.sum(result == 1) == 5
        assert np.sum(result == 0) == 5

    def test_no_action_zero_utility(self):
        """When action=0, utility should be 0 regardless of bin."""
        u = PartitionedUtility(
            thresholds=[0.5],
            values=[1, 0],
            threshold_type='percentile'
        )
        ground_truth = np.array([0.1, 0.9])  # one in each bin
        actions = np.array([0, 0])
        result = u.compute(ground_truth, actions)
        np.testing.assert_array_equal(result, [0, 0])

    def test_absolute_thresholds(self):
        """Absolute threshold mode uses actual values, not percentiles."""
        u = PartitionedUtility(
            thresholds=[50],
            values=[1, 0],
            threshold_type='absolute'
        )
        ground_truth = np.array([30, 70])  # 30 < 50, 70 > 50
        actions = np.array([1, 1])
        result = u.compute(ground_truth, actions)
        assert result[0] == 1  # below threshold
        assert result[1] == 0  # above threshold
