"""Tests for allocation policies."""
import numpy as np
import pytest

from rvp.policies import RankingPolicy
from rvp.utilities import CRRAUtility
from rvp.constraints import CoverageConstraint


class TestRankingPolicy:
    """Tests for ranking-based allocation policy."""

    def test_instantiation(self):
        """Policy can be created with default parameters."""
        policy = RankingPolicy()
        assert policy.rank_by == 'prediction'
        assert policy.ascending is False

    def test_ascending_mode(self):
        """ascending=True ranks lowest first (poverty targeting)."""
        policy = RankingPolicy(ascending=True)
        assert policy.ascending is True

    def test_invalid_rank_by(self):
        """Invalid rank_by should raise error."""
        with pytest.raises(ValueError):
            RankingPolicy(rank_by='invalid')

    def test_allocates_to_highest_by_default(self):
        """Default policy allocates to highest predictions."""
        policy = RankingPolicy(ascending=False)
        predictions = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        constraint = CoverageConstraint(max_coverage=2)  # Allocate to 2

        actions = policy(predictions, constraint)

        # Should allocate to indices with highest predictions (0.9 and 0.7)
        assert actions[2] == 1  # 0.9
        assert actions[4] == 1  # 0.7
        assert np.sum(actions) == 2

    def test_allocates_to_lowest_when_ascending(self):
        """ascending=True allocates to lowest predictions."""
        policy = RankingPolicy(ascending=True)
        predictions = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        constraint = CoverageConstraint(max_coverage=2)

        actions = policy(predictions, constraint)

        # Should allocate to indices with lowest predictions (0.1 and 0.3)
        assert actions[0] == 1  # 0.1
        assert actions[3] == 1  # 0.3
        assert np.sum(actions) == 2

    def test_respects_capacity_constraint(self):
        """Policy should not exceed capacity."""
        policy = RankingPolicy()
        predictions = np.random.rand(100)
        constraint = CoverageConstraint(max_coverage=20)

        actions = policy(predictions, constraint)

        assert np.sum(actions) == 20

    def test_min_prediction_filter(self):
        """min_prediction filters out low predictions."""
        policy = RankingPolicy(min_prediction=0.5)
        predictions = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        constraint = CoverageConstraint(max_coverage=3)

        actions = policy(predictions, constraint)

        # Only indices 2, 3, 4 are eligible (>= 0.5)
        assert actions[0] == 0
        assert actions[1] == 0
        assert np.sum(actions) == 3

    def test_returns_binary_actions(self):
        """Actions should be 0 or 1."""
        policy = RankingPolicy()
        predictions = np.random.rand(50)
        constraint = CoverageConstraint(max_coverage=10)

        actions = policy(predictions, constraint)

        assert np.all((actions == 0) | (actions == 1))

    def test_rank_by_utility(self):
        """Can rank by expected utility instead of prediction."""
        # ascending=False picks highest utility scores first
        policy = RankingPolicy(rank_by='utility', ascending=False)
        utility = CRRAUtility(rho=2.0, b=100)
        predictions = np.array([0.5, 1.0, 2.0])  # Lower y = higher utility gain
        constraint = CoverageConstraint(max_coverage=1)

        actions = policy(predictions, constraint, utility)

        # CRRA: lower y gives higher gain, so y=0.5 has highest utility score
        # ascending=False picks highest first, so should pick index 0
        assert actions[0] == 1
        assert np.sum(actions) == 1
