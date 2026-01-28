"""Tests for constraint classes."""
import numpy as np
import pytest

from rvp.constraints import CoverageConstraint


class TestCoverageConstraint:
    """Tests for coverage constraint."""

    def test_instantiation_fractional(self):
        """Coverage constraint can be created with fractional coverage."""
        c = CoverageConstraint(max_coverage=0.2, population_size=100)
        assert c.get_capacity() == 20

    def test_instantiation_absolute(self):
        """Coverage constraint can be created with absolute coverage."""
        c = CoverageConstraint(max_coverage=50)
        assert c.get_capacity() == 50

    def test_invalid_zero_coverage(self):
        """Zero coverage should raise ValueError."""
        with pytest.raises(ValueError):
            CoverageConstraint(max_coverage=0.0, population_size=100)

    def test_full_coverage_absolute(self):
        """Full coverage using absolute number."""
        c = CoverageConstraint(max_coverage=100)
        assert c.get_capacity() == 100

    def test_unit_costs_are_ones(self):
        """All unit costs should be 1."""
        c = CoverageConstraint(max_coverage=0.2, population_size=100)
        costs = c.get_unit_costs(100)
        assert np.all(costs == 1)
        assert len(costs) == 100
