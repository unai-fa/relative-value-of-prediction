import numpy as np
import pandas as pd
import pytest

from rvp import (
    AllocationData,
    AllocationProblem,
    BenefitDimension,
    CapacityDimension,
    DesignDimension,
    DesignSpace,
    PredictionSetDimension,
)
from rvp.constraints import CoverageConstraint
from rvp.policies import RankingPolicy
from rvp.utilities import CRRAUtility


def _data(predictions=(0.1, 0.2, 0.3, 0.4), ground_truth=(1.0, 2.0, 3.0, 4.0)):
    return AllocationData(
        pd.DataFrame(
            {
                "predictions": list(predictions),
                "ground_truth": list(ground_truth),
            }
        )
    )


def _problem():
    data = _data()
    return AllocationProblem(
        data=data,
        utility=CRRAUtility(rho=2.0, b=100.0),
        constraint=CoverageConstraint(max_coverage=0.5, population_size=data.n),
        policy=RankingPolicy(ascending=False),
    )


class _UnknownDimension(DesignDimension):
    name = "unknown"
    target_component = "policy"

    def at(self, problem, theta):
        return problem

    @property
    def bounds(self):
        return (0.0, 1.0)


def test_continuous_dimension_grid_uses_bounds():
    dim = CapacityDimension(bounds=(0.1, 0.5), name="alpha")

    np.testing.assert_allclose(dim.grid(3), [0.1, 0.3, 0.5])


def test_discrete_dimension_grid_uses_values_and_ignores_n():
    low = _data(predictions=(0.1, 0.1, 0.1, 0.1))
    high = _data(predictions=(0.9, 0.9, 0.9, 0.9))
    dim = PredictionSetDimension(name="features", data_by_value={"low": low, "high": high})

    assert list(dim.grid(100)) == ["low", "high"]
    assert dim.is_discrete is True


def test_design_space_applies_dimensions_without_mutating_base_problem():
    base = _problem()
    capacity_dim = CapacityDimension(bounds=(0.25, 0.75), name="alpha")
    benefit_dim = BenefitDimension(bounds=(50.0, 200.0), name="benefit")
    space = DesignSpace(base, [capacity_dim, benefit_dim])

    modified = space.at({capacity_dim: 0.25, benefit_dim: 200.0})

    assert modified.constraint.get_capacity() == 1
    assert modified.utility.b == 200.0
    assert base.constraint.get_capacity() == 2
    assert base.utility.b == 100.0


def test_design_space_rejects_duplicate_target_components():
    with pytest.raises(ValueError, match="same component|both target component"):
        DesignSpace(_problem(), [CapacityDimension(name="a"), CapacityDimension(name="b")])


def test_design_space_rejects_unknown_dimensions():
    space = DesignSpace(_problem(), [CapacityDimension(name="alpha")])

    with pytest.raises(KeyError):
        space.at({_UnknownDimension(): 0.5})


def test_welfare_surface_shape_matches_discrete_and_continuous_axes():
    base = _problem()
    capacity_dim = CapacityDimension(bounds=(0.25, 0.75), name="alpha")
    prediction_dim = PredictionSetDimension(
        name="features",
        data_by_value={
            "baseline": _data(predictions=(0.1, 0.2, 0.3, 0.4)),
            "improved": _data(predictions=(0.4, 0.3, 0.2, 0.1)),
        },
    )
    space = DesignSpace(base, [capacity_dim, prediction_dim])

    x_vals, y_vals, welfare = space.welfare_surface(
        dim_x=capacity_dim,
        dim_y=prediction_dim,
        n=4,
        metric="total_utility",
    )

    assert len(x_vals) == 4
    assert list(y_vals) == ["baseline", "improved"]
    assert welfare.shape == (2, 4)


def test_prediction_set_dimension_resolves_numeric_keys_only_when_close():
    dim = PredictionSetDimension(
        name="features",
        data_by_value={
            0.1: _data(predictions=(0.1, 0.1, 0.1, 0.1)),
            0.2: _data(predictions=(0.2, 0.2, 0.2, 0.2)),
        },
    )
    problem = _problem()

    resolved = dim.at(problem, np.float64(0.1))
    np.testing.assert_allclose(resolved.data.predictions, [0.1, 0.1, 0.1, 0.1])

    with pytest.raises(KeyError):
        dim.at(problem, 0.15)


def test_welfare_max_respects_fixed_dimensions_and_per_dimension_grid_sizes():
    base = _problem()
    capacity_dim = CapacityDimension(bounds=(0.25, 0.75), name="alpha")
    benefit_dim = BenefitDimension(bounds=(50.0, 200.0), name="benefit")
    space = DesignSpace(base, [capacity_dim, benefit_dim])

    best = space.welfare_max(
        dims=[capacity_dim],
        fixed={benefit_dim: 200.0},
        n={capacity_dim: 3},
        metric="total_utility",
    )
    surface_best = max(
        space.welfare_at(
            {capacity_dim: alpha, benefit_dim: 200.0},
            metric="total_utility",
        )
        for alpha in capacity_dim.grid(3)
    )

    assert best == surface_best
