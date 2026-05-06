"""Tests for exact-budget frontier optimization."""

import numpy as np

from rvp.comparison import optimize_budget, optimize_budget_frontier
from rvp.design.cost import CostSurface
from rvp.design.dimension import DesignDimension


class _ContinuousDim(DesignDimension):
    def __init__(self, name, bounds, target_component):
        self.name = name
        self.bounds_ = bounds
        self.target_component = target_component

    def at(self, problem, theta):
        raise NotImplementedError("Test dimensions are evaluated by _ToySpace")

    @property
    def bounds(self):
        return self.bounds_


class _ToySpace:
    def __init__(self, dimensions):
        self.dimensions = list(dimensions)

    def welfare_at(self, config, metric="mean_utility"):
        del metric
        alpha_dim, benefit_dim = self.dimensions
        return float(config[alpha_dim]) + 0.01 * float(config[benefit_dim])


def _make_frontier_problem():
    alpha_dim = _ContinuousDim("alpha", (1.0, 4.0), "constraint")
    benefit_dim = _ContinuousDim("benefit", (1.0, 10.0), "utility")
    space = _ToySpace([alpha_dim, benefit_dim])
    cost_surface = CostSurface(
        lambda config: float(config[alpha_dim]) * float(config[benefit_dim])
    )
    return space, cost_surface, alpha_dim, benefit_dim


def test_frontier_total_cost_spends_exact_budget():
    space, cost_surface, alpha_dim, benefit_dim = _make_frontier_problem()

    results, candidates = optimize_budget_frontier(
        space=space,
        cost_surface=cost_surface,
        solve_dim=benefit_dim,
        budgets=[8.0, 16.0],
        dims=[alpha_dim, benefit_dim],
        n=4,
        return_candidates=True,
    )

    assert len(results) == 2
    np.testing.assert_allclose(results["optimal_cost"], results["budget"], atol=1e-8)
    np.testing.assert_allclose(results["unspent_budget"], 0.0, atol=1e-8)
    np.testing.assert_allclose(results["optimal_alpha_theta"], 4.0)
    np.testing.assert_allclose(
        results["optimal_benefit_theta"],
        results["budget"] / results["optimal_alpha_theta"],
        atol=1e-8,
    )
    np.testing.assert_allclose(candidates["cost"], candidates["budget"], atol=1e-8)


def test_frontier_infeasible_budget_returns_nans():
    space, cost_surface, alpha_dim, benefit_dim = _make_frontier_problem()

    results = optimize_budget_frontier(
        space=space,
        cost_surface=cost_surface,
        solve_dim=benefit_dim,
        budgets=[0.5],
        dims=[alpha_dim, benefit_dim],
        n=4,
    )

    assert np.isnan(results.loc[0, "optimal_welfare"])
    assert np.isnan(results.loc[0, "optimal_cost"])
    assert np.isnan(results.loc[0, "optimal_alpha_theta"])
    assert np.isnan(results.loc[0, "optimal_benefit_theta"])


def test_frontier_incremental_cost_skips_below_status_quo():
    space, cost_surface, alpha_dim, benefit_dim = _make_frontier_problem()
    status_quo = {alpha_dim: 2.0, benefit_dim: 2.0}

    results, candidates = optimize_budget_frontier(
        space=space,
        cost_surface=cost_surface,
        solve_dim=benefit_dim,
        budgets=[8.0],
        dims=[alpha_dim, benefit_dim],
        status_quo=status_quo,
        n=4,
        return_candidates=True,
    )

    np.testing.assert_allclose(results.loc[0, "optimal_cost"], 8.0, atol=1e-8)
    assert results.loc[0, "optimal_alpha_theta"] >= status_quo[alpha_dim]
    assert results.loc[0, "optimal_benefit_theta"] >= status_quo[benefit_dim]
    assert np.all(candidates["alpha_theta"] >= status_quo[alpha_dim])
    assert np.all(candidates["benefit_theta"] >= status_quo[benefit_dim])


def test_existing_grid_optimizer_behavior_is_unchanged():
    space, cost_surface, alpha_dim, benefit_dim = _make_frontier_problem()

    results = optimize_budget(
        space=space,
        cost_surface=cost_surface,
        budgets=[8.0, 16.0],
        dims=[alpha_dim, benefit_dim],
        n=4,
    )

    assert list(results["budget"]) == [8.0, 16.0]
    assert set(results.columns) == {
        "budget",
        "optimal_welfare",
        "optimal_cost",
        "unspent_budget",
        "optimal_alpha_theta",
        "optimal_benefit_theta",
    }


def test_grid_optimizer_uses_incremental_cost_and_skips_below_status_quo():
    space, cost_surface, alpha_dim, benefit_dim = _make_frontier_problem()
    status_quo = {alpha_dim: 2.0, benefit_dim: 2.0}

    results, candidates = optimize_budget(
        space=space,
        cost_surface=cost_surface,
        budgets=[4.0],
        dims=[alpha_dim, benefit_dim],
        status_quo=status_quo,
        n=4,
        return_candidates=True,
    )

    assert np.all(
        candidates.loc[~candidates["below_status_quo"], "alpha_theta"]
        >= status_quo[alpha_dim]
    )
    assert results.loc[0, "optimal_alpha_theta"] >= status_quo[alpha_dim]
    assert results.loc[0, "optimal_benefit_theta"] >= status_quo[benefit_dim]
    assert results.loc[0, "optimal_cost"] <= results.loc[0, "budget"]
