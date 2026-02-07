"""Tests for the comparison module."""
import matplotlib
matplotlib.use('Agg')

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
    CRRABenefitLever,
    DataLabelingLever,
)
from rvp.comparison import (
    LeverComparison,
    optimize_budget_allocation,
    plot_budget_shares,
    plot_budget_thetas,
    compute_welfare_difference,
    find_breakeven,
    plot_welfare_difference,
    compute_welfare_curve,
    plot_welfare_curve,
    compute_welfare_ratio,
    plot_welfare_heatmap,
    compute_equivalent_cost,
    plot_equivalent_cost,
)
from rvp.comparison.budget_optimization import _generate_simplex_grid


def _make_problem(n=100, seed=42):
    """Create a basic allocation problem for testing."""
    np.random.seed(seed)
    df = pd.DataFrame({
        'predictions': np.random.rand(n),
        'ground_truth': np.random.rand(n) + 0.1,
    })
    data = AllocationData(df=df)
    return AllocationProblem(
        data=data,
        utility=CRRAUtility(rho=2.0, b=100),
        constraint=CoverageConstraint(max_coverage=0.2, population_size=n),
        policy=RankingPolicy(ascending=True),
    )


# ---------------------------------------------------------------------------
# Simplex grid
# ---------------------------------------------------------------------------
class TestSimplexGrid:
    """Tests for the budget simplex grid generator."""

    def test_points_sum_to_budget(self):
        """Every tuple should sum to total_budget for 1, 2, and 3 levers."""
        budget = 1000.0
        for n_levers in [1, 2, 3]:
            for point in _generate_simplex_grid(n_levers, budget, grid_density=5):
                assert abs(sum(point) - budget) < 1e-10, (
                    f"n={n_levers}, point={point} sums to {sum(point)}"
                )

    def test_correct_count(self):
        """2 levers → d+1 points, 3 levers → C(d+2,2) points."""
        d = 8
        points_2 = list(_generate_simplex_grid(2, 100.0, d))
        assert len(points_2) == d + 1

        points_3 = list(_generate_simplex_grid(3, 100.0, d))
        expected = (d + 2) * (d + 1) // 2  # C(d+2, 2)
        assert len(points_3) == expected

    def test_boundary_points_present(self):
        """For 3 levers, corners (B,0,0), (0,B,0), (0,0,B) should exist."""
        B = 500.0
        points = list(_generate_simplex_grid(3, B, grid_density=5))
        corners = {(B, 0.0, 0.0), (0.0, B, 0.0), (0.0, 0.0, B)}
        points_set = set(points)
        for corner in corners:
            assert corner in points_set, f"Missing corner {corner}"

    def test_all_non_negative(self):
        """No component should be negative."""
        for point in _generate_simplex_grid(3, 200.0, grid_density=6):
            for component in point:
                assert component >= 0, f"Negative component in {point}"


# ---------------------------------------------------------------------------
# Budget optimization
# ---------------------------------------------------------------------------
class TestBudgetOptimization:
    """Tests for optimize_budget_allocation."""

    def test_single_lever_gets_all_budget(self):
        """With one lever, share should always be 1.0."""
        problem = _make_problem()
        lever = ExpandCoverageLever(
            name="Capacity", coverage_increase=0.05, marginal_cost_per_person=100
        )
        results = optimize_budget_allocation(
            problem,
            levers=[lever],
            budget_range=(100, 1000),
            n_budget_points=5,
            grid_density=5,
        )
        shares = results['optimal_Capacity_share'].values
        np.testing.assert_allclose(shares, 1.0, atol=1e-10)

    def test_shares_sum_to_one(self):
        """With two levers, shares should sum to ~1 for each budget level."""
        problem = _make_problem()
        lever1 = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        lever2 = CRRABenefitLever(name="Ben", new_benefit=50, marginal=True)
        results = optimize_budget_allocation(
            problem,
            levers=[lever1, lever2],
            budget_range=(100, 2000),
            n_budget_points=5,
            grid_density=5,
        )
        share_sum = results['optimal_Cap_share'] + results['optimal_Ben_share']
        np.testing.assert_allclose(share_sum.values, 1.0, atol=1e-10)

    def test_optimal_beats_equal_split(self):
        """Optimal allocation should achieve >= welfare of a 50/50 split."""
        problem = _make_problem()
        lever1 = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        lever2 = CRRABenefitLever(name="Ben", new_benefit=50, marginal=True)

        results = optimize_budget_allocation(
            problem,
            levers=[lever1, lever2],
            budget_range=(500, 500),  # single budget point
            n_budget_points=1,
            grid_density=10,
        )
        optimal_utility = results['optimal_utility'].iloc[0]

        # Evaluate 50/50 manually
        B = 500.0
        l1 = lever1.for_budget(B / 2, problem)
        p1 = l1.apply(problem)
        l2 = lever2.for_budget(B / 2, p1)
        p2 = l2.apply(p1)
        equal_split_utility = p2.evaluate()['total_utility']

        assert optimal_utility >= equal_split_utility - 1e-6


# ---------------------------------------------------------------------------
# LeverComparison
# ---------------------------------------------------------------------------
class TestLeverComparison:
    """Tests for the LeverComparison class."""

    def test_parameterized_detection(self):
        """Correctly detects whether levers are parameterized."""
        problem = _make_problem()
        lever_param = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        lever_pred = PredictionImprovementLever(name="Pred", error_reduction=0.3)

        comp = LeverComparison(problem, lever_a=lever_param, lever_b=lever_pred)
        assert comp.a_is_parameterized is True
        assert comp.b_is_parameterized is True

    def test_cost_mapping_detection(self):
        """ExpandCoverageLever has cost, PredictionImprovementLever does not."""
        problem = _make_problem()
        lever_cap = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        lever_pred = PredictionImprovementLever(name="Pred", error_reduction=0.3)

        comp = LeverComparison(problem, lever_a=lever_pred, lever_b=lever_cap)
        assert comp.a_has_cost_mapping is False
        assert comp.b_has_cost_mapping is True

    def test_can_plot_guards(self):
        """can_plot_equivalent_cost is False when reference has no cost; calling plot raises."""
        problem = _make_problem()
        lever_pred = PredictionImprovementLever(name="Pred", error_reduction=0.3)
        lever_cap = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        # sweep a (pred), reference b (cap) — b has cost → should work
        comp = LeverComparison(problem, lever_a=lever_pred, lever_b=lever_cap)
        assert comp.can_plot_equivalent_cost(swept_lever='a') is True

        # sweep b (cap), reference a (pred) — a has no cost → should fail
        assert comp.can_plot_equivalent_cost(swept_lever='b') is False
        with pytest.raises(TypeError):
            comp.plot_equivalent_cost(theta_range=(0, 0.3), swept_lever='b')

    def test_baseline_caching(self):
        """get_baseline_welfare returns same value on repeated calls."""
        problem = _make_problem()
        lever1 = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        lever2 = PredictionImprovementLever(name="Pred", error_reduction=0.3)
        comp = LeverComparison(problem, lever_a=lever1, lever_b=lever2)

        w1 = comp.get_baseline_welfare()
        w2 = comp.get_baseline_welfare()
        assert w1 == w2

    def test_invalid_swept_lever_raises(self):
        """Invalid swept_lever value should raise ValueError."""
        problem = _make_problem()
        lever1 = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        lever2 = PredictionImprovementLever(name="Pred", error_reduction=0.3)
        comp = LeverComparison(problem, lever_a=lever1, lever_b=lever2)

        with pytest.raises(ValueError):
            comp.can_plot_welfare_difference(swept_lever='c')


# ---------------------------------------------------------------------------
# Welfare difference
# ---------------------------------------------------------------------------
class TestWelfareDifference:
    """Tests for welfare difference computation."""

    def test_better_lever_has_positive_difference(self):
        """Sweeping a lever that improves welfare should give positive difference vs a fixed worse lever."""
        problem = _make_problem()
        # PredictionImprovement at high theta strictly improves welfare
        # vs ExpandCoverage at a small fixed theta
        lever_pred = PredictionImprovementLever(name="Pred", error_reduction=0.5)
        lever_cap = ExpandCoverageLever(
            name="Cap", coverage_increase=0.01, marginal_cost_per_person=100
        )
        comp = LeverComparison(problem, lever_a=lever_pred, lever_b=lever_cap)

        df = compute_welfare_difference(
            comp, theta_range=(0.3, 0.9), swept_lever='a', n_points=5
        )
        # At high error_reduction, prediction improvement should dominate
        # a very small capacity increase
        assert df['welfare_difference'].iloc[-1] > 0

    def test_breakeven_exists_when_levers_cross(self):
        """find_breakeven should return a float when levers cross."""
        problem = _make_problem()
        # Sweep prediction from 0 to 1. At theta=0 it's worse than reference,
        # at theta=1 it's better (perfect predictions).
        lever_pred = PredictionImprovementLever(name="Pred", error_reduction=0.0)
        lever_cap = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        comp = LeverComparison(problem, lever_a=lever_pred, lever_b=lever_cap)

        breakeven = find_breakeven(comp, theta_range=(0.0, 1.0), swept_lever='a', n_points=50)
        assert breakeven is not None
        assert 0.0 < breakeven < 1.0

    def test_no_breakeven_when_one_dominates(self):
        """find_breakeven should return None when swept lever always dominates."""
        problem = _make_problem()
        # Sweep prediction from 0.8 to 1.0 — should always beat a tiny capacity increase
        lever_pred = PredictionImprovementLever(name="Pred", error_reduction=0.8)
        lever_cap = ExpandCoverageLever(
            name="Cap", coverage_increase=0.001, marginal_cost_per_person=100
        )
        comp = LeverComparison(problem, lever_a=lever_pred, lever_b=lever_cap)

        breakeven = find_breakeven(comp, theta_range=(0.8, 1.0), swept_lever='a', n_points=20)
        assert breakeven is None


# ---------------------------------------------------------------------------
# Equivalent cost
# ---------------------------------------------------------------------------
class TestEquivalentCost:
    """Tests for equivalent cost computation."""

    def test_equivalent_cost_increases_with_theta(self):
        """As swept lever improves, the cost to match with reference should grow."""
        problem = _make_problem()
        lever_pred = PredictionImprovementLever(name="Pred", error_reduction=0.3)
        lever_cap = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        comp = LeverComparison(problem, lever_a=lever_pred, lever_b=lever_cap)

        df = compute_equivalent_cost(
            comp, theta_range=(0.01, 0.5), swept_lever='a', n_points=5,
            reference_theta_search_range=(0.0, 0.5),
        )
        costs = df['equivalent_cost'].dropna().values
        if len(costs) >= 2:
            # Should generally be non-decreasing (or NaN when unachievable)
            assert costs[-1] >= costs[0] - 1e-6

    def test_low_theta_gives_low_cost(self):
        """At small swept theta, equivalent cost should be near zero."""
        problem = _make_problem()
        lever_pred = PredictionImprovementLever(name="Pred", error_reduction=0.01)
        lever_cap = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        comp = LeverComparison(problem, lever_a=lever_pred, lever_b=lever_cap)

        df = compute_equivalent_cost(
            comp, theta_range=(0.0, 0.01), swept_lever='a', n_points=3,
            reference_theta_search_range=(0.0, 0.3),
        )
        first_cost = df['equivalent_cost'].iloc[0]
        # At theta=0, swept welfare = baseline, so equivalent cost = 0
        assert first_cost < 1.0 or np.isnan(first_cost)


# ---------------------------------------------------------------------------
# Welfare curve
# ---------------------------------------------------------------------------
class TestWelfareCurve:
    """Tests for welfare curve computation."""

    def test_welfare_increases_with_coverage(self):
        """Expanding coverage should increase total welfare."""
        problem = _make_problem()
        lever = ExpandCoverageLever(
            name="Cap", coverage_increase=0.0, marginal_cost_per_person=100
        )
        df = compute_welfare_curve(
            problem, lever, theta_range=(0.0, 0.3), n_points=5
        )
        # Welfare gain at max theta should be positive
        assert df['welfare_gain'].iloc[-1] > 0

    def test_subgroup_mask_changes_welfare(self):
        """Passing a subgroup mask should change welfare values."""
        problem = _make_problem()
        lever = ExpandCoverageLever(
            name="Cap", coverage_increase=0.0, marginal_cost_per_person=100
        )
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True

        df_full = compute_welfare_curve(
            problem, lever, theta_range=(0.0, 0.2), n_points=3
        )
        df_sub = compute_welfare_curve(
            problem, lever, theta_range=(0.0, 0.2), n_points=3,
            subgroup_mask=mask,
        )
        # Subgroup welfare should differ from full welfare
        assert df_full['welfare'].iloc[-1] != df_sub['welfare'].iloc[-1]


# ---------------------------------------------------------------------------
# Welfare heatmap
# ---------------------------------------------------------------------------
class TestWelfareHeatmap:
    """Tests for welfare ratio computation."""

    def test_row_count(self):
        """Should have n_points_a * n_points_b rows."""
        problem = _make_problem()
        lever_a = PredictionImprovementLever(name="Pred", error_reduction=0.3)
        lever_b = ExpandCoverageLever(
            name="Cap", coverage_increase=0.05, marginal_cost_per_person=100
        )
        comp = LeverComparison(problem, lever_a=lever_a, lever_b=lever_b)

        df = compute_welfare_ratio(
            comp,
            theta_a_range=(0.0, 0.5),
            theta_b_range=(0.0, 0.3),
            n_points_a=4,
            n_points_b=3,
        )
        assert len(df) == 4 * 3

    def test_ratio_reflects_lever_dominance(self):
        """When lever_a gain > lever_b gain, ratio should be > 1."""
        problem = _make_problem()
        lever_a = PredictionImprovementLever(name="Pred", error_reduction=0.5)
        lever_b = ExpandCoverageLever(
            name="Cap", coverage_increase=0.01, marginal_cost_per_person=100
        )
        comp = LeverComparison(problem, lever_a=lever_a, lever_b=lever_b)

        df = compute_welfare_ratio(
            comp,
            theta_a_range=(0.5, 0.9),  # strong prediction improvement
            theta_b_range=(0.001, 0.01),  # tiny capacity increase
            n_points_a=3,
            n_points_b=3,
        )
        finite_ratios = df['ratio'][np.isfinite(df['ratio'])]
        if len(finite_ratios) > 0:
            assert finite_ratios.mean() > 1.0