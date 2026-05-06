from rvp.design.cost import AdditiveCostSurface, CostSurface
from rvp.design.dimensions import BenefitDimension, CapacityDimension


def test_cost_surface_reports_total_and_improvement_costs():
    alpha = CapacityDimension(name="alpha")
    benefit = BenefitDimension(name="benefit")
    cost = CostSurface(lambda config: 100.0 * config[alpha] + 2.0 * config[benefit])

    start = {alpha: 0.1, benefit: 50.0}
    end = {alpha: 0.2, benefit: 75.0}

    assert cost.cost(start) == 110.0
    assert cost.cost(end) == 170.0
    assert cost.improvement_cost(end, start) == 60.0


def test_additive_cost_surface_sums_configured_dimensions_only():
    alpha = CapacityDimension(name="alpha")
    benefit = BenefitDimension(name="benefit")
    cost = AdditiveCostSurface(
        {
            alpha: lambda theta, config: 1000.0 * theta,
            benefit: lambda theta, config: config[alpha] * theta,
        }
    )

    assert cost.cost({alpha: 0.2, benefit: 100.0}) == 220.0
    assert cost.cost({alpha: 0.2}) == 200.0

