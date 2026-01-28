# RVP - Relative Value of Prediction

A Python toolkit for comparing policy levers in resource allocation problems. Given predictions, how should policymakers allocate limited resources across different policy levers (e.g., data collection, capacity expansion, benefit increases) to maximize social welfare?

![](figures/relative-value-of-prediction.png)

## Quick Start

**1. Load your data** — a DataFrame with predictions and ground truth outcomes:

```python
from rvp import AllocationData

data = AllocationData(
    df=your_dataframe,
    predictions_col='predictions',
    ground_truth_col='groundtruth'
)
```

**2. Define the allocation problem** — specify utility function, capacity constraint, and allocation policy:

```python
from rvp import AllocationProblem
from rvp.utilities import CRRAUtility
from rvp.constraints import CoverageConstraint
from rvp.policies import RankingPolicy

problem = AllocationProblem(
    data=data,
    utility=CRRAUtility(rho=2.0, b=100),        # CRRA utility with baseline benefit
    constraint=CoverageConstraint(0.1, data.n),  # Can treat 10% of population
    policy=RankingPolicy(ascending=True)         # Target lowest predicted values
)
```

**3. Evaluate** — compare prediction-based targeting vs random allocation:

```python
results = problem.evaluate()
print(results['total_utility'])   # Total welfare achieved
print(results['mean_utility'])    # Average utility per treated individual
print(results['utility_ratio'])   # e.g., 1.47 = 47% better than random
```

## Examples

The `examples/` folder contains two case studies:

### Poverty Targeting (`examples/poverty-targeting/`)

Cash transfer targeting using consumption predictions.

To reproduce:

1. Download the Ethiopia’s 2015 Living Standards Measurement Survey from the [World Bank Micro Data Library](https://microdata.worldbank.org/index.php/catalog/2783)

2. Run `poverty-targeting-preprocessing-and-model.ipynb` to process survey data and train the prediction model

3. Open `poverty-targeting.ipynb` to run the lever comparison analysis

### Employment Office (`examples/unemployment-targeting/`)

Job seeker profiling for employment services.

*Note: Due to the sensitive nature of the data, the underlying records are not publicly available but can be applied for at the [Research Data Centre (FDZ) of the Institute for Employment Research (IAB)](https://fdz.iab.de/en/our-data-products/individual-and-household-data/siab/).*

## Core Concepts

### The Allocation Problem

An [`AllocationProblem`](rvp/problem.py) consists of:

| Component | Description | Example |
|-----------|-------------|---------|
| [`AllocationData`](rvp/data.py) | Predictions and ground truth outcomes | DataFrame with `predicted_benefit`, `actual_benefit` |
| [`Utility`](rvp/utilities/base.py) | How we value outcomes | `CRRAUtility` for diminishing returns |
| [`Constraint`](rvp/constraints/base.py) | Resource limits | `CoverageConstraint` for capacity limits |
| [`Policy`](rvp/policies/base.py) | How predictions map to allocations | `RankingPolicy` to treat top-k |

### Policy Levers

Levers modify the allocation problem. Each lever has an associated cost model. For example:

| Lever | What it does | Key parameter | Cost model |
|-------|--------------|---------------|------------|
| [`DataLabelingLever`](rvp/levers/data_labeling.py) | Controls share of population with usable predictions | `label_share` | `n × label_share × cost_per_label` |
| [`ExpandCoverageLever`](rvp/levers/expand_coverage.py) | Increases program capacity | `coverage_increase` | `n × coverage_increase × marginal_cost_per_person` |
| [`CRRABenefitLever`](rvp/levers/crra_benefit.py) | Changes transfer amount per beneficiary | `new_benefit` | `n_beneficiaries × new_benefit` |
| [`PredictionImprovementLever`](rvp/levers/prediction_improvement.py) | Interpolates predictions toward ground truth | `error_reduction` | Fixed cost (no cost-to-theta mapping) |

**Applying a lever** — levers return a modified problem:

```python
from rvp.levers import ExpandCoverageLever

capacity_lever = ExpandCoverageLever(name="Capacity", coverage_increase=0.05, marginal_cost_per_person=100)
expanded_problem = capacity_lever.apply(problem)  # New problem with increased capacity
expanded_problem.evaluate()
```

## Comparing Policy Levers

The toolkit supports three types of comparisons based on available cost information.

**Setup** — define levers and comparison:

```python
from rvp.comparison import LeverComparison
from rvp.levers import DataLabelingLever, ExpandCoverageLever, PredictionImprovementLever

data_lever = DataLabelingLever.from_data(
    data=data,
    label_share=0.2,         # Start at 20% labeled
    cost_per_label=13.0,
    ascending=True
)

capacity_lever = ExpandCoverageLever(
    name="Capacity",
    coverage_increase=0.1,
    marginal_cost_per_person=100
)

prediction_lever = PredictionImprovementLever(
    name="Prediction",
    error_reduction=0.2
)

comparison = LeverComparison(problem, lever_a=prediction_lever, lever_b=capacity_lever)
```

### Q1: How should a fixed budget be allocated across levers?

When costs are known for all levers, find optimal budget allocation:

```python
# Two-lever optimization (requires levers with cost mappings)
comparison_with_costs = LeverComparison(problem, lever_a=data_lever, lever_b=capacity_lever)
results = comparison_with_costs.optimize_budget(budget_range=[0, 100000])
comparison_with_costs.plot_budget_optimization(results)
```

For three levers (with benefit increase as residual lever implying CRRA utility):

```python
from rvp.comparison import optimize_budget_with_residual_benefit, plot_budget_allocation_stacked

results = optimize_budget_with_residual_benefit(
    problem,
    lever1=data_lever,
    lever2=capacity_lever,
    budget_range=[0, 100000]
)
plot_budget_allocation_stacked(results)
```


### Q2: What improvement would one lever need to match another?

When costs are known for one lever but uncertain for another:

```python
# Compare prediction improvement (unknown cost) vs capacity (known cost)
comparison = LeverComparison(problem, lever_a=prediction_lever, lever_b=capacity_lever)

# Welfare difference as lever A varies
comparison.plot_welfare_difference(
    theta_range=(0, 0.5),
    swept_lever='a'
)

# What budget for lever B achieves the same welfare as lever A at each theta?
comparison.plot_equivalent_cost(
    theta_range=(0, 0.5),
    swept_lever='a'  # Sweep prediction, find equivalent capacity cost
)
```

### Q3: How do levers compare in relative welfare impact?

When costs are unavailable for both levers:

```python
from rvp.comparison import plot_welfare_curve

# How does welfare change as we vary a single lever?
plot_welfare_curve(
    problem=problem,
    lever=capacity_lever,
    theta_range=(0, 0.5),
    welfare_metric='mean_utility'
)

# 2D heatmap comparing two levers
comparison.plot_welfare_heatmap(
    theta_a_range=(0, 1.0),
    theta_b_range=(0, 0.5),
    vmin=-5, vmax=5
)
```
