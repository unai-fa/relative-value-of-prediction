# rvp

<p align="center">
  <img src="assets/relative-value-of-prediction.png" alt="rvp design-space overview" width="70%">
</p>

`rvp` is a Python toolkit for studying the meta-design of allocation
problems. It helps evaluate how downstream welfare changes when a planner
varies design choices such as program capacity, data availability, prediction
inputs, or treatment size.

The package is under active development.

## Installation

From the repository root:

```bash
pip install -e .
```

To run the test suite:

```bash
python -m pytest -q
```

## Core Idea

The basic object in `rvp` is an `AllocationProblem`. It ties together the data
available to the planner, the welfare objective, the resource constraint, and
the allocation policy used to assign the scarce resource.

| Component | Role |
| --- | --- |
| `AllocationData` | predictions and ground-truth outcomes |
| utility | how outcomes and allocations are valued, e.g. `CRRAUtility` |
| constraint | how many units can receive the resource, e.g. `CoverageConstraint` |
| policy | how scores are turned into allocations, e.g. `RankingPolicy` |

Together, these components define the allocation problem to be solved. A design
space then defines axes along which this problem can vary. For example, one
axis might expand capacity, while another reveals predictions for a larger
share of the population. The welfare surface evaluates the resulting allocation
problem at each design configuration.

## Minimal Example

```python
import pandas as pd

from rvp import AllocationData, AllocationProblem
from rvp.constraints import CoverageConstraint
from rvp.policies import RankingPolicy
from rvp.utilities import CRRAUtility

df = pd.DataFrame(
    {
        "predictions": predictions,
        "ground_truth": outcomes,
    }
)

data = AllocationData(df=df)
problem = AllocationProblem(
    data=data,
    utility=CRRAUtility(rho=3.0, b=100.0),
    constraint=CoverageConstraint(max_coverage=0.10, population_size=data.n),
    policy=RankingPolicy(ascending=True),
)

result = problem.evaluate()
```

## Design Spaces

Design dimensions transform one component of an allocation problem. The current
public dimensions are:

| Dimension | Varies |
| --- | --- |
| `CapacityDimension` | the capacity constraint |
| `BenefitDimension` | the transfer amount in CRRA utility |
| `DataLabelingDimension` | the share of units with revealed predictions |
| `PredictionSetDimension` | a discrete set of prediction/data variants |

Example:

```python
from rvp import CapacityDimension, DataLabelingDimension, DesignSpace

capacity_dim = CapacityDimension(bounds=(0.01, 0.50), name="capacity")
labeling_dim = DataLabelingDimension(
    base_problem=problem,
    bounds=(0.10, 1.00),
    seed=42,
    name="label_share",
)

space = DesignSpace(
    base_problem=problem,
    dimensions=[capacity_dim, labeling_dim],
)

welfare = space.welfare_at(
    {capacity_dim: 0.20, labeling_dim: 0.50},
    metric="mean_utility",
)
```

## Plotting Welfare Surfaces

`plot_welfare_surface` evaluates a two-dimensional slice of a design space and
plots the resulting welfare surface.

```python
from rvp import plot_welfare_surface

fig, ax = plot_welfare_surface(
    space,
    dim_x=capacity_dim,
    dim_y=labeling_dim,
    welfare_metric="mean_utility",
    normalize="max",
    n=40,
    contour_labels=True,
    cbar_label="Normalized welfare",
)
```

The plotting helper also supports status-quo markers, local welfare contours,
cost-frontier overlays, and expansion paths returned by the budget optimizers.

## Costs and Budget Optimization

Cost surfaces are separate from welfare surfaces. This lets the same evaluated
design space be inspected under different assumptions about implementation
costs.

```python
from rvp import CostSurface
from rvp.comparison import optimize_budget_frontier

cost_surface = CostSurface(
    lambda config: population_size
    * (
        cost_per_label * float(config[labeling_dim])
        + transfer_amount * float(config[capacity_dim])
    )
)

budget_results = optimize_budget_frontier(
    space=space,
    cost_surface=cost_surface,
    solve_dim=capacity_dim,
    budget_range=(min_budget, max_budget),
    n_budget_points=25,
    dims=[capacity_dim, labeling_dim],
    n=50,
    welfare_metric="mean_utility",
)
```

Use `optimize_budget` for grid-based optimization over feasible configurations,
and `optimize_budget_frontier` when one continuous dimension can be solved to
spend each budget exactly.

## Case-Study Notebooks

The standard empirical notebooks are:

| Notebook | Setting | Data access |
| --- | --- | --- |
| [`examples/poverty-targeting/poverty-targeting.ipynb`](examples/poverty-targeting/poverty-targeting.ipynb) | Targeting cash transfers using poverty scorecard predictions | Ethiopia LSMS data from the [World Bank Microdata Library](https://microdata.worldbank.org/index.php/catalog/2783) |
| [`examples/unemployment-targeting/unemployment-targeting.ipynb`](examples/unemployment-targeting/unemployment-targeting.ipynb) | Prioritizing job seekers at risk of long-term unemployment | SIAB data from the [Research Data Centre of the Institute for Employment Research](https://fdz.iab.de/en/our-data-products/individual-and-household-data/siab/) |

## Repository Structure

```text
rvp/
  data.py                    # AllocationData
  problem.py                 # AllocationProblem
  constraints/               # resource constraints
  policies/                  # allocation policies
  utilities/                 # welfare/utility functions
  design/                    # design dimensions, spaces, cost surfaces
  comparison/                # budget optimization
  plotting/                  # welfare surface plotting

examples/
  poverty-targeting/         # case-study notebook
  unemployment-targeting/    # case-study notebook

tests/
  test_*.py                  # focused package tests
```

## License

Apache-2.0
