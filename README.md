`rvp` is a Python toolkit for evaluating the value of prediction for guiding the allocation of scarce resources.

When predictions guide who receives limited societal resources, for example, cash transfers, job training or medical screenings, policymakers face a practical question: how valuable is improving predictions compared to other investments, such as expanding program capacity or increasing benefit levels? `rvp` provides the tools to answer this empirically, using your own data.

> Companion to our paper *Empirically Understanding the Value of Prediction in Allocation* (Under Review, ICML 2026).

---

## Installation

```bash
git clone https://github.com/rvp-toolkit/rvp.git
cd rvp
pip install -e .
```

---

## Getting Started

The best way to get started is the **[ACS Income notebook](examples/folktables-income/folktables-income.ipynb)**, which walks through the full toolkit step by step using US Census data (downloaded via the `folktables` package).

---

## How It Works

An `AllocationProblem` combines four components:

| Component | What it encodes | Example |
|---|---|---|
| `AllocationData` | Predictions + ground truth | DataFrame with `predictions`, `ground_truth` |
| `UtilityFunction` | How outcomes are valued | `CRRAUtility`, `PartitionedUtility` |
| `ResourceConstraint` | Capacity limits | `CoverageConstraint` |
| `Policy` | How predictions map to allocations | `RankingPolicy` (allocate to top-k) |

**Policy levers** modify one or more components. Policy lever are typically parameterized by an intensity $\theta$ with an optional cost model:

| Lever | Modifies | $\theta$ controls |
|---|---|---|
| `PredictionImprovementLever` | Predictions | Error reduction (interpolation toward ground truth) |
| `ExpandCoverageLever` | Constraint | Additional coverage (pp increase) |
| `DataLabelingLever` | Predictions | Share of population with predictions |
| `CRRABenefitLever` | Utility | Transfer amount per beneficiary |

Applying a lever returns a new problem:

```python
from rvp.levers import ExpandCoverageLever

lever = ExpandCoverageLever(name="Capacity", coverage_increase=0.05, marginal_cost_per_person=100)
new_problem = lever.apply(problem)
new_problem.evaluate()
```

---

## Comparing Policy Levers

The toolkit supports three modes of comparison, depending on what cost information is available.

### Budget optimization — costs known for all levers

Find the welfare-maximizing split of a fixed budget across any number of levers:

```python
from rvp.comparison import optimize_budget_allocation, plot_budget_shares

results = optimize_budget_allocation(
    problem,
    levers=[data_lever, capacity_lever],
    budget_range=(0, 100_000),
    n_budget_points=20,
    grid_density=10,
)
plot_budget_shares(results)
```

### Equivalent cost — costs known for one lever

What investment in lever B matches the welfare gain of lever A?

```python
from rvp.comparison import LeverComparison

comparison = LeverComparison(problem, lever_a=prediction_lever, lever_b=capacity_lever)
comparison.plot_welfare_difference(theta_range=(0, 0.5), swept_lever="a")
comparison.plot_equivalent_cost(theta_range=(0, 0.5), swept_lever="a")
```

### Welfare curves — no cost information

Compare relative welfare impact across lever intensities:

```python
from rvp.comparison import plot_welfare_curve

plot_welfare_curve(problem, lever=capacity_lever, theta_range=(0, 0.5))
comparison.plot_welfare_heatmap(theta_a_range=(0, 1.0), theta_b_range=(0, 0.5))
```

---

## Examples

| Example | Data | Access |
|---|---|---|
| **[ACS Income](examples/folktables-income/)** | Folktables (census) | Auto-downloaded |
| **[Poverty Targeting](examples/poverty-targeting/)** | Ethiopia LSMS | [World Bank](https://microdata.worldbank.org/index.php/catalog/2783) |
| **[Employment Office](examples/unemployment-targeting/)** | IAB SIAB | [FDZ](https://fdz.iab.de/en/our-data-products/individual-and-household-data/siab/) (restricted) |

---

## Citation

```bibtex
@inproceedings{rvp2026,
  title     = {Empirically Understanding the Value of Prediction in Allocation},
  author    = {Anonymous},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {Under review}
}
```

## License

MIT