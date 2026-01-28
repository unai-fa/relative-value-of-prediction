"""Budget optimization with residual benefit allocation."""

from typing import TYPE_CHECKING, Tuple, Optional
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..problem import AllocationProblem
    from ..levers import ParameterizedLever


def optimize_budget_with_residual_benefit(
    problem: 'AllocationProblem',
    lever1: 'ParameterizedLever',
    lever2: 'ParameterizedLever',
    budget_range: Tuple[float, float],
    n_budget_points: int = 20,
    grid_density: int = 10,
    max_datasets: Optional[int] = None,
    welfare_metric: str = 'total_utility',
    verbose: bool = False,
) -> pd.DataFrame:
    """Find optimal budget allocation with benefit as residual.

    Iterates over spending on lever1 and lever2, then computes benefit
    as (budget - lever1_cost - lever2_cost) / n_beneficiaries.

    Requires:
    - problem.utility is CRRAUtility (to get baseline benefit)
    - problem.constraint is CoverageConstraint (to get baseline coverage)
    - lever1 and lever2 implement for_budget() and compute_cost()

    Args:
        problem: Allocation problem with CRRAUtility and CoverageConstraint
        lever1: First lever
        lever2: Second lever
        budget_range: (min, max) total budget to sweep
        n_budget_points: Number of budget points to evaluate
        grid_density: Number of grid points per lever dimension
        max_datasets: If set, only use first N datasets
        welfare_metric: Which metric to optimize
        verbose: If True, print progress

    Returns:
        DataFrame with columns for budget, optimal shares, thetas, and utility
    """
    from ..problem import AllocationProblem
    from ..data import AllocationData
    from ..utilities import CRRAUtility
    from ..constraints.coverage import CoverageConstraint

    # Validate problem setup
    if not isinstance(problem.utility, CRRAUtility):
        raise TypeError(f"Requires CRRAUtility, got {type(problem.utility).__name__}")
    if not isinstance(problem.constraint, CoverageConstraint):
        raise TypeError(f"Requires CoverageConstraint, got {type(problem.constraint).__name__}")

    # Optionally limit datasets
    if max_datasets is not None and problem.data.n_datasets > max_datasets:
        limited_dfs = [problem.data._dfs[i] for i in range(max_datasets)]
        limited_data = AllocationData(
            df=limited_dfs,
            covariate_cols=problem.data.covariate_cols,
            ground_truth_col=problem.data.ground_truth_col,
            predictions_col=problem.data.predictions_col,
        )
        problem = AllocationProblem(
            data=limited_data,
            utility=problem.utility,
            constraint=problem.constraint,
            policy=problem.policy,
        )

    budgets = np.linspace(budget_range[0], budget_range[1], n_budget_points)
    results = []
    n_datasets = problem.data.n_datasets

    for i, B in enumerate(budgets):
        if verbose:
            print(f"Budget {i+1}/{n_budget_points}: ${B:.0f}", end='\r')

        # Track best allocation for each dataset
        best_per_dataset = [
            {
                'utility': -np.inf,
                'lever1_spend': None, 'lever2_spend': None, 'benefit_spend': None,
                'theta1': None, 'theta2': None, 'benefit': None,
            }
            for _ in range(n_datasets)
        ]

        # Grid search over lever1 and lever2 spending
        for lever1_spend in np.linspace(0, B, grid_density):
            # Get lever1 at this budget
            try:
                lever1_at_budget = lever1.for_budget(lever1_spend, problem)
            except Exception:
                continue

            lever1_cost = lever1_at_budget.compute_cost(problem)

            # Remaining budget for lever2 + benefit
            remaining_after_lever1 = B - lever1_cost

            for lever2_spend in np.linspace(0, remaining_after_lever1, grid_density):
                try:
                    lever2_at_budget = lever2.for_budget(lever2_spend, problem)
                except Exception:
                    continue

                lever2_cost = lever2_at_budget.compute_cost(problem)

                # Benefit budget is residual
                benefit_budget = B - lever1_cost - lever2_cost

                if benefit_budget < 0:
                    continue

                # Apply lever1 and lever2
                try:
                    from ..levers import CRRABenefitLever

                    problem_modified = lever1_at_budget.apply(problem)
                    problem_modified = lever2_at_budget.apply(problem_modified)

                    # Read n_beneficiaries from modified problem's constraint
                    n_beneficiaries = problem_modified.constraint.get_capacity()

                    if n_beneficiaries <= 0:
                        continue

                    # Benefit per person from residual
                    benefit_per_person = benefit_budget / n_beneficiaries

                    # Apply benefit increment (added on top of baseline)
                    benefit_lever = CRRABenefitLever(
                        name="benefit",
                        new_benefit=benefit_per_person,
                        marginal=True,
                    )
                    problem_modified = benefit_lever.apply(problem_modified)

                    # Evaluate each dataset
                    for dataset_idx in range(n_datasets):
                        result = problem_modified._evaluate_single(dataset_idx)
                        utility = result[welfare_metric]

                        if utility is not None and utility > best_per_dataset[dataset_idx]['utility']:
                            best_per_dataset[dataset_idx] = {
                                'utility': utility,
                                'lever1_spend': lever1_cost,
                                'lever2_spend': lever2_cost,
                                'benefit_spend': benefit_budget,
                                'theta1': lever1_at_budget.theta,
                                'theta2': lever2_at_budget.theta,
                                'benefit': benefit_per_person,
                            }
                except Exception as e:
                    if verbose:
                        print(f"\nError at B={B}: {e}")
                    continue

        # Average optimal decisions across datasets
        valid_datasets = [d for d in best_per_dataset if d['lever1_spend'] is not None]
        if valid_datasets:
            n_valid = len(valid_datasets)

            # Compute average shares (as fraction of total budget)
            avg_lever1_share = sum(d['lever1_spend'] for d in valid_datasets) / n_valid / B if B > 0 else 0
            avg_lever2_share = sum(d['lever2_spend'] for d in valid_datasets) / n_valid / B if B > 0 else 0
            avg_benefit_share = sum(d['benefit_spend'] for d in valid_datasets) / n_valid / B if B > 0 else 0

            avg_theta1 = sum(d['theta1'] for d in valid_datasets) / n_valid
            avg_theta2 = sum(d['theta2'] for d in valid_datasets) / n_valid
            avg_benefit = sum(d['benefit'] for d in valid_datasets) / n_valid
            avg_utility = sum(d['utility'] for d in valid_datasets) / n_valid

            results.append({
                'budget': B,
                f'optimal_{lever1.name}_share': avg_lever1_share,
                f'optimal_{lever2.name}_share': avg_lever2_share,
                'optimal_benefit_share': avg_benefit_share,
                f'optimal_{lever1.name}_theta': avg_theta1,
                f'optimal_{lever2.name}_theta': avg_theta2,
                'optimal_benefit_theta': avg_benefit,
                'optimal_utility': avg_utility,
            })

    if verbose:
        print()

    return pd.DataFrame(results)


def plot_budget_allocation_stacked(
    results: pd.DataFrame,
    labels: Optional[list] = None,
    ax=None,
    figsize: Tuple[float, float] = (8, 5),
    xlabel: str = 'Budget',
    ylabel: str = 'Budget Share',
    colors: list = None,
    alpha: float = 0.8,
):
    """Plot budget allocation as a stacked area chart.

    Single plot showing how budget is split between three levers,
    with stacked colored bands showing each lever's share.

    Lever names are automatically extracted from DataFrame columns
    (looks for columns matching 'optimal_*_share' pattern).

    Args:
        results: DataFrame from optimize_budget_with_residual_benefit
        labels: List of 3 labels for legend. If None, uses lever names from columns.
        ax: Optional matplotlib axis. If None, creates new figure.
        figsize: Figure size if creating new figure
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: List of 3 colors for the levers (default: blue, pink, orange)
        alpha: Transparency of filled areas

    Returns:
        matplotlib axis
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import re

    # Extract lever names from columns
    share_cols = [c for c in results.columns if c.startswith('optimal_') and c.endswith('_share')]
    if len(share_cols) != 3:
        raise ValueError(f"Expected 3 share columns, found {len(share_cols)}: {share_cols}")

    # Extract lever names from column names (optimal_<name>_share -> <name>)
    lever_names = []
    for col in share_cols:
        match = re.match(r'optimal_(.+)_share', col)
        if match:
            lever_names.append(match.group(1))

    if ax is None:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['legend.fontsize'] = 14

        _, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = ['#2E86AB', '#A23B72', '#F18F01']

    if labels is None:
        labels = lever_names

    x = results['budget'].values
    share1 = results[share_cols[0]].values
    share2 = results[share_cols[1]].values
    share3 = results[share_cols[2]].values

    # Stacked area chart
    ax.fill_between(x, 0, share1, color=colors[0], alpha=alpha, label=labels[0])
    ax.fill_between(x, share1, share1 + share2, color=colors[1], alpha=alpha, label=labels[1])
    ax.fill_between(x, share1 + share2, share1 + share2 + share3, color=colors[2], alpha=alpha, label=labels[2])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    ax.figure.tight_layout()

    return ax
