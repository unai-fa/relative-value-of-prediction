"""Budget optimization across two parameterized levers."""

from typing import TYPE_CHECKING, Tuple, Optional, List
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..problem import AllocationProblem
    from ..levers import ParameterizedLever


def optimize_two_lever_budget(
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
    """Find optimal budget split between two parameterized levers.

    For each total budget, searches over how to split it between
    lever1 and lever2 to maximize utility.

    Both levers are always applied (even at budget=0, which gives theta=0).
    The baseline state has no investment in either lever.

    Both levers must implement:
    - compute_cost(problem) -> float
    - for_budget(budget, problem) -> ParameterizedLever

    Args:
        problem: Allocation problem with (potentially multiple) datasets
        lever1: First parameterized lever (with cost mapping)
        lever2: Second parameterized lever (with cost mapping)
        budget_range: (min, max) total budget to sweep
        n_budget_points: Number of budget points to evaluate
        grid_density: Number of split options to try per budget
        max_datasets: If set, only use first N datasets (for faster runs)
        welfare_metric: Which metric to optimize ('total_utility', 'utility_ratio', etc.)
        verbose: If True, print progress

    Returns:
        DataFrame with columns:
        - budget: total budget
        - optimal_lever1_share: fraction of budget on lever1
        - optimal_lever1_theta: theta value for lever1
        - optimal_lever2_theta: theta value for lever2
        - optimal_utility: utility at optimal allocation
    """
    from ..problem import AllocationProblem
    from ..data import AllocationData

    # Validate levers have cost mapping
    try:
        lever1.for_budget(1.0, problem)
    except NotImplementedError:
        raise ValueError(f"{lever1.name} doesn't support for_budget()")

    try:
        lever2.for_budget(1.0, problem)
    except NotImplementedError:
        raise ValueError(f"{lever2.name} doesn't support for_budget()")

    # Optionally limit datasets for faster runs
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
            {'utility': -np.inf, 'share': None, 'theta1': None, 'theta2': None}
            for _ in range(n_datasets)
        ]

        # Apply levers once per grid point, evaluate all datasets
        for lever1_share in np.linspace(0, 1, grid_density):
            budget1 = B * lever1_share
            budget2 = B - budget1

            try:
                # Apply lever1 first
                lever1_at_budget = lever1.for_budget(budget1, problem)
                problem_after_lever1 = lever1_at_budget.apply(problem)

                # Use modified problem for lever2's cost calculation (handles coupled costs)
                lever2_at_budget = lever2.for_budget(budget2, problem_after_lever1)
                problem_modified = lever2_at_budget.apply(problem_after_lever1)

                lever1_theta = lever1_at_budget.theta
                lever2_theta = lever2_at_budget.theta

                # Evaluate each dataset and track best per dataset
                for dataset_idx in range(n_datasets):
                    result = problem_modified._evaluate_single(dataset_idx)
                    utility = result[welfare_metric]

                    if utility is not None and utility > best_per_dataset[dataset_idx]['utility']:
                        best_per_dataset[dataset_idx] = {
                            'utility': utility,
                            'share': lever1_share,
                            'theta1': lever1_theta,
                            'theta2': lever2_theta,
                        }
            except Exception as e:
                if verbose:
                    print(f"\nError at B={B}, split={lever1_share}: {e}")
                continue

        # Average the optimal decisions across datasets
        valid_datasets = [d for d in best_per_dataset if d['share'] is not None]
        if valid_datasets:
            n_valid = len(valid_datasets)
            avg_share = sum(d['share'] for d in valid_datasets) / n_valid
            avg_theta1 = sum(d['theta1'] for d in valid_datasets) / n_valid
            avg_theta2 = sum(d['theta2'] for d in valid_datasets) / n_valid
            avg_utility = sum(d['utility'] for d in valid_datasets) / n_valid

            results.append({
                'budget': B,
                f'optimal_{lever1.name}_share': avg_share,
                f'optimal_{lever1.name}_theta': avg_theta1,
                f'optimal_{lever2.name}_theta': avg_theta2,
                'optimal_utility': avg_utility,
            })

    if verbose:
        print()  # newline after progress

    return pd.DataFrame(results)


def plot_two_lever_optimization(
    results: pd.DataFrame,
    lever1_name: str,
    lever2_name: str,
    axes: Optional[List] = None,
    figsize: Tuple[float, float] = (5, 4),
    xlabels: Optional[List[Optional[str]]] = None,
    ylabels: Optional[List[Optional[str]]] = None,
    color: str = '#2E86AB',
    linestyle: str = '-',
    label: Optional[str] = None,
):
    """Plot optimal two-lever budget allocation results.

    Creates 3 plots showing:
    1. Share of budget spent on lever1 vs budget
    2. Optimal lever1 theta vs budget
    3. Optimal lever2 theta vs budget

    Args:
        results: DataFrame from optimize_two_lever_budget
        lever1_name: Name of first lever (for column lookup)
        lever2_name: Name of second lever (for column lookup)
        axes: Optional list of 3 axes. If None, creates 3 separate figures.
        figsize: Figure size for each plot if creating new figures
        xlabels: List of 3 x-axis labels (None for default)
        ylabels: List of 3 y-axis labels (None for default)
        color: Line color
        linestyle: Line style ('-', '--', ':', '-.')

    Returns:
        List of 3 matplotlib axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    created_figs = False
    if axes is None:
        # Set up publication-quality plotting
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['legend.fontsize'] = 14

        axes = []
        for _ in range(3):
            _, ax = plt.subplots(figsize=figsize)
            axes.append(ax)
        created_figs = True

    # Default labels
    default_xlabels = ['Budget', 'Budget', 'Budget']
    default_ylabels = ['Share', 'Theta', 'Theta']

    if xlabels is None:
        xlabels = default_xlabels
    if ylabels is None:
        ylabels = default_ylabels

    x = results['budget']

    # Plot 1: Share on lever1
    share_col = f'optimal_{lever1_name}_share'
    axes[0].plot(x, results[share_col], linewidth=6, color=color, linestyle=linestyle, label=label)
    if xlabels[0]:
        axes[0].set_xlabel(xlabels[0])
    if ylabels[0]:
        axes[0].set_ylabel(ylabels[0])
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    axes[0].grid(alpha=0.3)

    # Plot 2: Lever1 theta
    theta1_col = f'optimal_{lever1_name}_theta'
    axes[1].plot(x, results[theta1_col], linewidth=6, color=color, linestyle=linestyle, label=label)
    if xlabels[1]:
        axes[1].set_xlabel(xlabels[1])
    if ylabels[1]:
        axes[1].set_ylabel(ylabels[1])
    axes[1].grid(alpha=0.3)

    # Plot 3: Lever2 theta
    theta2_col = f'optimal_{lever2_name}_theta'
    axes[2].plot(x, results[theta2_col], linewidth=6, color=color, linestyle=linestyle, label=label)
    if xlabels[2]:
        axes[2].set_xlabel(xlabels[2])
    if ylabels[2]:
        axes[2].set_ylabel(ylabels[2])
    axes[2].grid(alpha=0.3)

    if created_figs:
        for ax in axes:
            ax.figure.tight_layout()

    return axes
