"""Budget optimization across parameterized levers."""

from typing import TYPE_CHECKING, Tuple, Optional, List, Callable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..problem import AllocationProblem
    from ..levers import ParameterizedLever

# Type alias for lever linkage function
LeverLinkage = Callable[
    ['ParameterizedLever', 'AllocationProblem', List['ParameterizedLever']],
    'ParameterizedLever'
]


def _generate_simplex_grid(n_levers: int, total_budget: float, grid_density: int):
    """Generate uniform grid points on the budget simplex.

    For n_levers, generates all tuples (b_1, ..., b_n) where sum = total_budget,
    uniformly covering the simplex. Uses integer compositions:
    all (i_1, ..., i_n) with i_k >= 0 and sum = grid_density.

    Number of points: C(grid_density + n_levers - 1, n_levers - 1)
    For n=2, d=30: 31 points. For n=3, d=30: 496 points.

    Args:
        n_levers: Number of levers
        total_budget: Total budget to allocate
        grid_density: Number of steps (higher = finer grid)

    Yields:
        Tuples of (budget_1, budget_2, ..., budget_n) that sum to total_budget
    """
    if n_levers == 1:
        yield (total_budget,)
        return

    if n_levers == 2:
        for i in range(grid_density + 1):
            b1 = total_budget * i / grid_density
            b2 = total_budget - b1
            yield (b1, b2)
        return

    # General case: enumerate all integer compositions summing to grid_density
    def _compositions(n, total, prefix=()):
        if n == 1:
            yield prefix + (total_budget * total / grid_density,)
            return
        for i in range(total + 1):
            yield from _compositions(
                n - 1, total - i,
                prefix + (total_budget * i / grid_density,)
            )

    yield from _compositions(n_levers, grid_density)


def optimize_budget_allocation(
    problem: 'AllocationProblem',
    levers: List['ParameterizedLever'],
    budget_range: Tuple[float, float],
    n_budget_points: int = 20,
    grid_density: int = 10,
    lever_linkage: Optional[LeverLinkage] = None,
    max_datasets: Optional[int] = None,
    welfare_metric: str = 'total_utility',
    verbose: bool = False,
) -> pd.DataFrame:
    """Optimize budget allocation across multiple levers.

    Grid searches over budget splits to find the allocation that
    maximizes welfare. Levers are applied sequentially in list order.

    All levers must implement:
    - compute_cost(problem) -> float
    - for_budget(budget, problem) -> ParameterizedLever

    Args:
        problem: Allocation problem with (potentially multiple) datasets
        levers: List of parameterized levers (1-3 recommended, more gets slow)
        budget_range: (min, max) total budget to sweep
        n_budget_points: Number of budget points to evaluate
        grid_density: Grid points per lever dimension
        lever_linkage: Optional function to update lever[i]'s cost parameters
            based on problem state after applying levers[0..i-1].
            Signature: (lever, problem_after_previous, applied_levers) -> lever
        max_datasets: If set, only use first N datasets (for faster runs)
        welfare_metric: Which metric to optimize ('total_utility', 'utility_ratio', etc.)
        verbose: If True, print progress

    Returns:
        DataFrame with columns:
        - budget: total budget
        - optimal_{lever.name}_share: fraction of budget on each lever
        - optimal_{lever.name}_theta: theta value for each lever
        - optimal_utility: utility at optimal allocation
    """
    from ..problem import AllocationProblem
    from ..data import AllocationData

    if len(levers) == 0:
        raise ValueError("Must provide at least one lever")

    # Validate levers have cost mapping
    for lever in levers:
        try:
            lever.for_budget(1.0, problem)
        except NotImplementedError:
            raise ValueError(f"{lever.name} doesn't support for_budget()")

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
    n_levers = len(levers)

    for i, B in enumerate(budgets):
        if verbose:
            print(f"Budget {i+1}/{n_budget_points}: ${B:.0f}", end='\r')

        # Track best allocation for each dataset
        best_per_dataset = [
            {'utility': -np.inf, 'budgets': None, 'thetas': None}
            for _ in range(n_datasets)
        ]

        # Grid search over budget splits
        for budget_split in _generate_simplex_grid(n_levers, B, grid_density):
            try:
                current_problem = problem
                applied_levers = []
                lever_thetas = []

                for j, (lever, budget_j) in enumerate(zip(levers, budget_split)):
                    # Apply linkage to update cost parameters (for levers after first)
                    lever_for_budget = lever
                    if lever_linkage is not None and j > 0:
                        lever_for_budget = lever_linkage(lever, current_problem, applied_levers)

                    # Get lever at this budget and apply
                    lever_at_budget = lever_for_budget.for_budget(budget_j, current_problem)
                    current_problem = lever_at_budget.apply(current_problem)
                    applied_levers.append(lever_at_budget)
                    lever_thetas.append(lever_at_budget.theta)

                # Evaluate each dataset and track best per dataset
                for dataset_idx in range(n_datasets):
                    result = current_problem._evaluate_single(dataset_idx)
                    utility = result[welfare_metric]

                    if utility is not None and utility > best_per_dataset[dataset_idx]['utility']:
                        best_per_dataset[dataset_idx] = {
                            'utility': utility,
                            'budgets': budget_split,
                            'thetas': lever_thetas,
                        }
            except Exception as e:
                if verbose:
                    print(f"\nError at B={B}, split={budget_split}: {e}")
                continue

        # Average the optimal decisions across datasets
        valid_datasets = [d for d in best_per_dataset if d['budgets'] is not None]
        if valid_datasets:
            n_valid = len(valid_datasets)

            row = {'budget': B}

            # Compute averages for each lever
            for j, lever in enumerate(levers):
                avg_budget = sum(d['budgets'][j] for d in valid_datasets) / n_valid
                avg_theta = sum(d['thetas'][j] for d in valid_datasets) / n_valid
                share = avg_budget / B if B > 0 else 0

                row[f'optimal_{lever.name}_share'] = share
                row[f'optimal_{lever.name}_theta'] = avg_theta

            avg_utility = sum(d['utility'] for d in valid_datasets) / n_valid
            row['optimal_utility'] = avg_utility

            results.append(row)

    if verbose:
        print()  # newline after progress

    return pd.DataFrame(results)


# Default color palette for plotting
DEFAULT_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#8338EC', '#06D6A0']


def _extract_lever_names(results: pd.DataFrame, suffix: str) -> List[str]:
    """Extract lever names from columns matching 'optimal_*_{suffix}'."""
    import re
    names = []
    pattern = rf'optimal_(.+)_{suffix}'
    for col in results.columns:
        match = re.match(pattern, col)
        if match:
            names.append(match.group(1))
    return names


def _setup_plot_style():
    """Set up publication-quality plotting defaults."""
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 14


def plot_budget_shares(
    results: pd.DataFrame,
    ax=None,
    stacked: bool = True,
    show_levers: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    xlabel: str = 'Budget',
    ylabel: str = 'Budget Share',
    alpha: float = 0.8,
    figsize: Tuple[float, float] = (8, 5),
    linewidth: float = 6,
):
    """Plot budget shares for levers.

    Works for any number of levers (auto-detected from columns).

    Args:
        results: DataFrame from optimize_budget_allocation
        ax: Matplotlib axis. If None, creates new figure.
        stacked: If True, stacked area chart. If False, line plot.
        show_levers: List of lever names to include. If None, show all.
        labels: Custom labels (default: lever names from columns)
        colors: Custom colors (default: preset palette)
        xlabel: X-axis label
        ylabel: Y-axis label
        alpha: Transparency for stacked areas
        figsize: Figure size if creating new figure
        linewidth: Line width for line plots

    Returns:
        Matplotlib axis
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    all_lever_names = _extract_lever_names(results, 'share')

    if show_levers is not None:
        lever_names = [n for n in all_lever_names if n in show_levers]
        # Stacked area with a subset is misleading (won't sum to 1)
        if stacked and len(lever_names) < len(all_lever_names):
            stacked = False
    else:
        lever_names = all_lever_names

    n_levers = len(lever_names)

    if n_levers == 0:
        raise ValueError("No share columns found in results")

    if colors is None:
        colors = DEFAULT_COLORS[:n_levers]
    if labels is None:
        labels = lever_names

    if ax is None:
        _setup_plot_style()
        _, ax = plt.subplots(figsize=figsize)

    x = results['budget'].values
    share_cols = [f'optimal_{name}_share' for name in lever_names]
    shares = [results[col].values for col in share_cols]

    if stacked:
        # Stacked area chart
        cumulative = np.zeros_like(x)
        for i, (share, color, label) in enumerate(zip(shares, colors, labels)):
            ax.fill_between(x, cumulative, cumulative + share,
                           color=color, alpha=alpha, label=label)
            cumulative = cumulative + share
    else:
        # Line plot
        for share, color, label in zip(shares, colors, labels):
            ax.plot(x, share, linewidth=linewidth, color=color, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_budget_thetas(
    results: pd.DataFrame,
    show_levers: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    xlabel: str = 'Budget',
    ylabels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 5),
    linewidth: float = 6,
) -> List:
    """Plot optimal theta vs budget, one subplot per lever.

    Each lever gets its own full-size plot with independent y-axis,
    since thetas have different scales and units across levers.

    Args:
        results: DataFrame from optimize_budget_allocation
        show_levers: List of lever names to include. If None, show all.
        labels: Custom title labels per subplot (default: lever names)
        colors: Custom colors per subplot (default: preset palette)
        xlabel: X-axis label for all subplots
        ylabels: Custom y-axis labels per subplot (default: 'Optimal {name}')
        figsize: Figure size per subplot
        linewidth: Line width

    Returns:
        List of matplotlib axes, one per lever
    """
    import matplotlib.pyplot as plt

    lever_names = _extract_lever_names(results, 'theta')

    if show_levers is not None:
        lever_names = [n for n in lever_names if n in show_levers]

    n_levers = len(lever_names)

    if n_levers == 0:
        raise ValueError("No theta columns found in results")

    if colors is None:
        colors = DEFAULT_COLORS[:n_levers]
    if labels is None:
        labels = lever_names
    if ylabels is None:
        ylabels = [f'Optimal {name}' for name in lever_names]

    _setup_plot_style()
    axes = []
    x = results['budget'].values

    for name, color, label, ylabel in zip(lever_names, colors, labels, ylabels):
        fig, ax = plt.subplots(figsize=figsize)
        theta_col = f'optimal_{name}_theta'
        ax.plot(x, results[theta_col].values, linewidth=linewidth, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        axes.append(ax)

    return axes


