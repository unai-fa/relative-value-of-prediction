"""Welfare curve plot for a single parameterized lever."""

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from ..problem import AllocationProblem
    from ..levers import ParameterizedLever


def compute_welfare_curve(
    problem: 'AllocationProblem',
    lever: 'ParameterizedLever',
    theta_range: Tuple[float, float],
    n_points: int = 50,
    welfare_metric: str = 'total_utility',
    subgroup_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute welfare across theta range for a single lever.

    Args:
        problem: Baseline allocation problem
        lever: Parameterized lever to sweep
        theta_range: (min, max) range for theta
        n_points: Number of points to evaluate
        welfare_metric: Metric to use for welfare
        subgroup_mask: Optional boolean mask to compute welfare only for a subgroup

    Returns:
        DataFrame with columns:
        - theta: theta value
        - welfare: welfare at this theta
        - welfare_gain: welfare - baseline_welfare
        - baseline_welfare: welfare with no lever applied
    """
    theta_values = np.linspace(theta_range[0], theta_range[1], n_points)

    # Get baseline welfare
    baseline_result = problem.evaluate(subgroup_mask=subgroup_mask)
    baseline_welfare = baseline_result[welfare_metric]

    rows = []
    for theta in theta_values:
        lever_at_theta = lever.with_theta(theta)
        problem_modified = lever_at_theta.apply(problem)
        result = problem_modified.evaluate(subgroup_mask=subgroup_mask)
        welfare = result[welfare_metric]

        rows.append({
            'theta': theta,
            'welfare': welfare,
            'welfare_gain': welfare - baseline_welfare,
            'baseline_welfare': baseline_welfare,
        })

    return pd.DataFrame(rows)


def plot_welfare_curve(
    problem: 'AllocationProblem',
    lever: 'ParameterizedLever',
    theta_range: Tuple[float, float],
    n_points: int = 50,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 6),
    xlabel: Optional[str] = None,
    ylabel: str = 'Welfare Gain',
    welfare_metric: str = 'total_utility',
    show_baseline: bool = False,
    color: str = '#2E86AB',
    linestyle: str = '-',
    ylim_max: Optional[float] = None,
    label: Optional[str] = None,
    subgroup_mask: Optional[np.ndarray] = None,
) -> plt.Axes:
    """Plot welfare gain vs theta for a single lever.

    Args:
        problem: Baseline allocation problem
        lever: Parameterized lever to sweep
        theta_range: (min, max) range for theta
        n_points: Number of points to evaluate
        ax: Matplotlib axes (creates new if None)
        figsize: Figure size if creating new axes
        xlabel: Custom x-axis label
        ylabel: Y-axis label
        welfare_metric: Metric to use for welfare
        show_baseline: If True, show absolute welfare; if False, show gain
        subgroup_mask: Optional boolean mask to compute welfare only for a subgroup

    Returns:
        Matplotlib axes with the plot
    """
    created_fig = False
    if ax is None:
        # Set up publication-quality plotting only when creating new figure
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 24
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['legend.fontsize'] = 14
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Compute welfare curve
    df = compute_welfare_curve(
        problem, lever, theta_range, n_points, welfare_metric, subgroup_mask
    )

    # Plot
    y_col = 'welfare' if show_baseline else 'welfare_gain'
    ax.plot(df['theta'], df[y_col], linewidth=6, color=color, linestyle=linestyle, label=label if label else lever.name)

    if not show_baseline:
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_xlim(theta_range[0], theta_range[1])
    if ylim_max is not None:
        ax.set_ylim(top=ylim_max)

    # Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(rf'$\theta$ ({lever.name})')

    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    if created_fig:
        plt.tight_layout()

    return ax
