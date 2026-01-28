"""Equivalent cost plot for lever comparison."""

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

if TYPE_CHECKING:
    from .two_lever_comparison import LeverComparison


def _find_equivalent_theta(
    comparison: 'LeverComparison',
    target_welfare: float,
    reference_lever: str,
    welfare_metric: str,
    theta_search_range: Tuple[float, float],
) -> Optional[float]:
    """Find theta for reference lever that achieves target welfare.

    Args:
        comparison: LeverComparison instance
        target_welfare: Target welfare to match
        reference_lever: Which lever to find theta for ('a' or 'b')
        welfare_metric: Metric to use for welfare
        theta_search_range: Range to search for theta

    Returns:
        theta that achieves target welfare, or None if not achievable
    """
    baseline_welfare = comparison.get_baseline_welfare(welfare_metric)
    target_gain = target_welfare - baseline_welfare

    if target_gain <= 0:
        return 0.0

    lever = comparison.lever_a if reference_lever == 'a' else comparison.lever_b

    def welfare_diff(theta):
        lever_at_theta = lever.with_theta(theta)
        problem_modified = lever_at_theta.apply(comparison.problem)
        result = problem_modified.evaluate()
        welfare = result[welfare_metric]
        return (welfare - baseline_welfare) - target_gain

    try:
        theta = brentq(welfare_diff, theta_search_range[0], theta_search_range[1])
        return theta
    except ValueError:
        # Target not achievable in range
        return None


def compute_equivalent_cost(
    comparison: 'LeverComparison',
    theta_range: Tuple[float, float],
    swept_lever: str = 'a',
    n_points: int = 50,
    welfare_metric: str = 'total_utility',
    reference_theta_search_range: Tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Compute equivalent cost of reference lever to match swept lever's welfare.

    For each theta of swept lever, finds theta of reference lever such that
    welfare matches, then computes cost of reference lever at that theta.

    Args:
        comparison: LeverComparison instance
        theta_range: (min, max) range for swept lever's theta
        swept_lever: Which lever to sweep ('a' or 'b')
        n_points: Number of points to evaluate
        welfare_metric: Metric to use for welfare
        reference_theta_search_range: Range to search for reference lever's theta

    Returns:
        DataFrame with columns:
        - theta: theta value of swept lever
        - swept_welfare: welfare from swept lever at this theta
        - reference_theta_equivalent: theta of reference lever that matches welfare
        - equivalent_cost: cost of reference lever at equivalent theta
        - baseline_welfare: welfare with no lever applied
    """
    theta_values = np.linspace(theta_range[0], theta_range[1], n_points)
    baseline_welfare = comparison.get_baseline_welfare(welfare_metric)

    # Determine which is swept vs reference
    if swept_lever == 'a':
        swept = comparison.lever_a
        reference = comparison.lever_b
        reference_lever_id = 'b'
    else:
        swept = comparison.lever_b
        reference = comparison.lever_a
        reference_lever_id = 'a'

    rows = []
    for theta in theta_values:
        # Evaluate swept lever at this theta
        swept_at_theta = swept.with_theta(theta)
        problem_swept = swept_at_theta.apply(comparison.problem)
        result_swept = problem_swept.evaluate()
        swept_welfare = result_swept[welfare_metric]

        # Find theta of reference lever that matches this welfare
        ref_theta_eq = _find_equivalent_theta(
            comparison, swept_welfare, reference_lever_id,
            welfare_metric, reference_theta_search_range
        )

        # Compute cost at equivalent theta
        if ref_theta_eq is not None:
            ref_at_theta = reference.with_theta(ref_theta_eq)
            equiv_cost = ref_at_theta.compute_cost(comparison.problem)
        else:
            equiv_cost = np.nan

        rows.append({
            'theta': theta,
            'swept_welfare': swept_welfare,
            'reference_theta_equivalent': ref_theta_eq if ref_theta_eq is not None else np.nan,
            'equivalent_cost': equiv_cost if equiv_cost is not None else np.nan,
            'baseline_welfare': baseline_welfare,
        })

    return pd.DataFrame(rows)


def plot_equivalent_cost(
    comparison: 'LeverComparison',
    theta_range: Tuple[float, float],
    swept_lever: str = 'a',
    n_points: int = 50,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 6),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    welfare_metric: str = 'total_utility',
    reference_theta_search_range: Tuple[float, float] = (0.0, 1.0),
) -> plt.Axes:
    """Plot equivalent cost of reference lever to match swept lever's welfare.

    For each theta of swept lever (x-axis), shows the cost of reference lever
    needed to achieve the same welfare gain (y-axis).

    Args:
        comparison: LeverComparison instance
        theta_range: (min, max) range for swept lever's theta
        swept_lever: Which lever to sweep ('a' or 'b')
        n_points: Number of points to evaluate
        ax: Matplotlib axes (creates new if None)
        figsize: Figure size if creating new axes
        xlabel: Custom x-axis label
        ylabel: Custom y-axis label
        welfare_metric: Metric to use for welfare
        reference_theta_search_range: Range to search for reference lever's theta

    Returns:
        Matplotlib axes with the plot
    """
    # Set up publication-quality plotting
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 14

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Compute equivalent costs
    df = compute_equivalent_cost(
        comparison, theta_range, swept_lever, n_points,
        welfare_metric, reference_theta_search_range
    )

    # Determine lever names
    if swept_lever == 'a':
        swept_name = comparison.lever_a.name
        reference_name = comparison.lever_b.name
    else:
        swept_name = comparison.lever_b.name
        reference_name = comparison.lever_a.name

    # Plot equivalent cost
    ax.plot(df['theta'], df['equivalent_cost'],
            linewidth=5, color='#2E86AB')
    ax.set_xlim(theta_range[0], theta_range[1])
    ax.set_ylim(bottom=0)

    # Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(rf'$\theta$ ({swept_name})')

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(f'Equivalent Cost ({reference_name})')

    ax.grid(alpha=0.3)

    plt.tight_layout()
    return ax
