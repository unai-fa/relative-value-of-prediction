"""Welfare difference plot for lever comparison."""

from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .two_lever_comparison import LeverComparison


def compute_welfare_difference(
    comparison: 'LeverComparison',
    theta_range: Tuple[float, float],
    swept_lever: str = 'a',
    n_points: int = 50,
    welfare_metric: str = 'total_utility',
) -> pd.DataFrame:
    """Compute welfare difference when sweeping one lever's theta.

    Args:
        comparison: LeverComparison instance
        theta_range: (min, max) range for swept lever's theta
        swept_lever: Which lever to sweep ('a' or 'b')
        n_points: Number of points to evaluate
        welfare_metric: Metric to use for welfare

    Returns:
        DataFrame with columns:
        - theta: theta value of swept lever
        - swept_welfare: welfare from swept lever at this theta
        - reference_welfare: welfare from reference lever (fixed)
        - welfare_difference: swept_welfare - reference_welfare
        - baseline_welfare: welfare with no lever applied
    """
    theta_values = np.linspace(theta_range[0], theta_range[1], n_points)

    # Determine which lever is swept vs reference
    if swept_lever == 'a':
        swept = comparison.lever_a
        reference = comparison.lever_b
    else:
        swept = comparison.lever_b
        reference = comparison.lever_a

    # Get baseline welfare
    baseline_welfare = comparison.get_baseline_welfare(welfare_metric)

    # Evaluate reference lever (fixed)
    reference_problem = reference.apply(comparison.problem)
    reference_result = reference_problem.evaluate()
    reference_welfare = reference_result[welfare_metric]

    rows = []
    for theta in theta_values:
        # Create swept lever at this theta
        swept_at_theta = swept.with_theta(theta)
        swept_problem = swept_at_theta.apply(comparison.problem)
        swept_result = swept_problem.evaluate()
        swept_welfare = swept_result[welfare_metric]

        rows.append({
            'theta': theta,
            'swept_welfare': swept_welfare,
            'reference_welfare': reference_welfare,
            'welfare_difference': swept_welfare - reference_welfare,
            'baseline_welfare': baseline_welfare,
        })

    return pd.DataFrame(rows)


def find_breakeven(
    comparison: 'LeverComparison',
    theta_range: Tuple[float, float],
    swept_lever: str = 'a',
    n_points: int = 100,
    welfare_metric: str = 'total_utility',
) -> Optional[float]:
    """Find theta value where swept lever matches reference welfare.

    Args:
        comparison: LeverComparison instance
        theta_range: (min, max) range for swept lever's theta
        swept_lever: Which lever to sweep ('a' or 'b')
        n_points: Number of points for search
        welfare_metric: Metric to use for welfare

    Returns:
        Break-even theta value, or None if no break-even in range
    """
    df = compute_welfare_difference(
        comparison, theta_range, swept_lever, n_points, welfare_metric
    )

    # Find sign changes in welfare_difference
    sign_changes = np.where(np.diff(np.sign(df['welfare_difference'])))[0]

    if len(sign_changes) == 0:
        return None

    # Take first sign change and interpolate
    idx = sign_changes[0]
    x1, x2 = df.iloc[idx]['theta'], df.iloc[idx + 1]['theta']
    y1, y2 = df.iloc[idx]['welfare_difference'], df.iloc[idx + 1]['welfare_difference']

    # Linear interpolation: x = x1 - y1 * (x2 - x1) / (y2 - y1)
    breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)

    return breakeven


def plot_welfare_difference(
    comparison: 'LeverComparison',
    theta_range: Tuple[float, float],
    swept_lever: str = 'a',
    n_points: int = 50,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 6),
    xlabel: Optional[str] = None,
    ylabel: str = 'Welfare Difference',
    welfare_metric: str = 'total_utility',
) -> plt.Axes:
    """Plot welfare difference when sweeping one lever's theta.

    Shows where the swept lever becomes better than the reference lever.
    Shades regions to indicate which lever to prioritize.

    Args:
        comparison: LeverComparison instance
        theta_range: (min, max) range for swept lever's theta
        swept_lever: Which lever to sweep ('a' or 'b')
        n_points: Number of points to evaluate
        ax: Matplotlib axes (creates new if None)
        figsize: Figure size if creating new axes
        xlabel: Custom x-axis label
        ylabel: Y-axis label
        welfare_metric: Metric to use for welfare

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

    # Compute welfare difference
    df = compute_welfare_difference(
        comparison, theta_range, swept_lever, n_points, welfare_metric
    )

    # Get lever names for labels
    if swept_lever == 'a':
        swept_name = comparison.lever_a.name
        reference_name = comparison.lever_b.name
    else:
        swept_name = comparison.lever_b.name
        reference_name = comparison.lever_a.name

    # Plot welfare difference
    ax.plot(df['theta'], df['welfare_difference'],
            linewidth=3, color='#2E86AB')

    # Zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Shade regions
    ax.fill_between(
        df['theta'], 0, df['welfare_difference'],
        where=(df['welfare_difference'] >= 0),
        alpha=0.2, color='green',
        label=f'Prioritize {swept_name}'
    )
    ax.fill_between(
        df['theta'], 0, df['welfare_difference'],
        where=(df['welfare_difference'] < 0),
        alpha=0.2, color='red',
        label=f'Prioritize {reference_name}'
    )

    # Mark break-even point
    breakeven = find_breakeven(
        comparison, theta_range, swept_lever, n_points * 2, welfare_metric
    )
    if breakeven is not None:
        ax.axvline(x=breakeven, color='orange', linestyle=':', linewidth=2.5, zorder=5)
        ax.scatter([breakeven], [0], color='orange', s=150, zorder=5)

    # Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(r'$\theta$')

    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return ax
