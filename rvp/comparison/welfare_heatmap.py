"""Welfare heatmap for two parameterized levers."""

from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from .two_lever_comparison import LeverComparison

# Custom diverging colormap (blue -> yellow -> orange/red)
_DIVERGING_COLORS = ["#313695", "#4575b4", "#91bfdb", "#ffffbf", "#fdae61", "#f46d43", "#a50026"]
WELFARE_RATIO_CMAP = LinearSegmentedColormap.from_list("welfare_ratio", _DIVERGING_COLORS)


def compute_welfare_ratio(
    comparison: 'LeverComparison',
    theta_a_range: Tuple[float, float],
    theta_b_range: Tuple[float, float],
    n_points_a: int = 30,
    n_points_b: int = 30,
    welfare_metric: str = 'total_utility',
) -> pd.DataFrame:
    """Compute welfare gain ratio (lever_a / lever_b) for each (theta_a, theta_b).

    Args:
        comparison: LeverComparison instance with two parameterized levers
        theta_a_range: (min, max) range for lever_a's theta
        theta_b_range: (min, max) range for lever_b's theta
        n_points_a: Number of points for lever_a
        n_points_b: Number of points for lever_b
        welfare_metric: Metric to use for welfare

    Returns:
        DataFrame with columns: theta_a, theta_b, welfare_gain_a, welfare_gain_b, ratio
    """
    baseline_welfare = comparison.get_baseline_welfare(welfare_metric)

    theta_a_values = np.linspace(theta_a_range[0], theta_a_range[1], n_points_a)
    theta_b_values = np.linspace(theta_b_range[0], theta_b_range[1], n_points_b)

    results = []
    for theta_a in theta_a_values:
        for theta_b in theta_b_values:
            # Welfare gain from lever_a alone
            lever_a_at_theta = comparison.lever_a.with_theta(theta_a)
            problem_after_a = lever_a_at_theta.apply(comparison.problem)
            welfare_a = problem_after_a.evaluate()[welfare_metric]
            welfare_gain_a = welfare_a - baseline_welfare

            # Welfare gain from lever_b alone
            lever_b_at_theta = comparison.lever_b.with_theta(theta_b)
            problem_after_b = lever_b_at_theta.apply(comparison.problem)
            welfare_b = problem_after_b.evaluate()[welfare_metric]
            welfare_gain_b = welfare_b - baseline_welfare

            # Ratio (handle division by zero)
            if welfare_gain_b != 0:
                ratio = welfare_gain_a / welfare_gain_b
            else:
                ratio = np.inf if welfare_gain_a > 0 else (np.nan if welfare_gain_a == 0 else -np.inf)

            results.append({
                'theta_a': theta_a,
                'theta_b': theta_b,
                'welfare_gain_a': welfare_gain_a,
                'welfare_gain_b': welfare_gain_b,
                'ratio': ratio,
            })

    return pd.DataFrame(results)


def plot_welfare_heatmap(
    comparison: 'LeverComparison',
    theta_a_range: Tuple[float, float],
    theta_b_range: Tuple[float, float],
    n_points_a: int = 30,
    n_points_b: int = 30,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 6),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    welfare_metric: str = 'total_utility',
    cmap=None,  # defaults to custom diverging colormap
    show_colorbar: bool = True,
    colorbar_label: str = 'Welfare Gain Ratio (A / B)',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Axes:
    """Plot welfare gain ratio heatmap for two parameterized levers.

    Shows ratio of welfare_gain(lever_a) / welfare_gain(lever_b) at each
    (theta_a, theta_b) combination. Colormap is centered on 1 (equal value).

    Args:
        comparison: LeverComparison instance with two parameterized levers
        theta_a_range: (min, max) range for lever_a's theta (x-axis)
        theta_b_range: (min, max) range for lever_b's theta (y-axis)
        n_points_a: Number of points for lever_a
        n_points_b: Number of points for lever_b
        ax: Matplotlib axes (creates new if None)
        figsize: Figure size if creating new axes
        xlabel: Custom x-axis label
        ylabel: Custom y-axis label
        welfare_metric: Metric to use for welfare
        cmap: Colormap for heatmap (default custom blue-yellow-red)
        show_colorbar: Whether to show colorbar
        colorbar_label: Label for colorbar
        vmin: Min value for colormap (default: auto symmetric around 1)
        vmax: Max value for colormap (default: auto symmetric around 1)

    Returns:
        Matplotlib axes with the plot
    """
    created_fig = False
    if ax is None:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 24
        plt.rcParams['axes.labelsize'] = 24
        plt.rcParams['axes.titlesize'] = 24
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['legend.fontsize'] = 14
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Compute the ratio data
    df = compute_welfare_ratio(
        comparison, theta_a_range, theta_b_range,
        n_points_a, n_points_b, welfare_metric
    )

    # Pivot to 2D grid
    theta_a_values = df['theta_a'].unique()
    theta_b_values = df['theta_b'].unique()
    Z = df.pivot(index='theta_b', columns='theta_a', values='ratio').values

    # Create meshgrid for plotting
    X, Y = np.meshgrid(theta_a_values, theta_b_values)

    # Set up symmetric colormap centered on 1
    if vmin is None or vmax is None:
        # Find max deviation from 1 for symmetric colormap
        finite_Z = Z[np.isfinite(Z)]
        if len(finite_Z) > 0:
            max_dev = max(abs(finite_Z.max() - 1), abs(finite_Z.min() - 1))
            vmin = 1 - max_dev if vmin is None else vmin
            vmax = 1 + max_dev if vmax is None else vmax
        else:
            vmin, vmax = 0, 2

    # Create normalizer centered on 1
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)

    # Use custom colormap if not specified
    if cmap is None:
        cmap = WELFARE_RATIO_CMAP

    # Plot heatmap (no interpolation)
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto')

    # Add contour line at ratio = 1 (equal value) - black dashed
    ax.contour(X, Y, Z, levels=[1], colors='black', linewidths=2.5, linestyles='--')

    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)
        # Set ticks to include values below and above 1
        tick_values = [vmin, (vmin + 1) / 2, 1, (vmax + 1) / 2, vmax]
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{v:.1f}' for v in tick_values])

    # Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(rf'$\theta$ ({comparison.lever_a.name})')

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(rf'$\theta$ ({comparison.lever_b.name})')

    ax.set_xlim(theta_a_range[0], theta_a_range[1])
    ax.set_ylim(theta_b_range[0], theta_b_range[1])

    if created_fig:
        plt.tight_layout()

    return ax
