"""Lever comparison framework."""

from typing import Optional, Tuple, TYPE_CHECKING

from ..levers import ParameterizedLever

if TYPE_CHECKING:
    from ..problem import AllocationProblem
    from ..levers import PolicyLever


class LeverComparison:
    """Compare two policy levers.

    Provides methods to analyze and plot comparisons between levers,
    with availability depending on lever types and cost mappings.

    Example:
        >>> comparison = LeverComparison(
        ...     problem=problem,
        ...     lever_a=prediction_lever,
        ...     lever_b=capacity_lever,
        ... )
        >>> comparison.plot_welfare_difference(theta_range=(0.0, 0.5))
    """

    def __init__(
        self,
        problem: 'AllocationProblem',
        lever_a: 'PolicyLever',
        lever_b: 'PolicyLever',
    ):
        """Initialize lever comparison.

        Args:
            problem: Baseline allocation problem (includes policy)
            lever_a: First lever to compare
            lever_b: Second lever to compare
        """
        self.problem = problem
        self.lever_a = lever_a
        self.lever_b = lever_b

        # Cache for baseline welfare
        self._baseline_welfare: Optional[float] = None

    @property
    def a_is_parameterized(self) -> bool:
        """Whether lever_a is a ParameterizedLever."""
        return isinstance(self.lever_a, ParameterizedLever)

    @property
    def b_is_parameterized(self) -> bool:
        """Whether lever_b is a ParameterizedLever."""
        return isinstance(self.lever_b, ParameterizedLever)

    def _has_cost_mapping(self, lever: 'PolicyLever') -> bool:
        """Check if lever has a cost mapping (cost depends on theta)."""
        if not isinstance(lever, ParameterizedLever):
            return False
        cost = lever.compute_cost(self.problem)
        return cost is not None

    @property
    def a_has_cost_mapping(self) -> bool:
        """Whether lever_a has a cost mapping."""
        return self._has_cost_mapping(self.lever_a)

    @property
    def b_has_cost_mapping(self) -> bool:
        """Whether lever_b has a cost mapping."""
        return self._has_cost_mapping(self.lever_b)

    def get_baseline_welfare(self, welfare_metric: str = 'total_utility') -> float:
        """Get baseline welfare (cached)."""
        if self._baseline_welfare is None:
            result = self.problem.evaluate()
            self._baseline_welfare = result[welfare_metric]
        return self._baseline_welfare

    def can_plot_welfare_difference(self, swept_lever: str = 'a') -> bool:
        """Check if welfare difference plot is available.

        Requires the swept lever to be parameterized.
        """
        if swept_lever == 'a':
            return self.a_is_parameterized
        elif swept_lever == 'b':
            return self.b_is_parameterized
        else:
            raise ValueError(f"swept_lever must be 'a' or 'b', got {swept_lever}")

    def can_plot_equivalent_cost(self, swept_lever: str = 'a') -> bool:
        """Check if equivalent cost plot is available.

        Requires:
        - swept lever: parameterized (we sweep its theta)
        - reference lever: parameterized with cost mapping (we find cost to match welfare)

        Args:
            swept_lever: Which lever to sweep ('a' or 'b')
        """
        if swept_lever == 'a':
            return (
                self.a_is_parameterized
                and self.b_is_parameterized
                and self.b_has_cost_mapping
            )
        elif swept_lever == 'b':
            return (
                self.b_is_parameterized
                and self.a_is_parameterized
                and self.a_has_cost_mapping
            )
        else:
            raise ValueError(f"swept_lever must be 'a' or 'b', got {swept_lever}")

    def plot_welfare_difference(
        self,
        theta_range: Tuple[float, float],
        swept_lever: str = 'a',
        n_points: int = 50,
        ax=None,
        figsize: tuple = (10, 6),
        xlabel: Optional[str] = None,
        ylabel: str = 'Welfare Difference',
        welfare_metric: str = 'total_utility',
    ):
        """Plot welfare difference when sweeping one lever's theta.

        Sweeps the specified lever's theta and compares welfare against
        the other lever (held fixed at its current theta).

        Args:
            theta_range: (min, max) range for swept lever's theta
            swept_lever: Which lever to sweep ('a' or 'b')
            n_points: Number of points to evaluate
            ax: Matplotlib axes (creates new if None)
            figsize: Figure size if creating new axes
            xlabel: Custom x-axis label
            ylabel: Y-axis label
            welfare_metric: Metric to use for welfare comparison

        Returns:
            Matplotlib axes with the plot
        """
        from .welfare_difference import plot_welfare_difference

        if not self.can_plot_welfare_difference(swept_lever):
            lever = self.lever_a if swept_lever == 'a' else self.lever_b
            raise TypeError(
                f"Cannot plot welfare difference: {lever.name} is not parameterized"
            )

        return plot_welfare_difference(
            comparison=self,
            theta_range=theta_range,
            swept_lever=swept_lever,
            n_points=n_points,
            ax=ax,
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            welfare_metric=welfare_metric,
        )

    def plot_equivalent_cost(
        self,
        theta_range: Tuple[float, float],
        swept_lever: str = 'a',
        n_points: int = 50,
        ax=None,
        figsize: tuple = (10, 6),
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        welfare_metric: str = 'total_utility',
        reference_theta_search_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """Plot equivalent cost of reference lever to match swept lever's welfare.

        For each theta of swept lever (x-axis), shows the cost of reference lever
        needed to achieve the same welfare gain (y-axis).

        Args:
            theta_range: (min, max) range for swept lever's theta
            swept_lever: Which lever to sweep ('a' or 'b')
            n_points: Number of points to evaluate
            ax: Matplotlib axes (creates new if None)
            figsize: Figure size if creating new axes
            xlabel: Custom x-axis label
            ylabel: Custom y-axis label
            welfare_metric: Metric to use for welfare comparison
            reference_theta_search_range: Range to search for reference lever's theta

        Returns:
            Matplotlib axes with the plot
        """
        from .equivalent_cost import plot_equivalent_cost

        if not self.can_plot_equivalent_cost(swept_lever):
            if swept_lever == 'a':
                raise TypeError(
                    f"Cannot plot equivalent cost: requires lever_a to be parameterized, "
                    f"and lever_b to be parameterized with cost mapping. "
                    f"Got: lever_a parameterized={self.a_is_parameterized}; "
                    f"lever_b parameterized={self.b_is_parameterized}, "
                    f"cost_mapping={self.b_has_cost_mapping}"
                )
            else:
                raise TypeError(
                    f"Cannot plot equivalent cost: requires lever_b to be parameterized, "
                    f"and lever_a to be parameterized with cost mapping. "
                    f"Got: lever_b parameterized={self.b_is_parameterized}; "
                    f"lever_a parameterized={self.a_is_parameterized}, "
                    f"cost_mapping={self.a_has_cost_mapping}"
                )

        return plot_equivalent_cost(
            comparison=self,
            theta_range=theta_range,
            swept_lever=swept_lever,
            n_points=n_points,
            ax=ax,
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            welfare_metric=welfare_metric,
            reference_theta_search_range=reference_theta_search_range,
        )

    def can_plot_welfare_heatmap(self) -> bool:
        """Check if welfare heatmap plot is available.

        Requires both levers to be parameterized.
        """
        return self.a_is_parameterized and self.b_is_parameterized

    def plot_welfare_heatmap(
        self,
        theta_a_range: Tuple[float, float],
        theta_b_range: Tuple[float, float],
        n_points_a: int = 30,
        n_points_b: int = 30,
        ax=None,
        figsize: tuple = (10, 8),
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        welfare_metric: str = 'total_utility',
        cmap=None,  # defaults to custom diverging colormap
        show_colorbar: bool = True,
        colorbar_label: str = 'Welfare Gain Ratio (A / B)',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Plot welfare heatmap for both levers.

        Shows welfare gain at each (theta_a, theta_b) combination,
        applying both levers sequentially.

        Args:
            theta_a_range: (min, max) range for lever_a's theta (x-axis)
            theta_b_range: (min, max) range for lever_b's theta (y-axis)
            n_points_a: Number of points for lever_a
            n_points_b: Number of points for lever_b
            ax: Matplotlib axes (creates new if None)
            figsize: Figure size if creating new axes
            xlabel: Custom x-axis label
            ylabel: Custom y-axis label
            welfare_metric: Metric to use for welfare
            cmap: Colormap for heatmap
            show_colorbar: Whether to show colorbar
            colorbar_label: Label for colorbar
            vmin: Min value for colormap (default: auto symmetric around 1)
            vmax: Max value for colormap (default: auto symmetric around 1)

        Returns:
            Matplotlib axes with the plot
        """
        from .welfare_heatmap import plot_welfare_heatmap

        if not self.can_plot_welfare_heatmap():
            raise TypeError(
                f"Cannot plot welfare heatmap: both levers must be parameterized. "
                f"Got: lever_a parameterized={self.a_is_parameterized}; "
                f"lever_b parameterized={self.b_is_parameterized}"
            )

        return plot_welfare_heatmap(
            comparison=self,
            theta_a_range=theta_a_range,
            theta_b_range=theta_b_range,
            n_points_a=n_points_a,
            n_points_b=n_points_b,
            ax=ax,
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            welfare_metric=welfare_metric,
            cmap=cmap,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
            vmin=vmin,
            vmax=vmax,
        )

    def can_optimize_budget(self) -> bool:
        """Check if budget optimization is available.

        Requires both levers to be parameterized with cost mappings.
        """
        return (
            self.a_is_parameterized
            and self.b_is_parameterized
            and self.a_has_cost_mapping
            and self.b_has_cost_mapping
        )

    def optimize_budget(
        self,
        budget_range: Tuple[float, float],
        n_budget_points: int = 20,
        grid_density: int = 10,
        max_datasets: Optional[int] = None,
        welfare_metric: str = 'total_utility',
        verbose: bool = False,
    ):
        """Find optimal budget split between the two levers.

        For each total budget, searches over how to split it between
        lever_a and lever_b to maximize welfare.

        Optimizes per-dataset then averages the optimal decisions across datasets.

        Args:
            budget_range: (min, max) total budget to sweep
            n_budget_points: Number of budget points to evaluate
            grid_density: Number of split options to try per budget
            max_datasets: If set, only use first N datasets (for faster runs)
            welfare_metric: Which metric to optimize
            verbose: If True, print progress

        Returns:
            DataFrame with optimal allocations at each budget level
        """
        from .budget_optimization import optimize_two_lever_budget

        if not self.can_optimize_budget():
            raise TypeError(
                f"Cannot optimize budget: both levers must be parameterized "
                f"with cost mappings. Got: lever_a has_cost_mapping={self.a_has_cost_mapping}; "
                f"lever_b has_cost_mapping={self.b_has_cost_mapping}"
            )

        return optimize_two_lever_budget(
            problem=self.problem,
            lever1=self.lever_a,
            lever2=self.lever_b,
            budget_range=budget_range,
            n_budget_points=n_budget_points,
            grid_density=grid_density,
            max_datasets=max_datasets,
            welfare_metric=welfare_metric,
            verbose=verbose,
        )

    def plot_budget_optimization(
        self,
        results=None,
        budget_range: Optional[Tuple[float, float]] = None,
        n_budget_points: int = 20,
        grid_density: int = 10,
        max_datasets: Optional[int] = None,
        welfare_metric: str = 'total_utility',
        axes=None,
        figsize: Tuple[float, float] = (10, 6),
        xlabels: Optional[list] = None,
        ylabels: Optional[list] = None,
        color: str = '#2E86AB',
        linestyle: str = '-',
        label: Optional[str] = None,
        verbose: bool = False,
    ):
        """Plot optimal budget allocation results.

        Either pass pre-computed results or provide budget_range to compute them.

        Args:
            results: Pre-computed results from optimize_budget (optional)
            budget_range: (min, max) budget range (required if results not provided)
            n_budget_points: Number of budget points
            grid_density: Grid density for search
            max_datasets: Limit datasets for faster runs
            welfare_metric: Metric to optimize
            axes: Optional list of 3 axes. If None, creates 3 separate figures.
            figsize: Figure size for each plot if creating new figures
            xlabels: List of 3 x-axis labels (None for default)
            ylabels: List of 3 y-axis labels (None for default)
            color: Line color
            linestyle: Line style ('-', '--', ':', '-.')
            label: Label for legend
            verbose: Print progress

        Returns:
            List of 3 matplotlib axes
        """
        from .budget_optimization import plot_two_lever_optimization

        if results is None:
            if budget_range is None:
                raise ValueError("Must provide either results or budget_range")
            results = self.optimize_budget(
                budget_range=budget_range,
                n_budget_points=n_budget_points,
                grid_density=grid_density,
                max_datasets=max_datasets,
                welfare_metric=welfare_metric,
                verbose=verbose,
            )

        return plot_two_lever_optimization(
            results=results,
            lever1_name=self.lever_a.name,
            lever2_name=self.lever_b.name,
            axes=axes,
            figsize=figsize,
            xlabels=xlabels,
            ylabels=ylabels,
            color=color,
            linestyle=linestyle,
            label=label,
        )
