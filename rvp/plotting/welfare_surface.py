"""Welfare surface heatmap.

The primary visualization of the design space: a 2D welfare heatmap with
optional contour lines and an optional status-quo marker.
"""

from typing import Any, Iterable, Literal, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from ..design.dimension import DesignDimension
from ..design.space import DesignSpace


Normalization = Literal["max", None]


def plot_welfare_surface(
    space: DesignSpace,
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    fixed: Optional[Mapping[DesignDimension, Any]] = None,
    welfare_metric: str = "mean_utility",
    subgroup_mask: Optional[Any] = None,
    normalize: Optional[Normalization] = None,
    normalize_by: Optional[float] = None,
    status_quo: Optional[
        Union[
            Mapping[DesignDimension, Any],
            Iterable[Mapping[DesignDimension, Any]],
        ]
    ] = None,
    cost_surface=None,
    budget: Optional[float] = None,
    budget_overlay_n: int = 300,
    path_results=None,
    path_color: str = "#d7191c",
    path_underlay_color: str = "white",
    path_linewidth: float = 4.0,
    path_underlay_linewidth: float = 6.0,
    path_linestyle: str = "--",
    path_alpha: float = 1.0,
    path_label: Optional[str] = None,
    path_legend_loc: str = "lower right",
    path_break_threshold: Optional[float] = None,
    path_bridge_breaks: bool = False,
    path_bridge_color: str = "#2d004b",
    path_bridge_alpha: float = 0.85,
    path_cmap: Optional[str] = None,
    path_marker_size: float = 0.0,
    n: int = 50,
    contour_levels: Optional[np.ndarray] = None,
    contour_labels: bool = True,
    n_contours: int = 9,
    discrete_contour_guides: bool = False,
    discrete_contour_linewidth: float = 4.0,
    discrete_contour_alpha: float = 0.55,
    discrete_contour_marker_size: float = 46.0,
    status_quo_contour_clearance: float = 0.5,
    status_quo_gradient_label: bool = True,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    ax=None,
    cmap: str = "viridis",
    cbar_label: Optional[str] = None,
    tick_labelsize: int = 18,
    cbar_labelsize: int = 20,
    cbar_tick_labelsize: int = 18,
):
    """Plot welfare as a 2D heatmap over (dim_x, dim_y).

    Parameters
    ----------
    space : DesignSpace
    dim_x, dim_y : DesignDimension
        Axes of the heatmap; must be in ``space``.
    fixed : dict, optional
        Values for the remaining dimensions.
    welfare_metric : str
        Key into ``AllocationProblem.evaluate()`` (e.g. 'mean_utility',
        'total_utility', 'utility_ratio').
    subgroup_mask : array-like or callable, optional
        If provided, report welfare only for this subgroup. Callables are
        passed each dataset DataFrame and must return a boolean mask.
    normalize : {'max', None}
        Optional normalization. ``None`` → raw welfare. ``'max'`` divides by
        the maximum welfare value on the plotted design-space grid.
    normalize_by : float, optional
        Explicit normalization denominator. Use this to keep color scales
        comparable across smaller slices of the same design space. If provided,
        this takes precedence over ``normalize``.
    status_quo : dict or iterable of dicts, optional
        Current design config(s). Rendered as red stars. For continuous plotted
        dimensions, the plot also highlights the welfare contour through each
        status quo and draws the local welfare-gradient direction. Pass a single
        dict for one reference point or an iterable of dicts for several.
        Cost-related overlays (cost frontier, infeasible region, cost gradient)
        use only the first status quo when a sequence is supplied.
    cost_surface : CostSurface, optional
        Cost model over design-space configs. If provided with ``budget``,
        infeasible regions are shaded. If provided with ``status_quo``, the
        same-cost frontier through the status quo is highlighted.
    budget : float, optional
        Maximum allowed cost.
    budget_overlay_n : int
        Resolution for drawing the budget feasibility overlay. This only
        evaluates costs, not welfare.
    path_results : DataFrame, optional
        Output from ``optimize_budget``. If provided, the optimal path is
        projected onto the plotted ``dim_x`` / ``dim_y`` surface using columns
        named ``optimal_<dimension_name>_theta``.
    path_color, path_underlay_color : str
        Foreground and underlay colors for the projected path.
    path_linewidth, path_underlay_linewidth : float
        Foreground and underlay line widths.
    path_linestyle : str
        Matplotlib line style for the foreground path.
    path_alpha : float
        Foreground path opacity.
    path_label : str, optional
        Legend label for the path. If omitted, no legend entry is added.
    path_legend_loc : str
        Legend location when ``path_label`` is provided.
    n : int
        Grid resolution for continuous dimensions.
    contour_levels : array-like, optional
        Explicit contour levels; otherwise ``n_contours`` evenly-spaced
        levels are drawn.
    contour_labels : bool
        Whether to label contour lines.
    discrete_contour_guides : bool
        If exactly one plotted axis is discrete, draw equal-welfare guide
        curves by solving the continuous axis row-by-row instead of using
        Matplotlib's smooth contour interpolation across discrete categories.
        The resulting guide data are attached to the axes as
        ``ax._rvp_discrete_contour_guides`` for notebook-level annotations.
    status_quo_contour_clearance : float
        When ``status_quo`` is given and ``contour_levels`` is auto-generated,
        any auto level within ``clearance * spacing`` of the status-quo welfare
        level is dropped (where ``spacing`` is the distance between adjacent
        auto levels). ``0.5`` keeps at least half a contour gap of clearance;
        set to ``0`` to disable filtering.
    xlim, ylim : tuple, optional
        Axis limits to apply after plotting. If omitted, limits are inferred
        from the plotted design grid.
    ax : matplotlib axes, optional
    cmap : str
    cbar_label : str, optional
    tick_labelsize : int
    cbar_labelsize : int
    cbar_tick_labelsize : int

    Returns
    -------
    fig, ax
    """
    fixed = dict(fixed) if fixed else {}
    status_quos = _normalize_status_quos(status_quo)
    primary_status_quo = status_quos[0] if status_quos else None

    x_vals, y_vals, W = space.welfare_surface(
        dim_x=dim_x,
        dim_y=dim_y,
        fixed=fixed,
        n=n,
        metric=welfare_metric,
        subgroup_mask=subgroup_mask,
    )

    is_normalized = normalize is not None or normalize_by is not None
    normalization_denominator = None
    if normalize_by is not None:
        normalization_denominator = float(normalize_by)
        W = _normalize_by(W, denominator=normalize_by)
    elif normalize is not None:
        normalization_denominator = float(np.nanmax(W))
        W = _normalize(W, mode=normalize)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    x_plot, x_tick_labels = _axis_coordinates(x_vals, dim_x)
    y_plot, y_tick_labels = _axis_coordinates(y_vals, dim_y)
    display_x, display_y, display_W = x_plot, y_plot, W
    if is_normalized:
        display_W = np.clip(display_W, 0.0, 1.0)
    XX, YY = np.meshgrid(display_x, display_y)
    shading = "nearest" if (dim_x.is_discrete or dim_y.is_discrete) else "auto"

    vmin = 0.0 if is_normalized else None
    vmax = 1.0 if is_normalized else None

    pcm = ax.pcolormesh(
        XX, YY, display_W, cmap=cmap, shading=shading, vmin=vmin, vmax=vmax
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(
        cbar_label or _default_cbar_label(welfare_metric, normalize, normalize_by),
        fontsize=cbar_labelsize,
    )
    cbar.ax.tick_params(labelsize=cbar_tick_labelsize)

    # Contour lines
    use_discrete_guides = (
        discrete_contour_guides
        and (dim_x.is_discrete != dim_y.is_discrete)
        and min(display_W.shape) >= 2
    )
    if contour_levels is None:
        lo = float(np.nanmin(display_W))
        hi = float(np.nanmax(display_W))
        if hi > lo:
            contour_levels = np.linspace(lo, hi, n_contours + 2)[1:-1]
            if (
                status_quos
                and status_quo_contour_clearance > 0
            ):
                sq_levels = []
                for sq in status_quos:
                    sq_level = _status_quo_display_welfare(
                        space=space,
                        status_quo=sq,
                        fixed=fixed,
                        dim_x=dim_x,
                        dim_y=dim_y,
                        welfare_metric=welfare_metric,
                        subgroup_mask=subgroup_mask,
                        normalization_denominator=normalization_denominator,
                    )
                    if np.isfinite(sq_level):
                        sq_levels.append(sq_level)
                if sq_levels:
                    spacing = (hi - lo) / (n_contours + 1)
                    min_gap = float(status_quo_contour_clearance) * spacing
                    contour_levels = np.asarray(
                        [
                            lvl for lvl in contour_levels
                            if all(abs(lvl - s) >= min_gap for s in sq_levels)
                        ],
                        dtype=float,
                    )
    if (
        use_discrete_guides
        and contour_levels is not None
        and len(contour_levels) > 0
    ):
        ax._rvp_discrete_contour_guides = _draw_discrete_contour_guides(
            ax=ax,
            dim_x=dim_x,
            dim_y=dim_y,
            x_plot=x_plot,
            y_plot=y_plot,
            display_W=display_W,
            contour_levels=np.asarray(contour_levels, dtype=float),
            linewidth=float(discrete_contour_linewidth),
            alpha=float(discrete_contour_alpha),
            marker_size=float(discrete_contour_marker_size),
        )
    elif contour_levels is not None and len(contour_levels) > 0 and min(display_W.shape) >= 2:
        cs = ax.contour(XX, YY, display_W, levels=contour_levels, colors="black", linewidths=2)
        if contour_labels:
            ax.clabel(cs, fmt="%.2f", fontsize=13, inline=True, inline_spacing=6)
    if budget is not None:
        if cost_surface is None:
            raise ValueError("cost_surface must be provided when budget is provided")
        BXX, BYY, bx_vals, by_vals = _budget_overlay_grid(
            x_vals=x_vals,
            y_vals=y_vals,
            x_plot=x_plot,
            y_plot=y_plot,
            x_tick_labels=x_tick_labels,
            y_tick_labels=y_tick_labels,
            n=budget_overlay_n,
        )
        infeasible = _infeasible_mask(
            cost_surface=cost_surface,
            budget=float(budget),
            x_vals=bx_vals,
            y_vals=by_vals,
            dim_x=dim_x,
            dim_y=dim_y,
            fixed=fixed,
            status_quo=primary_status_quo,
        )
        _draw_infeasible_overlay(ax, BXX, BYY, infeasible)
    elif cost_surface is not None and primary_status_quo is None:
        raise ValueError(
            "cost_surface without budget requires status_quo so the same-cost "
            "frontier can be drawn"
        )

    if cost_surface is not None and primary_status_quo is not None:
        BXX, BYY, bx_vals, by_vals = _budget_overlay_grid(
            x_vals=x_vals,
            y_vals=y_vals,
            x_plot=x_plot,
            y_plot=y_plot,
            x_tick_labels=x_tick_labels,
            y_tick_labels=y_tick_labels,
            n=budget_overlay_n,
        )
        _draw_status_quo_cost_frontier(
            ax=ax,
            cost_surface=cost_surface,
            status_quo=primary_status_quo,
            x_vals=bx_vals,
            y_vals=by_vals,
            XX=BXX,
            YY=BYY,
            dim_x=dim_x,
            dim_y=dim_y,
            fixed=fixed,
            color="#dc5b0b",
            linewidth=4.0,
        )

    ax.set_xlabel(_axis_label(dim_x), fontsize=24)
    ax.set_ylabel(_axis_label(dim_y), fontsize=24)
    _format_axis(ax, "x", x_plot, x_tick_labels, tick_labelsize)
    _format_axis(ax, "y", y_plot, y_tick_labels, tick_labelsize)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    for idx, sq in enumerate(status_quos):
        is_primary = idx == 0
        _draw_status_quo_gradient(
            ax=ax,
            space=space,
            status_quo=sq,
            dim_x=dim_x,
            dim_y=dim_y,
            fixed=fixed,
            welfare_metric=welfare_metric,
            subgroup_mask=subgroup_mask,
            display_x=display_x,
            display_y=display_y,
            display_W=display_W,
            normalization_denominator=normalization_denominator,
            color="#d7191c",
            length_px=90.0,
            label=(
                r"$\nabla V_*$"
                if (
                    is_primary
                    and status_quo_gradient_label
                    and not (dim_x.is_discrete or dim_y.is_discrete)
                )
                else None
            ),
            contour_color="black",
            contour_linewidth=(
                float(discrete_contour_linewidth)
                if use_discrete_guides
                else 4.0
            ),
        )
        if is_primary and cost_surface is not None:
            _draw_status_quo_cost_gradient(
                ax=ax,
                cost_surface=cost_surface,
                status_quo=sq,
                dim_x=dim_x,
                dim_y=dim_y,
                fixed=fixed,
                color="#dc5b0b",
                length_px=90.0,
                label=r"$\nabla c$" if not (dim_x.is_discrete or dim_y.is_discrete) else None,
            )

    if path_results is not None:
        _draw_budget_path(
            ax=ax,
            results=path_results,
            dim_x=dim_x,
            dim_y=dim_y,
            color=path_color,
            underlay_color=path_underlay_color,
            linewidth=path_linewidth,
            underlay_linewidth=path_underlay_linewidth,
            linestyle=path_linestyle,
            alpha=path_alpha,
            label=path_label,
            legend_loc=path_legend_loc,
            break_threshold=path_break_threshold,
            bridge_breaks=path_bridge_breaks,
            bridge_color=path_bridge_color,
            bridge_alpha=path_bridge_alpha,
            cmap=path_cmap,
            marker_size=path_marker_size,
        )

    # Status-quo markers
    for idx, sq in enumerate(status_quos):
        x0 = _status_quo_coordinate(sq, dim_x, x_vals, x_plot)
        y0 = _status_quo_coordinate(sq, dim_y, y_vals, y_plot)
        if np.isfinite(x0) and np.isfinite(y0):
            ax.plot(
                x0, y0, "*",
                ms=22, color="#d7191c",
                markeredgecolor="white", markeredgewidth=1.5,
                zorder=12,
                label="Current system" if idx == 0 else None,
            )

    return fig, ax


# -------- helpers --------------------------------------------------------


def _normalize_status_quos(
    status_quo,
) -> list[dict]:
    """Accept either a single Mapping or an iterable of Mappings; return a list of dicts."""
    if status_quo is None:
        return []
    if isinstance(status_quo, Mapping):
        return [dict(status_quo)]
    return [dict(sq) for sq in status_quo]


def _draw_discrete_contour_guides(
    ax,
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    display_W: np.ndarray,
    contour_levels: np.ndarray,
    linewidth: float,
    alpha: float,
    marker_size: float,
):
    if dim_y.is_discrete and not dim_x.is_discrete:
        guides = _discrete_contour_guides_for_rows(
            continuous_coords=x_plot,
            discrete_coords=y_plot,
            welfare_grid=display_W,
            contour_levels=contour_levels,
        )
        orientation = "y_discrete"
    elif dim_x.is_discrete and not dim_y.is_discrete:
        guides = _discrete_contour_guides_for_rows(
            continuous_coords=y_plot,
            discrete_coords=x_plot,
            welfare_grid=display_W.T,
            contour_levels=contour_levels,
        )
        orientation = "x_discrete"
    else:
        return {"orientation": None, "levels": []}

    for guide in guides:
        rows = np.array(sorted(guide["crossings"]), dtype=int)
        continuous = np.array([guide["crossings"][idx] for idx in rows], dtype=float)
        discrete = np.asarray(guide["discrete_coords"], dtype=float)[rows]
        if orientation == "y_discrete":
            xs, ys = continuous, discrete
        else:
            xs, ys = discrete, continuous

        ax.plot(
            xs,
            ys,
            color="black",
            linestyle=":",
            linewidth=linewidth,
            alpha=alpha,
            zorder=8,
        )
        ax.scatter(
            xs,
            ys,
            s=marker_size,
            color="black",
            edgecolor="white",
            linewidth=0.9,
            alpha=0.9,
            zorder=9,
        )

    return {"orientation": orientation, "levels": guides}


def _discrete_contour_guides_for_rows(
    continuous_coords: np.ndarray,
    discrete_coords: np.ndarray,
    welfare_grid: np.ndarray,
    contour_levels: np.ndarray,
):
    guides = []
    for level in contour_levels:
        crossings = {
            row_idx: _continuous_coordinate_at_level(
                continuous_coords=continuous_coords,
                values=row_welfare,
                level=float(level),
            )
            for row_idx, row_welfare in enumerate(welfare_grid)
        }
        crossings = {
            row_idx: crossing
            for row_idx, crossing in crossings.items()
            if np.isfinite(crossing)
        }
        if len(crossings) >= 2:
            guides.append(
                {
                    "level": float(level),
                    "crossings": crossings,
                    "continuous_coords": np.asarray(continuous_coords, dtype=float),
                    "discrete_coords": np.asarray(discrete_coords, dtype=float),
                }
            )
    return guides


def _continuous_coordinate_at_level(
    continuous_coords: np.ndarray,
    values: np.ndarray,
    level: float,
) -> float:
    finite = np.isfinite(continuous_coords) & np.isfinite(values)
    x = np.asarray(continuous_coords, dtype=float)[finite]
    y = np.asarray(values, dtype=float)[finite]
    if len(x) < 2 or level < np.nanmin(y) or level > np.nanmax(y):
        return np.nan

    diff = y - float(level)
    exact = np.where(np.isclose(diff, 0.0, rtol=1e-6, atol=1e-8))[0]
    if len(exact) > 0:
        return float(x[int(exact[0])])

    crossings = np.where(diff[:-1] * diff[1:] < 0)[0]
    if len(crossings) == 0:
        return np.nan

    i = int(crossings[0])
    if np.isclose(y[i], y[i + 1]):
        return float(x[i])
    t = (float(level) - y[i]) / (y[i + 1] - y[i])
    return float(x[i] + t * (x[i + 1] - x[i]))


def _normalize(
    W: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Normalize welfare grid by the maximum plotted value."""
    if mode != "max":
        raise ValueError(f"Unknown normalize mode: {mode}")
    denom = float(np.nanmax(W))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom != 0, W / denom, np.nan)


def _normalize_by(W: np.ndarray, denominator: float) -> np.ndarray:
    """Normalize welfare grid by an explicit denominator."""
    denom = float(denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom != 0, W / denom, np.nan)


def _infeasible_mask(
    cost_surface,
    budget: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    fixed: Mapping[DesignDimension, Any],
    status_quo: Optional[Mapping[DesignDimension, Any]],
) -> np.ndarray:
    mask = np.empty((len(y_vals), len(x_vals)), dtype=bool)
    from_config = dict(status_quo) if status_quo is not None else None
    for j, y in enumerate(y_vals):
        for i, x in enumerate(x_vals):
            config = {**fixed, dim_x: x, dim_y: y}
            if from_config is not None and _is_below_status_quo(config, from_config):
                mask[j, i] = True
            elif from_config is None:
                cost = cost_surface.cost(config)
                mask[j, i] = cost > budget
            else:
                cost = cost_surface.improvement_cost(config, from_config)
                mask[j, i] = cost > budget
    return mask


def _budget_overlay_grid(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    x_tick_labels,
    y_tick_labels,
    n: int,
):
    if x_tick_labels is None and y_tick_labels is None and n > 1:
        config_x = np.linspace(float(x_plot[0]), float(x_plot[-1]), n)
        config_y = np.linspace(float(y_plot[0]), float(y_plot[-1]), n)
        plot_x = config_x
        plot_y = config_y
    else:
        config_x = x_vals
        config_y = y_vals
        plot_x = x_plot
        plot_y = y_plot
    XX, YY = np.meshgrid(plot_x, plot_y)
    return XX, YY, config_x, config_y


def _is_below_status_quo(config: Mapping[DesignDimension, Any], status_quo: Mapping[DesignDimension, Any]) -> bool:
    for dim, status_quo_value in status_quo.items():
        if dim not in config:
            continue
        value = config[dim]
        if _is_numeric(value) and _is_numeric(status_quo_value):
            if float(value) < float(status_quo_value):
                return True
    return False


def _draw_infeasible_overlay(ax, XX: np.ndarray, YY: np.ndarray, infeasible: np.ndarray):
    if not np.any(infeasible):
        return

    Z = infeasible.astype(float)
    ax.contourf(
        XX,
        YY,
        Z,
        levels=[0.5, 1.5],
        colors=["white"],
        alpha=0.55,
        zorder=4,
    )

    if np.any(infeasible) and np.any(~infeasible) and min(infeasible.shape) >= 2:
        ax.contour(
            XX,
            YY,
            Z,
            levels=[0.5],
            colors=["#f28e2b"],
            linewidths=2.5,
            zorder=5,
        )


def _draw_status_quo_cost_frontier(
    ax,
    cost_surface,
    status_quo: Mapping[DesignDimension, Any],
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    XX: np.ndarray,
    YY: np.ndarray,
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    fixed: Mapping[DesignDimension, Any],
    color: str,
    linewidth: float,
):
    if dim_x.is_discrete and dim_y.is_discrete:
        return

    status_config = _slice_config(status_quo=status_quo, fixed=fixed, dim_x=dim_x, dim_y=dim_y)
    target_cost = cost_surface.cost(status_config)

    if dim_x.is_discrete or dim_y.is_discrete:
        _draw_discrete_status_quo_cost_frontier(
            ax=ax,
            cost_surface=cost_surface,
            target_cost=float(target_cost),
            x_vals=x_vals,
            y_vals=y_vals,
            dim_x=dim_x,
            dim_y=dim_y,
            fixed=fixed,
            color=color,
            linewidth=linewidth,
        )
        return

    costs = np.empty((len(y_vals), len(x_vals)), dtype=float)
    for j, y in enumerate(y_vals):
        for i, x in enumerate(x_vals):
            config = {**fixed, dim_x: x, dim_y: y}
            costs[j, i] = cost_surface.cost(config)

    finite = np.isfinite(costs)
    if (
        min(costs.shape) < 2
        or not np.any(finite)
        or float(target_cost) < float(np.nanmin(costs))
        or float(target_cost) > float(np.nanmax(costs))
    ):
        return

    ax.contour(
        XX,
        YY,
        costs,
        levels=[float(target_cost)],
        colors=["white"],
        linewidths=linewidth + 2.5,
        zorder=5.6,
    )
    ax.contour(
        XX,
        YY,
        costs,
        levels=[float(target_cost)],
        colors=[color],
        linewidths=linewidth,
        zorder=5.7,
    )


def _draw_discrete_status_quo_cost_frontier(
    ax,
    cost_surface,
    target_cost: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    fixed: Mapping[DesignDimension, Any],
    color: str,
    linewidth: float,
):
    if dim_y.is_discrete and not dim_x.is_discrete:
        continuous_dim = dim_x
        discrete_dim = dim_y
        continuous_bounds = dim_x.bounds
        discrete_values = list(y_vals)
        orientation = "y_discrete"
    elif dim_x.is_discrete and not dim_y.is_discrete:
        continuous_dim = dim_y
        discrete_dim = dim_x
        continuous_bounds = dim_y.bounds
        discrete_values = list(x_vals)
        orientation = "x_discrete"
    else:
        return

    boundaries = []
    for value in discrete_values:
        base_config = {**fixed, discrete_dim: value}
        crossing = _cost_frontier_continuous_crossing(
            cost_surface=cost_surface,
            target_cost=float(target_cost),
            config=base_config,
            continuous_dim=continuous_dim,
            bounds=continuous_bounds,
        )
        boundaries.append(crossing)

    boundaries = np.asarray(boundaries, dtype=float)
    if np.count_nonzero(np.isfinite(boundaries)) < 2:
        return

    _draw_discrete_cost_tradeoff_guide(
        ax=ax,
        boundaries=boundaries,
        orientation=orientation,
        color=color,
        linewidth=max(1.5, linewidth - 1.5),
    )

    low, high = (float(value) for value in continuous_bounds)
    boundaries = np.clip(boundaries, low, high)
    row_edges = np.arange(-0.5, len(discrete_values) + 0.5, 1.0)

    step_x = []
    step_y = []
    for row_idx, boundary in enumerate(boundaries):
        if not np.isfinite(boundary):
            continue
        if orientation == "y_discrete":
            step_x.extend([boundary, boundary])
            step_y.extend([row_edges[row_idx], row_edges[row_idx + 1]])
        else:
            step_x.extend([row_edges[row_idx], row_edges[row_idx + 1]])
            step_y.extend([boundary, boundary])

        if row_idx < len(boundaries) - 1 and np.isfinite(boundaries[row_idx + 1]):
            next_boundary = boundaries[row_idx + 1]
            if orientation == "y_discrete":
                step_x.extend([boundary, next_boundary])
                step_y.extend([row_edges[row_idx + 1], row_edges[row_idx + 1]])
            else:
                step_x.extend([row_edges[row_idx + 1], row_edges[row_idx + 1]])
                step_y.extend([boundary, next_boundary])

    if len(step_x) < 2:
        return

    ax.plot(
        step_x,
        step_y,
        color="white",
        linewidth=linewidth + 3.0,
        alpha=0.9,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=5.6,
    )
    ax.plot(
        step_x,
        step_y,
        color=color,
        linewidth=linewidth,
        alpha=0.98,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=5.7,
    )


def _draw_discrete_cost_tradeoff_guide(
    ax,
    boundaries: np.ndarray,
    orientation: str,
    color: str,
    linewidth: float,
):
    rows = np.arange(len(boundaries), dtype=float)
    finite = np.isfinite(boundaries)
    if np.count_nonzero(finite) < 2:
        return

    if orientation == "y_discrete":
        xs, ys = boundaries[finite], rows[finite]
    else:
        xs, ys = rows[finite], boundaries[finite]

    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth + 2.5,
        linestyle=":",
        alpha=0.95,
        zorder=10.4,
    )
    ax.scatter(
        xs,
        ys,
        s=58,
        color=color,
        edgecolor="white",
        linewidth=1.1,
        alpha=0.98,
        zorder=10.5,
    )


def _cost_frontier_continuous_crossing(
    cost_surface,
    target_cost: float,
    config: Mapping[DesignDimension, Any],
    continuous_dim: DesignDimension,
    bounds,
) -> float:
    low, high = (float(value) for value in bounds)

    def gap(value):
        return float(cost_surface.cost({**config, continuous_dim: value}) - target_cost)

    g_low = gap(low)
    g_high = gap(high)
    if not np.isfinite(g_low) or not np.isfinite(g_high):
        return np.nan
    if np.isclose(g_low, 0.0, rtol=1e-8, atol=1e-10):
        return low
    if np.isclose(g_high, 0.0, rtol=1e-8, atol=1e-10):
        return high
    if g_low * g_high > 0:
        return np.nan

    lo, hi = low, high
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g_mid = gap(mid)
        if not np.isfinite(g_mid):
            return np.nan
        if np.isclose(g_mid, 0.0, rtol=1e-8, atol=1e-10):
            return mid
        if g_low * g_mid <= 0:
            hi = mid
            g_high = g_mid
        else:
            lo = mid
            g_low = g_mid
    return 0.5 * (lo + hi)


def _status_quo_display_welfare(
    space: DesignSpace,
    status_quo: Mapping[DesignDimension, Any],
    fixed: Mapping[DesignDimension, Any],
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    welfare_metric: str,
    subgroup_mask,
    normalization_denominator: Optional[float],
) -> float:
    config = _slice_config(status_quo=status_quo, fixed=fixed, dim_x=dim_x, dim_y=dim_y)
    welfare = float(space.welfare_at(config, metric=welfare_metric, subgroup_mask=subgroup_mask))
    if normalization_denominator is not None and normalization_denominator != 0:
        welfare = welfare / float(normalization_denominator)
    return welfare


def _draw_status_quo_gradient(
    ax,
    space: DesignSpace,
    status_quo: Mapping[DesignDimension, Any],
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    fixed: Mapping[DesignDimension, Any],
    welfare_metric: str,
    subgroup_mask,
    display_x: np.ndarray,
    display_y: np.ndarray,
    display_W: np.ndarray,
    normalization_denominator: Optional[float],
    color: str,
    length_px: float,
    label: Optional[str],
    contour_color: str,
    contour_linewidth: float,
):
    if dim_x.is_discrete or dim_y.is_discrete:
        _draw_discrete_status_quo_gradient(
            ax=ax,
            space=space,
            status_quo=status_quo,
            dim_x=dim_x,
            dim_y=dim_y,
            fixed=fixed,
            welfare_metric=welfare_metric,
            subgroup_mask=subgroup_mask,
            display_x=display_x,
            display_y=display_y,
            display_W=display_W,
            normalization_denominator=normalization_denominator,
            color=color,
            length_px=length_px,
            label=label,
            contour_color=contour_color,
            contour_linewidth=contour_linewidth,
        )
        return

    config = _slice_config(status_quo=status_quo, fixed=fixed, dim_x=dim_x, dim_y=dim_y)
    x0 = float(config[dim_x])
    y0 = float(config[dim_y])

    welfare_at_status_quo = space.welfare_at(
        config,
        metric=welfare_metric,
        subgroup_mask=subgroup_mask,
    )
    contour_level = float(welfare_at_status_quo)
    if normalization_denominator is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            contour_level = contour_level / float(normalization_denominator)

    if (
        min(display_W.shape) >= 2
        and np.isfinite(contour_level)
        and float(np.nanmin(display_W)) <= contour_level <= float(np.nanmax(display_W))
    ):
        XX, YY = np.meshgrid(display_x, display_y)
        ax.contour(
            XX,
            YY,
            display_W,
            levels=[contour_level],
            colors=[contour_color],
            linewidths=contour_linewidth,
            zorder=7.5,
        )

    gradient = _finite_difference_welfare_gradient(
        space=space,
        config=config,
        dim_x=dim_x,
        dim_y=dim_y,
        welfare_metric=welfare_metric,
        subgroup_mask=subgroup_mask,
    )
    if not np.all(np.isfinite(gradient)) or np.linalg.norm(gradient) == 0:
        return

    _draw_gradient_arrow(
        ax=ax,
        x0=x0,
        y0=y0,
        gradient=gradient,
        length_px=float(length_px),
        color=color,
        label=label,
        label_offset_points=(6, 8),
        label_verticalalignment="bottom",
    )


def _draw_discrete_status_quo_gradient(
    ax,
    space: DesignSpace,
    status_quo: Mapping[DesignDimension, Any],
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    fixed: Mapping[DesignDimension, Any],
    welfare_metric: str,
    subgroup_mask,
    display_x: np.ndarray,
    display_y: np.ndarray,
    display_W: np.ndarray,
    normalization_denominator: Optional[float],
    color: str,
    length_px: float,
    label: Optional[str],
    contour_color: str,
    contour_linewidth: float,
):
    if dim_x.is_discrete == dim_y.is_discrete:
        return

    config = _slice_config(status_quo=status_quo, fixed=fixed, dim_x=dim_x, dim_y=dim_y)
    welfare_at_status_quo = space.welfare_at(
        config,
        metric=welfare_metric,
        subgroup_mask=subgroup_mask,
    )
    contour_level = float(welfare_at_status_quo)
    if normalization_denominator is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            contour_level = contour_level / float(normalization_denominator)
    if not np.isfinite(contour_level):
        return

    if dim_y.is_discrete:
        discrete_dim = dim_y
        continuous_dim = dim_x
        continuous_coords = display_x
        discrete_coords = display_y
        welfare_rows = display_W
        row_idx = _discrete_value_index(discrete_dim, config[discrete_dim])
        if row_idx is None or row_idx < 0 or row_idx >= len(welfare_rows):
            return
        origin = (float(config[continuous_dim]), float(discrete_coords[row_idx]))
        make_point = lambda crossing, row: (crossing, float(discrete_coords[row]))
    else:
        discrete_dim = dim_x
        continuous_dim = dim_y
        continuous_coords = display_y
        discrete_coords = display_x
        welfare_rows = display_W.T
        row_idx = _discrete_value_index(discrete_dim, config[discrete_dim])
        if row_idx is None or row_idx < 0 or row_idx >= len(welfare_rows):
            return
        origin = (float(discrete_coords[row_idx]), float(config[continuous_dim]))
        make_point = lambda crossing, row: (float(discrete_coords[row]), crossing)

    crossings = {
        idx: _continuous_coordinate_at_level(
            continuous_coords=continuous_coords,
            values=row_welfare,
            level=contour_level,
        )
        for idx, row_welfare in enumerate(welfare_rows)
    }
    crossings = {
        idx: crossing
        for idx, crossing in crossings.items()
        if np.isfinite(crossing)
    }
    crossings[row_idx] = float(config[continuous_dim])
    _draw_discrete_status_quo_guide(
        ax=ax,
        crossings=crossings,
        discrete_coords=discrete_coords,
        orientation="y_discrete" if dim_y.is_discrete else "x_discrete",
        color=contour_color,
        linewidth=contour_linewidth,
    )

    next_rows = [idx for idx in crossings if idx > row_idx]
    if not next_rows:
        return

    next_row = min(next_rows)
    tangent_end = make_point(float(crossings[next_row]), next_row)
    _draw_normal_to_screen_segment(
        ax=ax,
        origin=origin,
        tangent_start=origin,
        tangent_end=tangent_end,
        length_px=float(length_px),
        color=color,
        label=label,
        label_offset_points=(6, 8),
        label_verticalalignment="bottom",
    )


def _draw_discrete_status_quo_guide(
    ax,
    crossings: Mapping[int, float],
    discrete_coords: np.ndarray,
    orientation: str,
    color: str,
    linewidth: float,
):
    if len(crossings) < 2:
        return

    rows = np.array(sorted(crossings), dtype=int)
    continuous = np.array([crossings[row] for row in rows], dtype=float)
    discrete = np.asarray(discrete_coords, dtype=float)[rows]
    if orientation == "y_discrete":
        xs, ys = continuous, discrete
    else:
        xs, ys = discrete, continuous

    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=":",
        linewidth=linewidth,
        alpha=0.55,
        zorder=8.6,
    )
    ax.scatter(
        xs,
        ys,
        s=46,
        color=color,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.9,
        zorder=8.7,
    )


def _draw_normal_to_screen_segment(
    ax,
    origin: tuple[float, float],
    tangent_start: tuple[float, float],
    tangent_end: tuple[float, float],
    length_px: float,
    color: str,
    label: Optional[str],
    label_offset_points: tuple[float, float],
    label_verticalalignment: str,
):
    origin_px = ax.transData.transform(origin)
    tangent_px = ax.transData.transform(tangent_end) - ax.transData.transform(tangent_start)
    if not np.all(np.isfinite(tangent_px)) or np.linalg.norm(tangent_px) == 0:
        return

    normal_px = np.array([-tangent_px[1], tangent_px[0]], dtype=float)
    normal_px = normal_px / np.linalg.norm(normal_px)
    if normal_px[0] < 0 or normal_px[1] < 0:
        normal_px = -normal_px

    end = ax.transData.inverted().transform(origin_px + normal_px * float(length_px))
    ax.annotate(
        "",
        xy=end,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", color=color, lw=4, mutation_scale=30),
        zorder=10,
    )
    if label is None:
        return

    ax.annotate(
        label,
        xy=end,
        xytext=label_offset_points,
        textcoords="offset points",
        ha="left",
        va=label_verticalalignment,
        fontsize=22,
        color=color,
        fontweight="bold",
        zorder=12,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="black",
            alpha=1.0,
        ),
    )


def _draw_status_quo_cost_gradient(
    ax,
    cost_surface,
    status_quo: Mapping[DesignDimension, Any],
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    fixed: Mapping[DesignDimension, Any],
    color: str,
    length_px: float,
    label: Optional[str],
):
    config = _slice_config(status_quo=status_quo, fixed=fixed, dim_x=dim_x, dim_y=dim_y)

    if dim_x.is_discrete and dim_y.is_discrete:
        return

    if dim_x.is_discrete or dim_y.is_discrete:
        x0 = _config_axis_coordinate(config, dim_x)
        y0 = _config_axis_coordinate(config, dim_y)
        gradient = _mixed_discrete_cost_gradient(
            cost_surface=cost_surface,
            config=config,
            dim_x=dim_x,
            dim_y=dim_y,
        )
    else:
        x0 = float(config[dim_x])
        y0 = float(config[dim_y])
        gradient = _finite_difference_cost_gradient(
            cost_surface=cost_surface,
            config=config,
            dim_x=dim_x,
            dim_y=dim_y,
        )

    if not np.all(np.isfinite(gradient)) or np.linalg.norm(gradient) == 0:
        return

    _draw_gradient_arrow(
        ax=ax,
        x0=x0,
        y0=y0,
        gradient=gradient,
        length_px=float(length_px),
        color=color,
        label=label,
        label_offset_points=(6, -8),
        label_verticalalignment="top",
    )


def _draw_gradient_arrow(
    ax,
    x0: float,
    y0: float,
    gradient: np.ndarray,
    length_px: float,
    color: str,
    label: Optional[str],
    label_offset_points: tuple[float, float],
    label_verticalalignment: str,
):
    dx, dy = _screen_normal_delta(
        ax=ax,
        x0=x0,
        y0=y0,
        gradient=gradient,
        length_px=length_px,
    )
    ax.annotate(
        "",
        xy=(x0 + dx, y0 + dy),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=4, mutation_scale=30),
        zorder=10,
    )
    if label is None:
        return

    ax.annotate(
        label,
        xy=(x0 + dx, y0 + dy),
        xytext=label_offset_points,
        textcoords="offset points",
        ha="left",
        va=label_verticalalignment,
        fontsize=22,
        color=color,
        fontweight="bold",
        zorder=12,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="black",
            alpha=1.0,
        ),
    )


def _finite_difference_welfare_gradient(
    space: DesignSpace,
    config: Mapping[DesignDimension, Any],
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    welfare_metric: str,
    subgroup_mask,
) -> np.ndarray:
    return np.asarray(
        [
            _finite_difference(
                lambda value: space.welfare_at(
                    {**config, dim_x: value},
                    metric=welfare_metric,
                    subgroup_mask=subgroup_mask,
                ),
                x0=float(config[dim_x]),
                bounds=dim_x.bounds,
            ),
            _finite_difference(
                lambda value: space.welfare_at(
                    {**config, dim_y: value},
                    metric=welfare_metric,
                    subgroup_mask=subgroup_mask,
                ),
                x0=float(config[dim_y]),
                bounds=dim_y.bounds,
            ),
        ],
        dtype=float,
    )


def _finite_difference_cost_gradient(
    cost_surface,
    config: Mapping[DesignDimension, Any],
    dim_x: DesignDimension,
    dim_y: DesignDimension,
) -> np.ndarray:
    return np.asarray(
        [
            _finite_difference(
                lambda value: cost_surface.cost({**config, dim_x: value}),
                x0=float(config[dim_x]),
                bounds=dim_x.bounds,
            ),
            _finite_difference(
                lambda value: cost_surface.cost({**config, dim_y: value}),
                x0=float(config[dim_y]),
                bounds=dim_y.bounds,
            ),
        ],
        dtype=float,
    )


def _mixed_discrete_cost_gradient(
    cost_surface,
    config: Mapping[DesignDimension, Any],
    dim_x: DesignDimension,
    dim_y: DesignDimension,
) -> np.ndarray:
    return np.asarray(
        [
            _cost_axis_difference(cost_surface, config, dim_x),
            _cost_axis_difference(cost_surface, config, dim_y),
        ],
        dtype=float,
    )


def _cost_axis_difference(
    cost_surface,
    config: Mapping[DesignDimension, Any],
    dim: DesignDimension,
) -> float:
    if not dim.is_discrete:
        return _finite_difference(
            lambda value: cost_surface.cost({**config, dim: value}),
            x0=float(config[dim]),
            bounds=dim.bounds,
        )

    values = list(dim.values)
    idx = _discrete_value_index(dim, config[dim])
    if idx is None:
        return np.nan

    current_cost = cost_surface.cost(config)
    if idx < len(values) - 1:
        next_config = {**config, dim: values[idx + 1]}
        return float(cost_surface.cost(next_config) - current_cost)
    if idx > 0:
        prev_config = {**config, dim: values[idx - 1]}
        return float(current_cost - cost_surface.cost(prev_config))
    return np.nan


def _finite_difference(fn, x0: float, bounds) -> float:
    low, high = (float(value) for value in bounds)
    width = high - low
    h = 0.01 * width
    x0 = float(x0)

    if h <= 0:
        return np.nan

    if low <= x0 - h and x0 + h <= high:
        return (float(fn(x0 + h)) - float(fn(x0 - h))) / (2.0 * h)
    if x0 + h <= high:
        return (float(fn(x0 + h)) - float(fn(x0))) / h
    if low <= x0 - h:
        return (float(fn(x0)) - float(fn(x0 - h))) / h
    return np.nan


def _screen_normal_delta(ax, x0: float, y0: float, gradient: np.ndarray, length_px: float):
    origin = ax.transData.transform((x0, y0))
    x_scale = ax.transData.transform((x0 + 1, y0))[0] - origin[0]
    y_scale = ax.transData.transform((x0, y0 + 1))[1] - origin[1]
    screen_normal = np.array([gradient[0] / x_scale, gradient[1] / y_scale])
    norm = np.linalg.norm(screen_normal)
    if norm == 0 or not np.isfinite(norm):
        return 0.0, 0.0
    screen_normal = float(length_px) * screen_normal / norm
    tip = ax.transData.inverted().transform(origin + screen_normal)
    return float(tip[0] - x0), float(tip[1] - y0)


def _slice_config(
    status_quo: Mapping[DesignDimension, Any],
    fixed: Mapping[DesignDimension, Any],
    dim_x: DesignDimension,
    dim_y: DesignDimension,
) -> dict[DesignDimension, Any]:
    config = {**fixed, **dict(status_quo)}
    missing = [dim.name for dim in (dim_x, dim_y) if dim not in config]
    if missing:
        raise ValueError(f"status_quo must include plotted dimensions: {missing}")
    return config


def _draw_budget_path(
    ax,
    results,
    dim_x: DesignDimension,
    dim_y: DesignDimension,
    color: str,
    underlay_color: str,
    linewidth: float,
    underlay_linewidth: float,
    linestyle: str,
    alpha: float,
    label: Optional[str],
    legend_loc: str,
    break_threshold: Optional[float] = None,
    bridge_breaks: bool = False,
    bridge_color: str = "#2d004b",
    bridge_alpha: float = 0.85,
    cmap: Optional[str] = None,
    marker_size: float = 0.0,
):
    x_col = _theta_column(dim_x)
    y_col = _theta_column(dim_y)
    missing = [col for col in (x_col, y_col) if col not in results.columns]
    if missing:
        raise ValueError(f"Missing optimizer result columns for path overlay: {missing}")

    x_values = np.asarray(results[x_col], dtype=object)
    y_values = np.asarray(results[y_col], dtype=object)
    keep = np.asarray([
        not (_is_missing(x) or _is_missing(y))
        for x, y in zip(x_values, y_values)
    ])
    if not np.any(keep):
        return

    x = _path_axis_coordinates(x_values[keep], dim_x)
    y = _path_axis_coordinates(y_values[keep], dim_y)
    if len(x) < 2:
        return

    budgets = None
    if "budget" in results.columns:
        budgets = np.asarray(results["budget"], dtype=float)[keep]

    segments = _split_path_at_breaks(ax, x, y, break_threshold)
    if bridge_breaks:
        _draw_path_break_bridges(
            ax=ax,
            x=x,
            y=y,
            segments=segments,
            color=bridge_color,
            linewidth=max(1.0, float(linewidth) * 0.75),
            alpha=bridge_alpha,
        )

    if cmap is not None and budgets is not None and len(np.unique(budgets)) > 1:
        _draw_path_gradient(
            ax=ax,
            x=x, y=y,
            budgets=budgets,
            segments=segments,
            cmap_name=cmap,
            linewidth=linewidth,
            underlay_color=underlay_color,
            underlay_linewidth=underlay_linewidth,
            linestyle=linestyle,
            alpha=alpha,
            marker_size=marker_size,
            label=label,
        )
    else:
        first_drawn_label = label
        for seg in segments:
            if len(seg) < 2:
                continue
            ax.plot(
                x[seg], y[seg],
                color=underlay_color,
                linewidth=underlay_linewidth,
                linestyle="-",
                solid_capstyle="round",
                zorder=5.5,
            )
            ax.plot(
                x[seg], y[seg],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                solid_capstyle="round",
                zorder=6,
                label=first_drawn_label,
            )
            first_drawn_label = None
        if marker_size > 0:
            ax.scatter(
                x, y,
                s=float(marker_size),
                c=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=6.5,
            )

    if label is not None:
        leg = ax.legend(
            loc=legend_loc,
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
        )
        leg.set_zorder(10)


def _draw_path_break_bridges(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    segments,
    color: str,
    linewidth: float,
    alpha: float,
):
    if len(segments) < 2:
        return

    for left, right in zip(segments[:-1], segments[1:]):
        if len(left) == 0 or len(right) == 0:
            continue

        bridge_x = [x[left[-1]], x[right[0]]]
        bridge_y = [y[left[-1]], y[right[0]]]
 
        ax.plot(
            bridge_x,
            bridge_y,
            color=color,
            linewidth=linewidth,
            linestyle=":",
            solid_capstyle="round",
            alpha=alpha,
            zorder=5.5,
        )


def _split_path_at_breaks(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    break_threshold: Optional[float],
) -> list[np.ndarray]:
    """Return index groups where consecutive points stay within ``break_threshold``
    of the axis diagonal (in normalized data coords). ``None`` → one segment."""
    indices = np.arange(len(x))
    if break_threshold is None or break_threshold <= 0 or len(x) < 2:
        return [indices]

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x_range = float(x_hi - x_lo) or 1.0
    y_range = float(y_hi - y_lo) or 1.0

    dx = np.diff(x) / x_range
    dy = np.diff(y) / y_range
    dist = np.sqrt(dx * dx + dy * dy)
    breaks = np.where(dist > float(break_threshold))[0]
    if len(breaks) == 0:
        return [indices]
    return np.split(indices, breaks + 1)


def _draw_path_gradient(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    budgets: np.ndarray,
    segments,
    cmap_name: str,
    linewidth: float,
    underlay_color: str,
    underlay_linewidth: float,
    linestyle: str,
    alpha: float,
    marker_size: float,
    label: Optional[str],
):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    cmap_obj = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=float(np.nanmin(budgets)), vmax=float(np.nanmax(budgets)))

    legend_proxy_drawn = False
    for seg in segments:
        if len(seg) < 2:
            continue
        seg_pts = np.column_stack([x[seg], y[seg]])
        lc_segs = np.stack([seg_pts[:-1], seg_pts[1:]], axis=1)
        seg_budgets = (budgets[seg[:-1]] + budgets[seg[1:]]) / 2.0

        ax.plot(
            x[seg], y[seg],
            color=underlay_color,
            linewidth=underlay_linewidth,
            linestyle="-",
            solid_capstyle="round",
            zorder=5.5,
        )

        lc = LineCollection(
            lc_segs,
            cmap=cmap_obj,
            norm=norm,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=6,
        )
        lc.set_array(seg_budgets)
        ax.add_collection(lc)
        if not legend_proxy_drawn and label is not None:
            mid_color = cmap_obj(norm(float(np.nanmedian(budgets))))
            ax.plot(
                [], [],
                color=mid_color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=label,
            )
            legend_proxy_drawn = True

    if marker_size > 0:
        ax.scatter(
            x, y,
            c=budgets,
            cmap=cmap_obj,
            norm=norm,
            s=float(marker_size),
            edgecolor="white",
            linewidth=1.0,
            zorder=7,
        )


def _theta_column(dim: DesignDimension) -> str:
    return f"optimal_{dim.name}_theta"


def _path_axis_coordinates(values: np.ndarray, dim: DesignDimension) -> np.ndarray:
    if dim.is_discrete:
        dim_values = list(dim.values)
        if all(_is_numeric(value) for value in dim_values):
            return np.asarray(values, dtype=float)
        mapping = {value: i for i, value in enumerate(dim_values)}
        return np.asarray([mapping[value] for value in values], dtype=float)
    return np.asarray(values, dtype=float)


def _is_missing(value) -> bool:
    return value is None or (_is_numeric(value) and np.isnan(float(value)))


def _axis_label(dim: DesignDimension) -> str:
    return dim.name


def _default_cbar_label(
    metric: str,
    normalize: Optional[Normalization],
    normalize_by: Optional[float] = None,
) -> str:
    if normalize_by is not None:
        return "Welfare / reference max"
    if normalize == "max":
        return "Welfare / max"
    return metric.replace("_", " ").capitalize()


def _axis_coordinates(values: np.ndarray, dim: DesignDimension):
    if dim.is_discrete:
        if all(_is_numeric(v) for v in values):
            coords = np.asarray(values, dtype=float)
            labels = None
        else:
            coords = np.arange(len(values), dtype=float)
            labels = [str(v) for v in values]
        return coords, labels
    return np.asarray(values, dtype=float), None


def _format_axis(ax, axis: str, coords: np.ndarray, tick_labels, tick_labelsize: int):
    if len(coords) == 0:
        return

    if tick_labels is not None:
        ticks, labels = _select_categorical_ticks(coords, tick_labels)
        if axis == "x":
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)

    ax.tick_params(axis=axis, labelsize=tick_labelsize)

    lo, hi = _axis_limits(coords)
    if axis == "x":
        ax.set_xlim(lo, hi)
    else:
        ax.set_ylim(lo, hi)


def _select_categorical_ticks(coords: np.ndarray, labels, max_ticks: int = 12):
    if len(coords) <= max_ticks:
        return coords, labels
    idx = np.unique(np.linspace(0, len(coords) - 1, max_ticks).round().astype(int))
    return coords[idx], [labels[i] for i in idx]


def _axis_limits(coords: np.ndarray):
    if len(coords) == 1:
        return float(coords[0] - 0.5), float(coords[0] + 0.5)
    sorted_coords = np.sort(np.asarray(coords, dtype=float))
    left_step = sorted_coords[1] - sorted_coords[0]
    right_step = sorted_coords[-1] - sorted_coords[-2]
    return float(sorted_coords[0] - left_step / 2), float(sorted_coords[-1] + right_step / 2)


def _status_quo_coordinate(status_quo, dim, values, coords) -> float:
    if dim not in status_quo:
        return np.nan
    value = status_quo[dim]
    if dim.is_discrete:
        for candidate, coord in zip(values, coords):
            if _values_equal(value, candidate):
                return float(coord)
        raise ValueError(
            f"status_quo value {value!r} is not valid for discrete dimension "
            f"{dim.name!r}; expected one of {list(values)!r}"
        )
    return float(value)


def _config_axis_coordinate(config: Mapping[DesignDimension, Any], dim: DesignDimension) -> float:
    if dim.is_discrete:
        idx = _discrete_value_index(dim, config[dim])
        return np.nan if idx is None else float(idx)
    return float(config[dim])


def _discrete_value_index(dim: DesignDimension, value) -> Optional[int]:
    try:
        return list(dim.values).index(value)
    except ValueError:
        return None


def _values_equal(left, right) -> bool:
    if _is_numeric(left) and _is_numeric(right):
        return bool(np.isclose(float(left), float(right)))
    return left == right


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
