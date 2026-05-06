"""Budget optimization over design spaces."""

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from itertools import product
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ..design.dimension import DesignDimension
from ..design.space import DesignSpace


Config = Mapping[DesignDimension, Any]
GridSize = Union[int, Sequence[int], Mapping[DesignDimension, int]]


def optimize_budget(
    space: DesignSpace,
    cost_surface,
    budgets: Optional[Sequence[float]] = None,
    budget_range: Optional[Tuple[float, float]] = None,
    n_budget_points: int = 50,
    dims: Optional[Sequence[DesignDimension]] = None,
    fixed: Optional[Config] = None,
    status_quo: Optional[Config] = None,
    n: GridSize = 50,
    welfare_metric: str = "mean_utility",
    return_candidates: bool = False,
):
    """Find the welfare-maximizing design config at each budget.

    The optimizer evaluates a design grid once, then reuses those candidate
    configs for every budget. ``cost_surface`` is treated as a pure total-cost
    function. If ``status_quo`` is provided, budgets are interpreted as
    incremental and feasibility uses ``cost_surface.improvement_cost``.

    Exact welfare ties are broken by choosing the lowest-cost configuration.
    """
    budget_values = _resolve_budgets(
        budgets=budgets,
        budget_range=budget_range,
        n_budget_points=n_budget_points,
    )
    fixed = dict(fixed) if fixed else {}
    status_quo = dict(status_quo) if status_quo else None
    dims = _resolve_dims(space=space, dims=dims, fixed=fixed)

    candidates = _candidate_grid(
        space=space,
        cost_surface=cost_surface,
        dims=dims,
        fixed=fixed,
        status_quo=status_quo,
        n=n,
        welfare_metric=welfare_metric,
    )
    results = _optimize_over_budgets(
        candidates=candidates,
        budgets=budget_values,
        dims=dims,
    )

    if return_candidates:
        return results, candidates
    return results


def optimize_budget_frontier(
    space: DesignSpace,
    cost_surface,
    solve_dim: DesignDimension,
    budgets: Optional[Sequence[float]] = None,
    budget_range: Optional[Tuple[float, float]] = None,
    n_budget_points: int = 50,
    dims: Optional[Sequence[DesignDimension]] = None,
    fixed: Optional[Config] = None,
    status_quo: Optional[Config] = None,
    n: GridSize = 50,
    welfare_metric: str = "mean_utility",
    return_candidates: bool = False,
):
    """Optimize along exact-spend continuous budget frontiers.

    This optimizer grids all swept dimensions except ``solve_dim``. For each
    budget and partial grid config, it solves ``solve_dim`` so the candidate
    spends the budget exactly. It is intended for smooth, monotone continuous
    cost surfaces where exact budget frontiers are meaningful.
    """
    budget_values = _resolve_budgets(
        budgets=budgets,
        budget_range=budget_range,
        n_budget_points=n_budget_points,
    )
    fixed = dict(fixed) if fixed else {}
    status_quo = dict(status_quo) if status_quo else None
    dims = _resolve_dims(space=space, dims=dims, fixed=fixed)
    _validate_frontier_solve_dim(solve_dim=solve_dim, dims=dims)

    result_rows = []
    candidate_rows = []
    for budget in budget_values:
        candidates = _frontier_candidates_for_budget(
            space=space,
            cost_surface=cost_surface,
            budget=float(budget),
            dims=dims,
            solve_dim=solve_dim,
            fixed=fixed,
            status_quo=status_quo,
            n=n,
            welfare_metric=welfare_metric,
        )
        candidate_rows.extend(candidates)
        result_rows.append(
            _optimize_frontier_budget(
                candidates=candidates,
                budget=float(budget),
                dims=dims,
            )
        )

    results = pd.DataFrame(result_rows)
    if return_candidates:
        return results, pd.DataFrame(candidate_rows)
    return results


def _resolve_budgets(
    budgets: Optional[Sequence[float]],
    budget_range: Optional[Tuple[float, float]],
    n_budget_points: int,
) -> np.ndarray:
    if budgets is not None and budget_range is not None:
        raise ValueError("Use only one of budgets or budget_range")
    if budgets is None and budget_range is None:
        raise ValueError("Provide either budgets or budget_range")

    if budgets is not None:
        values = np.asarray(budgets, dtype=float)
        if values.ndim != 1:
            raise ValueError("budgets must be one-dimensional")
        return values

    if n_budget_points < 1:
        raise ValueError(f"n_budget_points must be >= 1, got {n_budget_points}")
    if len(budget_range) != 2:
        raise ValueError("budget_range must be a (low, high) tuple")
    low, high = budget_range
    return np.linspace(float(low), float(high), int(n_budget_points))


def _resolve_dims(
    space: DesignSpace,
    dims: Optional[Sequence[DesignDimension]],
    fixed: Config,
) -> Sequence[DesignDimension]:
    if dims is None:
        resolved = [dim for dim in space.dimensions if dim not in fixed]
    else:
        resolved = list(dims)

    for dim in resolved:
        if dim not in space.dimensions:
            raise KeyError(f"{dim!r} is not a dimension of this space")
        if dim in fixed:
            raise ValueError(f"{dim!r} cannot be both swept and fixed")
    for dim in fixed:
        if dim not in space.dimensions:
            raise KeyError(f"{dim!r} is not a dimension of this space")
    return resolved


def _validate_frontier_solve_dim(
    solve_dim: DesignDimension,
    dims: Sequence[DesignDimension],
) -> None:
    if solve_dim not in dims:
        raise ValueError("solve_dim must be one of the swept dims")
    if solve_dim.is_discrete:
        raise ValueError("solve_dim must be continuous")

    low, high = solve_dim.bounds
    if not all(np.isfinite([float(low), float(high)])):
        raise ValueError("solve_dim bounds must be finite")
    if float(low) >= float(high):
        raise ValueError("solve_dim bounds must satisfy low < high")


def _candidate_grid(
    space: DesignSpace,
    cost_surface,
    dims: Sequence[DesignDimension],
    fixed: Config,
    status_quo: Optional[Config],
    n: GridSize,
    welfare_metric: str,
) -> pd.DataFrame:
    grids = [
        dim.grid(_n_for_dim(n=n, dim=dim, index=i))
        for i, dim in enumerate(dims)
    ]
    rows = []
    for values in product(*grids):
        config = {**fixed, **dict(zip(dims, values))}
        below_status_quo = (
            status_quo is not None
            and _is_below_status_quo(config=config, status_quo=status_quo, dims=dims)
        )
        cost = (
            cost_surface.improvement_cost(config, status_quo)
            if status_quo is not None
            else cost_surface.cost(config)
        )
        row = {
            "config": config,
            "welfare": space.welfare_at(config, metric=welfare_metric),
            "cost": float(cost),
            "below_status_quo": bool(below_status_quo),
        }
        for dim, value in zip(dims, values):
            row[_candidate_theta_column(dim)] = value
        rows.append(row)

    return pd.DataFrame(rows)


def _frontier_candidates_for_budget(
    space: DesignSpace,
    cost_surface,
    budget: float,
    dims: Sequence[DesignDimension],
    solve_dim: DesignDimension,
    fixed: Config,
    status_quo: Optional[Config],
    n: GridSize,
    welfare_metric: str,
) -> list[dict[str, Any]]:
    grid_dims = [dim for dim in dims if dim != solve_dim]
    grid_indices = [dims.index(dim) for dim in grid_dims]
    grids = [
        dim.grid(_n_for_dim(n=n, dim=dim, index=index))
        for dim, index in zip(grid_dims, grid_indices)
    ]
    grid_values = product(*grids) if grids else [()]

    rows = []
    for values in grid_values:
        partial_config = {**fixed, **dict(zip(grid_dims, values))}
        solved_value = _solve_frontier_theta(
            cost_surface=cost_surface,
            budget=budget,
            solve_dim=solve_dim,
            partial_config=partial_config,
            status_quo=status_quo,
        )
        if solved_value is None:
            continue

        config = {**partial_config, solve_dim: solved_value}
        below_status_quo = (
            status_quo is not None
            and _is_below_status_quo(config=config, status_quo=status_quo, dims=dims)
        )
        if below_status_quo:
            continue

        cost = (
            cost_surface.improvement_cost(config, status_quo)
            if status_quo is not None
            else cost_surface.cost(config)
        )
        if not np.isfinite(float(cost)):
            continue

        row = {
            "budget": float(budget),
            "config": config,
            "welfare": space.welfare_at(config, metric=welfare_metric),
            "cost": float(cost),
            "below_status_quo": False,
        }
        for dim in dims:
            row[_candidate_theta_column(dim)] = config[dim]
        rows.append(row)

    return rows


def _solve_frontier_theta(
    cost_surface,
    budget: float,
    solve_dim: DesignDimension,
    partial_config: Config,
    status_quo: Optional[Config],
) -> Optional[float]:
    low, high = (float(value) for value in solve_dim.bounds)
    if (
        status_quo is not None
        and solve_dim in status_quo
        and _is_numeric(status_quo[solve_dim])
    ):
        low = max(low, float(status_quo[solve_dim]))
    if low > high:
        return None

    def objective(theta: float) -> float:
        config = {**partial_config, solve_dim: theta}
        cost = (
            cost_surface.improvement_cost(config, status_quo)
            if status_quo is not None
            else cost_surface.cost(config)
        )
        return float(cost) - float(budget)

    try:
        f_low = objective(low)
        f_high = objective(high)
    except (ArithmeticError, ValueError, TypeError, KeyError, OverflowError):
        return None

    if not np.isfinite(f_low) or not np.isfinite(f_high):
        return None
    if np.isclose(f_low, 0.0, rtol=1e-10, atol=1e-8):
        return float(low)
    if np.isclose(f_high, 0.0, rtol=1e-10, atol=1e-8):
        return float(high)
    if np.signbit(f_low) == np.signbit(f_high):
        return None

    try:
        return float(brentq(objective, low, high, xtol=1e-10, rtol=1e-10))
    except (ArithmeticError, ValueError, TypeError, RuntimeError, OverflowError):
        return None


def _optimize_frontier_budget(
    candidates: Sequence[Mapping[str, Any]],
    budget: float,
    dims: Sequence[DesignDimension],
) -> dict[str, Any]:
    row = {"budget": float(budget)}
    if not candidates:
        row.update(
            {
                "optimal_welfare": np.nan,
                "optimal_cost": np.nan,
                "unspent_budget": np.nan,
            }
        )
        for dim in dims:
            row[_result_theta_column(dim)] = np.nan
        return row

    welfare = np.asarray([candidate["welfare"] for candidate in candidates], dtype=float)
    costs = np.asarray([candidate["cost"] for candidate in candidates], dtype=float)
    feasible = np.isfinite(welfare) & np.isfinite(costs)
    if not np.any(feasible):
        row.update(
            {
                "optimal_welfare": np.nan,
                "optimal_cost": np.nan,
                "unspent_budget": np.nan,
            }
        )
        for dim in dims:
            row[_result_theta_column(dim)] = np.nan
        return row

    feasible_idx = np.flatnonzero(feasible)
    best_local = np.lexsort((costs[feasible_idx], -welfare[feasible_idx]))[0]
    best = candidates[int(feasible_idx[best_local])]

    row.update(
        {
            "optimal_welfare": float(best["welfare"]),
            "optimal_cost": float(best["cost"]),
            "unspent_budget": float(budget) - float(best["cost"]),
        }
    )
    for dim in dims:
        row[_result_theta_column(dim)] = best[_candidate_theta_column(dim)]
    return row


def _optimize_over_budgets(
    candidates: pd.DataFrame,
    budgets: np.ndarray,
    dims: Sequence[DesignDimension],
) -> pd.DataFrame:
    rows = []
    feasible_base = ~candidates["below_status_quo"].to_numpy(dtype=bool)
    costs = candidates["cost"].to_numpy(dtype=float)
    welfare = candidates["welfare"].to_numpy(dtype=float)

    for budget in budgets:
        feasible = feasible_base & (costs <= float(budget))
        row = {"budget": float(budget)}
        if not np.any(feasible):
            row.update(
                {
                    "optimal_welfare": np.nan,
                    "optimal_cost": np.nan,
                    "unspent_budget": np.nan,
                }
            )
            for dim in dims:
                row[_result_theta_column(dim)] = np.nan
            rows.append(row)
            continue

        feasible_idx = np.flatnonzero(feasible)
        best_local = np.lexsort((costs[feasible_idx], -welfare[feasible_idx]))[0]
        best_idx = feasible_idx[best_local]
        best = candidates.iloc[int(best_idx)]

        row.update(
            {
                "optimal_welfare": float(best["welfare"]),
                "optimal_cost": float(best["cost"]),
                "unspent_budget": float(budget) - float(best["cost"]),
            }
        )
        for dim in dims:
            row[_result_theta_column(dim)] = best[_candidate_theta_column(dim)]
        rows.append(row)

    return pd.DataFrame(rows)


def _n_for_dim(n: GridSize, dim: DesignDimension, index: int) -> int:
    if isinstance(n, int):
        return int(n)
    if isinstance(n, MappingABC):
        return int(n.get(dim, 50))
    if isinstance(n, SequenceABC) and not isinstance(n, (str, bytes)):
        return int(n[index])
    return int(n)


def _is_below_status_quo(
    config: Config,
    status_quo: Config,
    dims: Sequence[DesignDimension],
) -> bool:
    for dim in dims:
        if dim not in status_quo or dim not in config:
            continue
        value = config[dim]
        status_quo_value = status_quo[dim]
        if _is_numeric(value) and _is_numeric(status_quo_value):
            if float(value) < float(status_quo_value):
                return True
    return False


def _candidate_theta_column(dim: DesignDimension) -> str:
    return f"{dim.name}_theta"


def _result_theta_column(dim: DesignDimension) -> str:
    return f"optimal_{dim.name}_theta"


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
