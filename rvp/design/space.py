"""Design space — a collection of independent design dimensions.

The space holds a base AllocationProblem and a list of DesignDimensions.
Each ``config`` is a dict ``{dimension: theta}``; the space evaluates welfare
at any such config by composing the individual dimensions' transformations.
"""

from collections.abc import Callable, Mapping as MappingABC, Sequence as SequenceABC
from itertools import product
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..problem import AllocationProblem
from .dimension import DesignDimension


# Type aliases
Config = Mapping[DesignDimension, Any]


class DesignSpace:
    """A collection of design dimensions over a common base problem.

    Parameters
    ----------
    base_problem : AllocationProblem
        The reference problem; treated as immutable. Every call to ``at()``
        starts from this and applies dimensions in order.
    dimensions : Iterable[DesignDimension]
        The dimensions defining the space. No two may share the same
        ``target_component`` — this is validated at construction time.

    Notes
    -----
    Dimensions are applied in any order (they are required to operate on
    disjoint components of the AllocationProblem). The ``base_problem`` is
    never mutated.
    """

    def __init__(
        self,
        base_problem: AllocationProblem,
        dimensions: Iterable[DesignDimension],
    ):
        self.base_problem = base_problem
        self.dimensions = list(dimensions)
        self._validate_disjoint_targets()

    # -- public API -----------------------------------------------------

    def at(self, config: Config) -> AllocationProblem:
        """Return the AllocationProblem at the given config.

        Missing dimensions are not modified — they keep the base problem's
        component as-is. This is useful for partial slices.
        """
        problem = self.base_problem
        for dim, theta in config.items():
            if dim not in self.dimensions:
                raise KeyError(f"{dim!r} is not a dimension of this space")
            problem = dim.at(problem, theta)
        return problem

    def welfare_at(
        self,
        config: Config,
        metric: str = "mean_utility",
        subgroup_mask: Optional[np.ndarray | Callable] = None,
    ) -> float:
        """Evaluate welfare at the given config."""
        return float(self.at(config).evaluate(subgroup_mask=subgroup_mask)[metric])

    def welfare_surface(
        self,
        dim_x: DesignDimension,
        dim_y: DesignDimension,
        fixed: Optional[Config] = None,
        n: int = 50,
        metric: str = "mean_utility",
        subgroup_mask: Optional[np.ndarray | Callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate welfare on a 2D grid over (dim_x, dim_y).

        Parameters
        ----------
        dim_x, dim_y : DesignDimension
            Must both be in the space and distinct.
        fixed : dict, optional
            Values for the remaining dimensions. Any dimension not in
            ``fixed`` and not equal to ``dim_x`` / ``dim_y`` is left at its
            base-problem state.
        n : int
            Grid resolution for continuous dims. Ignored for discrete dims
            (those use their full ``values`` list).
        metric : str
            Welfare metric to read from ``AllocationProblem.evaluate()``.
        subgroup_mask : array-like or callable, optional
            If provided, evaluate welfare only for this subgroup. Callables are
            passed each dataset DataFrame and must return a boolean mask.

        Returns
        -------
        x_vals : np.ndarray (nx,)
        y_vals : np.ndarray (ny,)
        W : np.ndarray (ny, nx)
            ``W[j, i] = welfare at (dim_x=x_vals[i], dim_y=y_vals[j], **fixed)``.
        """
        if dim_x is dim_y:
            raise ValueError("dim_x and dim_y must be different dimensions")
        for dim in (dim_x, dim_y):
            if dim not in self.dimensions:
                raise KeyError(f"{dim!r} is not a dimension of this space")

        x_vals = self._grid_for(dim_x, n)
        y_vals = self._grid_for(dim_y, n)
        fixed = dict(fixed) if fixed else {}

        W = np.empty((len(y_vals), len(x_vals)), dtype=float)
        for j, y in enumerate(y_vals):
            for i, x in enumerate(x_vals):
                cfg = {**fixed, dim_x: x, dim_y: y}
                W[j, i] = self.welfare_at(
                    cfg,
                    metric=metric,
                    subgroup_mask=subgroup_mask,
                )
        return x_vals, y_vals, W

    def welfare_max(
        self,
        dims: Optional[Sequence[DesignDimension]] = None,
        fixed: Optional[Config] = None,
        n: Union[int, Sequence[int], Mapping[DesignDimension, int]] = 50,
        metric: str = "mean_utility",
    ) -> float:
        """Return the maximum welfare over a grid of design configurations.

        This is mainly useful for computing a shared normalization denominator
        once, then reusing it across multiple plots.
        """
        dims = list(self.dimensions if dims is None else dims)
        fixed = dict(fixed) if fixed else {}

        for dim in dims:
            if dim not in self.dimensions:
                raise KeyError(f"{dim!r} is not a dimension of this space")
            if dim in fixed:
                raise ValueError(f"{dim!r} cannot be both swept and fixed")

        if len(dims) == 0:
            return self.welfare_at(fixed, metric=metric)

        grids = [
            self._grid_for(dim, self._n_for_dim(n=n, dim=dim, index=i))
            for i, dim in enumerate(dims)
        ]
        max_welfare = -np.inf
        for values in product(*grids):
            cfg = {**fixed, **dict(zip(dims, values))}
            max_welfare = max(max_welfare, self.welfare_at(cfg, metric=metric))
        return float(max_welfare)

    # -- helpers --------------------------------------------------------

    def _validate_disjoint_targets(self) -> None:
        seen: Dict[str, DesignDimension] = {}
        for dim in self.dimensions:
            if dim.target_component in seen:
                raise ValueError(
                    f"Dimensions {seen[dim.target_component].name!r} and "
                    f"{dim.name!r} both target component "
                    f"{dim.target_component!r}. A DesignSpace may not contain "
                    f"two dimensions modifying the same component."
                )
            seen[dim.target_component] = dim

    @staticmethod
    def _grid_for(dim: DesignDimension, n: int) -> np.ndarray:
        return dim.grid(n)

    @staticmethod
    def _n_for_dim(
        n: Union[int, Sequence[int], Mapping[DesignDimension, int]],
        dim: DesignDimension,
        index: int,
    ) -> int:
        if isinstance(n, int):
            return int(n)
        if isinstance(n, MappingABC):
            return int(n.get(dim, 50))
        if isinstance(n, SequenceABC) and not isinstance(n, (str, bytes)):
            return int(n[index])
        return int(n)

    def __repr__(self) -> str:
        names = [d.name for d in self.dimensions]
        return f"DesignSpace(dimensions={names})"
