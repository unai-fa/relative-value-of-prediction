"""Cost surface — a global cost function over a design space.

This is intentionally minimal. Detailed semantics for budget optimization
(marginal costs, cost interactions, status-quo-relative pricing strategies)
will be revisited when budget optimization is designed.
"""

from typing import Any, Callable, Dict, Mapping

from .dimension import DesignDimension


Config = Mapping[DesignDimension, Any]


class CostSurface:
    """A global cost function over a design space.

    Wraps a callable ``fn(config) -> float`` that maps a configuration
    (``{dimension: theta}``) to a non-negative cost.

    Parameters
    ----------
    fn : Callable[[Config], float]
        The cost function. Receives the full config dict so that arbitrary
        cross-dimension interactions are expressible.
    """

    def __init__(self, fn: Callable[[Config], float]):
        self.fn = fn

    def cost(self, config: Config) -> float:
        return float(self.fn(config))

    def improvement_cost(self, config: Config, from_config: Config) -> float:
        """Cost of moving from ``from_config`` to ``config``.

        For now, just the difference of total costs. Refine later if needed.
        """
        return self.cost(config) - self.cost(from_config)


class AdditiveCostSurface(CostSurface):
    """Cost as a sum of per-dimension contributions.

    Each dimension is mapped to a callable ``g(theta, config) -> float``
    that returns its contribution. Dimensions absent from a given config
    contribute zero.

    Parameters
    ----------
    contributions : dict[DesignDimension, Callable[[Any, Config], float]]
        Per-dimension cost contribution functions.
    """

    def __init__(
        self,
        contributions: Dict[DesignDimension, Callable[[Any, Config], float]],
    ):
        self.contributions = dict(contributions)

        def _fn(config: Config) -> float:
            total = 0.0
            for dim, theta in config.items():
                g = self.contributions.get(dim)
                if g is not None:
                    total += float(g(theta, config))
            return total

        super().__init__(_fn)
