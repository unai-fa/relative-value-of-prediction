"""Comparison and optimization helpers.

Budget optimization is intentionally deferred until the design-space cost
semantics are finalized.
"""

from .budget_optimization import optimize_budget, optimize_budget_frontier

__all__ = ["optimize_budget", "optimize_budget_frontier"]
