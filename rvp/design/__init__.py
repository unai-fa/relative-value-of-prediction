"""Design space abstractions: dimensions, spaces, costs.

Public API:
    DesignDimension      — abstract base class for one axis of the space
    DesignSpace          — collection of independent dimensions over a base problem
    CostSurface          — global cost function over a design space
    AdditiveCostSurface  — convenience: sum of per-dimension cost contributions

Concrete dimensions:
    CapacityDimension       — coverage-fraction (alpha)
    BenefitDimension        — CRRA transfer benefit (b)
    PredictionSetDimension  — discrete; swap AllocationData
    DataLabelingDimension   — fraction of population with revealed predictions
"""

from .cost import AdditiveCostSurface, CostSurface
from .dimension import DesignDimension
from .dimensions import (
    BenefitDimension,
    CapacityDimension,
    DataLabelingDimension,
    PredictionSetDimension,
)
from .space import DesignSpace

__all__ = [
    "DesignDimension",
    "DesignSpace",
    "CostSurface",
    "AdditiveCostSurface",
    "BenefitDimension",
    "CapacityDimension",
    "DataLabelingDimension",
    "PredictionSetDimension",
]
