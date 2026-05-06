"""Allocation Toolkit - A framework for allocation problems under resource constraints."""

from .data import AllocationData
from .problem import AllocationProblem
from .utilities import UtilityFunction, PartitionedUtility
from .constraints import ResourceConstraint, CoverageConstraint
from .policies import RankingPolicy
from .design import (
    AdditiveCostSurface,
    BenefitDimension,
    CapacityDimension,
    CostSurface,
    DataLabelingDimension,
    DesignDimension,
    DesignSpace,
    PredictionSetDimension,
)
from .plotting import plot_welfare_surface

__all__ = [
    # Core
    "AllocationData",
    "AllocationProblem",
    # Utilities
    "UtilityFunction",
    "PartitionedUtility",
    # Constraints
    "ResourceConstraint",
    "CoverageConstraint",
    # Policies
    "RankingPolicy",
    # Design spaces
    "DesignDimension",
    "DesignSpace",
    "CostSurface",
    "AdditiveCostSurface",
    # Dimensions
    "BenefitDimension",
    "CapacityDimension",
    "DataLabelingDimension",
    "PredictionSetDimension",
    # Plotting
    "plot_welfare_surface",
]

__version__ = "0.1.0"
