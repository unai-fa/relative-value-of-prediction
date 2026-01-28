"""Allocation Toolkit - A framework for allocation problems under resource constraints."""

from .data import AllocationData
from .problem import AllocationProblem
from .utilities import UtilityFunction, PartitionedUtility
from .constraints import ResourceConstraint, CoverageConstraint
from .policies import RankingPolicy

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
]

__version__ = "0.1.0"
