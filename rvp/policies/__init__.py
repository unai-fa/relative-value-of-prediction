"""Allocation policies."""

from .base import Policy
from .ranking import RankingPolicy

__all__ = [
    "Policy",
    "RankingPolicy",
]
