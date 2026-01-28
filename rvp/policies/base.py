"""Base class for allocation policies."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ..utilities import UtilityFunction
from ..constraints import ResourceConstraint


class Policy(ABC):
    """Base class for allocation policies.

    A policy transforms predictions into binary allocation actions.
    """

    @abstractmethod
    def __call__(
        self,
        predictions: np.ndarray,
        constraint: ResourceConstraint,
        utility: Optional[UtilityFunction] = None,
    ) -> np.ndarray:
        """Compute allocation given predictions and constraints.

        Args:
            predictions: Predicted outcomes of shape (n,)
            constraint: Resource constraint to satisfy
            utility: Utility function (optional, for policies that rank by expected utility)

        Returns:
            Binary actions array of shape (n,)
        """
        pass
