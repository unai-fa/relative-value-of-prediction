from abc import ABC, abstractmethod
import numpy as np

class UtilityFunction(ABC):
    """Abstract base class for utility functions.

    A utility function defines how to compute the value obtained from
    taking actions on units based on their true outcomes.
    """

    @abstractmethod
    def compute(self, y: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute utilities for each unit given outcomes and actions.

        Args:
            y: Ground truth outcomes of shape (n,)
            actions: Actions taken on each unit of shape (n,)

        Returns:
            Array of utilities of shape (n,)
        """
        pass
