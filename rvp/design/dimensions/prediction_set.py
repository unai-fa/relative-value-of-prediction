"""Prediction set dimension — discrete, externally-supplied prediction data."""

from typing import Hashable, Mapping

import numpy as np

from ...data import AllocationData
from ...problem import AllocationProblem
from ..dimension import DesignDimension


class PredictionSetDimension(DesignDimension):
    """Discrete dimension that swaps the ``AllocationData`` of the problem.

    The user supplies a mapping from a parameter value (e.g. feature
    fraction, training sample size, ``R^2``, ...) to a fully-formed
    ``AllocationData``. Each ``AllocationData`` may itself wrap multiple
    prediction sets (for stability averaging) — ``AllocationProblem.evaluate``
    will average across them automatically.

    Self-contained: the dimension carries all data it ever serves, so
    ``at(p, theta)`` is idempotent and never reads predictions from ``p``.

    Parameters
    ----------
    name : str
        Identifier for the dimension (e.g. "feature_fraction", "n_train").
    data_by_value : dict[hashable, AllocationData]
        Maps each admissible value of theta to the AllocationData to use.

    Examples
    --------
    >>> ff_dim = PredictionSetDimension(
    ...     name="feature_fraction",
    ...     data_by_value={
    ...         0.04: AllocationData(df=[df_run0, ..., df_run28]),
    ...         0.10: AllocationData(df=[df_run0, ..., df_run28]),
    ...         ...
    ...     }
    ... )
    """

    target_component = "data"

    def __init__(self, name: str, data_by_value: Mapping[Hashable, AllocationData]):
        if not data_by_value:
            raise ValueError("data_by_value must contain at least one entry")
        self.name = name
        # Preserve caller order, except for uniformly sortable numeric keys where
        # increasing order is the most useful default for plotting.
        keys = list(data_by_value.keys())
        if all(_is_numeric(key) for key in keys):
            keys = sorted(keys)
        self._data_by_value = {key: data_by_value[key] for key in keys}

    def at(self, problem: AllocationProblem, theta) -> AllocationProblem:
        if theta not in self._data_by_value:
            theta = self._resolve_numeric_key(theta)
        return AllocationProblem(
            data=self._data_by_value[theta],
            utility=problem.utility,
            constraint=problem.constraint,
            policy=problem.policy,
        )

    @property
    def values(self) -> np.ndarray:
        return np.array(list(self._data_by_value.keys()), dtype=object)

    @property
    def is_discrete(self) -> bool:
        return True

    def _resolve_numeric_key(self, theta):
        if not _is_numeric(theta):
            raise KeyError(
                f"theta={theta!r} is not in data_by_value "
                f"(keys: {list(self._data_by_value.keys())})"
            )

        keys = list(self._data_by_value.keys())
        numeric_keys = [key for key in keys if _is_numeric(key)]
        if len(numeric_keys) != len(keys):
            raise KeyError(
                f"theta={theta!r} is not in data_by_value "
                f"(keys: {list(self._data_by_value.keys())})"
            )

        key_array = np.asarray(numeric_keys, dtype=float)
        idx = int(np.argmin(np.abs(key_array - float(theta))))
        nearest = numeric_keys[idx]
        if not np.isclose(float(nearest), float(theta)):
            raise KeyError(
                f"theta={theta!r} is not in data_by_value "
                f"(keys: {list(self._data_by_value.keys())})"
            )
        return nearest


def _is_numeric(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
