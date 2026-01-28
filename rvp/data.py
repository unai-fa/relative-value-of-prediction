from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional, List, Union


@dataclass
class AllocationData:
    """Data container for allocation problems.

    Supports single DataFrame or list of DataFrames. When multiple DataFrames
    are provided, evaluation methods will average results across them.

    Examples:
        # Single dataset
        data = AllocationData(df, ground_truth_col='y', predictions_col='pred')

        # Multiple datasets (e.g., different train/test seeds)
        data = AllocationData(
            df=[df_seed_0, df_seed_1, ...],
            ground_truth_col='y',
            predictions_col='pred'
        )
    """
    df: Union[pd.DataFrame, List[pd.DataFrame]]
    covariate_cols: Optional[List[str]] = None
    ground_truth_col: str = 'ground_truth'
    predictions_col: str = 'predictions'

    def __post_init__(self) -> None:
        # Normalize to list internally
        if isinstance(self.df, pd.DataFrame):
            self._dfs = [self.df]
        else:
            self._dfs = list(self.df)

        if len(self._dfs) == 0:
            raise ValueError("Must provide at least one DataFrame")

        # Validate each DataFrame
        for i, df in enumerate(self._dfs):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"df[{i}] must be a pandas DataFrame")

            if df.empty:
                raise ValueError(f"DataFrame {i} cannot be empty")

            if self.ground_truth_col not in df.columns:
                raise ValueError(
                    f"Column '{self.ground_truth_col}' not found in DataFrame {i}"
                )

            if self.predictions_col not in df.columns:
                raise ValueError(
                    f"Column '{self.predictions_col}' not found in DataFrame {i}"
                )

        # Infer covariate columns from first DataFrame if not specified
        if self.covariate_cols is None:
            reserved = {self.ground_truth_col, self.predictions_col}
            self.covariate_cols = [
                col for col in self._dfs[0].columns
                if col not in reserved
            ]

        # Validate covariate columns exist in all DataFrames
        for i, df in enumerate(self._dfs):
            missing_cols = set(self.covariate_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"Covariate columns not found in DataFrame {i}: {missing_cols}"
                )

        # Validate all datasets have same size (required for meaningful averaging)
        if len(self._dfs) > 1:
            sizes = [len(df) for df in self._dfs]
            if len(set(sizes)) > 1:
                raise ValueError(
                    f"All datasets must have the same size for averaging. "
                    f"Got sizes: {sizes[:5]}{'...' if len(sizes) > 5 else ''}"
                )

    @property
    def n_datasets(self) -> int:
        """Number of datasets."""
        return len(self._dfs)

    def get_dataset(self, index: int = 0) -> 'AllocationData':
        """Get a single-dataset AllocationData for dataset at index.

        Args:
            index: Index of the dataset (default: 0).

        Returns:
            AllocationData wrapping just that DataFrame.
        """
        return AllocationData(
            df=self._dfs[index],
            covariate_cols=self.covariate_cols,
            ground_truth_col=self.ground_truth_col,
            predictions_col=self.predictions_col,
        )

    @property
    def X(self) -> np.ndarray:
        """Feature matrix of shape (n, d). Uses first dataset."""
        return self._dfs[0][self.covariate_cols].values

    @property
    def y(self) -> np.ndarray:
        """Ground truth outcomes of shape (n,). Uses first dataset."""
        return self._dfs[0][self.ground_truth_col].values

    @property
    def predictions(self) -> np.ndarray:
        """Predictions of shape (n,). Uses first dataset."""
        return self._dfs[0][self.predictions_col].values

    @property
    def n(self) -> int:
        """Number of samples in first dataset."""
        return len(self._dfs[0])

    @property
    def df_single(self) -> pd.DataFrame:
        """The DataFrame (first one if multiple). For backwards compatibility."""
        return self._dfs[0]
