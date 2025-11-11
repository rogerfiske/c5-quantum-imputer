"""
Base Imputation Class for Quantum-Inspired Feature Engineering

This module defines the abstract base class that all quantum-inspired imputation
strategies must inherit from. It provides a standardized interface for transforming
raw binary quantum state data into feature-engineered representations suitable for
ranking model training.

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.1 - Base Imputation Class
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)


class BaseImputer(ABC):
    """
    Abstract base class for quantum-inspired imputation strategies.

    This class defines the standard interface that all imputation methods
    must implement. Each imputation strategy transforms raw binary quantum
    state data into feature-engineered representations suitable for ranking
    model training.

    The imputation workflow follows scikit-learn's transformer pattern:
    1. fit(X): Learn strategy-specific parameters from training data
    2. transform(X): Apply learned transformation to data
    3. fit_transform(X): Convenience method combining fit and transform

    All concrete imputation classes must implement the abstract methods
    _fit() and _transform() which contain the strategy-specific logic.

    Attributes:
        name (str): Human-readable name of the imputation strategy
        config (Dict[str, Any]): Configuration parameters for the strategy
        fitted_ (bool): Whether the imputer has been fitted to training data

    Examples:
        >>> # Example concrete implementation
        >>> class MyImputer(BaseImputer):
        ...     def _fit(self, X):
        ...         # Learn strategy-specific parameters
        ...         self.mean_values_ = X.mean()
        ...
        ...     def _transform(self, X):
        ...         # Apply transformation
        ...         return X.values * self.mean_values_.values
        ...
        >>> # Usage
        >>> from src.data_loader import load_dataset
        >>> df = load_dataset()
        >>> imputer = MyImputer(name="my_strategy")
        >>> imputer.fit(df)
        >>> features = imputer.transform(df)
        >>> print(features.shape)  # (n_samples, n_features)

    Notes:
        - Input data must be a pandas DataFrame with 40 columns:
          event-ID, QV_1, QV_2, ..., QV_39
        - Output is always a numpy array of shape (n_samples, n_features)
        - n_features varies by strategy and is documented in each subclass
        - All QV columns must be binary (0 or 1) with exactly 5 ones per row
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base imputer.

        Args:
            name: Human-readable name of the imputation strategy
            config: Optional configuration dictionary for strategy-specific parameters

        Examples:
            >>> imputer = BaseImputer(name="my_strategy", config={"param1": 10})
        """
        self.name = name
        self.config = config if config is not None else {}
        self.fitted_ = False

        logger.debug(f"Initialized {self.name} imputer with config: {self.config}")

    @abstractmethod
    def _fit(self, X: pd.DataFrame) -> None:
        """
        Learn imputation parameters from training data.

        This is an abstract method that must be implemented by all concrete
        imputation strategies. It should learn any strategy-specific parameters
        from the training data (e.g., position frequencies, statistical moments).

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Raises:
            NotImplementedError: If not implemented by subclass

        Notes:
            - This method should store learned parameters as instance attributes
              with trailing underscores (e.g., self.position_frequencies_)
            - Input validation is performed by the public fit() method
            - This method should not return anything; it modifies self in-place
        """
        pass

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using learned parameters.

        This is an abstract method that must be implemented by all concrete
        imputation strategies. It applies the learned transformation to input
        data to produce feature-engineered representations.

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Returns:
            Transformed feature matrix of shape (n_samples, n_features)
            where n_features depends on the specific imputation strategy.
            Data type: np.float64

        Raises:
            NotImplementedError: If not implemented by subclass
            RuntimeError: If called before fit()

        Notes:
            - Output must be a numpy array (not DataFrame)
            - Output must not contain NaN or Inf values
            - n_features is strategy-specific and documented in each subclass
            - Input validation is performed by the public transform() method
        """
        pass

    def fit(self, X: pd.DataFrame) -> "BaseImputer":
        """
        Fit the imputer to training data (public method with validation).

        This method validates the input data, then calls the strategy-specific
        _fit() method to learn parameters from the training data.

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Returns:
            self: The fitted imputer instance (enables method chaining)

        Raises:
            ValueError: If input data format is invalid

        Examples:
            >>> from src.data_loader import load_dataset
            >>> df = load_dataset()
            >>> imputer = MyImputer(name="my_strategy")
            >>> imputer.fit(df)  # Returns self
            >>> print(imputer.fitted_)  # True
        """
        logger.info(f"Fitting {self.name} imputer on {len(X)} samples")

        # Validate input data format
        self._validate_input(X)

        # Call strategy-specific fit implementation
        self._fit(X)

        # Mark as fitted
        self.fitted_ = True

        logger.info(f"{self.name} imputer fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted parameters (public method with validation).

        This method validates that the imputer has been fitted, validates the
        input data, then calls the strategy-specific _transform() method to
        apply the learned transformation.

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Returns:
            Transformed feature matrix of shape (n_samples, n_features)
            Data type: np.float64

        Raises:
            RuntimeError: If imputer has not been fitted yet
            ValueError: If input data format is invalid
            ValueError: If output contains NaN or Inf values

        Examples:
            >>> from src.data_loader import load_dataset
            >>> df = load_dataset()
            >>> imputer = MyImputer(name="my_strategy")
            >>> imputer.fit(df)
            >>> features = imputer.transform(df)
            >>> print(features.shape)  # (11581, n_features)
        """
        # Check if fitted
        if not self.fitted_:
            raise RuntimeError(
                f"{self.name} imputer must be fitted before calling transform(). "
                f"Call fit() first to learn parameters from training data."
            )

        logger.info(f"Transforming {len(X)} samples with {self.name} imputer")

        # Validate input data format
        self._validate_input(X)

        # Call strategy-specific transform implementation
        features = self._transform(X)

        # Validate output
        self._validate_output(features)

        logger.info(f"Transformation complete: output shape {features.shape}")
        return features

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit to data, then transform it (convenience method).

        This is a convenience method that combines fit() and transform() into
        a single call. It is equivalent to calling fit(X) followed by transform(X).

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Returns:
            Transformed feature matrix of shape (n_samples, n_features)
            Data type: np.float64

        Raises:
            ValueError: If input data format is invalid
            ValueError: If output contains NaN or Inf values

        Examples:
            >>> from src.data_loader import load_dataset
            >>> df = load_dataset()
            >>> imputer = MyImputer(name="my_strategy")
            >>> features = imputer.fit_transform(df)  # Fit and transform in one step
            >>> print(features.shape)  # (11581, n_features)
        """
        logger.info(f"Fit-transform with {self.name} imputer on {len(X)} samples")
        return self.fit(X).transform(X)

    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input DataFrame matches expected format.

        This method checks that the input data has the correct structure for
        quantum state imputation:
        - Must be a pandas DataFrame
        - Must have exactly 40 columns (event-ID + QV_1-39)
        - QV columns must be binary (0 or 1)
        - Must have no missing values
        - Each row must have exactly 5 active positions (sum of QV = 5)

        Args:
            X: Input DataFrame to validate

        Raises:
            ValueError: If input format is invalid (with detailed error message)

        Notes:
            - This reuses validation patterns from src.data_loader module
            - Error messages are user-friendly (NFR2 compliance)
        """
        # Check it's a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"Input must be a pandas DataFrame, got {type(X).__name__}. "
                f"Use pd.read_csv() to load your data first."
            )

        # Check number of columns
        if len(X.columns) != 40:
            raise ValueError(
                f"Input must have exactly 40 columns (event-ID + QV_1-39), "
                f"but got {len(X.columns)} columns. "
                f"Columns found: {list(X.columns)[:5]}... "
                f"Please check your dataset format."
            )

        # Check required columns exist
        required_columns = ['event-ID'] + [f'QV_{i}' for i in range(1, 40)]
        if list(X.columns) != required_columns:
            raise ValueError(
                f"Input columns do not match expected format. "
                f"Expected: ['event-ID', 'QV_1', 'QV_2', ..., 'QV_39']. "
                f"Got: {list(X.columns)[:5]}... "
                f"Please check your dataset format."
            )

        # Check for missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            raise ValueError(
                f"Input data contains {missing_count} missing values (NaN). "
                f"All values must be present. "
                f"Please clean your dataset before imputation."
            )

        # Check QV columns are binary
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        qv_data = X[qv_columns]

        # Check all values are 0 or 1
        non_binary_mask = ~qv_data.isin([0, 1])
        if non_binary_mask.any().any():
            non_binary_count = non_binary_mask.sum().sum()
            raise ValueError(
                f"QV columns must contain only binary values (0 or 1), "
                f"but found {non_binary_count} non-binary values. "
                f"Please check your dataset for invalid values."
            )

        # Check each row has exactly 5 active positions
        qv_sums = qv_data.sum(axis=1)
        invalid_rows = qv_sums != 5
        if invalid_rows.any():
            num_invalid = invalid_rows.sum()
            invalid_indices = X.index[invalid_rows].tolist()[:5]  # Show first 5
            raise ValueError(
                f"Each row must have exactly 5 active positions (QV sum = 5), "
                f"but found {num_invalid} rows with invalid sums. "
                f"Example invalid row indices: {invalid_indices}... "
                f"This is a quantum state constraint violation."
            )

        logger.debug(f"Input validation passed: {len(X)} samples, 40 columns")

    def _validate_output(self, features: np.ndarray) -> None:
        """
        Validate transformation output quality.

        This method checks that the transformed features are valid:
        - Must be a numpy array
        - Must not contain NaN values
        - Must not contain Inf values
        - Must have correct number of samples

        Args:
            features: Output array from _transform()

        Raises:
            ValueError: If output is invalid (with detailed error message)

        Notes:
            - This is called automatically by transform()
            - Helps catch implementation bugs early
        """
        # Check it's a numpy array
        if not isinstance(features, np.ndarray):
            raise ValueError(
                f"Output must be a numpy array, got {type(features).__name__}. "
                f"Use np.array() or similar in your _transform() implementation."
            )

        # Check for NaN values
        if np.isnan(features).any():
            num_nan = np.isnan(features).sum()
            raise ValueError(
                f"Output contains {num_nan} NaN values. "
                f"This indicates a bug in the {self.name} implementation. "
                f"Please check your transformation logic."
            )

        # Check for Inf values
        if np.isinf(features).any():
            num_inf = np.isinf(features).sum()
            raise ValueError(
                f"Output contains {num_inf} Inf values. "
                f"This indicates a numerical instability in the {self.name} implementation. "
                f"Please check for division by zero or overflow."
            )

        logger.debug(f"Output validation passed: shape {features.shape}, "
                    f"range [{features.min():.4f}, {features.max():.4f}]")

    def __repr__(self) -> str:
        """
        Return string representation of imputer for debugging.

        Returns:
            String representation including name, fitted status, and config

        Examples:
            >>> imputer = MyImputer(name="my_strategy", config={"param": 10})
            >>> print(imputer)
            MyImputer(name='my_strategy', fitted=False, config={'param': 10})
        """
        fitted_status = "fitted" if self.fitted_ else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {fitted_status}, config={self.config})"
