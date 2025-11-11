"""
Base Ranker Abstract Class

This module defines the abstract base class for all ranking models in Epic 3.
All rankers must implement the fit() and predict_top_k() methods to ensure
a consistent interface across different ranking approaches.

Pattern:
This follows the same design pattern as BaseImputer from Epic 2, ensuring
consistency across the codebase and making it easy to swap rankers in
experiments (Epic 5).

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 3, Story 3.2 - Frequency-Based Baseline Rankers
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


class BaseRanker(ABC):
    """
    Abstract base class for all ranking models.

    All ranking models in this project must inherit from this class and
    implement the fit() and predict_top_k() methods. This ensures a
    consistent interface that allows for easy experimentation with different
    ranking approaches.

    Design Philosophy:
        - Consistency: All rankers follow the same interface pattern
        - Modularity: Easy to add new ranker types without changing infrastructure
        - Testability: Common interface makes testing straightforward
        - Reproducibility: fit() must be deterministic for same input data

    Usage Pattern:
        >>> ranker = SomeRanker()
        >>> ranker.fit(X_train, y_train)
        >>> predictions = ranker.predict_top_k(X_test, k=20)

    Attributes:
        is_fitted_: bool
            Indicates whether the ranker has been fitted (trained) on data.
            Set to True after successful fit() call.
    """

    def __init__(self):
        """Initialize base ranker."""
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None):
        """
        Train the ranker on training data.

        This method must be implemented by all subclasses. It should learn
        the necessary parameters from the training data to enable ranking
        predictions on new data.

        Args:
            X_train: Training features (imputed data from Epic 2)
                Shape: (n_samples, n_features)
                Expected columns: imputed features from one of the 5 strategies
            y_train: Optional training labels
                For supervised rankers: target ranking or active positions
                For unsupervised rankers: can be None
                Shape: (n_samples,) or (n_samples, n_targets)

        Returns:
            self: Returns self to enable method chaining (e.g., ranker.fit().predict())

        Raises:
            ValueError: If X_train is empty or has invalid shape
            RuntimeError: If training fails to converge (for ML models)

        Notes:
            - This method should set self.is_fitted_ = True upon successful completion
            - For statistical methods (e.g., frequency baselines), this extracts statistics
            - For ML methods (e.g., GBDT, neural nets), this performs gradient descent
            - Should be deterministic: same input → same learned parameters
        """
        pass

    @abstractmethod
    def predict_top_k(self, X_test: pd.DataFrame, k: int = 20) -> np.ndarray:
        """
        Predict top-k ranked list for each test sample.

        This method must be implemented by all subclasses. It should return
        the k most likely quantum positions for each input sample, ranked
        in descending order of likelihood.

        Args:
            X_test: Test features (imputed data)
                Shape: (n_samples, n_features)
            k: Number of top predictions to return (default: 20)
                Must be in range [1, 39] (39 is the total number of positions)

        Returns:
            predictions: Array of ranked position predictions
                Shape: (n_samples, k)
                Each row contains k position indices (1-39) ranked by likelihood
                Example: [[5, 12, 33, ...], [7, 20, 1, ...], ...]

        Raises:
            RuntimeError: If predict_top_k called before fit()
            ValueError: If k is not in range [1, 39]
            ValueError: If X_test has wrong number of features

        Notes:
            - Position indices are 1-based (1 to 39), not 0-based
            - Top position is at index 0, second at index 1, etc.
            - For ties, use consistent ordering (e.g., lower position index first)
        """
        pass

    def get_name(self) -> str:
        """
        Return ranker name for logging and identification.

        This is useful for:
        - Logging: Track which ranker is being used in experiments
        - Metadata: Store ranker type in model artifacts
        - Debugging: Identify ranker in error messages

        Returns:
            name: Human-readable name of the ranker class

        Example:
            >>> ranker = FrequencyRanker()
            >>> ranker.get_name()
            'FrequencyRanker'
        """
        return self.__class__.__name__

    def fit_transform(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None,
                      k: int = 20) -> np.ndarray:
        """
        Fit the ranker and immediately predict on training data.

        Convenience method that combines fit() and predict_top_k() in one call.
        Useful for quick prototyping and testing.

        Args:
            X_train: Training features
            y_train: Optional training labels
            k: Number of top predictions to return

        Returns:
            predictions: Top-k predictions on training data
                Shape: (n_samples, k)

        Example:
            >>> ranker = FrequencyRanker()
            >>> train_predictions = ranker.fit_transform(X_train, k=20)
        """
        self.fit(X_train, y_train)
        return self.predict_top_k(X_train, k=k)

    def _check_is_fitted(self):
        """
        Check if ranker has been fitted.

        This is a helper method that should be called at the start of predict_top_k()
        to ensure the ranker has been trained before making predictions.

        Raises:
            RuntimeError: If ranker has not been fitted

        Example:
            def predict_top_k(self, X_test, k=20):
                self._check_is_fitted()  # Verify ranker is trained
                # ... prediction logic ...
        """
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.get_name()} must be fitted before calling predict_top_k(). "
                "Call fit(X_train, y_train) first."
            )

    def _validate_k(self, k: int):
        """
        Validate that k is in valid range [1, 39].

        Helper method to ensure k parameter is valid for quantum state prediction.

        Args:
            k: Number of top predictions requested

        Raises:
            ValueError: If k is not in range [1, 39]

        Example:
            def predict_top_k(self, X_test, k=20):
                self._validate_k(k)  # Ensure k is valid
                # ... prediction logic ...
        """
        if not (1 <= k <= 39):
            raise ValueError(
                f"k must be in range [1, 39], got {k}. "
                "There are only 39 possible quantum positions in the C₃₉ group."
            )

    def __repr__(self) -> str:
        """
        String representation of the ranker.

        Returns:
            repr: String representation including fit status

        Example:
            >>> ranker = FrequencyRanker(method='cumulative')
            >>> print(ranker)
            FrequencyRanker(method='cumulative', fitted=False)
        """
        fitted_status = "fitted" if self.is_fitted_ else "not fitted"
        return f"{self.get_name()}({fitted_status})"
