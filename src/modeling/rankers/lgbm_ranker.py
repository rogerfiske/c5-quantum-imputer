"""
Gradient Boosting Decision Tree (GBDT) Ranker using LightGBM

This module implements a learning-to-rank model using LightGBM's LGBMRanker.
GBDT models are state-of-the-art for tabular data and provide fast training
with excellent performance on structured features.

The ranker uses LambdaRank objective to optimize ranking metrics directly,
making it well-suited for predicting the top-k most likely quantum positions.

Key Features:
    - Uses LightGBM's LGBMRanker with lambdarank objective
    - Engineered features: position frequencies, circular distances, DFT harmonics
    - Performance timing tracking (for RunPod usage decisions per NFR1)
    - Hyperparameter configurability with sensible defaults

Author: BMad Dev Agent (James)
Date: 2025-10-14
Story: Epic 3, Story 3.3 - Gradient Boosting Ranker
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from .base_ranker import BaseRanker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default hyperparameters (sensible defaults from handoff doc)
DEFAULT_LGBM_PARAMS = {
    'objective': 'lambdarank',  # Learning to rank objective
    'metric': 'ndcg',  # Normalized Discounted Cumulative Gain
    'ndcg_eval_at': [5, 10, 20],  # Evaluate NDCG at top-5, 10, 20
    'n_estimators': 100,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,  # No limit
    'min_child_samples': 20,
    'verbosity': -1,  # Suppress LightGBM output
    'random_state': 42,  # For reproducibility
}


class LGBMRanker(BaseRanker):
    """
    Gradient Boosting Decision Tree ranker using LightGBM.

    This ranker uses LightGBM's LGBMRanker with the lambdarank objective
    to learn to rank quantum positions based on their likelihood. The model
    learns from patterns in imputed features to predict which positions are
    most likely to be active in future events.

    The approach:
    1. For each training event, we create 39 candidate positions
    2. Each candidate gets features describing that position
    3. Labels indicate whether that position was actually active (1) or not (0)
    4. LGBMRanker learns to rank positions by their likelihood
    5. At inference, we score all 39 positions and return top-k

    Engineered Features:
        - Position frequency features: How often each position appears in training
        - Circular distance features: Distances on the C₃₉ ring structure
        - DFT harmonic features: Frequency domain features from graph/cycle encoding
        - Imputed features: Original features from Epic 2 imputation strategies

    Args:
        params: Dict of LightGBM hyperparameters
            If None, uses DEFAULT_LGBM_PARAMS
            Can override specific params, e.g., {'n_estimators': 200}
        track_time: Whether to track and log training time
            Default: True (required for RunPod decisions per NFR1)

    Attributes:
        model_: lgb.LGBMRanker
            The trained LightGBM model (set after fit())
        training_time_: float
            Training time in seconds (set after fit() if track_time=True)
        feature_importance_: Dict[str, float]
            Feature importance scores (set after fit())
        position_frequencies_: Dict[int, float]
            Learned position frequencies from training data

    Example:
        >>> # Train with default hyperparameters
        >>> ranker = LGBMRanker()
        >>> ranker.fit(X_train, y_train)
        >>> predictions = ranker.predict_top_k(X_test, k=20)

        >>> # Train with custom hyperparameters
        >>> ranker = LGBMRanker(params={'n_estimators': 200, 'learning_rate': 0.05})
        >>> ranker.fit(X_train, y_train)
        >>> print(f"Training took {ranker.training_time_:.2f} seconds")

        >>> # Save trained model
        >>> ranker.save_model('models/lgbm_ranker_v1.joblib')

    Notes:
        - Requires LightGBM ≥4.0.0 to be installed
        - Training time is tracked automatically (per NFR1 requirement)
        - Model can be saved/loaded using save_model() and load_model()
        - Feature engineering happens automatically during fit()
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, track_time: bool = True):
        """
        Initialize LightGBM ranker.

        Args:
            params: LightGBM hyperparameters (uses defaults if None)
            track_time: Whether to track training time (default: True)

        Raises:
            ImportError: If LightGBM is not installed
        """
        super().__init__()

        # Check LightGBM availability
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Please install it with:\n"
                "  pip install lightgbm>=4.0.0\n"
                "Or use XGBoost as a fallback (see docs/handoff-epic3-to-dev.md)."
            )

        # Set hyperparameters (merge with defaults)
        self.params = DEFAULT_LGBM_PARAMS.copy()
        if params is not None:
            self.params.update(params)

        self.track_time = track_time

        # Model and training artifacts (set during fit)
        self.model_: Optional[lgb.LGBMRanker] = None
        self.training_time_: Optional[float] = None
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.position_frequencies_: Optional[Dict[int, float]] = None
        self.feature_names_: Optional[list] = None

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None):
        """
        Train the LightGBM ranker on training data.

        This method:
        1. Validates input data
        2. Engineers features (position frequencies, circular distances, etc.)
        3. Transforms data into LightGBM ranking format (one row per position)
        4. Trains the LGBMRanker model
        5. Tracks training time (if enabled)
        6. Extracts feature importance

        For ranking, we transform each event into 39 candidate positions:
        - Each position gets engineered features
        - Labels indicate if position was active (1) or inactive (0)
        - Groups ensure positions from same event are ranked together

        Args:
            X_train: Training features from imputed data
                Shape: (n_events, n_features)
                Expected: Imputed data from Epic 2 with 'event-ID' column
            y_train: Not directly used (labels extracted from X_train)
                For consistency with BaseRanker interface

        Returns:
            self: Returns self for method chaining

        Raises:
            ValueError: If X_train is empty or has wrong format
            RuntimeError: If training fails

        Notes:
            - Training time is automatically logged if track_time=True
            - If training exceeds 1 hour, suggests using RunPod (per NFR1)
        """
        # Validate input
        if X_train.empty:
            raise ValueError("X_train cannot be empty")

        if len(X_train) < 10:
            raise ValueError(
                f"X_train too small: {len(X_train)} samples. "
                "Need at least 10 samples for meaningful training."
            )

        logger.info(f"Training LGBMRanker on {len(X_train)} events...")

        # Start timing (if enabled)
        start_time = time.time() if self.track_time else None

        try:
            # Step 1: Extract active positions from training data
            # This tells us which positions were actually active in each event
            active_positions_list = self._extract_active_positions(X_train)

            # Step 2: Compute position frequencies (for feature engineering)
            self._compute_position_frequencies(active_positions_list)

            # Step 3: Transform data into ranking format
            # Each event becomes 39 rows (one per candidate position)
            X_rank, y_rank, groups = self._transform_to_ranking_format(
                X_train, active_positions_list
            )

            logger.info(f"Transformed to ranking format: {X_rank.shape[0]} samples "
                       f"({len(groups)} groups of 39 positions each)")

            # Step 4: Train LGBMRanker
            logger.info("Training LightGBM model...")

            # Create LGBMRanker with specified hyperparameters
            self.model_ = lgb.LGBMRanker(**self.params)

            # Fit the model
            # group parameter tells LightGBM which rows belong to same query
            self.model_.fit(
                X_rank, y_rank,
                group=groups,
                eval_set=[(X_rank, y_rank)],
                eval_group=[groups],
                eval_metric='ndcg',
                callbacks=[lgb.log_evaluation(period=0)]  # Suppress iteration logs
            )

            # Store feature names for interpretability
            self.feature_names_ = list(X_rank.columns)

            # Step 5: Extract feature importance
            self._extract_feature_importance()

            # Step 6: Track training time
            if self.track_time:
                self.training_time_ = time.time() - start_time
                logger.info(f"✓ Training completed in {self.training_time_:.2f} seconds")

                # Warn if training exceeded 1 hour (RunPod suggestion per NFR1)
                if self.training_time_ > 3600:
                    logger.warning(
                        f"⚠️ Training exceeded 1 hour ({self.training_time_/3600:.2f} hours). "
                        "Consider using RunPod for Epic 5 experiments (see docs/architecture.md section 5)."
                    )

            self.is_fitted_ = True
            logger.info("✓ LGBMRanker training successful")

            return self

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(
                f"LGBMRanker training failed: {e}. "
                "Check data format and hyperparameters."
            ) from e

    def fit_with_features(
        self,
        X_features: pd.DataFrame,
        y_positions: np.ndarray,
        feature_prefix: str = ""
    ) -> 'LGBMRanker':
        """
        Train LightGBM ranker on pre-imputed features (modular imputer support).

        This method enables training on features that have already been computed
        by an external imputer (e.g., HistoricalAmplitudeEmbedding,
        CylindricalAmplitudeEmbedding). This supports modular comparison of
        different imputation strategies.

        Unlike fit(), this method:
        - Does NOT perform internal feature engineering
        - Accepts pre-computed feature matrices directly
        - Requires explicit target positions as input
        - Enables A/B testing of imputation methods

        For ranking, we transform the pre-imputed features into ranking format:
        - Each event's features are replicated 39 times (one per position)
        - Position-specific features are added (position_id, position_frequency)
        - Labels indicate if position was active (1) or inactive (0)
        - Groups ensure positions from same event are ranked together

        Args:
            X_features: Pre-imputed feature matrix from external imputer
                Shape: (n_events, n_features)
                Must be output from an imputer's transform() method
            y_positions: Active position indices for each event
                Shape: (n_events, 5) - the 5 winning positions per event
                Values: 1-39 (position indices)
            feature_prefix: Optional prefix for feature names (default: "")
                Useful for tracking which imputer generated features

        Returns:
            self: Returns self for method chaining

        Raises:
            ValueError: If X_features or y_positions has wrong shape
            RuntimeError: If training fails

        Example:
            >>> # Train with baseline imputer
            >>> baseline_imputer = HistoricalAmplitudeEmbedding()
            >>> X_baseline = baseline_imputer.fit_transform(train_ids)
            >>> y_positions = train_df[['q_1', 'q_2', 'q_3', 'q_4', 'q_5']].values
            >>> ranker = LGBMRanker()
            >>> ranker.fit_with_features(X_baseline, y_positions, feature_prefix="baseline_")

            >>> # Train with cylindrical imputer
            >>> cylindrical_imputer = CylindricalAmplitudeEmbedding()
            >>> X_cylindrical = cylindrical_imputer.fit_transform(train_ids)
            >>> ranker = LGBMRanker()
            >>> ranker.fit_with_features(X_cylindrical, y_positions, feature_prefix="cylindrical_")

            >>> # Compare performance on holdout
            >>> predictions = ranker.predict_top_k_from_features(X_test, k=20)

        Notes:
            - Use this method for validating new imputation strategies
            - Use standard fit() for production workflows with built-in imputation
            - Requires predict_top_k_from_features() for inference
        """
        # Validate input shapes
        if X_features.empty:
            raise ValueError("X_features cannot be empty")

        if len(X_features) < 10:
            raise ValueError(
                f"X_features too small: {len(X_features)} samples. "
                "Need at least 10 samples for meaningful training."
            )

        n_events = len(X_features)

        if y_positions.shape != (n_events, 5):
            raise ValueError(
                f"y_positions has wrong shape: {y_positions.shape}. "
                f"Expected: ({n_events}, 5) - 5 winning positions per event."
            )

        logger.info(f"Training LGBMRanker with pre-imputed features on {n_events} events...")
        logger.info(f"Feature matrix shape: {X_features.shape}")

        # Start timing
        start_time = time.time() if self.track_time else None

        try:
            # Step 1: Convert y_positions to list format (for compatibility)
            active_positions_list = []
            for event_positions in y_positions:
                positions = [int(pos) for pos in event_positions]
                active_positions_list.append(positions)

            # Step 2: Compute position frequencies (for minimal feature engineering)
            self._compute_position_frequencies(active_positions_list)

            # Step 3: Transform to ranking format (with pre-imputed features)
            X_rank, y_rank, groups = self._transform_features_to_ranking_format(
                X_features, active_positions_list, feature_prefix
            )

            logger.info(f"Transformed to ranking format: {X_rank.shape[0]} samples "
                       f"({len(groups)} groups of 39 positions each)")

            # Step 4: Train LGBMRanker
            logger.info("Training LightGBM model...")

            self.model_ = lgb.LGBMRanker(**self.params)

            self.model_.fit(
                X_rank, y_rank,
                group=groups,
                eval_set=[(X_rank, y_rank)],
                eval_group=[groups],
                eval_metric='ndcg',
                callbacks=[lgb.log_evaluation(period=0)]
            )

            # Store feature names
            self.feature_names_ = list(X_rank.columns)

            # Step 5: Extract feature importance
            self._extract_feature_importance()

            # Step 6: Track training time
            if self.track_time:
                self.training_time_ = time.time() - start_time
                logger.info(f"✓ Training completed in {self.training_time_:.2f} seconds")

            self.is_fitted_ = True
            logger.info("✓ LGBMRanker training successful (fit_with_features)")

            return self

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(
                f"LGBMRanker training failed (fit_with_features): {e}. "
                "Check feature matrix format and target positions."
            ) from e

    def _transform_features_to_ranking_format(
        self,
        X_features: pd.DataFrame,
        active_positions_list: list,
        feature_prefix: str = ""
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Transform pre-imputed features into LightGBM ranking format.

        For each event, we create 39 rows (one per candidate position).
        Each row has:
        - Pre-imputed features (shared across all 39 positions)
        - Position-specific features (position_id, position_frequency)
        - Label: 1 if position was active, 0 if not

        Args:
            X_features: Pre-imputed features (n_events × n_features)
            active_positions_list: Active positions per event
            feature_prefix: Optional prefix for imputed feature names

        Returns:
            Tuple of (X_rank, y_rank, groups):
                X_rank: Features in ranking format (n_events * 39 rows)
                y_rank: Labels (1 if position was active, 0 if not)
                groups: Array indicating group sizes (all 39)
        """
        X_rank_list = []
        y_rank_list = []
        groups = []

        for event_idx, (_, feature_row) in enumerate(X_features.iterrows()):
            if event_idx >= len(active_positions_list):
                continue

            active_positions = set(active_positions_list[event_idx])

            # Create 39 candidate rows for this event
            for position in range(1, 40):
                # Start with position-specific features
                features = {
                    'position_id': position,
                    'position_frequency': self.position_frequencies_.get(position, 1e-6)
                }

                # Add pre-imputed features (with optional prefix)
                for feat_name, feat_value in feature_row.items():
                    prefixed_name = f"{feature_prefix}{feat_name}" if feature_prefix else feat_name
                    features[prefixed_name] = feat_value

                X_rank_list.append(features)

                # Label: 1 if position was active, 0 otherwise
                label = 1 if position in active_positions else 0
                y_rank_list.append(label)

            groups.append(39)

        X_rank = pd.DataFrame(X_rank_list)
        y_rank = np.array(y_rank_list)
        groups = np.array(groups)

        return X_rank, y_rank, groups

    def predict_top_k(self, X_test: pd.DataFrame, k: int = 20) -> np.ndarray:
        """
        Predict top-k ranked positions for each test event.

        For each test event:
        1. Create 39 candidate positions with engineered features
        2. Score each position using the trained model
        3. Rank positions by score (descending)
        4. Return top-k positions

        Args:
            X_test: Test features (same format as X_train)
                Shape: (n_events, n_features)
            k: Number of top predictions to return
                Default: 20
                Must be in range [1, 39]

        Returns:
            predictions: Top-k ranked position predictions
                Shape: (n_events, k)
                Each row contains k position indices (1-39) ranked by likelihood
                Most likely position at index 0, second at index 1, etc.

        Raises:
            RuntimeError: If called before fit()
            ValueError: If k not in range [1, 39]
            ValueError: If X_test has wrong format

        Example:
            >>> predictions = ranker.predict_top_k(X_test, k=20)
            >>> predictions.shape
            (2317, 20)  # 2317 test events, 20 predictions each
            >>> predictions[0]  # First event's top-20 predictions
            array([ 5, 12, 33, 7, 20, ...])
        """
        # Validation
        self._check_is_fitted()
        self._validate_k(k)

        if X_test.empty:
            raise ValueError("X_test cannot be empty")

        n_events = len(X_test)
        logger.info(f"Predicting top-{k} for {n_events} test events...")

        # Create candidate positions for each test event
        # Each event gets 39 rows (one per position)
        X_rank_test = self._create_ranking_features(X_test)

        # Score all candidates using trained model
        scores = self.model_.predict(X_rank_test)

        # Reshape scores: (n_events * 39,) -> (n_events, 39)
        scores_matrix = scores.reshape(n_events, 39)

        # Rank positions by score (descending)
        # argsort returns indices of smallest to largest, so use negative scores
        ranked_indices = np.argsort(-scores_matrix, axis=1)

        # Convert 0-based indices to 1-based position numbers and take top-k
        predictions = ranked_indices[:, :k] + 1  # +1 for 1-based indexing

        logger.info(f"✓ Predictions generated for {n_events} events")

        return predictions

    def predict_top_k_from_features(
        self,
        X_features: pd.DataFrame,
        k: int = 20,
        feature_prefix: str = ""
    ) -> np.ndarray:
        """
        Predict top-k ranked positions from pre-imputed features.

        This is the companion inference method to fit_with_features().
        Use this when the model was trained using fit_with_features() and
        you have pre-imputed test features.

        For each test event:
        1. Create 39 candidate positions with pre-imputed features
        2. Add position-specific features (position_id, position_frequency)
        3. Score each position using the trained model
        4. Rank positions by score and return top-k

        Args:
            X_features: Pre-imputed test features from external imputer
                Shape: (n_events, n_features)
                Must match the feature format used in fit_with_features()
            k: Number of top predictions to return (default: 20)
                Must be in range [1, 39]
            feature_prefix: Optional prefix for feature names (default: "")
                Must match the prefix used in fit_with_features()

        Returns:
            predictions: Top-k ranked position predictions
                Shape: (n_events, k)
                Each row contains k position indices (1-39) ranked by likelihood

        Raises:
            RuntimeError: If called before fit_with_features()
            ValueError: If k not in range [1, 39]
            ValueError: If X_features has wrong format

        Example:
            >>> # Train with cylindrical imputer
            >>> cylindrical_imputer = CylindricalAmplitudeEmbedding()
            >>> X_train = cylindrical_imputer.fit_transform(train_ids)
            >>> ranker = LGBMRanker()
            >>> ranker.fit_with_features(X_train, y_positions, feature_prefix="cylindrical_")

            >>> # Predict on test set with same imputer
            >>> X_test = cylindrical_imputer.transform(test_ids)
            >>> predictions = ranker.predict_top_k_from_features(X_test, k=20, feature_prefix="cylindrical_")
            >>> predictions.shape
            (1000, 20)  # 1000 test events, 20 predictions each

        Notes:
            - feature_prefix must match what was used in fit_with_features()
            - X_features must come from the same imputer used for training
            - Feature matrix shape (n_features) must match training
        """
        # Validation
        self._check_is_fitted()
        self._validate_k(k)

        if X_features.empty:
            raise ValueError("X_features cannot be empty")

        n_events = len(X_features)
        logger.info(f"Predicting top-{k} for {n_events} test events (from features)...")

        # Create ranking features for test set
        X_rank_test = self._create_ranking_features_from_imputed(X_features, feature_prefix)

        # Score all candidates using trained model
        scores = self.model_.predict(X_rank_test)

        # Reshape scores: (n_events * 39,) -> (n_events, 39)
        scores_matrix = scores.reshape(n_events, 39)

        # Rank positions by score (descending)
        ranked_indices = np.argsort(-scores_matrix, axis=1)

        # Convert 0-based indices to 1-based position numbers and take top-k
        predictions = ranked_indices[:, :k] + 1

        logger.info(f"✓ Predictions generated for {n_events} events (from features)")

        return predictions

    def _create_ranking_features_from_imputed(
        self,
        X_features: pd.DataFrame,
        feature_prefix: str = ""
    ) -> pd.DataFrame:
        """
        Create ranking features from pre-imputed test features.

        For each test event, creates 39 rows of features (one per position).
        Similar to _transform_features_to_ranking_format but without labels.

        Args:
            X_features: Pre-imputed test features (n_events × n_features)
            feature_prefix: Optional prefix for imputed feature names

        Returns:
            Feature DataFrame with n_events * 39 rows
        """
        X_rank_list = []

        for _, feature_row in X_features.iterrows():
            # Create 39 candidate rows for this event
            for position in range(1, 40):
                # Start with position-specific features
                features = {
                    'position_id': position,
                    'position_frequency': self.position_frequencies_.get(position, 1e-6)
                }

                # Add pre-imputed features (with optional prefix)
                for feat_name, feat_value in feature_row.items():
                    prefixed_name = f"{feature_prefix}{feat_name}" if feature_prefix else feat_name
                    features[prefixed_name] = feat_value

                X_rank_list.append(features)

        X_rank = pd.DataFrame(X_rank_list)

        # Ensure column order matches training
        if self.feature_names_ is not None:
            # Reorder columns to match training
            X_rank = X_rank[self.feature_names_]

        return X_rank

    def _extract_active_positions(self, X: pd.DataFrame) -> list:
        """
        Extract active position indices from target columns (DATA LEAKAGE FIX).

        Uses q_1...q_5 columns which contain the 5 winning positions for each event.
        This replaces the previous approach that used QV columns (which contained
        the answer and caused data leakage).

        Args:
            X: Feature DataFrame with q_1, q_2, q_3, q_4, q_5 columns

        Returns:
            List of lists, each containing active position indices (1-39) for one event
            Example: [[8, 21, 22, 28, 38], [19, 29, 35, 37, 39], ...]

        Raises:
            ValueError: If target columns are missing

        Notes:
            - Target columns must contain winning positions (not binary indicators)
            - Position indices are 1-based (1 to 39)
            - DATA LEAKAGE FIX: Uses explicit targets, not QV feature columns
        """
        # Use explicit target columns (q_1 through q_5)
        target_cols = ['q_1', 'q_2', 'q_3', 'q_4', 'q_5']

        # Verify target columns exist
        missing_cols = [col for col in target_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(
                f"Missing target columns in data. Expected q_1 through q_5. "
                f"Missing: {missing_cols}. "
                f"These columns should contain the 5 winning positions per event."
            )

        active_positions_list = []

        # Extract winning positions from target columns
        for _, row in X[target_cols].iterrows():
            # q_* columns contain position numbers (1-39)
            positions = [int(row[col]) for col in target_cols]
            active_positions_list.append(positions)

        return active_positions_list

    def _compute_position_frequencies(self, active_positions_list: list):
        """
        Compute position frequency statistics from training data.

        These frequencies are used as features to help the model learn
        which positions are generally more/less likely to be active.

        Args:
            active_positions_list: List of active positions per training event
        """
        from collections import Counter

        # Count occurrences of each position
        position_counts = Counter()
        for positions in active_positions_list:
            position_counts.update(positions)

        # Normalize to frequencies (probabilities)
        total_count = sum(position_counts.values())
        self.position_frequencies_ = {
            pos: count / total_count
            for pos, count in position_counts.items()
        }

        # Ensure all positions 1-39 have an entry (default to small value)
        for pos in range(1, 40):
            if pos not in self.position_frequencies_:
                self.position_frequencies_[pos] = 1e-6  # Small epsilon

        logger.info(f"Computed position frequencies: {len(self.position_frequencies_)} positions")

    def _transform_to_ranking_format(
        self, X: pd.DataFrame, active_positions_list: list
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Transform data into LightGBM ranking format.

        For each event, we create 39 rows (one per candidate position).
        Each row has:
        - Features describing that position
        - Label: 1 if position was active, 0 if not

        Args:
            X: Original feature DataFrame (one row per event)
            active_positions_list: Active positions per event

        Returns:
            Tuple of (X_rank, y_rank, groups):
                X_rank: Features in ranking format (n_events * 39 rows)
                y_rank: Labels (1 if position was active, 0 if not)
                groups: Array indicating group sizes (all 39)
        """
        X_rank_list = []
        y_rank_list = []
        groups = []

        for event_idx, (_, event_row) in enumerate(X.iterrows()):
            if event_idx >= len(active_positions_list):
                # Skip if no active positions extracted
                continue

            active_positions = set(active_positions_list[event_idx])

            # Create 39 candidate rows for this event (one per position)
            for position in range(1, 40):
                # Create features for this candidate position
                features = self._create_position_features(position, event_row)
                X_rank_list.append(features)

                # Label: 1 if position was active, 0 otherwise
                label = 1 if position in active_positions else 0
                y_rank_list.append(label)

            # Group size: 39 positions per event
            groups.append(39)

        # Convert to arrays
        X_rank = pd.DataFrame(X_rank_list)
        y_rank = np.array(y_rank_list)
        groups = np.array(groups)

        return X_rank, y_rank, groups

    def _create_position_features(self, position: int, event_row: pd.Series) -> Dict[str, float]:
        """
        Create engineered features for a candidate position (DATA LEAKAGE FIX).

        Creates features from imputed data only - does NOT include QV columns
        which contain the answer (which positions appeared).

        Features include:
        1. Position frequency: How often this position appears in training
        2. Circular distance features: Distances to other positions on C₃₉ ring
        3. Position indicator: One-hot style indicator for position identity
        4. Ring quadrant: Which quarter of the ring (0-3)
        5. Imputed features: Continuous features from Epic 2 imputation strategies
           (rho_*, eigenvalue_*, purity, amplitude_*, etc.)

        Args:
            position: Position index (1-39)
            event_row: Original event features (imputed features + targets, NO QV columns)

        Returns:
            Dictionary of features for this position candidate

        Notes:
            - DATA LEAKAGE FIX: QV columns excluded from features
            - Only uses imputed features that don't reveal the answer
        """
        features = {}

        # ============================================
        # Engineered Features (original features)
        # ============================================

        # Feature 1: Position frequency (learned from training data)
        features['position_frequency'] = self.position_frequencies_.get(position, 1e-6)

        # Feature 2: Position identity (help model learn position-specific patterns)
        features['position_id'] = position

        # Feature 3: Circular distance features
        # Compute distances to positions 1, 10, 20, 30 (representatives on ring)
        for ref_pos in [1, 10, 20, 30]:
            dist = self._circular_distance(position, ref_pos, n_positions=39)
            features[f'circ_dist_{ref_pos}'] = dist

        # Feature 4: Position in ring quadrant (which quarter of the ring)
        features['ring_quadrant'] = (position - 1) // 10  # 0, 1, 2, 3

        # ============================================
        # Add Imputed Features (DATA LEAKAGE FIX)
        # ============================================
        # Include continuous features from Epic 2 imputation strategies
        # These capture learned patterns (amplitude, basis, angle, density, graph)
        #
        # CRITICAL: QV columns are NOT included (they contain the answer!)
        # Only use imputed features: rho_*, eigenvalue_*, purity, etc.

        # Define columns to exclude (targets and metadata only)
        target_cols = ['q_1', 'q_2', 'q_3', 'q_4', 'q_5']  # Target columns (winning position indices)
        qv_cols = [f'QV_{i}' for i in range(1, 40)]  # QV columns (ACTUAL WINNERS - must exclude!)
        exclude_cols = set(target_cols + qv_cols + ['event-ID'])

        # Add all remaining columns as features
        # This includes imputed features but excludes targets
        for col in event_row.index:
            if col not in exclude_cols and col not in features:
                features[col] = event_row[col]

        return features

    def _circular_distance(self, pos1: int, pos2: int, n_positions: int = 39) -> int:
        """
        Compute circular (ring) distance between two positions.

        On a ring of n_positions, the distance between two positions
        is the minimum of clockwise and counter-clockwise distances.

        Args:
            pos1: First position (1-based)
            pos2: Second position (1-based)
            n_positions: Total positions on ring (default: 39)

        Returns:
            Circular distance (0 to n_positions//2)

        Example:
            >>> _circular_distance(1, 3, 39)
            2  # Direct distance
            >>> _circular_distance(1, 39, 39)
            1  # Wraps around: 39 -> 1
        """
        # Convert to 0-based for modular arithmetic
        p1 = pos1 - 1
        p2 = pos2 - 1

        # Compute both clockwise and counter-clockwise distances
        clockwise = (p2 - p1) % n_positions
        counter_clockwise = (p1 - p2) % n_positions

        # Return minimum distance
        return min(clockwise, counter_clockwise)

    def _create_ranking_features(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Create ranking features for test events.

        For each test event, creates 39 rows of features (one per position).
        Similar to _transform_to_ranking_format but without labels.

        Args:
            X_test: Test event features

        Returns:
            Feature DataFrame with n_events * 39 rows
        """
        X_rank_list = []

        for _, event_row in X_test.iterrows():
            # Create 39 candidate rows for this event
            for position in range(1, 40):
                features = self._create_position_features(position, event_row)
                X_rank_list.append(features)

        X_rank = pd.DataFrame(X_rank_list)

        # Ensure column order matches training
        if self.feature_names_ is not None:
            # Reorder columns to match training (important for model.predict)
            X_rank = X_rank[self.feature_names_]

        return X_rank

    def _extract_feature_importance(self):
        """
        Extract feature importance from trained model.

        Feature importance helps understand which features the model
        relies on most for ranking decisions. Useful for:
        - Model interpretability
        - Feature engineering insights
        - Debugging unexpected behavior
        """
        if self.model_ is None:
            return

        # Get feature importance (gain-based)
        importance_scores = self.model_.feature_importances_

        # Create dictionary mapping feature names to importance scores
        self.feature_importance_ = {
            feature: float(score)
            for feature, score in zip(self.feature_names_, importance_scores)
        }

        # Sort by importance (descending)
        self.feature_importance_ = dict(
            sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True)
        )

        # Log top-5 most important features
        top_features = list(self.feature_importance_.items())[:5]
        logger.info("Top-5 most important features:")
        for feat, score in top_features:
            logger.info(f"  {feat}: {score:.2f}")

    def save_model(self, model_path: Path, save_metadata: bool = True):
        """
        Save trained model to disk.

        Saves the model and optionally training metadata (hyperparameters,
        training time, feature importance) for reproducibility and analysis.

        Args:
            model_path: Path to save model file (e.g., 'models/lgbm_ranker_v1.joblib')
            save_metadata: Whether to save metadata as sidecar JSON file
                Default: True

        Raises:
            RuntimeError: If called before fit()
            OSError: If file write fails

        Example:
            >>> ranker.save_model(Path('models/lgbm_ranker_v1.joblib'))
            >>> # Creates: models/lgbm_ranker_v1.joblib
            >>> # Creates: models/lgbm_ranker_v1.joblib.meta.json
        """
        self._check_is_fitted()

        # Ensure parent directory exists
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model using joblib
        logger.info(f"Saving model to {model_path}")
        joblib.dump(self.model_, model_path)
        logger.info(f"✓ Model saved successfully")

        # Save metadata if requested
        if save_metadata:
            metadata_path = Path(str(model_path) + '.meta.json')
            metadata = {
                'ranker_type': 'lgbm',
                'lightgbm_version': lgb.__version__,
                'hyperparameters': self.params,
                'training_time_seconds': self.training_time_,
                'n_features': len(self.feature_names_) if self.feature_names_ else None,
                'feature_names': self.feature_names_,
                'feature_importance': self.feature_importance_,
                'position_frequencies': self.position_frequencies_,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            logger.info(f"Saving metadata to {metadata_path}")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Metadata saved successfully")

    @classmethod
    def load_model(cls, model_path: Path) -> 'LGBMRanker':
        """
        Load trained model from disk.

        Args:
            model_path: Path to saved model file

        Returns:
            Loaded LGBMRanker instance with trained model

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails

        Example:
            >>> ranker = LGBMRanker.load_model(Path('models/lgbm_ranker_v1.joblib'))
            >>> predictions = ranker.predict_top_k(X_test, k=20)
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please train and save model first using save_model()."
            )

        # Load model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Create LGBMRanker instance and set model
        instance = cls()
        instance.model_ = model
        instance.is_fitted_ = True

        # Try to load metadata if available
        metadata_path = Path(str(model_path) + '.meta.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                instance.params = metadata.get('hyperparameters', {})
                instance.training_time_ = metadata.get('training_time_seconds')
                instance.feature_names_ = metadata.get('feature_names')
                instance.feature_importance_ = metadata.get('feature_importance')

                # Restore position_frequencies (needed for prediction)
                position_freq_data = metadata.get('position_frequencies')
                if position_freq_data is not None:
                    # Convert string keys back to integers
                    instance.position_frequencies_ = {
                        int(k): v for k, v in position_freq_data.items()
                    }
                logger.info("✓ Metadata loaded")

        logger.info(f"✓ Model loaded successfully")
        return instance

    def __repr__(self) -> str:
        """String representation with training status and time."""
        fitted_status = "fitted" if self.is_fitted_ else "not fitted"
        if self.training_time_ is not None:
            return f"LGBMRanker({fitted_status}, trained in {self.training_time_:.2f}s)"
        else:
            return f"LGBMRanker({fitted_status})"
