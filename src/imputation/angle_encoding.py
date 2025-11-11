"""
Angle Encoding Imputation Strategy

This module implements the Angle Encoding quantum-inspired imputation method,
which maps binary quantum state vectors to rotation angles on the Bloch sphere.
Each position corresponds to a rotation angle, creating features that capture
angular relationships and cyclic symmetries.

The quantum-inspired concept:
In quantum computing, the Bloch sphere is a geometric representation of quantum
states. A point on the sphere represents a pure quantum state, specified by
rotation angles (θ, φ). For our binary quantum states with 39 positions, we
map each position to an angle on a circle:
    θᵢ = 2π * (i-1) / 39  for position QV_i

This creates a natural cyclic structure (C₃₉ symmetry group) where position 1
and position 39 are adjacent on the circle. Active positions correspond to
specific angular orientations, and we extract both direct angles and
trigonometric features to capture rotational patterns.

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.4 - Angle Encoding Strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Literal

from src.imputation.base_imputer import BaseImputer
from src.imputation.proximity_features import ProximityFeatures

# Set up logging
logger = logging.getLogger(__name__)


class AngleEncoding(BaseImputer):
    """
    Angle Encoding imputation strategy.

    Maps binary quantum state vectors to rotation angles on a circular
    representation (Bloch sphere analogy). Each of the 39 positions corresponds
    to a specific angle on a circle, and active positions determine the overall
    rotational characteristics of the quantum state.

    Quantum Mechanics Background:
    ------------------------------
    In quantum computing, the Bloch sphere is a geometric representation where
    any pure qubit state can be represented as a point on a unit sphere. The
    state is determined by two angles (θ, φ) representing rotations:
        |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩

    For our 39-position quantum states, we map each position to an angle on a
    circle (one-dimensional analog of Bloch sphere):
        Position i → Angle θᵢ = 2π * (i-1) / 39

    This creates a cyclic structure where positions are evenly distributed
    around a circle, respecting the C₃₉ cyclic group symmetry.

    Angular Feature Strategy:
    --------------------------
    This implementation creates three types of features:

    1. **Direct Angle Features** (39 features):
       The angle θᵢ for each position. Zero for inactive positions,
       non-zero (the assigned angle) for active positions.
       - Position 1 → θ₁ = 0
       - Position 2 → θ₂ = 2π/39 ≈ 0.161
       - Position 39 → θ₃₉ = 2π*(38)/39 ≈ 6.122

    2. **Trigonometric Features** (78 features):
       For each position, compute sin(θᵢ) and cos(θᵢ). These capture
       the (x, y) coordinates of the position on the unit circle.
       - 39 sine features: sin(θᵢ) for i=1..39
       - 39 cosine features: cos(θᵢ) for i=1..39
       Trigonometric features are naturally bounded [-1, 1] and capture
       rotational symmetries.

    3. **Aggregated Angle Features** (6 features):
       Summary statistics across the 5 active positions:
       - sum_sin: Σ sin(θᵢ) for active positions
       - sum_cos: Σ cos(θᵢ) for active positions
       - mean_sin: mean of sin(θᵢ) for active positions
       - mean_cos: mean of cos(θᵢ) for active positions
       - resultant_magnitude: ||sum_sin, sum_cos|| (vector length)
       - resultant_angle: atan2(sum_sin, sum_cos) (average direction)

       These features capture the "center of mass" of active positions on
       the circle, indicating whether active positions are clustered or
       spread out.

    Cyclic Symmetry (C₃₉ Group):
    ----------------------------
    The 39 positions form a cyclic group C₃₉ where position 1 is adjacent
    to position 39 (wrapping around). This is naturally encoded in the
    angular representation:
    - θ₁ = 0 and θ₃₉ ≈ 6.122 are close on the circle (differ by 2π/39)
    - Rotational operations preserve distances on the circle
    - Trigonometric features capture these circular relationships

    Parameters:
    -----------
    name : str, optional (default="angle_encoding")
        Human-readable name for this imputation strategy

    include_trig_features : bool, optional (default=True)
        Whether to include trigonometric (sin/cos) features.
        - If True: includes both angle and trig features
        - If False: only includes direct angle features

    include_aggregated_features : bool, optional (default=True)
        Whether to include aggregated angle statistics.
        - If True: adds 6 summary features across active positions
        - If False: omits aggregated features

    Output Dimensions:
    ------------------
    - With all features (default): (n_samples, 123)
      = 39 (angles) + 78 (sin/cos) + 6 (aggregated) = 123 features
    - Without trig features: (n_samples, 45)
      = 39 (angles) + 6 (aggregated) = 45 features
    - Without aggregated features: (n_samples, 117)
      = 39 (angles) + 78 (sin/cos) = 117 features
    - Only angles: (n_samples, 39)

    Attributes:
    -----------
    position_angles_ : np.ndarray, shape (39,)
        Pre-computed angles for each position: θᵢ = 2π * (i-1) / 39
        Set during initialization (not learned from data).

    Examples:
    ---------
    >>> from src.data_loader import load_dataset
    >>> from src.imputation.angle_encoding import AngleEncoding
    >>>
    >>> # Load dataset
    >>> df = load_dataset()
    >>>
    >>> # Full angle encoding (default)
    >>> imputer = AngleEncoding()
    >>> features = imputer.fit_transform(df)
    >>> print(features.shape)  # (11581, 123)
    >>>
    >>> # Only angle and trigonometric features
    >>> imputer_no_agg = AngleEncoding(include_aggregated_features=False)
    >>> features_no_agg = imputer_no_agg.fit_transform(df)
    >>> print(features_no_agg.shape)  # (11581, 117)
    >>>
    >>> # Only direct angle features
    >>> imputer_angles_only = AngleEncoding(
    ...     include_trig_features=False,
    ...     include_aggregated_features=False
    ... )
    >>> features_angles = imputer_angles_only.fit_transform(df)
    >>> print(features_angles.shape)  # (11581, 39)

    Notes:
    ------
    - Angles are in radians, range [0, 2π)
    - This strategy requires no learning from training data (angles are fixed)
    - The fit() method is still required but does minimal work
    - Captures cyclic symmetry of the C₃₉ group structure
    - Trigonometric features are naturally bounded and rotation-invariant
    """

    def __init__(
        self,
        name: str = "angle_encoding",
        include_trig_features: bool = True,
        include_aggregated_features: bool = True,
        include_proximity_features: bool = True
    ):
        """
        Initialize Angle Encoding imputer.

        Args:
            name: Human-readable name for this imputation strategy
            include_trig_features: Whether to include sin/cos features
            include_aggregated_features: Whether to include aggregated statistics
            include_proximity_features: Whether to include proximity-based quantum features
                                       (adds 195 features, increases total from 127 to 322)

        Examples:
            >>> # Default: all features including proximity
            >>> imputer1 = AngleEncoding()
            >>>
            >>> # Without aggregated features
            >>> imputer2 = AngleEncoding(include_aggregated_features=False)
            >>>
            >>> # Only direct angles
            >>> imputer3 = AngleEncoding(
            ...     include_trig_features=False,
            ...     include_aggregated_features=False
            ... )
            >>>
            >>> # Without proximity features (original 127 features)
            >>> imputer4 = AngleEncoding(include_proximity_features=False)
        """
        # Initialize parent class with config
        super().__init__(
            name=name,
            config={
                "include_trig_features": include_trig_features,
                "include_aggregated_features": include_aggregated_features,
                "include_proximity_features": include_proximity_features
            }
        )

        # Store parameters as instance attributes
        self.include_trig_features = include_trig_features
        self.include_aggregated_features = include_aggregated_features
        self.include_proximity_features = include_proximity_features

        # Pre-compute position angles (fixed, not learned)
        # θᵢ = 2π * (i-1) / 39 for position i (i=1..39)
        self.position_angles_ = 2 * np.pi * np.arange(39) / 39

        # Proximity feature calculator (initialized if needed)
        if self.include_proximity_features:
            self.proximity_calculator = ProximityFeatures(N=39)

        logger.debug(
            f"Initialized AngleEncoding with "
            f"include_trig_features={include_trig_features}, "
            f"include_aggregated_features={include_aggregated_features}, "
            f"include_proximity_features={include_proximity_features}"
        )
        logger.debug(
            f"Position angles range: [{self.position_angles_.min():.4f}, "
            f"{self.position_angles_.max():.4f}] radians"
        )

    def _fit(self, X: pd.DataFrame) -> None:
        """
        Learn statistics from training data (optional for angle encoding).

        Angle encoding uses fixed angular mappings based on position indices,
        so there is no learning required from the training data. However, we
        implement this method to satisfy the BaseImputer interface.

        Future extensions could learn angle adjustments based on position
        frequencies or co-occurrence patterns, but the base implementation
        uses fixed uniform angles.

        Args:
            X: Training DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Side Effects:
            None (angles are pre-computed during __init__)

        Notes:
            - This method is called by the public fit() method
            - Satisfies the abstract method requirement
            - Could be extended to learn angle adjustments in future versions
        """
        # No learning required for basic angle encoding
        # Angles are fixed based on position indices
        logger.info(
            f"Angle encoding uses fixed angular mapping (no learning from "
            f"{len(X)} training samples)"
        )

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform binary quantum states to angle encoding features.

        This method creates angular feature representations by:
        1. Mapping each position to its pre-computed angle θᵢ
        2. Computing trigonometric features (sin, cos) for each position
        3. Computing aggregated features across active positions

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Returns:
            Feature matrix of shape:
            - (n_samples, 123) if all features included (default)
            - (n_samples, 117) if no aggregated features
            - (n_samples, 45) if no trig features
            - (n_samples, 39) if only angle features
            Data type: np.float64

        Algorithm:
            For each sample:
            1. Extract binary QV vector (39 dims)
            2. Create angle features: angle[i] = θᵢ * QV[i]
               (0 for inactive, θᵢ for active)
            3. Create trig features: sin_feat[i] = sin(θᵢ) * QV[i]
                                     cos_feat[i] = cos(θᵢ) * QV[i]
            4. Compute aggregated features over active positions:
               - sum_sin, sum_cos: Total sin/cos across active positions
               - mean_sin, mean_cos: Average sin/cos across active positions
               - resultant_magnitude: ||sum vector||
               - resultant_angle: Direction of sum vector
            5. Concatenate all feature types

        Mathematical Details:
            Direct angle features:
                angle_feat[i] = θᵢ × QV[i]
                where θᵢ = 2π(i-1)/39

            Trigonometric features:
                sin_feat[i] = sin(θᵢ) × QV[i]
                cos_feat[i] = cos(θᵢ) × QV[i]

            Aggregated features (for 5 active positions):
                sum_sin = Σ sin(θⱼ) for j in active positions
                sum_cos = Σ cos(θⱼ) for j in active positions
                mean_sin = sum_sin / 5
                mean_cos = sum_cos / 5
                resultant_magnitude = sqrt(sum_sin² + sum_cos²)
                resultant_angle = atan2(sum_sin, sum_cos)

            Resultant vector interpretation:
                If active positions are evenly distributed around the circle,
                resultant_magnitude ≈ 0 (vectors cancel out).
                If active positions are clustered, resultant_magnitude is large
                and resultant_angle points toward the cluster center.

        Notes:
            - Output validation (NaN/Inf) performed by parent class
            - Each sample is processed independently
            - Angles are in radians [0, 2π)
        """
        # Extract QV columns
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        qv_data = X[qv_columns].values  # Shape: (n_samples, 39)

        # Start building feature list
        features_list = []

        # 1. Direct Angle Features (39 dims)
        # Multiply each position by its angle (0 for inactive, θᵢ for active)
        angle_features = qv_data * self.position_angles_  # Broadcasting
        features_list.append(angle_features)

        # 2. Trigonometric Features (78 dims) - optional
        if self.include_trig_features:
            # Compute sin and cos of position angles
            sin_angles = np.sin(self.position_angles_)
            cos_angles = np.cos(self.position_angles_)

            # Multiply by QV (0 for inactive, sin/cos for active)
            sin_features = qv_data * sin_angles  # Shape: (n_samples, 39)
            cos_features = qv_data * cos_angles  # Shape: (n_samples, 39)

            features_list.extend([sin_features, cos_features])

        # 3. Aggregated Angle Features (6 dims) - optional
        if self.include_aggregated_features:
            # For aggregated features, we need sin/cos values
            sin_angles = np.sin(self.position_angles_)
            cos_angles = np.cos(self.position_angles_)

            # Weighted sin/cos (0 for inactive, value for active)
            sin_values = qv_data * sin_angles  # Shape: (n_samples, 39)
            cos_values = qv_data * cos_angles  # Shape: (n_samples, 39)

            # Sum across active positions (axis=1)
            sum_sin = sin_values.sum(axis=1, keepdims=True)  # Shape: (n_samples, 1)
            sum_cos = cos_values.sum(axis=1, keepdims=True)  # Shape: (n_samples, 1)

            # Mean across active positions (divide by number of active = 5)
            # We know each row has exactly 5 active positions
            n_active = 5
            mean_sin = sum_sin / n_active  # Shape: (n_samples, 1)
            mean_cos = sum_cos / n_active  # Shape: (n_samples, 1)

            # Resultant vector magnitude (length of sum vector)
            # ||R|| = sqrt(sum_sin² + sum_cos²)
            resultant_magnitude = np.sqrt(sum_sin**2 + sum_cos**2)  # Shape: (n_samples, 1)

            # Resultant vector angle (direction of sum vector)
            # atan2 returns angle in [-π, π]
            resultant_angle = np.arctan2(sum_sin, sum_cos)  # Shape: (n_samples, 1)

            # Combine aggregated features
            aggregated_features = np.hstack([
                sum_sin, sum_cos,
                mean_sin, mean_cos,
                resultant_magnitude, resultant_angle
            ])  # Shape: (n_samples, 6)

            features_list.append(aggregated_features)

        # ========================================================================
        # SOPHISTICATED QUANTUM FEATURES (Added 2025-10-21)
        # ========================================================================
        # These features capture phase coherence and angular momentum properties
        # that go beyond simple angle-position mappings
        # ========================================================================

        # --- I. PHASE COHERENCE (Order parameter for angular distribution) ---
        # Measures how "aligned" the active positions are on the circle

        # I1. Phase coherence magnitude (0 = spread out, 1 = perfectly aligned)
        # Order parameter: |Σⱼ e^(iθⱼ)| / n_active for active positions
        # We already have sum_sin and sum_cos from aggregated features
        sin_angles = np.sin(self.position_angles_)
        cos_angles = np.cos(self.position_angles_)
        sin_values = qv_data * sin_angles
        cos_values = qv_data * cos_angles
        sum_sin = sin_values.sum(axis=1, keepdims=True)
        sum_cos = cos_values.sum(axis=1, keepdims=True)

        n_active = 5
        phase_coherence_magnitude = np.sqrt(sum_sin**2 + sum_cos**2) / n_active
        features_list.append(phase_coherence_magnitude)

        # I2. Phase coherence angle (direction of alignment)
        # This is the same as resultant_angle but included explicitly for clarity
        phase_coherence_angle = np.arctan2(sum_sin, sum_cos)
        features_list.append(phase_coherence_angle)

        # --- J. ANGULAR MOMENTUM / WINDING (Topological properties) ---

        # J1. Total angular momentum (sum of active angles)
        # L = Σⱼ θⱼ for j in active positions
        # This is like total "phase" accumulated around the circle
        active_angle_sum = angle_features.sum(axis=1, keepdims=True)  # Already computed above
        features_list.append(active_angle_sum)

        # J2. Winding number (normalized angular momentum)
        # Winding = L / (2π) measures how many times we "wind around" the circle
        winding_number = active_angle_sum / (2 * np.pi)
        features_list.append(winding_number)

        # ========================================================================
        # END SOPHISTICATED QUANTUM FEATURES
        # Feature count added: 2 (phase coherence) + 2 (angular momentum) = 4
        # Total features: 123 (original) + 4 (new) = 127
        # ========================================================================

        # ========================================================================
        # PROXIMITY FEATURES (Added 2025-10-23)
        # ========================================================================
        # Proximity features model how winning positions (QV=1) influence ALL
        # positions through spatial relationships on the C₃₉ cyclic ring.
        # ========================================================================

        if self.include_proximity_features:
            # Compute proximity features for each event
            proximity_features_list = []
            for qv_vector in qv_data:
                prox_features, _ = self.proximity_calculator.compute_all_features(
                    qv_vector,
                    include_interference=False  # Exclude interference (0.11% importance)
                )
                proximity_features_list.append(prox_features)

            proximity_features = np.array(proximity_features_list)  # Shape: (n_samples, 195)
            features_list.append(proximity_features)

        # ========================================================================
        # END PROXIMITY FEATURES
        # Feature count added: 195 (min_dist=39, tunnel=39, density=117)
        # Total features: 127 (original) + 195 (proximity) = 322
        # ========================================================================

        # Concatenate all features horizontally
        result = np.hstack(features_list)

        logger.debug(
            f"Transformed {len(X)} samples to shape {result.shape}. "
            f"Features: angle={True}, trig={self.include_trig_features}, "
            f"aggregated={self.include_aggregated_features}"
        )

        return result

    def get_feature_names(self) -> list:
        """
        Get human-readable names for output features.

        Returns:
            List of feature names, length depends on configuration:
            - 123 names if all features included
            - 117 names if no aggregated features
            - 45 names if no trig features
            - 39 names if only angle features

        Examples:
            >>> imputer = AngleEncoding()
            >>> imputer.fit(df)
            >>> names = imputer.get_feature_names()
            >>> print(names[:5])
            ['qv_1_angle', 'qv_2_angle', ..., 'qv_5_angle']
            >>> print(names[39:44])
            ['qv_1_sin', 'qv_2_sin', ..., 'qv_5_sin']
            >>> print(names[78:83])
            ['qv_1_cos', 'qv_2_cos', ..., 'qv_5_cos']
            >>> print(names[-6:])
            ['sum_sin', 'sum_cos', 'mean_sin', 'mean_cos',
             'resultant_magnitude', 'resultant_angle']
        """
        feature_names = []

        # Angle feature names
        angle_names = [f'qv_{i}_angle' for i in range(1, 40)]
        feature_names.extend(angle_names)

        # Trigonometric feature names
        if self.include_trig_features:
            sin_names = [f'qv_{i}_sin' for i in range(1, 40)]
            cos_names = [f'qv_{i}_cos' for i in range(1, 40)]
            feature_names.extend(sin_names)
            feature_names.extend(cos_names)

        # Aggregated feature names
        if self.include_aggregated_features:
            aggregated_names = [
                'sum_sin', 'sum_cos',
                'mean_sin', 'mean_cos',
                'resultant_magnitude', 'resultant_angle'
            ]
            feature_names.extend(aggregated_names)

        # Sophisticated quantum feature names (added 2025-10-21)
        sophisticated_names = [
            'phase_coherence_magnitude',
            'phase_coherence_angle',
            'angular_momentum',
            'winding_number'
        ]
        feature_names.extend(sophisticated_names)

        return feature_names

    def get_position_angle(self, position: int) -> float:
        """
        Get the angle assigned to a specific position.

        This is a utility method for understanding the angular mapping.

        Args:
            position: Position index (1-39)

        Returns:
            Angle in radians [0, 2π)

        Raises:
            ValueError: If position is not in range [1, 39]

        Examples:
            >>> imputer = AngleEncoding()
            >>> angle_1 = imputer.get_position_angle(1)
            >>> print(f"{angle_1:.4f}")  # 0.0000
            >>> angle_20 = imputer.get_position_angle(20)
            >>> print(f"{angle_20:.4f}")  # ~3.0630 (just past π)
        """
        if position < 1 or position > 39:
            raise ValueError(f"Position must be in range [1, 39], got {position}")

        return self.position_angles_[position - 1]
