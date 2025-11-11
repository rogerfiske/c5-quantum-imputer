"""
Amplitude Embedding Imputation Strategy

This module implements the Amplitude Embedding quantum-inspired imputation method,
which represents quantum states as superpositions where amplitudes are distributed
across active positions. This captures the probabilistic nature of quantum
superposition states.

The quantum-inspired concept:
In quantum mechanics, a superposition state is written as:
    |ψ⟩ = α₁|1⟩ + α₂|2⟩ + ... + α₃₉|39⟩

where αᵢ are complex amplitudes satisfying |α₁|² + |α₂|² + ... + |α₃₉|² = 1.
The squared magnitude |αᵢ|² represents the probability of measuring position i.

For our binary quantum states with exactly 5 active positions, we create amplitude
vectors where the 5 active positions have non-zero amplitudes that are normalized.

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.3 - Amplitude Embedding Strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Literal

from src.imputation.base_imputer import BaseImputer
from src.imputation.proximity_features import ProximityFeatures

# Set up logging
logger = logging.getLogger(__name__)


class AmplitudeEmbedding(BaseImputer):
    """
    Amplitude Embedding imputation strategy.

    Represents quantum states as superpositions with amplitudes distributed
    across active positions. This captures the probabilistic nature of quantum
    superposition states.

    Quantum Mechanics Background:
    ------------------------------
    In quantum mechanics, a pure quantum state |ψ⟩ is represented as a
    superposition of basis states with complex amplitudes:

        |ψ⟩ = α₁|1⟩ + α₂|2⟩ + ... + α₃₉|39⟩

    Key properties:
    1. Amplitudes αᵢ are complex numbers (we use real for simplicity)
    2. Normalization: Σᵢ |αᵢ|² = 1 (sum of squared amplitudes equals 1)
    3. Born rule: |αᵢ|² = probability of measuring position i

    For our binary quantum states:
    - 5 positions are active (QV=1), 34 are inactive (QV=0)
    - Active positions get non-zero amplitudes
    - Inactive positions get zero amplitudes
    - Amplitudes are normalized so squared sum = 1

    Normalization Strategies:
    -------------------------
    **Uniform Normalization** (default):
    - All active positions get equal amplitude: αᵢ = 1/√5 ≈ 0.447
    - This represents a "uniform superposition" over 5 basis states
    - Reflects maximum uncertainty: all active positions equally likely
    - Example: If positions [1,5,8,30,38] are active:
      α₁ = α₅ = α₈ = α₃₀ = α₃₈ = 1/√5 ≈ 0.447
      All other αᵢ = 0

    **Weighted Normalization**:
    - Amplitudes proportional to position frequencies from training data
    - More frequent positions get larger amplitudes
    - Normalized so squared sum = 1
    - Example: If QV_1 appears in 30% of training, QV_5 in 10%:
      α₁ will be larger than α₅ (after normalization)

    Feature Engineering Strategy:
    -----------------------------
    This implementation creates two types of features:

    1. **Amplitude Features** (39 features):
       The amplitude αᵢ for each position. Zero for inactive positions,
       non-zero (normalized) for active positions.

    2. **Probability Features** (39 features):
       The squared amplitude |αᵢ|² for each position. This represents
       the "measurement probability" in quantum mechanics. Also called
       the Born rule probabilities.

    Parameters:
    -----------
    name : str, optional (default="amplitude_embedding")
        Human-readable name for this imputation strategy

    normalization : {"uniform", "weighted"}, optional (default="uniform")
        Normalization strategy for amplitudes:
        - "uniform": Equal amplitudes (1/√n_active) for all active positions
        - "weighted": Amplitudes proportional to position frequencies

    include_probability_features : bool, optional (default=True)
        Whether to include probability (amplitude squared) features.
        - If True: output has 78 features (39 amplitudes + 39 probabilities)
        - If False: output has 39 features (39 amplitudes only)

    Output Dimensions:
    ------------------
    - With probability features (default): (n_samples, 78)
    - Without probability features: (n_samples, 39)

    Attributes:
    -----------
    position_frequencies_ : np.ndarray, shape (39,)
        Learned frequency of each position (used for weighted normalization).
        Only set after calling fit().

    Examples:
    ---------
    >>> from src.data_loader import load_dataset
    >>> from src.imputation.amplitude_embedding import AmplitudeEmbedding
    >>>
    >>> # Load dataset
    >>> df = load_dataset()
    >>>
    >>> # Uniform superposition (default)
    >>> imputer = AmplitudeEmbedding()
    >>> features = imputer.fit_transform(df)
    >>> print(features.shape)  # (11581, 78)
    >>>
    >>> # Weighted by training frequencies
    >>> imputer_weighted = AmplitudeEmbedding(normalization="weighted")
    >>> features_weighted = imputer_weighted.fit_transform(df)
    >>> print(features_weighted.shape)  # (11581, 78)
    >>>
    >>> # Without probability features
    >>> imputer_simple = AmplitudeEmbedding(include_probability_features=False)
    >>> features_simple = imputer_simple.fit_transform(df)
    >>> print(features_simple.shape)  # (11581, 39)

    Notes:
    ------
    - Amplitudes are normalized: sum of squared amplitudes = 1.0 for each row
    - Uniform normalization: amplitude = 1/√5 ≈ 0.447 for active positions
    - Weighted normalization uses position frequencies from training data
    - This captures quantum superposition concept in classical features
    """

    def __init__(
        self,
        name: str = "amplitude_embedding",
        normalization: Literal["uniform", "weighted"] = "uniform",
        include_probability_features: bool = True,
        include_proximity_features: bool = True,
        include_low_boundary_features: bool = True,
        temporal_decay_rate: float = 0.0,
        lookback_window: int = 50
    ):
        """
        Initialize Amplitude Embedding imputer.

        Args:
            name: Human-readable name for this imputation strategy
            normalization: "uniform" or "weighted" amplitude normalization
            include_probability_features: Whether to include |amplitude|² features
            include_proximity_features: Whether to include proximity-based quantum features
                                       (adds 195 features, increases total from 182 to 377)
            include_low_boundary_features: Whether to include LOW_BOUNDARY improvement features
                                          (adds 22 features for QV 2-3 targeting)
            temporal_decay_rate: Exponential decay rate for temporal weighting (Story 13.1)
                                - 0.0 = no temporal weighting (all events weighted equally)
                                - 0.05 = moderate temporal weighting (recommended)
                                - 0.10 = strong temporal weighting (recent events heavily favored)
                                Higher values give more weight to recent events.
            lookback_window: Number of historical events to consider for imputation (Story 13.2)
                            - Default: 50 events
                            - Previous studies suggest 30-40 or 90+ may be more effective
                            - Shorter windows (30-40) capture recent trends
                            - Longer windows (90+) provide more stable patterns

        Raises:
            ValueError: If normalization is not "uniform" or "weighted"

        Examples:
            >>> # Default: uniform normalization with probabilities and proximity
            >>> imputer1 = AmplitudeEmbedding()
            >>>
            >>> # Weighted normalization
            >>> imputer2 = AmplitudeEmbedding(normalization="weighted")
            >>>
            >>> # Without probability features
            >>> imputer3 = AmplitudeEmbedding(include_probability_features=False)
            >>>
            >>> # Without proximity features (original 182 features)
            >>> imputer4 = AmplitudeEmbedding(include_proximity_features=False)
            >>>
            >>> # With temporal weighting (Story 13.1)
            >>> imputer5 = AmplitudeEmbedding(temporal_decay_rate=0.05)
        """
        # Validate normalization parameter
        if normalization not in ["uniform", "weighted"]:
            raise ValueError(
                f"normalization must be 'uniform' or 'weighted', got '{normalization}'"
            )

        # Initialize parent class with config
        super().__init__(
            name=name,
            config={
                "normalization": normalization,
                "include_probability_features": include_probability_features,
                "include_proximity_features": include_proximity_features,
                "include_low_boundary_features": include_low_boundary_features,
                "temporal_decay_rate": temporal_decay_rate,
                "lookback_window": lookback_window
            }
        )

        # Store parameters as instance attributes
        self.normalization = normalization
        self.include_probability_features = include_probability_features
        self.include_proximity_features = include_proximity_features
        self.include_low_boundary_features = include_low_boundary_features
        self.temporal_decay_rate = temporal_decay_rate
        self.lookback_window = lookback_window

        # Learned parameter (set during fit(), used for weighted normalization)
        self.position_frequencies_ = None

        # Proximity feature calculator (initialized if needed)
        if self.include_proximity_features:
            self.proximity_calculator = ProximityFeatures(N=39)

        logger.debug(
            f"Initialized AmplitudeEmbedding with normalization={normalization}, "
            f"include_probability_features={include_probability_features}, "
            f"include_proximity_features={include_proximity_features}, "
            f"temporal_decay_rate={temporal_decay_rate}"
        )

    def _fit(self, X: pd.DataFrame) -> None:
        """
        Learn position frequency statistics for weighted normalization.

        For uniform normalization, this method doesn't need to learn anything
        (amplitudes are always 1/√5). However, for weighted normalization,
        we learn position frequencies from training data to weight amplitudes.

        Args:
            X: Training DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Side Effects:
            Sets self.position_frequencies_ to numpy array of shape (39,)
            containing position frequencies (only used if normalization="weighted")

        Notes:
            - For uniform normalization: frequencies are computed but not used
            - For weighted normalization: frequencies determine amplitude weights
            - Called automatically by the public fit() method
        """
        # Calculate position frequencies (same as Basis Embedding)
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        position_counts = X[qv_columns].sum(axis=0)
        self.position_frequencies_ = position_counts.values / len(X)

        logger.info(
            f"Learned position frequencies from {len(X)} samples. "
            f"Normalization mode: {self.normalization}"
        )

        if self.normalization == "weighted":
            logger.debug(
                f"Frequency range for weighting: "
                f"[{self.position_frequencies_.min():.4f}, "
                f"{self.position_frequencies_.max():.4f}]"
            )

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform binary quantum states to amplitude embedding features.

        This method creates amplitude vectors where:
        1. Active positions (QV=1) get non-zero amplitudes
        2. Inactive positions (QV=0) get zero amplitudes
        3. Amplitudes are normalized so sum of squares = 1 for each row
        4. Optionally compute probability features (amplitude squared)

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Returns:
            Feature matrix of shape:
            - (n_samples, 78) if include_probability_features=True
            - (n_samples, 39) if include_probability_features=False
            Data type: np.float64

        Algorithm:
            For each sample:
            1. Identify which positions are active (QV=1)
            2. Assign raw amplitudes based on normalization mode:
               - uniform: all active get value 1.0
               - weighted: active get position_frequencies_[i]
            3. Normalize so sum of squared amplitudes = 1:
               normalized_amp[i] = raw_amp[i] / sqrt(sum(raw_amp²))
            4. Compute probabilities: prob[i] = normalized_amp[i]²
            5. Concatenate amplitude and probability features

        Mathematical Details:
            Uniform normalization for n_active=5:
                α = 1/√5 ≈ 0.447 for each active position
                Σᵢ α² = 5 × (1/√5)² = 5 × (1/5) = 1.0 ✓

            Weighted normalization:
                raw_αᵢ = frequency[i] × QV[i]
                α = raw_α / ||raw_α|| where ||·|| is L2 norm
                Ensures Σᵢ α² = 1.0

        Notes:
            - Output validation (NaN/Inf) performed by parent class
            - Each row is independently normalized
        """
        # Extract QV columns
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        qv_data = X[qv_columns].values  # Shape: (n_samples, 39)

        # Create amplitude features based on normalization mode
        if self.normalization == "uniform":
            # Uniform: all active positions get equal raw amplitude (1.0)
            raw_amplitudes = qv_data.astype(np.float64)
        else:  # weighted
            # Weighted: active positions weighted by frequency
            # Broadcasting: (n_samples, 39) * (39,) → (n_samples, 39)
            raw_amplitudes = qv_data * self.position_frequencies_

        # Normalize amplitudes so sum of squares = 1 for each row
        # L2 norm per row: sqrt(sum of squared amplitudes)
        amplitude_norms = np.sqrt((raw_amplitudes ** 2).sum(axis=1, keepdims=True))

        # Avoid division by zero (shouldn't happen with valid data, but defensive)
        amplitude_norms = np.where(amplitude_norms > 0, amplitude_norms, 1.0)

        # Normalized amplitudes
        amplitudes = raw_amplitudes / amplitude_norms  # Shape: (n_samples, 39)

        # Start with amplitude features
        features_list = [amplitudes]

        # Add probability features if requested
        if self.include_probability_features:
            # Probability = amplitude squared (Born rule)
            probabilities = amplitudes ** 2  # Shape: (n_samples, 39)
            features_list.append(probabilities)

        # ========================================================================
        # SOPHISTICATED QUANTUM FEATURES (Added 2025-10-21)
        # ========================================================================
        # These features capture genuine quantum properties beyond trivial QV
        # transformations: interference, entanglement, and collective behavior
        # ========================================================================

        # --- A. INTERFERENCE TERMS (Non-trivial cross-position relationships) ---
        # Quantum interference: ψᵢ × ψⱼ captures correlations between positions

        interference_features = []

        # A1. Adjacent position interference (38 features)
        # Captures nearest-neighbor correlations on C₃₉ ring
        for i in range(39):
            j = (i + 1) % 39  # Next position (wraps around)
            interference_adjacent = amplitudes[:, i] * amplitudes[:, j]
            interference_features.append(interference_adjacent)

        # A2. Opposite position interference (19 features)
        # Captures antipodal correlations (positions ~180° apart on ring)
        for i in range(19):
            j = i + 19  # Opposite side of ring
            interference_opposite = amplitudes[:, i] * amplitudes[:, j]
            interference_features.append(interference_opposite)

        # A3. Harmonic position interference (39 features)
        # Captures C₃ and C₁₃ subgroup symmetries
        # C₁₃ subgroup: every 3rd position (13 positions total)
        c13_positions = [i for i in range(0, 39, 3)]  # [0, 3, 6, ..., 36]
        # C₃ subgroup: every 13th position (3 positions total)
        c3_positions = [0, 13, 26]

        for i in c13_positions:
            for j in c3_positions:
                if i != j:
                    interference_harmonic = amplitudes[:, i] * amplitudes[:, j]
                    interference_features.append(interference_harmonic)

        # Stack interference features: (n_samples, ~96 features)
        interference_matrix = np.column_stack(interference_features)
        features_list.append(interference_matrix)

        # --- B. ENTANGLEMENT MEASURES (System-level quantum properties) ---

        # Compute density matrix ρ = |ψ⟩⟨ψ| for entanglement calculations
        # Note: We compute per-sample using vectorized operations

        # B1. Von Neumann Entropy: S = -Tr(ρ log ρ)
        # Measures mixedness of quantum state (0 = pure, log(N) = maximally mixed)
        probabilities_born = amplitudes ** 2  # Born rule probabilities
        # Handle zeros in log (add small epsilon)
        epsilon = 1e-10
        safe_probs = np.where(probabilities_born > epsilon, probabilities_born, epsilon)
        von_neumann_entropy = -np.sum(safe_probs * np.log(safe_probs), axis=1, keepdims=True)
        features_list.append(von_neumann_entropy)

        # B2. Participation Ratio: PR = 1 / Σᵢ|ψᵢ|⁴
        # Measures delocalization (5 = uniform over 5 states, 1 = localized)
        amplitude_4th_power = amplitudes ** 4
        sum_4th_power = np.sum(amplitude_4th_power, axis=1, keepdims=True)
        participation_ratio = 1.0 / (sum_4th_power + epsilon)
        features_list.append(participation_ratio)

        # B3. Purity: Tr(ρ²) for pure state (using Tr(ρ²) = Σᵢ|ψᵢ|⁴)
        # For pure states: purity = 1, for mixed states: purity < 1
        purity = sum_4th_power  # Already computed above
        features_list.append(purity)

        # B4. Linear Entropy: S_L = 1 - Tr(ρ²)
        # Alternative entropy measure (0 = pure, 1-1/N = maximally mixed)
        linear_entropy = 1.0 - purity
        features_list.append(linear_entropy)

        # --- C. HIGHER MOMENTS (Statistical properties of position distribution) ---

        # Treat |ψᵢ|² as probability distribution over positions
        positions = np.arange(1, 40).reshape(1, -1)  # [1, 2, ..., 39]

        # C1. Mean position (weighted centroid)
        mean_position = np.sum(positions * probabilities_born, axis=1, keepdims=True)
        features_list.append(mean_position)

        # C2. Variance (spread of position distribution)
        variance = np.sum(positions**2 * probabilities_born, axis=1, keepdims=True) - mean_position**2
        variance = np.maximum(variance, epsilon)  # Ensure non-negative
        features_list.append(variance)

        # C3. Skewness (asymmetry of distribution)
        std_dev = np.sqrt(variance)
        skewness = np.sum((positions - mean_position)**3 * probabilities_born, axis=1, keepdims=True) / (std_dev**3 + epsilon)
        features_list.append(skewness)

        # C4. Kurtosis (tail heaviness of distribution)
        kurtosis = np.sum((positions - mean_position)**4 * probabilities_born, axis=1, keepdims=True) / (variance**2 + epsilon)
        features_list.append(kurtosis)

        # ========================================================================
        # END SOPHISTICATED QUANTUM FEATURES
        # Feature count added: ~96 (interference) + 4 (entanglement) + 4 (moments) = ~104
        # Total features: 78 (original) + 104 (new) = 182
        # ========================================================================

        # ========================================================================
        # POSITION-SPECIFIC TEMPORAL FEATURES (Added 2025-11-05, Story 11.2.1)
        # ========================================================================
        # These features capture position-specific temporal evolution patterns
        # for the hardest QVs identified in failure mode analysis.
        # Goal: Improve low-range (QVs 6-13) recall@20 from 15.4% to >35%
        #
        # Key insight from Task 11.2.1.1:
        # - Frequency is NOT the problem (low-range appears MORE than mid-range)
        # - Temporal dynamics differ: low-range has 22% longer inter-arrival time
        # - Need position-specific features, not just global features
        # ========================================================================

        # UPDATED 2025-11-07 based on 1000-event diagnostic (Tool 1):
        # TEMP: Use positions matching lgbm_ranker_enhanced.pkl training
        # Original: [2, 6, 8, 9, 11, 13, 15, 27]
        # Updated (2025-11-07): [2, 3, 5, 6, 9, 10, 11, 12, 14, 15]  # Hardest 10 QVs
        # For diagnostic tools compatibility, using original positions
        target_positions = [2, 6, 8, 9, 11, 13, 15, 27]  # Match enhanced model training

        # For position-specific temporal features, we need historical context
        # We'll compute these features by looking at recent history
        position_temporal_features = []

        for pos_idx in target_positions:
            qv_idx = pos_idx - 1  # 0-indexed for array access

            # Feature 1: Recent activation count (last 3 events)
            # How many times did this position appear recently?
            recent_activations = []
            for i in range(len(qv_data)):
                if i < 3:
                    # Not enough history
                    recent_activations.append(0.0)
                else:
                    # Count activations in last 3 events
                    count = np.sum(qv_data[i-3:i, qv_idx])
                    recent_activations.append(float(count))

            position_temporal_features.append(np.array(recent_activations))

            # Feature 2: Time since last activation (in events)
            # How long ago did this position last appear?
            time_since_last = []
            for i in range(len(qv_data)):
                # Look backwards to find last activation
                found_distance = 0
                for lookback in range(1, min(i+1, 10)):  # Look back up to 10 events
                    if qv_data[i-lookback, qv_idx] == 1:
                        found_distance = lookback
                        break
                else:
                    # Not found in last 10 events
                    found_distance = 10

                # Normalize to [0, 1]
                time_since_last.append(found_distance / 10.0)

            position_temporal_features.append(np.array(time_since_last))

            # Feature 3: Temporal velocity (rate of change)
            # Is this position becoming more or less frequent?
            # FIX 2025-11-06: Use history only (i-1 vs i-2), not current event
            velocity = []
            for i in range(len(qv_data)):
                if i < 2:
                    velocity.append(0.0)
                else:
                    # Previous state (i-1) - state before that (i-2)
                    # This predicts velocity WITHOUT seeing current event
                    vel = float(qv_data[i-1, qv_idx] - qv_data[i-2, qv_idx])
                    velocity.append(vel)

            position_temporal_features.append(np.array(velocity))

            # Feature 4: Temporal acceleration (rate of change of velocity)
            # Is the velocity itself changing?
            # FIX 2025-11-06: Use history only, not current event
            acceleration = []
            for i in range(len(qv_data)):
                if i < 3:
                    acceleration.append(0.0)
                else:
                    # Velocity at i-1 - velocity at i-2
                    # This predicts acceleration WITHOUT seeing current event
                    vel_at_i_minus_1 = float(qv_data[i-1, qv_idx] - qv_data[i-2, qv_idx])
                    vel_at_i_minus_2 = float(qv_data[i-2, qv_idx] - qv_data[i-3, qv_idx])
                    accel = vel_at_i_minus_1 - vel_at_i_minus_2
                    acceleration.append(accel)

            position_temporal_features.append(np.array(acceleration))

            # Feature 5: Weighted recent history (exponential decay)
            # Recent events matter more than older events
            weighted_history = []
            for i in range(len(qv_data)):
                if i < 5:
                    # Not enough history
                    weighted_history.append(0.0)
                else:
                    # Exponential weights: 0.5^1, 0.5^2, 0.5^3, 0.5^4, 0.5^5
                    weights = np.array([0.5**k for k in range(1, 6)])  # [0.5, 0.25, 0.125, 0.0625, 0.03125]
                    history = qv_data[i-5:i, qv_idx]
                    weighted = np.sum(history * weights)
                    weighted_history.append(float(weighted))

            position_temporal_features.append(np.array(weighted_history))

        # Stack position-specific temporal features
        # Total: 10 positions × 5 features = 50 features (UPDATED 2025-11-07)
        if position_temporal_features:
            position_temporal_matrix = np.column_stack(position_temporal_features)
            features_list.append(position_temporal_matrix)

        # ========================================================================
        # END POSITION-SPECIFIC TEMPORAL FEATURES
        # Feature count added: 50 (10 positions × 5 temporal features each) [UPDATED 2025-11-07: was 40]
        # Total features: 182 (baseline) + 50 (position-temporal) = 232
        # ========================================================================

        # ========================================================================
        # ROLLING SEQUENCE / PATTERN FEATURES (Added 2025-11-05, Story 11.2.1)
        # ========================================================================
        # These features capture pattern triggers that predict specific QVs.
        # Goal: Capture sequences/patterns that reliably precede low-range QVs
        #
        # Key insight from Task 11.2.1.1 (H4: Pattern Trigger Hypothesis):
        # - Hardest QVs (9, 15, 2) may appear after specific patterns
        # - Need to capture "what typically follows this sequence?"
        # ========================================================================

        pattern_features = []

        # Feature 1: Last N pattern hash (sliding window signature)
        # Create unique fingerprint for recent sequence of QVs
        pattern_hashes = []
        for i in range(len(qv_data)):
            if i < 3:
                # Not enough history
                pattern_hashes.append(0.0)
            else:
                # Get last 3 events' active positions
                last_3_active = []
                for event_idx in range(i-3, i):
                    active_positions = tuple(np.where(qv_data[event_idx] == 1)[0])
                    last_3_active.append(active_positions)

                # Create hash from pattern (simplified: sum of position indices)
                # This creates a unique-ish signature for this pattern
                pattern_sig = sum([sum(pos_tuple) for pos_tuple in last_3_active])
                # Normalize to [0, 1] range (max possible sum ~= 3 * 5 * 39 = 585)
                pattern_hashes.append(pattern_sig / 585.0)

        pattern_features.append(np.array(pattern_hashes))

        # Feature 2: Pattern recency (how long since we saw this pattern?)
        pattern_recency = []
        seen_patterns = {}  # Track when we last saw each pattern

        for i in range(len(qv_data)):
            if i < 3:
                pattern_recency.append(1.0)  # Max recency (never seen)
            else:
                pattern_sig = int(pattern_hashes[i] * 585.0)  # Recover hash

                if pattern_sig in seen_patterns:
                    # How many events since we last saw this pattern?
                    events_ago = i - seen_patterns[pattern_sig]
                    recency = min(events_ago / 10.0, 1.0)  # Cap at 10 events
                    pattern_recency.append(recency)
                else:
                    # Never seen before
                    pattern_recency.append(1.0)

                # Update last seen
                seen_patterns[pattern_sig] = i

        pattern_features.append(np.array(pattern_recency))

        # Feature 3-10: Position-specific pattern trigger scores
        # "Given current pattern, how likely is position X to appear next?"
        for pos_idx in target_positions:
            qv_idx = pos_idx - 1

            trigger_scores = []
            pattern_to_next_counts = {}  # Count how often pattern → position

            for i in range(len(qv_data)):
                if i < 4:
                    # Need at least 4 events (3 for pattern + 1 for next)
                    trigger_scores.append(0.0)
                else:
                    # Get previous pattern
                    prev_pattern_sig = int(pattern_hashes[i-1] * 585.0)

                    # Did this position appear after this pattern?
                    if qv_data[i-1, qv_idx] == 1:  # Position appeared at i-1
                        if prev_pattern_sig not in pattern_to_next_counts:
                            pattern_to_next_counts[prev_pattern_sig] = {'hits': 0, 'total': 0}
                        pattern_to_next_counts[prev_pattern_sig]['hits'] += 1

                    # Track total occurrences of this pattern
                    if prev_pattern_sig in pattern_to_next_counts:
                        pattern_to_next_counts[prev_pattern_sig]['total'] += 1

                    # Current score: what's the historical trigger rate?
                    current_pattern_sig = int(pattern_hashes[i] * 585.0)
                    if current_pattern_sig in pattern_to_next_counts and pattern_to_next_counts[current_pattern_sig]['total'] > 0:
                        score = pattern_to_next_counts[current_pattern_sig]['hits'] / pattern_to_next_counts[current_pattern_sig]['total']
                        trigger_scores.append(score)
                    else:
                        trigger_scores.append(0.0)

            pattern_features.append(np.array(trigger_scores))

        # Stack pattern features
        # Total: 2 (global pattern) + 10 (position-specific triggers) = 12 features (UPDATED 2025-11-07)
        if pattern_features:
            pattern_matrix = np.column_stack(pattern_features)
            features_list.append(pattern_matrix)

        # ========================================================================
        # END ROLLING SEQUENCE / PATTERN FEATURES
        # Feature count added: 12 (2 global + 10 position-specific) [UPDATED 2025-11-07: was 10]
        # Total features: 232 (previous) + 12 (pattern) = 244
        # ========================================================================

        # ========================================================================
        # CYCLICAL DISTANCE & EDGE ASYMMETRY FEATURES (Added 2025-11-05, Story 11.2.1)
        # ========================================================================
        # These features address the edge asymmetry problem:
        # - edge_low (1-5): 20.8% recall@20
        # - edge_high (35-39): 61.2% recall@20  (3x better!)
        #
        # Hypothesis: Need wrap-around distance features to capture cyclic geometry
        # ========================================================================

        cyclical_features = []

        # Helper: Cylindrical/wrap-around distance
        def cylindrical_distance(pos_i, pos_j, n_positions=39):
            """Shortest distance on circular ring (wraps around)"""
            linear_dist = abs(pos_i - pos_j)
            wrap_dist = n_positions - linear_dist
            return min(linear_dist, wrap_dist)

        # Feature 1-8: Cylindrical distance to target positions
        # For each target position, compute wrap-around distance from each active position
        # FIX 2025-11-06: Use PREVIOUS event (i-1), not current event (i)
        for target_pos in target_positions:
            target_idx = target_pos - 1

            cylindrical_dists = []
            for i in range(len(qv_data)):
                if i < 1:
                    # No history yet
                    cylindrical_dists.append(0.5)  # Neutral value
                else:
                    # Get all active positions in PREVIOUS event (i-1), not current
                    active_positions = np.where(qv_data[i-1] == 1)[0]

                    if len(active_positions) == 0:
                        # No active positions (shouldn't happen, but defensive)
                        cylindrical_dists.append(0.0)
                    else:
                        # Compute cylindrical distance from each active position to target
                        distances = [cylindrical_distance(pos, target_idx) for pos in active_positions]
                        # Use minimum distance (closest active position to target)
                        min_dist = min(distances)
                        # Normalize to [0, 1] (max distance is 19 on C39 ring)
                        cylindrical_dists.append(min_dist / 19.0)

            cyclical_features.append(np.array(cylindrical_dists))

        # Feature 9: Edge proximity score (distance to nearest edge)
        # Captures: Am I near an edge (1-5 or 35-39)?
        # FIX 2025-11-06: Use PREVIOUS event (i-1), not current event (i)
        edge_proximity = []
        for i in range(len(qv_data)):
            if i < 1:
                edge_proximity.append(0.5)  # Neutral - no history
            else:
                # Use PREVIOUS event's active positions
                active_positions = np.where(qv_data[i-1] == 1)[0]

                if len(active_positions) == 0:
                    edge_proximity.append(0.5)  # Neutral
                else:
                    # Distance to nearest edge position (0-4 or 34-38 in 0-indexed)
                    edge_dists = []
                    for pos in active_positions:
                        # Distance to low edge (position 0)
                        dist_to_low_edge = min(pos, cylindrical_distance(pos, 0))
                        # Distance to high edge (position 38)
                        dist_to_high_edge = min(38 - pos, cylindrical_distance(pos, 38))
                        # Minimum distance to any edge
                        edge_dists.append(min(dist_to_low_edge, dist_to_high_edge))

                    # Normalize (closer to edge = higher score)
                    min_edge_dist = min(edge_dists)
                    # Invert: close to edge (0) → high score (1), far from edge (19) → low score (0)
                    edge_proximity.append(1.0 - (min_edge_dist / 19.0))

        cyclical_features.append(np.array(edge_proximity))

        # Feature 10: Harmonic resonance (captures C3, C13 subgroup structure)
        # Position modulo 3 (for C13 subgroup, every 3rd position)
        # FIX 2025-11-06: Use PREVIOUS event (i-1), not current event (i)
        harmonic_mod3 = []
        for i in range(len(qv_data)):
            if i < 1:
                harmonic_mod3.append(0.5)  # Neutral - no history
            else:
                # Use PREVIOUS event's active positions
                active_positions = np.where(qv_data[i-1] == 1)[0]
                if len(active_positions) > 0:
                    # Average position mod 3 (captures harmonic structure)
                    avg_mod3 = np.mean([pos % 3 for pos in active_positions])
                    harmonic_mod3.append(avg_mod3 / 2.0)  # Normalize to [0, 1]
                else:
                    harmonic_mod3.append(0.5)

        cyclical_features.append(np.array(harmonic_mod3))

        # Stack cyclical features
        # Total: 10 (cylindrical dists to targets) + 1 (edge proximity) + 1 (harmonic) = 12 features (UPDATED 2025-11-07)
        if cyclical_features:
            cyclical_matrix = np.column_stack(cyclical_features)
            features_list.append(cyclical_matrix)

        # ========================================================================
        # END CYCLICAL DISTANCE & EDGE ASYMMETRY FEATURES
        # Feature count added: 10 (8 cylindrical + 1 edge + 1 harmonic) [CORRECTED 2025-11-10]
        # Total features: 232 (previous) + 10 (cyclical) = 242
        # ========================================================================

        # ========================================================================
        # LOW_BOUNDARY IMPROVEMENT FEATURES (Added 2025-11-10, Option A)
        # ========================================================================
        # These features specifically target poor performance on QV 2-3 (56-66% recall).
        # Goal: Improve LOW_BOUNDARY zone (QV 1-5) from current 56-89% to >80% recall.
        #
        # Key insights from Tool 4 QV Difficulty Profiler:
        # - QV 2: 56.8% recall, 16.3% precision, avg_rank 15.7 when actual
        # - QV 3: 66.1% recall, 14.8% precision, avg_rank 15.0 when actual
        # - Problem: Model sees them but ranks them LOW (high avg_rank)
        # - Problem: Model over-predicts them (low precision)
        #
        # Strategy: Add QV-specific historical features to help model distinguish
        # LOW_BOUNDARY QVs from similar-looking positions
        # ========================================================================

        if self.include_low_boundary_features:
            low_boundary_features = []

            # Define QV zones for targeted features
            low_boundary_qvs = [1, 2, 3, 4, 5]  # 0-indexed: 0, 1, 2, 3, 4
            hard_qvs = [2, 3, 10, 11, 14, 15]  # 0-indexed: 1, 2, 9, 10, 13, 14

            # Feature 1-6: QV-specific historical activation rate (for hard QVs)
            # "How often has this specific QV appeared historically?"
            # Helps model learn some QVs are intrinsically rarer than others
            for qv_pos in hard_qvs:
                qv_idx = qv_pos - 1  # Convert to 0-indexed
    
                activation_rates = []
                for i in range(len(qv_data)):
                    if i < 10:
                        # Not enough history
                        activation_rates.append(0.13)  # Default ~= 5/39 baseline
                    else:
                        # Compute activation rate over lookback window (or all available history)
                        lookback = min(i, self.lookback_window)
                        historical_activations = qv_data[i-lookback:i, qv_idx]
                        rate = np.mean(historical_activations)
                        activation_rates.append(rate)
    
                low_boundary_features.append(np.array(activation_rates))
    
            # Feature 7-12: Recent momentum (for hard QVs)
            # "Is this QV becoming more or less frequent in recent events?"
            # Captures trend: activation rate last 10 events vs last 50 events
            for qv_pos in hard_qvs:
                qv_idx = qv_pos - 1
    
                momentum_scores = []
                for i in range(len(qv_data)):
                    if i < 50:
                        # Not enough history for momentum calculation
                        momentum_scores.append(0.0)  # Neutral
                    else:
                        # Recent rate (last 10 events)
                        recent_rate = np.mean(qv_data[i-10:i, qv_idx])
                        # Historical rate (events 50-10 ago, excluding recent 10)
                        historical_rate = np.mean(qv_data[i-50:i-10, qv_idx])
    
                        # Momentum: positive if increasing, negative if decreasing
                        if historical_rate > 0:
                            momentum = (recent_rate - historical_rate) / (historical_rate + 1e-10)
                        else:
                            momentum = recent_rate  # If no historical, use recent as signal
    
                        # Clip to [-1, 1] range
                        momentum = np.clip(momentum, -1.0, 1.0)
                        momentum_scores.append(momentum)
    
                low_boundary_features.append(np.array(momentum_scores))
    
            # Feature 13-17: Boundary zone activation density (for LOW_BOUNDARY QVs 1-5)
            # "How active is the LOW_BOUNDARY zone recently?"
            # Helps model understand zone-level patterns
            for qv_pos in low_boundary_qvs:
                qv_idx = qv_pos - 1
    
                zone_density = []
                for i in range(len(qv_data)):
                    if i < 5:
                        # Not enough history
                        zone_density.append(0.5)  # Neutral
                    else:
                        # Count activations in LOW_BOUNDARY zone in last 5 events
                        last_5_events = qv_data[i-5:i, :5]  # First 5 QVs (indices 0-4)
                        zone_activations = np.sum(last_5_events)
    
                        # Normalize by max possible (5 events × 5 positions = 25)
                        density = zone_activations / 25.0
                        zone_density.append(density)
    
                low_boundary_features.append(np.array(zone_density))
    
            # Feature 18-22: Position-specific inter-arrival time (for LOW_BOUNDARY QVs)
            # "How long since this specific QV last appeared?"
            # Longer wait time might predict imminent appearance
            for qv_pos in low_boundary_qvs:
                qv_idx = qv_pos - 1
    
                inter_arrival_times = []
                for i in range(len(qv_data)):
                    if i < 1:
                        inter_arrival_times.append(0.5)  # Neutral
                    else:
                        # Find how many events since last activation
                        events_since_last = 0
                        for lookback in range(1, min(i+1, 100)):
                            if qv_data[i-lookback, qv_idx] == 1:
                                events_since_last = lookback
                                break
                        else:
                            # Not found in last 100 events
                            events_since_last = 100
    
                        # Normalize to [0, 1] (cap at 100 events)
                        normalized_time = min(events_since_last / 100.0, 1.0)
                        inter_arrival_times.append(normalized_time)
    
                low_boundary_features.append(np.array(inter_arrival_times))
    
            # Stack LOW_BOUNDARY improvement features
            # Total: 6 (activation rates) + 6 (momentum) + 5 (zone density) + 5 (inter-arrival) = 22 features
            if low_boundary_features:
                low_boundary_matrix = np.column_stack(low_boundary_features)
                features_list.append(low_boundary_matrix)

        # ========================================================================
        # END LOW_BOUNDARY IMPROVEMENT FEATURES
        # Feature count added: 22 features (6+6+5+5)
        # Total features: 242 (previous) + 22 (low_boundary) = 264
        # ========================================================================

        # ========================================================================
        # PROXIMITY FEATURES (Added 2025-10-23)
        # ========================================================================
        # Proximity features model how winning positions (QV=1) influence ALL
        # positions through spatial relationships on the C₃₉ cyclic ring.
        # These features capture quantum-inspired effects: distance, tunneling,
        # and local density patterns.
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
        # Total features: 182 (original) + 195 (proximity) = 377
        # ========================================================================

        # Concatenate features horizontally
        result = np.hstack(features_list)

        logger.debug(
            f"Transformed {len(X)} samples to shape {result.shape}. "
            f"Normalization: {self.normalization}, "
            f"Features: {'amplitude + probability' if self.include_probability_features else 'amplitude only'}"
        )

        return result

    def get_feature_names(self) -> list:
        """
        Get human-readable names for output features.

        Returns:
            List of feature names, length ~182 (with sophisticated quantum features)

        Examples:
            >>> imputer = AmplitudeEmbedding()
            >>> imputer.fit(df)
            >>> names = imputer.get_feature_names()
            >>> print(names[:5])
            ['qv_1_amplitude', 'qv_2_amplitude', ..., 'qv_5_amplitude']
            >>> print(names[39:44])
            ['qv_1_probability', 'qv_2_probability', ..., 'qv_5_probability']
        """
        feature_names = []

        # Amplitude feature names
        amplitude_names = [f'qv_{i}_amplitude' for i in range(1, 40)]
        feature_names.extend(amplitude_names)

        if self.include_probability_features:
            # Probability feature names
            probability_names = [f'qv_{i}_probability' for i in range(1, 40)]
            feature_names.extend(probability_names)

        # --- INTERFERENCE FEATURES ---
        # A1. Adjacent interference (38 features)
        for i in range(39):
            j = (i + 1) % 39
            feature_names.append(f'interference_adj_{i+1}_{j+1}')

        # A2. Opposite interference (19 features)
        for i in range(19):
            j = i + 19
            feature_names.append(f'interference_opp_{i+1}_{j+1}')

        # A3. Harmonic interference (C₃ × C₁₃)
        c13_positions = [i for i in range(0, 39, 3)]
        c3_positions = [0, 13, 26]
        for i in c13_positions:
            for j in c3_positions:
                if i != j:
                    feature_names.append(f'interference_harmonic_{i+1}_{j+1}')

        # --- ENTANGLEMENT MEASURES ---
        feature_names.append('von_neumann_entropy')
        feature_names.append('participation_ratio')
        feature_names.append('purity')
        feature_names.append('linear_entropy')

        # --- HIGHER MOMENTS ---
        feature_names.append('mean_position')
        feature_names.append('variance')
        feature_names.append('skewness')
        feature_names.append('kurtosis')

        # --- POSITION-SPECIFIC TEMPORAL FEATURES (Story 11.2.1, Updated 2025-11-07) ---
        # UPDATED 2025-11-07 based on 1000-event diagnostic: 10 hardest QVs
        target_positions = [2, 6, 8, 9, 11, 13, 15, 27]  # Match enhanced model training
        for pos in target_positions:
            feature_names.append(f'pos_{pos}_recent_activations')
            feature_names.append(f'pos_{pos}_time_since_last')
            feature_names.append(f'pos_{pos}_velocity')
            feature_names.append(f'pos_{pos}_acceleration')
            feature_names.append(f'pos_{pos}_weighted_history')

        # --- ROLLING SEQUENCE / PATTERN FEATURES (Story 11.2.1, Updated 2025-11-07) ---
        feature_names.append('pattern_hash')
        feature_names.append('pattern_recency')
        for pos in target_positions:
            feature_names.append(f'pos_{pos}_pattern_trigger_score')

        # --- CYCLICAL DISTANCE & EDGE ASYMMETRY FEATURES (Story 11.2.1, Updated 2025-11-07) ---
        for pos in target_positions:
            feature_names.append(f'pos_{pos}_cylindrical_distance')
        feature_names.append('edge_proximity_score')
        feature_names.append('harmonic_mod3')

        # --- LOW_BOUNDARY IMPROVEMENT FEATURES (Option A, 2025-11-10) ---
        if self.include_low_boundary_features:
            hard_qvs = [2, 3, 10, 11, 14, 15]

            # Feature 1-6: QV-specific activation rates
            for qv in hard_qvs:
                feature_names.append(f'qv_{qv}_activation_rate')

            # Feature 7-12: Recent momentum
            for qv in hard_qvs:
                feature_names.append(f'qv_{qv}_momentum')

            # Feature 13-17: Zone density (LOW_BOUNDARY)
            for i in range(5):
                feature_names.append(f'zone_low_boundary_density_{i+1}')

            # Feature 18-22: Inter-arrival time
            for i in range(5):
                feature_names.append(f'low_boundary_inter_arrival_{i+1}')

        return feature_names

    def transform_single_event(self, current_event: pd.Series, historical_context: pd.DataFrame) -> np.ndarray:
        """
        Transform a SINGLE event with proper temporal isolation.

        This method computes features for ONE event, using only its historical context
        (past events) to compute temporal/pattern features. This ensures no data leakage
        in sequential evaluation.

        CRITICAL FIX (2025-11-06): Temporal features have been refactored to use ONLY
        historical data (row i-1 and earlier), not current event (row i). This allows
        us to safely include the current event in the dataframe for amplitude/probability
        feature computation while temporal features only look backward.

        Args:
            current_event: Single event as pandas Series with QV_1...QV_39 columns
            historical_context: DataFrame of previous events (can be empty for first event)

        Returns:
            Feature vector for this single event (shape: (1, n_features))

        Example:
            >>> imputer.fit(training_data)
            >>> for i in range(len(holdout)):
            >>>     event = holdout.iloc[i]
            >>>     history = holdout.iloc[:i]  # All previous holdout events
            >>>     features = imputer.transform_single_event(event, history)
        """
        # Combine history + current event
        # It's now safe to include current event because temporal features
        # have been refactored to only look at previous rows
        if len(historical_context) == 0:
            # First event - no history
            combined = pd.DataFrame([current_event])
        else:
            combined = pd.concat([historical_context, pd.DataFrame([current_event])], ignore_index=True)

        # Transform the combined data
        # - Amplitude/probability features: computed from current event's QVs (row i)
        # - Temporal features: computed from history only (rows i-1, i-2, etc.)
        # This gives us both types of features without leakage
        all_features = self.transform(combined)

        # Return ONLY the last row (current event's features)
        return all_features[-1:, :]

    def verify_normalization(self, features: np.ndarray) -> dict:
        """
        Verify that amplitude normalization is correct.

        This is a diagnostic method to check that amplitudes are properly
        normalized (sum of squared amplitudes = 1 for each row).

        Args:
            features: Output from transform(), shape (n_samples, n_features)

        Returns:
            Dictionary with normalization statistics:
            - "mean_norm": Average sum of squared amplitudes (should be ~1.0)
            - "max_deviation": Maximum deviation from 1.0
            - "all_normalized": True if all rows are normalized (within tolerance)

        Examples:
            >>> imputer = AmplitudeEmbedding()
            >>> features = imputer.fit_transform(df)
            >>> stats = imputer.verify_normalization(features)
            >>> print(stats["all_normalized"])  # Should be True
            True
        """
        # Extract amplitude features (first 39 columns)
        amplitudes = features[:, :39]

        # Compute sum of squared amplitudes for each row
        squared_sums = (amplitudes ** 2).sum(axis=1)

        # Statistics
        mean_norm = squared_sums.mean()
        max_deviation = np.abs(squared_sums - 1.0).max()
        all_normalized = np.allclose(squared_sums, 1.0, atol=1e-10)

        return {
            "mean_norm": float(mean_norm),
            "max_deviation": float(max_deviation),
            "all_normalized": all_normalized
        }

    def predict_with_imputation(
        self,
        historical_context: pd.DataFrame,
        ranker_model=None,
        num_predictions: int = 20
    ) -> list:
        """
        Predict top-N positions using ONLY historical data (NO LEAKAGE).

        This is the CORRECT prediction interface that prevents data leakage.
        Uses two-stage architecture:
        1. Impute probabilities from historical patterns (events 1 to N-1)
        2. Compute features from imputed probabilities (NOT actual QVs)
        3. Predict using ranker model (or fallback to probability-based ranking)

        CRITICAL: This method does NOT accept the current event's actual QV values.
        It generates predictions using ONLY historical context.

        Args:
            historical_context: Events 1 to N-1 (with actual QVs)
            ranker_model: Optional trained ranker model with predict_top_k_from_features() method
                         If None, falls back to probability-based ranking
            num_predictions: Number of top positions to return (default: 20)

        Returns:
            List of predicted positions (1-39)

        Examples:
            >>> # Simple probability-based prediction (no model)
            >>> imputer = AmplitudeEmbedding()
            >>> imputer.fit(train_df)
            >>> predictions = imputer.predict_with_imputation(historical_context)
            >>> print(predictions)  # [3, 7, 12, 15, ...]

            >>> # Model-based prediction (with trained LGBM ranker)
            >>> lgbm_model = joblib.load('models/lgbm_ranker.pkl')
            >>> predictions = imputer.predict_with_imputation(
            ...     historical_context, ranker_model=lgbm_model, num_predictions=20
            ... )
        """
        # Stage 1: Impute probabilities for next event from historical patterns
        imputed_probs = self._impute_next_event_probabilities(historical_context)

        # Stage 2: Compute FULL feature set from imputed probabilities
        features = self._compute_features_from_probabilities(imputed_probs, historical_context)

        # Stage 3: Predict using ranker model or fallback to probability-based ranking
        if ranker_model is not None:
            # Use trained ranker model
            # Convert features to DataFrame with proper column names
            feature_names = self.get_feature_names()
            features_df = pd.DataFrame([features], columns=feature_names)

            # Add dummy q_1 to q_5 columns (set to 0 - NO LEAKAGE!)
            # These columns are expected by the ranker model but contain no information
            # during prediction (they were used during training to identify winning positions)
            for i in range(5):
                features_df[f'q_{i+1}'] = 0

            # Call ranker's predict_top_k_from_features method
            predictions = ranker_model.predict_top_k_from_features(features_df, k=num_predictions)

            # Return first row (only one event)
            return predictions[0].tolist()
        else:
            # Fallback: Return positions with highest imputed probabilities
            top_positions = np.argsort(imputed_probs)[::-1][:num_predictions] + 1  # +1 for 1-indexed
            return top_positions.tolist()

    def _impute_next_event_probabilities(self, historical_context: pd.DataFrame) -> np.ndarray:
        """
        Impute probabilities for next event using historical frequencies.

        Computes each position's probability from historical activation rates,
        optionally with temporal weighting (Story 13.1) to give more weight
        to recent events.

        This is Stage 1 of the two-stage prediction architecture.

        Temporal Weighting (Story 13.1):
        ---------------------------------
        When temporal_decay_rate > 0, applies exponential decay weights where
        more recent events get higher weights:

            weight[i] = exp(decay_rate * i) / sum(exp(decay_rate * j))

        where i is the event index (0=oldest, lookback-1=newest).

        Example with decay_rate=0.05 and lookback=50:
        - Event -50 (oldest): weight ≈ 0.0082 (0.82%)
        - Event -25 (middle): weight ≈ 0.0152 (1.52%)
        - Event -1  (newest): weight ≈ 0.0282 (2.82%)

        Newest event gets ~3.4x more weight than oldest event.

        Args:
            historical_context: Historical events (with actual QVs)

        Returns:
            Array of 39 probabilities (one per position), normalized to sum to 5.0

        Examples:
            >>> # Without temporal weighting (all events weighted equally)
            >>> imputer = AmplitudeEmbedding(temporal_decay_rate=0.0)
            >>> probs = imputer._impute_next_event_probabilities(historical_df)
            >>> print(probs.shape)  # (39,)
            >>> print(probs.sum())  # ~5.0

            >>> # With temporal weighting (recent events weighted more)
            >>> imputer = AmplitudeEmbedding(temporal_decay_rate=0.05)
            >>> probs = imputer._impute_next_event_probabilities(historical_df)
            >>> print(probs.sum())  # ~5.0 (still normalized)
        """
        if len(historical_context) == 0:
            # No history: uniform distribution
            # 5 expected winners / 39 positions
            return np.full(39, 5.0 / 39.0)

        # Use last N events (or all if less than N)
        lookback = min(len(historical_context), self.lookback_window)
        recent_events = historical_context.iloc[-lookback:]

        # Compute activation rate for each position
        qv_cols = [f'QV_{i}' for i in range(1, 40)]
        qv_data = recent_events[qv_cols].values

        # Story 13.1: Apply temporal weighting if enabled
        if self.temporal_decay_rate > 0:
            # Create exponential decay weights (recent events get higher weight)
            # indices: [0, 1, 2, ..., lookback-1] where 0=oldest, lookback-1=newest
            indices = np.arange(lookback)
            weights = np.exp(self.temporal_decay_rate * indices)
            weights = weights / weights.sum()  # Normalize to sum to 1.0

            # Compute weighted activation rates
            activation_rates = np.average(qv_data, axis=0, weights=weights)
        else:
            # No temporal weighting: simple mean (all events equally weighted)
            activation_rates = np.mean(qv_data, axis=0)

        # Normalize to sum to 5.0 (expected 5 winners per event)
        if np.sum(activation_rates) > 0:
            probabilities = activation_rates * (5.0 / np.sum(activation_rates))
        else:
            # Edge case: no activations in recent history, use uniform
            probabilities = np.full(39, 5.0 / 39.0)

        return probabilities

    def _compute_features_from_probabilities(self, imputed_probs: np.ndarray,
                                            historical_context: pd.DataFrame) -> np.ndarray:
        """
        Compute FULL feature set from imputed probabilities (NOT actual QVs).

        This is Stage 2 of the two-stage prediction architecture.

        CRITICAL: This method computes features from IMPUTED probabilities,
        not from actual QV values. This prevents data leakage.

        The method creates a synthetic event from imputed probabilities and
        computes ALL features (amplitude, interference, entanglement, temporal)
        by combining with historical context and calling transform().

        Args:
            imputed_probs: Array of 39 imputed probabilities
            historical_context: Historical events for temporal features

        Returns:
            Feature array compatible with transform() output (full feature set)

        Examples:
            >>> imputer = AmplitudeEmbedding()
            >>> imputer.fit(train_df)
            >>> probs = imputer._impute_next_event_probabilities(historical_df)
            >>> features = imputer._compute_features_from_probabilities(probs, historical_df)
            >>> print(features.shape)  # (n_features,) - full feature set
        """
        # Create a synthetic event from imputed probabilities
        # We'll use the probabilities directly as continuous QV values
        # (instead of binary 0/1, we have continuous probabilities)

        # Create a DataFrame row with imputed probabilities as QV columns
        synthetic_event = {}
        for i in range(1, 40):
            synthetic_event[f'QV_{i}'] = imputed_probs[i-1]

        # Add a dummy event-ID (not used in feature computation)
        if len(historical_context) > 0:
            last_event_id = historical_context['event-ID'].iloc[-1]
            synthetic_event['event-ID'] = last_event_id + 1
        else:
            synthetic_event['event-ID'] = 1

        # Convert to Series
        synthetic_event_series = pd.Series(synthetic_event)

        # Combine historical context with synthetic event
        if len(historical_context) == 0:
            combined = pd.DataFrame([synthetic_event_series])
        else:
            combined = pd.concat([historical_context, pd.DataFrame([synthetic_event_series])], ignore_index=True)

        # Transform to get FULL feature set
        # The transform() method will compute:
        # - Amplitude features from the synthetic event's continuous QVs
        # - Interference features from the amplitudes
        # - Entanglement features (entropy, purity, etc.)
        # - Temporal features from historical context only
        all_features = self._transform(combined)

        # Return ONLY the last row (synthetic event's features)
        return all_features[-1, :]
