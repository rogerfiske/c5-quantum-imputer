"""
Graph/Cycle Encoding Imputation Strategy

This module implements the Graph/Cycle Encoding quantum-inspired imputation method,
which treats the 39 quantum positions as nodes in a cyclic graph (C₃₉ ring).
Uses Discrete Fourier Transform (DFT) to capture periodic patterns and harmonic
structure in the cyclic group, combined with graph-based features.

The quantum-inspired concept:
In quantum mechanics and group theory, cyclic groups appear naturally in periodic
systems. The C₃₉ cyclic group represents 39 positions arranged in a ring where
position i is adjacent to positions (i-1) mod 39 and (i+1) mod 39.

The DFT (implemented via FFT) decomposes the binary quantum state into frequency
components (harmonics), revealing periodic patterns that may not be obvious in
the spatial domain. This is analogous to how quantum systems can be analyzed
in momentum space (via Fourier transform of position space).

Author: BMad Dev Agent (James)
Date: 2025-10-13
Story: Epic 2, Story 2.6 - Graph/Cycle Encoding Strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Literal

from src.imputation.base_imputer import BaseImputer
from src.imputation.proximity_features import ProximityFeatures

# Set up logging
logger = logging.getLogger(__name__)


class GraphCycleEncoding(BaseImputer):
    """
    Graph/Cycle Encoding imputation strategy.

    Treats the 39 quantum positions as nodes in a cyclic graph (C₃₉ ring) and
    uses Discrete Fourier Transform (DFT) combined with graph-based features
    to capture periodic patterns and structural relationships.

    Mathematical Background:
    ------------------------
    **Cyclic Graph (C₃₉)**:
    A cyclic graph with 39 nodes where each node i is connected to:
    - Previous node: (i-1) mod 39
    - Next node: (i+1) mod 39

    This creates a ring topology where position 1 is adjacent to both position 2
    and position 39. Distances between positions are measured as the shortest
    path on the ring (can go either direction).

    **Discrete Fourier Transform (DFT)**:
    The DFT transforms the binary vector from spatial domain to frequency domain:
        F[k] = Σₙ x[n] * e^(-2πikn/N)

    For our binary vectors:
    - Input: Binary vector of length 39 (exactly 5 ones)
    - Output: Complex-valued frequency spectrum of length 39
    - k=0: DC component (average value, always 5/39 for our data)
    - k=1,2,...: Harmonics capturing periodic patterns

    We extract:
    - Magnitude spectrum: |F[k]| = sqrt(Re[F[k]]² + Im[F[k]]²)
    - Phase spectrum: angle(F[k]) = atan2(Im[F[k]], Re[F[k]])

    **Parseval's Theorem**:
    Energy conservation: Σₙ |x[n]|² = (1/N) Σₖ |F[k]|²
    For our binary vectors: left side = 5 (number of ones)

    Feature Engineering Strategy:
    ------------------------------
    This implementation creates three types of features:

    1. **DFT Magnitude Features** (n_harmonics features, default 20):
       The magnitude |F[k]| of the k-th harmonic. Captures the strength of
       periodic patterns at different frequencies. Lower harmonics (k=1,2,3)
       capture global patterns, higher harmonics capture local variations.

    2. **DFT Phase Features** (n_harmonics features, default 20):
       The phase angle(F[k]) of the k-th harmonic. Captures the alignment
       or shift of periodic patterns. Phases are in range [-π, π].

    3. **Graph Features** (~15-20 features):
       - Circular distances between active positions
       - Distance statistics (min, max, mean, std)
       - Clustering coefficient (how clustered active positions are)
       - Ring spread (angular span of active positions)
       - Symmetry measures

    Cyclic Group Properties:
    -------------------------
    The C₃₉ cyclic group has several interesting properties:
    - Order: 39 = 3 × 13 (not prime, has divisors)
    - Generators: Elements that can generate entire group
    - Subgroups: C₃ (every 13th position) and C₁₃ (every 3rd position)
    - DFT basis functions are the irreducible representations of C₃₉

    Parameters:
    -----------
    name : str, optional (default="graph_cycle_encoding")
        Human-readable name for this imputation strategy

    include_dft_features : bool, optional (default=True)
        Whether to include DFT magnitude and phase features.

    include_graph_features : bool, optional (default=True)
        Whether to include graph-based distance and clustering features.

    n_harmonics : int, optional (default=20)
        Number of DFT harmonics to keep (from 0 to n_harmonics-1).
        Must be in range [1, 39]. Lower values reduce dimensionality,
        higher values capture more detail.

    Output Dimensions:
    ------------------
    - With all features (default): (n_samples, 2*n_harmonics + ~17)
      = 40 (DFT magnitude) + 40 (DFT phase) + 17 (graph) = 57 features (n_harmonics=20)
    - Without DFT features: (n_samples, ~17) graph features only
    - Without graph features: (n_samples, 2*n_harmonics) DFT features only
    - Actual dimension depends on n_harmonics parameter

    Attributes:
    -----------
    n_harmonics : int
        Number of DFT harmonics to extract

    include_dft_features : bool
        Whether DFT features are included

    include_graph_features : bool
        Whether graph features are included

    Examples:
    ---------
    >>> from src.data_loader import load_dataset
    >>> from src.imputation.graph_cycle_encoding import GraphCycleEncoding
    >>>
    >>> # Load dataset
    >>> df = load_dataset()
    >>>
    >>> # Full graph/cycle encoding (default)
    >>> imputer = GraphCycleEncoding()
    >>> features = imputer.fit_transform(df)
    >>> print(features.shape)  # (11581, 57) with n_harmonics=20
    >>>
    >>> # More harmonics for higher resolution
    >>> imputer_full = GraphCycleEncoding(n_harmonics=39)
    >>> features_full = imputer_full.fit_transform(df)
    >>> print(features_full.shape)  # (11581, 95) = 78 DFT + 17 graph
    >>>
    >>> # Only DFT features
    >>> imputer_dft = GraphCycleEncoding(include_graph_features=False)
    >>> features_dft = imputer_dft.fit_transform(df)
    >>> print(features_dft.shape)  # (11581, 40) = 2*20 harmonics
    >>>
    >>> # Only graph features
    >>> imputer_graph = GraphCycleEncoding(include_dft_features=False)
    >>> features_graph = imputer_graph.fit_transform(df)
    >>> print(features_graph.shape)  # (11581, 17)

    Notes:
    ------
    - FFT is O(N log N) vs O(N²) for naive DFT, very efficient
    - DFT features are translation-invariant on the ring (rotation-invariant)
    - Low harmonics capture global structure, high harmonics capture fine details
    - Graph features complement DFT by providing explicit distance information
    - No learning from training data required (all features deterministic)
    """

    def __init__(
        self,
        name: str = "graph_cycle_encoding",
        include_dft_features: bool = True,
        include_graph_features: bool = True,
        n_harmonics: int = 20,
        include_proximity_features: bool = True
    ):
        """
        Initialize Graph/Cycle Encoding imputer.

        Args:
            name: Human-readable name for this imputation strategy
            include_dft_features: Whether to include DFT magnitude/phase features
            include_graph_features: Whether to include graph distance/clustering features
            n_harmonics: Number of DFT harmonics to extract (1-39)
            include_proximity_features: Whether to include proximity-based quantum features
                                       (adds 195 features, increases total from 82 to 277)

        Raises:
            ValueError: If n_harmonics not in range [1, 39]

        Examples:
            >>> # Default: 20 harmonics, all features including proximity
            >>> imputer1 = GraphCycleEncoding()
            >>>
            >>> # Full DFT resolution
            >>> imputer2 = GraphCycleEncoding(n_harmonics=39)
            >>>
            >>> # Only graph features
            >>> imputer3 = GraphCycleEncoding(include_dft_features=False)
            >>>
            >>> # Without proximity features (original 82 features)
            >>> imputer4 = GraphCycleEncoding(include_proximity_features=False)
        """
        # Validate n_harmonics parameter
        if not (1 <= n_harmonics <= 39):
            raise ValueError(
                f"n_harmonics must be in range [1, 39], got {n_harmonics}"
            )

        # Initialize parent class with config
        super().__init__(
            name=name,
            config={
                "include_dft_features": include_dft_features,
                "include_graph_features": include_graph_features,
                "n_harmonics": n_harmonics,
                "include_proximity_features": include_proximity_features
            }
        )

        # Store parameters as instance attributes
        self.include_dft_features = include_dft_features
        self.include_graph_features = include_graph_features
        self.n_harmonics = n_harmonics
        self.include_proximity_features = include_proximity_features

        # Proximity feature calculator (initialized if needed)
        if self.include_proximity_features:
            self.proximity_calculator = ProximityFeatures(N=39)

        logger.debug(
            f"Initialized GraphCycleEncoding with "
            f"include_dft_features={include_dft_features}, "
            f"include_graph_features={include_graph_features}, "
            f"n_harmonics={n_harmonics}"
        )

    def _fit(self, X: pd.DataFrame) -> None:
        """
        Learn statistics from training data (optional for graph/cycle encoding).

        Graph/Cycle encoding uses fixed DFT basis functions and geometric
        distances on the C₃₉ ring, so no learning from training data is required.
        However, we implement this method to satisfy the BaseImputer interface.

        Future extensions could learn:
        - Optimal number of harmonics based on energy distribution
        - Position-specific weights for graph features
        - Clustering patterns across training set

        Args:
            X: Training DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Side Effects:
            None (no parameters learned in base implementation)

        Notes:
            - This method is called by the public fit() method
            - Satisfies the abstract method requirement
            - Could be extended with learning in future versions
        """
        # No learning required for basic graph/cycle encoding
        # DFT basis functions are fixed, graph distances are geometric
        logger.info(
            f"Graph/Cycle encoding uses fixed DFT basis and geometric distances "
            f"(no learning from {len(X)} training samples)"
        )

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform binary quantum states to graph/cycle encoding features.

        This method creates feature representations by:
        1. Applying FFT to extract DFT magnitude and phase features
        2. Computing graph-based features on the C₃₉ ring
        3. Combining both feature types

        Args:
            X: Input DataFrame with columns [event-ID, QV_1, ..., QV_39]
               Shape: (n_samples, 40)

        Returns:
            Feature matrix of shape (n_samples, n_features) where n_features depends
            on configuration:
            - Default (n_harmonics=20): 57 features (40 DFT + 17 graph)
            - All harmonics (n_harmonics=39): 95 features (78 DFT + 17 graph)
            Data type: np.float64

        Algorithm:
            For each sample:
            1. Extract binary QV vector (39 dims)
            2. Apply FFT: F = np.fft.fft(qv_vector)
            3. Extract magnitude and phase of first n_harmonics:
               - magnitude[k] = |F[k]|
               - phase[k] = angle(F[k])
            4. Compute graph features:
               - Find active position indices
               - Compute circular distances between all pairs
               - Calculate distance statistics
               - Compute clustering and spread measures
            5. Concatenate DFT and graph features

        Mathematical Details:
            DFT via FFT:
                F[k] = Σₙ x[n] * e^(-2πikn/39)
                magnitude[k] = sqrt(Re[F[k]]² + Im[F[k]]²)
                phase[k] = atan2(Im[F[k]], Re[F[k]])

            Circular distance on C₃₉ ring:
                For positions i and j:
                d_forward = (j - i) mod 39
                d_backward = (i - j) mod 39
                circular_distance = min(d_forward, d_backward)

            Clustering coefficient (simplified):
                Measures how close active positions are relative to random
                distribution on the ring.

        Notes:
            - FFT is O(N log N), very efficient
            - Output validation (NaN/Inf) performed by parent class
            - Each sample is processed independently
        """
        # Extract QV columns
        qv_columns = [f'QV_{i}' for i in range(1, 40)]
        qv_data = X[qv_columns].values  # Shape: (n_samples, 39)

        # Start building feature list
        features_list = []

        # 1. DFT Features (magnitude and phase)
        if self.include_dft_features:
            # Apply FFT to each row
            # np.fft.fft expects input along last axis
            fft_result = np.fft.fft(qv_data, axis=1)  # Shape: (n_samples, 39), complex

            # Extract first n_harmonics
            fft_harmonics = fft_result[:, :self.n_harmonics]  # Shape: (n_samples, n_harmonics)

            # Magnitude spectrum
            magnitude = np.abs(fft_harmonics)  # Shape: (n_samples, n_harmonics)
            features_list.append(magnitude)

            # Phase spectrum
            phase = np.angle(fft_harmonics)  # Shape: (n_samples, n_harmonics)
            features_list.append(phase)

        # 2. Graph Features
        if self.include_graph_features:
            graph_features = self._compute_graph_features(qv_data)
            features_list.append(graph_features)

        # ========================================================================
        # SOPHISTICATED QUANTUM FEATURES (Added 2025-10-21)
        # ========================================================================
        # These features capture phase relationships, harmonic interactions,
        # and non-linear DFT properties beyond simple magnitude/phase extraction
        # ========================================================================

        if self.include_dft_features:
            # --- G. DFT PHASE RELATIONSHIPS (Relative phases, not absolute) ---
            # Phase differences capture periodic structure relationships

            # G1. Phase differences between consecutive harmonics (n_harmonics-1 features)
            # Δφₖ = φₖ - φₖ₋₁ captures how phase evolves across frequency spectrum
            phase_diffs = np.diff(phase, axis=1)  # Shape: (n_samples, n_harmonics-1)
            features_list.append(phase_diffs)

            # --- H. HARMONIC INTERACTIONS (Non-linear DFT features) ---
            # Cross-products of DFT modes capture non-linear periodic relationships

            # H1. Key harmonic interactions (C₃₉, C₁₃, C₃ subgroups)
            # These harmonics correspond to important cyclic symmetries
            key_harmonics = [1, 3, 13]  # Fundamental, C₁₃ period, C₃ period

            harmonic_interactions = []
            for k1_idx, k1 in enumerate(key_harmonics):
                for k2_idx, k2 in enumerate(key_harmonics):
                    if k1 < k2 and k1 < self.n_harmonics and k2 < self.n_harmonics:
                        # Interaction: |F[k1] * conj(F[k2])|
                        # Uses fft_result from earlier computation
                        fft_result = np.fft.fft(qv_data, axis=1)
                        interaction = np.abs(fft_result[:, k1] * np.conj(fft_result[:, k2]))
                        harmonic_interactions.append(interaction)

            if harmonic_interactions:
                harmonic_matrix = np.column_stack(harmonic_interactions)
                features_list.append(harmonic_matrix)

            # H2. Energy distribution across frequency bands
            # Total energy in low frequencies (k=1..5) vs high frequencies (k>5)
            fft_full = np.fft.fft(qv_data, axis=1)
            energy_low = np.sum(np.abs(fft_full[:, 1:6])**2, axis=1, keepdims=True)
            energy_high = np.sum(np.abs(fft_full[:, 6:self.n_harmonics])**2, axis=1, keepdims=True) if self.n_harmonics > 6 else np.zeros((qv_data.shape[0], 1))
            energy_ratio = energy_low / (energy_high + 1e-10)  # Avoid division by zero
            features_list.append(energy_low)
            features_list.append(energy_high)
            features_list.append(energy_ratio)

        # ========================================================================
        # END SOPHISTICATED QUANTUM FEATURES
        # Feature count added: (n_harmonics-1) phase diffs + 3 harmonic interactions + 3 energy = ~22
        # Total features: 57 (original) + 22 (new) = 79
        # NOTE: Actual total is 82 (not 79) based on production data
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
        # Total features: 82 (original) + 195 (proximity) = 277
        # ========================================================================

        # Concatenate all features horizontally
        result = np.hstack(features_list)

        logger.debug(
            f"Transformed {len(X)} samples to shape {result.shape}. "
            f"Features: DFT={self.include_dft_features} (n_harmonics={self.n_harmonics}), "
            f"graph={self.include_graph_features}"
        )

        return result

    def _compute_graph_features(self, qv_data: np.ndarray) -> np.ndarray:
        """
        Compute graph-based features for C₃₉ ring structure.

        Args:
            qv_data: Binary matrix of shape (n_samples, 39)

        Returns:
            Graph features of shape (n_samples, 17)

        Graph Features Computed:
        ------------------------
        1. Distance statistics (5 features):
           - min_circular_distance: Minimum distance between any two active positions
           - max_circular_distance: Maximum distance between any two active positions
           - mean_circular_distance: Average distance between all active pairs
           - std_circular_distance: Standard deviation of distances
           - median_circular_distance: Median distance

        2. Clustering measures (6 features):
           - clustering_coefficient: How clustered active positions are (0-1)
           - angular_span: Total angular span of active positions (0-39)
           - consecutive_pairs: Number of active positions that are adjacent
           - max_gap: Largest gap between consecutive active positions
           - min_gap: Smallest gap between consecutive active positions
           - gap_variance: Variance of gaps between active positions

        3. Symmetry measures (6 features):
           - centroid_position: Center of mass of active positions on ring
           - symmetry_score: How symmetric active positions are around centroid
           - reflection_symmetry: Measure of reflection symmetry
           - rotational_order_3: Alignment with C₃ subgroup (every 13 positions)
           - rotational_order_13: Alignment with C₁₃ subgroup (every 3 positions)
           - position_spread: Spread of positions relative to uniform distribution
        """
        n_samples = qv_data.shape[0]
        n_positions = 39
        n_active = 5  # Always 5 active positions

        # Initialize feature matrix
        graph_features = np.zeros((n_samples, 17), dtype=np.float64)

        for i in range(n_samples):
            # Get active position indices (0-indexed)
            active_positions = np.where(qv_data[i] == 1)[0]

            # 1. Compute all pairwise circular distances
            distances = []
            for j in range(n_active):
                for k in range(j + 1, n_active):
                    pos_j = active_positions[j]
                    pos_k = active_positions[k]
                    # Circular distance (shortest path on ring)
                    d_forward = (pos_k - pos_j) % n_positions
                    d_backward = (pos_j - pos_k) % n_positions
                    circular_dist = min(d_forward, d_backward)
                    distances.append(circular_dist)

            distances = np.array(distances)

            # Distance statistics (5 features)
            graph_features[i, 0] = distances.min()
            graph_features[i, 1] = distances.max()
            graph_features[i, 2] = distances.mean()
            graph_features[i, 3] = distances.std()
            graph_features[i, 4] = np.median(distances)

            # 2. Clustering measures
            # Sort active positions for gap analysis
            sorted_positions = np.sort(active_positions)

            # Angular span: distance from first to last active position
            angular_span_forward = (sorted_positions[-1] - sorted_positions[0]) % n_positions
            angular_span_backward = (sorted_positions[0] - sorted_positions[-1]) % n_positions
            angular_span = min(angular_span_forward, angular_span_backward)
            graph_features[i, 5] = angular_span

            # Consecutive pairs: count adjacent active positions
            consecutive_pairs = 0
            for j in range(n_active):
                pos_current = sorted_positions[j]
                pos_next = sorted_positions[(j + 1) % n_active]
                gap = (pos_next - pos_current) % n_positions
                if gap == 1:
                    consecutive_pairs += 1
            graph_features[i, 6] = consecutive_pairs

            # Gaps between consecutive active positions
            gaps = []
            for j in range(n_active):
                pos_current = sorted_positions[j]
                pos_next = sorted_positions[(j + 1) % n_active]
                gap = (pos_next - pos_current) % n_positions
                gaps.append(gap)
            gaps = np.array(gaps)

            graph_features[i, 7] = gaps.max()  # max_gap
            graph_features[i, 8] = gaps.min()  # min_gap
            graph_features[i, 9] = gaps.var()  # gap_variance

            # Clustering coefficient (0-1, higher = more clustered)
            # Based on ratio of angular span to full circle
            clustering_coef = 1.0 - (angular_span / (n_positions / 2))
            graph_features[i, 10] = max(0.0, clustering_coef)

            # 3. Symmetry measures
            # Centroid: weighted average position on ring
            # Treat positions as angles: θᵢ = 2π * i / 39
            angles = 2 * np.pi * active_positions / n_positions
            centroid_x = np.cos(angles).mean()
            centroid_y = np.sin(angles).mean()
            centroid_angle = np.arctan2(centroid_y, centroid_x)
            # Convert back to position index
            centroid_position = (centroid_angle / (2 * np.pi) * n_positions) % n_positions
            graph_features[i, 11] = centroid_position

            # Symmetry score: variance of distances from centroid
            # Lower variance = more symmetric around centroid
            centroid_distances = []
            for pos in active_positions:
                # Circular distance from centroid
                d_forward = (pos - centroid_position) % n_positions
                d_backward = (centroid_position - pos) % n_positions
                d = min(d_forward, d_backward)
                centroid_distances.append(d)
            symmetry_score = 1.0 / (1.0 + np.var(centroid_distances))
            graph_features[i, 12] = symmetry_score

            # Reflection symmetry: check if positions are symmetric around centroid
            # Compute reflection of each position and check if it's also active
            reflection_matches = 0
            for pos in active_positions:
                reflected_pos = (2 * centroid_position - pos) % n_positions
                # Find nearest active position to reflected position
                min_dist = min(abs(reflected_pos - ap) for ap in active_positions)
                if min_dist < 1.0:  # Exact match
                    reflection_matches += 1
            reflection_symmetry = reflection_matches / n_active
            graph_features[i, 13] = reflection_symmetry

            # Rotational alignment with subgroups
            # C₃ subgroup: positions 0, 13, 26 (every 13)
            c3_positions = np.array([0, 13, 26])
            c3_alignment = 0
            for pos in active_positions:
                min_dist = min(min((pos - cp) % n_positions, (cp - pos) % n_positions) for cp in c3_positions)
                if min_dist <= 2:  # Within 2 positions of C₃ element
                    c3_alignment += 1
            graph_features[i, 14] = c3_alignment / n_active

            # C₁₃ subgroup: every 3rd position (0, 3, 6, ..., 36)
            c13_alignment = sum(1 for pos in active_positions if pos % 3 == 0) / n_active
            graph_features[i, 15] = c13_alignment

            # Position spread: how spread out positions are
            # Use resultant vector length (similar to angle encoding)
            resultant_length = np.sqrt(centroid_x**2 + centroid_y**2)
            # Normalize by n_active (max length is n_active)
            position_spread = 1.0 - (resultant_length / n_active)
            graph_features[i, 16] = position_spread

        return graph_features

    def get_feature_names(self) -> list:
        """
        Get human-readable names for output features.

        Returns:
            List of feature names, length depends on configuration

        Examples:
            >>> imputer = GraphCycleEncoding(n_harmonics=20)
            >>> imputer.fit(df)
            >>> names = imputer.get_feature_names()
            >>> print(len(names))  # 57
            >>> print(names[:5])
            ['dft_magnitude_0', 'dft_magnitude_1', ..., 'dft_magnitude_4']
            >>> print(names[20:25])
            ['dft_phase_0', 'dft_phase_1', ..., 'dft_phase_4']
            >>> print(names[-5:])
            ['rotational_order_3', 'rotational_order_13', ...]
        """
        feature_names = []

        # DFT feature names
        if self.include_dft_features:
            magnitude_names = [f'dft_magnitude_{k}' for k in range(self.n_harmonics)]
            phase_names = [f'dft_phase_{k}' for k in range(self.n_harmonics)]
            feature_names.extend(magnitude_names)
            feature_names.extend(phase_names)

        # Graph feature names
        if self.include_graph_features:
            graph_names = [
                'min_circular_distance',
                'max_circular_distance',
                'mean_circular_distance',
                'std_circular_distance',
                'median_circular_distance',
                'angular_span',
                'consecutive_pairs',
                'max_gap',
                'min_gap',
                'gap_variance',
                'clustering_coefficient',
                'centroid_position',
                'symmetry_score',
                'reflection_symmetry',
                'rotational_order_3',
                'rotational_order_13',
                'position_spread'
            ]
            feature_names.extend(graph_names)

        # Sophisticated quantum features (added 2025-10-21)
        if self.include_dft_features:
            # Phase differences (n_harmonics-1 features)
            phase_diff_names = [f'phase_diff_{k}' for k in range(1, self.n_harmonics)]
            feature_names.extend(phase_diff_names)

            # Harmonic interactions (3 features for key harmonics 1,3,13)
            harmonic_interactions = [
                'harmonic_interact_1_3',
                'harmonic_interact_1_13',
                'harmonic_interact_3_13'
            ]
            feature_names.extend(harmonic_interactions)

            # Energy distribution (3 features)
            energy_names = [
                'energy_low_freq',
                'energy_high_freq',
                'energy_ratio'
            ]
            feature_names.extend(energy_names)

        return feature_names
