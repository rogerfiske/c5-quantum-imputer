"""
Range-Aware Bias Correction for Quantum Event Predictions

This module implements bias correction to reduce systematic over-prediction
of positions across LOW/MID/HIGH ranges in the C₃₉ cyclic group structure.

Background:
Epic 9A discovered that all imputation methods showed ~300% over-prediction
bias across all position ranges. This means models predict positions appearing
in predictions 3x more frequently than they actually occur in the dataset.

This bias correction applies range-specific downward adjustments to prediction
scores to bring the predicted distribution closer to the actual distribution
(approximately 33% per range: LOW/MID/HIGH).

Author: BMad Dev Agent (James)
Date: 2025-10-20
Epic: Epic 9B - Ensemble & Bias Correction
Story: Story 9B.2 - Range-Aware Bias Correction
NFR2: Non-programmer friendly - heavily commented with business rationale
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class RangeAwareBiasCorrection:
    """
    Applies range-aware bias correction to reduce systematic over-prediction.

    The C₃₉ cyclic group (39 quantum positions) is divided into three ranges:
    - LOW: positions 1-13
    - MID: positions 14-26
    - HIGH: positions 27-39

    Epic 9A analysis showed all models over-predict positions by ~300% in each
    range. This correction applies range-specific factors to reduce this bias.

    Correction Formula:
    ------------------
    Conservative (50% correction):
        correction_factor = 1 / (1 + bias_percent/100 * 0.5)
        corrected_score = original_score * correction_factor

    Aggressive (100% correction):
        correction_factor = 1 / (1 + bias_percent/100)
        corrected_score = original_score * correction_factor

    Why Conservative First?
    -----------------------
    - Reduces risk of overcorrection (hurting recall)
    - Epic 9B handoff specifies conservative as default
    - Can test aggressive if conservative insufficient

    Example:
    --------
    For LOW range with 297.5% bias:
    - Conservative: 1 / (1 + 2.975 * 0.5) = 1 / 2.4875 = 0.4020
    - Aggressive: 1 / (1 + 2.975) = 1 / 3.975 = 0.2516

    If a position in LOW range has score 0.8:
    - Conservative corrected: 0.8 * 0.4020 = 0.3216
    - Aggressive corrected: 0.8 * 0.2516 = 0.2013

    NFR2 Note:
    ----------
    This correction is necessary because models learned patterns from training
    data that may not generalize well. The bias suggests models are "too
    confident" in predicting positions, appearing in top-20 predictions far
    more often than those positions actually win. By downweighting scores in
    over-predicted ranges, we aim to produce more balanced, realistic predictions.
    """

    def __init__(self, bias_factors_file: str, mode: str = "conservative"):
        """
        Initialize bias correction with range-specific correction factors.

        Args:
            bias_factors_file: Path to CSV with columns:
                - range: low/mid/high
                - range_bounds: position ranges (e.g., "1-13")
                - actual_count: how many times positions in range appeared in holdout
                - predicted_count: how many times models predicted those positions
                - bias_pct: over-prediction percentage
                - correction_factor_conservative: 50% correction factor
                - correction_factor_aggressive: 100% correction factor

            mode: "conservative" (default) or "aggressive"
                - Conservative applies 50% of full correction (recommended start)
                - Aggressive applies 100% correction (use if conservative insufficient)

        Raises:
            FileNotFoundError: If bias_factors_file doesn't exist
            ValueError: If required columns missing or mode invalid
        """
        if mode not in ["conservative", "aggressive"]:
            raise ValueError(f"mode must be 'conservative' or 'aggressive', got: {mode}")

        self.mode = mode
        self.bias_factors_file = Path(bias_factors_file)

        if not self.bias_factors_file.exists():
            raise FileNotFoundError(f"Bias factors file not found: {bias_factors_file}")

        # Load bias factors from CSV
        self.bias_factors_df = pd.read_csv(self.bias_factors_file)

        # Validate required columns
        required_cols = [
            "range", "range_bounds", "actual_count", "predicted_count",
            "bias_pct", "correction_factor_conservative", "correction_factor_aggressive"
        ]
        missing_cols = set(required_cols) - set(self.bias_factors_df.columns)
        if missing_cols:
            raise ValueError(f"Bias factors file missing columns: {missing_cols}")

        # Create range -> correction factor mapping
        # Example: {"low": 0.4020, "mid": 0.4015, "high": 0.4022}
        factor_col = f"correction_factor_{mode}"
        self.range_factors = {}
        for _, row in self.bias_factors_df.iterrows():
            range_name = row["range"].lower()
            self.range_factors[range_name] = float(row[factor_col])

        # Define position ranges (C₃₉ structure)
        # Positions are 1-indexed (1-39)
        self.position_to_range = {}
        self.position_to_range.update({pos: "low" for pos in range(1, 14)})   # 1-13
        self.position_to_range.update({pos: "mid" for pos in range(14, 27)})  # 14-26
        self.position_to_range.update({pos: "high" for pos in range(27, 40)}) # 27-39

        # Store metadata for reporting
        self.metadata = {
            "mode": mode,
            "bias_factors_file": str(self.bias_factors_file),
            "range_factors": self.range_factors.copy(),
            "position_ranges": {
                "low": "1-13",
                "mid": "14-26",
                "high": "27-39"
            }
        }

    def correct_predictions(
        self,
        positions: List[int],
        scores: np.ndarray,
        k: int = 20
    ) -> Tuple[List[int], np.ndarray, Dict]:
        """
        Apply bias correction to prediction scores and re-rank.

        Algorithm:
        ----------
        1. For each position (1-39), determine its range (LOW/MID/HIGH)
        2. Apply range-specific correction factor to its score
        3. Re-rank positions by corrected scores
        4. Return top-k positions and correction metrics

        Args:
            positions: Original top-k positions (1-indexed, ranked by score)
            scores: Scores for all 39 positions (length 39, 0-indexed array)
            k: Number of top positions to return (default 20)

        Returns:
            corrected_positions: Top-k positions after correction (1-indexed)
            corrected_scores: Adjusted scores for all 39 positions
            correction_metrics: Dict with:
                - positions_moved: How many positions changed rank
                - range_distribution_before: Count of LOW/MID/HIGH in top-k before
                - range_distribution_after: Count of LOW/MID/HIGH in top-k after
                - avg_score_reduction_by_range: Average score reduction per range
                - positions_entered_top_k: Positions that entered top-k after correction
                - positions_exited_top_k: Positions that exited top-k after correction

        NFR2 Explanation:
        -----------------
        Why we do this: Models tend to predict certain positions too frequently
        (over-prediction bias). By reducing scores of over-predicted positions,
        we make predictions more balanced and potentially more realistic.

        Example: If position 5 (in LOW range with 297.5% bias) has score 0.8,
        it gets reduced to 0.8 * 0.4020 = 0.3216 (conservative mode). This makes
        it less likely to appear in top-20, giving other positions a chance.
        """
        # Validate inputs
        if len(scores) != 39:
            raise ValueError(f"scores must have length 39 (one per position), got: {len(scores)}")

        if k < 1 or k > 39:
            raise ValueError(f"k must be between 1 and 39, got: {k}")

        # Store original scores for comparison
        original_scores = scores.copy()

        # Apply correction factor to each position
        corrected_scores = np.zeros(39, dtype=float)
        score_reduction_by_range = {"low": [], "mid": [], "high": []}

        for pos_idx in range(39):
            pos_1indexed = pos_idx + 1  # Convert to 1-indexed position
            range_name = self.position_to_range[pos_1indexed]
            correction_factor = self.range_factors[range_name]

            # Apply correction
            corrected_scores[pos_idx] = original_scores[pos_idx] * correction_factor

            # Track reduction for metrics
            reduction = original_scores[pos_idx] - corrected_scores[pos_idx]
            score_reduction_by_range[range_name].append(reduction)

        # Re-rank positions by corrected scores
        # argsort gives indices in ascending order, [::-1] reverses to descending
        corrected_ranking = np.argsort(corrected_scores)[::-1]

        # Get top-k positions (convert to 1-indexed)
        corrected_top_k_positions = [pos + 1 for pos in corrected_ranking[:k]]

        # Calculate correction metrics
        correction_metrics = self._calculate_correction_metrics(
            original_positions=positions,
            corrected_positions=corrected_top_k_positions,
            original_scores=original_scores,
            corrected_scores=corrected_scores,
            score_reduction_by_range=score_reduction_by_range,
            k=k
        )

        return corrected_top_k_positions, corrected_scores, correction_metrics

    def _calculate_correction_metrics(
        self,
        original_positions: List[int],
        corrected_positions: List[int],
        original_scores: np.ndarray,
        corrected_scores: np.ndarray,
        score_reduction_by_range: Dict[str, List[float]],
        k: int
    ) -> Dict:
        """
        Calculate metrics about how correction changed the ranking.

        This is for transparency and debugging - helps understand:
        - How much did rankings change?
        - Which ranges were affected most?
        - Did correction achieve desired range distribution?

        Returns dict with detailed before/after comparison.
        """
        # Count positions that moved in/out of top-k
        original_set = set(original_positions[:k])
        corrected_set = set(corrected_positions[:k])

        positions_entered = sorted(list(corrected_set - original_set))
        positions_exited = sorted(list(original_set - corrected_set))
        positions_moved = len(positions_entered)

        # Count range distribution before/after
        def count_range_distribution(positions_list):
            counts = {"low": 0, "mid": 0, "high": 0}
            for pos in positions_list:
                range_name = self.position_to_range[pos]
                counts[range_name] += 1
            return counts

        range_dist_before = count_range_distribution(original_positions[:k])
        range_dist_after = count_range_distribution(corrected_positions[:k])

        # Average score reduction per range
        avg_reduction_by_range = {
            range_name: np.mean(reductions) if reductions else 0.0
            for range_name, reductions in score_reduction_by_range.items()
        }

        # Calculate score statistics
        original_top_k_scores = [original_scores[pos - 1] for pos in original_positions[:k]]
        corrected_top_k_scores = [corrected_scores[pos - 1] for pos in corrected_positions[:k]]

        return {
            "positions_moved": positions_moved,
            "positions_entered_top_k": positions_entered,
            "positions_exited_top_k": positions_exited,
            "range_distribution_before": range_dist_before,
            "range_distribution_after": range_dist_after,
            "avg_score_reduction_by_range": avg_reduction_by_range,
            "original_top_k_score_mean": float(np.mean(original_top_k_scores)),
            "original_top_k_score_std": float(np.std(original_top_k_scores)),
            "corrected_top_k_score_mean": float(np.mean(corrected_top_k_scores)),
            "corrected_top_k_score_std": float(np.std(corrected_top_k_scores)),
            "mode": self.mode
        }

    def analyze_correction_impact(
        self,
        before_positions: List[int],
        after_positions: List[int]
    ) -> Dict:
        """
        Analyze how correction changed the ranking between two position lists.

        Useful for understanding correction effectiveness at a high level.
        For example, comparing ensemble predictions before vs after correction
        across multiple events.

        Args:
            before_positions: Top-k positions before correction (1-indexed)
            after_positions: Top-k positions after correction (1-indexed)

        Returns:
            Dict with:
                - overlap: Number of positions in both lists
                - jaccard_similarity: Overlap / union (0-1)
                - rank_correlation: Spearman correlation of ranks
                - positions_unique_to_before: Positions only in before list
                - positions_unique_to_after: Positions only in after list

        NFR2 Note:
        ----------
        This helps answer: "How much did bias correction change predictions?"
        - High overlap (>90%): Correction made small adjustments
        - Low overlap (<70%): Correction significantly reordered predictions
        """
        before_set = set(before_positions)
        after_set = set(after_positions)

        overlap = len(before_set & after_set)
        union = len(before_set | after_set)
        jaccard = overlap / union if union > 0 else 0.0

        # Calculate rank correlation for overlapping positions
        # (positions in both lists)
        common_positions = before_set & after_set
        if len(common_positions) > 1:
            # Get ranks for common positions
            before_ranks = {pos: rank for rank, pos in enumerate(before_positions, 1)}
            after_ranks = {pos: rank for rank, pos in enumerate(after_positions, 1)}

            common_list = sorted(list(common_positions))
            ranks_before = [before_ranks[pos] for pos in common_list]
            ranks_after = [after_ranks[pos] for pos in common_list]

            # Spearman correlation
            from scipy.stats import spearmanr
            rank_corr, _ = spearmanr(ranks_before, ranks_after)
        else:
            rank_corr = 0.0

        return {
            "overlap": overlap,
            "jaccard_similarity": float(jaccard),
            "rank_correlation": float(rank_corr),
            "positions_unique_to_before": sorted(list(before_set - after_set)),
            "positions_unique_to_after": sorted(list(after_set - before_set)),
            "total_before": len(before_positions),
            "total_after": len(after_positions)
        }

    def get_correction_summary(self) -> Dict:
        """
        Get summary of correction configuration for reporting.

        Returns:
            Dict with mode, factors, ranges, and bias statistics.
        """
        summary = self.metadata.copy()

        # Add bias statistics from loaded CSV
        bias_stats = {}
        for _, row in self.bias_factors_df.iterrows():
            range_name = row["range"].lower()
            bias_stats[range_name] = {
                "range_bounds": row["range_bounds"],
                "actual_count": int(row["actual_count"]),
                "predicted_count": int(row["predicted_count"]),
                "bias_pct": float(row["bias_pct"]),
                "correction_factor": self.range_factors[range_name]
            }

        summary["bias_statistics"] = bias_stats
        return summary

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"RangeAwareBiasCorrection(mode='{self.mode}', "
            f"factors={self.range_factors})"
        )
