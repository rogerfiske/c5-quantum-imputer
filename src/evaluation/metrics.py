"""
Evaluation Metrics Module

This module implements custom metrics for evaluating quantum state prediction models.

Key Metric: "Wrong Predictions"
- Counts how many of the 5 actual quantum positions are NOT in the top-20 predictions
- Range: 0-5 wrong
  - 0 wrong = Excellent (all 5 actual in top-20)
  - 1-2 wrong = Good
  - 3-4 wrong = Poor
  - 5 wrong = Excellent (maximizing opposite - none in top-20)

This metric is particularly useful for understanding prediction quality distribution
and identifying where the model excels vs. struggles.

Author: BMad Dev Agent (James)
Date: 2025-10-14
Epic: Epic 4 - Evaluation and Reporting
Story: 4.2 - Wrong Predictions Metric
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WrongPredictionsMetric:
    """
    Computes the "wrong predictions" metric for quantum state prediction.

    This metric counts how many of the 5 actual quantum positions are NOT
    found in the top-k predictions (default k=20).

    Mathematical Definition:
        wrong_count = |actual_positions| - |actual_positions ∩ top_k_predictions|
        wrong_count = 5 - |actual_positions ∩ top_k_predictions|

    Where:
        - actual_positions: Set of 5 true quantum positions (from q_1 to q_5)
        - top_k_predictions: Set of k highest-ranked predicted positions
        - |·|: Set cardinality (size)
        - ∩: Set intersection

    Attributes:
        k (int): Number of top predictions to consider (default: 20)
        position_columns (List[str]): Column names for actual positions

    Example:
        >>> metric = WrongPredictionsMetric(k=20)
        >>> wrong_count = metric.compute_single_event(
        ...     actual_positions=[1, 15, 22, 30, 39],
        ...     predicted_positions=[1, 2, 15, 17, 22, ...]  # top-20
        ... )
        >>> print(wrong_count)
        2  # positions 30 and 39 not in top-20
    """

    def __init__(
        self,
        k: int = 20,
        position_columns: Optional[List[str]] = None
    ):
        """
        Initialize the wrong predictions metric.

        Args:
            k: Number of top predictions to consider (default: 20)
            position_columns: Column names for actual positions
                            (default: ['q_1', 'q_2', 'q_3', 'q_4', 'q_5'])
        """
        self.k = k

        if position_columns is None:
            self.position_columns = ['q_1', 'q_2', 'q_3', 'q_4', 'q_5']
        else:
            self.position_columns = position_columns

        logger.debug(f"Initialized WrongPredictionsMetric with k={k}")

    def compute_single_event(
        self,
        actual_positions: List[int],
        predicted_positions: List[int]
    ) -> int:
        """
        Compute wrong predictions count for a single event.

        Args:
            actual_positions: List of actual quantum positions (typically 5)
            predicted_positions: List of predicted positions (top-k ranked)

        Returns:
            Number of actual positions NOT in predictions (0 to len(actual_positions))

        Example:
            >>> metric.compute_single_event([1, 15, 22], [1, 2, 15, 17])
            1  # position 22 not in predictions
        """
        actual_set = set(actual_positions)
        predicted_set = set(predicted_positions[:self.k])

        # Count how many actual positions are NOT in top-k predictions
        missing_positions = actual_set - predicted_set
        wrong_count = len(missing_positions)

        return wrong_count

    def compute_batch(
        self,
        df: pd.DataFrame,
        predictions: List[List[int]]
    ) -> np.ndarray:
        """
        Compute wrong predictions counts for a batch of events.

        Args:
            df: Dataframe with actual positions in q_1, q_2, q_3, q_4, q_5 columns
            predictions: List of prediction lists (one per event, each with top-k positions)

        Returns:
            Array of wrong counts (one per event)

        Raises:
            ValueError: If number of predictions doesn't match number of events
            ValueError: If required position columns are missing
        """
        if len(predictions) != len(df):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) must match "
                f"number of events ({len(df)})"
            )

        # Validate required columns
        missing_cols = [col for col in self.position_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )

        wrong_counts = []

        for idx, row in df.iterrows():
            # Extract actual positions
            actual_positions = [int(row[col]) for col in self.position_columns]

            # Get predictions for this event
            predicted_positions = predictions[len(wrong_counts)]

            # Compute wrong count
            wrong_count = self.compute_single_event(actual_positions, predicted_positions)
            wrong_counts.append(wrong_count)

        return np.array(wrong_counts)

    def compute_distribution(
        self,
        wrong_counts: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute distribution of wrong predictions counts.

        Args:
            wrong_counts: Array of wrong counts (one per event)

        Returns:
            Dictionary mapping wrong_count to {count, percentage}

        Example:
            >>> distribution = metric.compute_distribution(wrong_counts)
            >>> print(distribution[0])
            {'count': 36, 'percentage': 3.60}
        """
        total_events = len(wrong_counts)

        if total_events == 0:
            logger.warning("No events to compute distribution")
            return {}

        distribution = {}

        # Count occurrences of each wrong_count (0-5)
        for wrong_count in range(6):
            count = int(np.sum(wrong_counts == wrong_count))
            percentage = (count / total_events) * 100.0

            distribution[wrong_count] = {
                'count': count,
                'percentage': percentage
            }

        return distribution

    def format_distribution_summary(
        self,
        distribution: Dict[int, Dict[str, float]],
        total_events: int,
        test_name: str = "Representative test"
    ) -> str:
        """
        Format distribution as the user-specified summary string.

        Expected format:
        HOLDOUT TEST SUMMARY - 1000 Events
        "Representative test name"
          ------------------------------------
          0 wrong: 36 events (3.60%)   ← All 5 actual values in top-20
          1 wrong: 166 events (16.60%)  ← 4 of 5 actual values in top-20
          2 wrong: 322 events (32.20%)  ← 3 of 5 actual values in top-20
          3 wrong: 313 events (31.30%)  ← 2 of 5 actual values in top-20
          4 wrong: 146 events (14.60%)  ← 1 of 5 actual values in top-20
          5 wrong: 17 events (1.70%)    ← 0 of 5 actual values in top-20

        Args:
            distribution: Distribution dictionary from compute_distribution()
            total_events: Total number of events
            test_name: Name of the test

        Returns:
            Formatted summary string
        """
        explanations = {
            0: f"All 5 actual values in top-{self.k}",
            1: f"4 of 5 actual values in top-{self.k}",
            2: f"3 of 5 actual values in top-{self.k}",
            3: f"2 of 5 actual values in top-{self.k}",
            4: f"1 of 5 actual values in top-{self.k}",
            5: f"0 of 5 actual values in top-{self.k}",
        }

        lines = [
            f"HOLDOUT TEST SUMMARY - {total_events} Events",
            f'"{test_name}"',
            "  ------------------------------------"
        ]

        for wrong_count in range(6):
            count = distribution[wrong_count]['count']
            percentage = distribution[wrong_count]['percentage']
            explanation = explanations[wrong_count]

            # Format: "  0 wrong: 36 events (3.60%)   ← All 5 actual values in top-20"
            line = f"  {wrong_count} wrong: {count} events ({percentage:.2f}%)".ljust(38)
            line += f" ← {explanation}"
            lines.append(line)

        return '\n'.join(lines)


def compute_top_k_accuracy(
    actual_positions: List[int],
    predicted_positions: List[int],
    k: int
) -> float:
    """
    Compute top-k accuracy for a single event.

    Top-k accuracy = (number of actual positions in top-k predictions) / k

    Args:
        actual_positions: List of actual quantum positions
        predicted_positions: List of predicted positions (ranked)
        k: Number of top predictions to consider

    Returns:
        Accuracy score between 0.0 and 1.0

    Example:
        >>> compute_top_k_accuracy([1, 15, 22, 30, 39], [1, 2, 15, ...], k=20)
        0.15  # 3 out of 5 actual in top-20, but measured as 3/20 = 0.15
    """
    actual_set = set(actual_positions)
    predicted_set = set(predicted_positions[:k])

    matches = len(actual_set & predicted_set)
    accuracy = matches / len(actual_positions)

    return accuracy


def compute_recall_at_k(
    actual_positions: List[int],
    predicted_positions: List[int],
    k: int
) -> float:
    """
    Compute recall@k for a single event.

    Recall@k = (number of actual positions in top-k) / (total actual positions)

    This is the proportion of actual positions successfully predicted in top-k.

    Args:
        actual_positions: List of actual quantum positions
        predicted_positions: List of predicted positions (ranked)
        k: Number of top predictions to consider

    Returns:
        Recall score between 0.0 and 1.0

    Example:
        >>> compute_recall_at_k([1, 15, 22, 30, 39], [1, 2, 15, ...], k=20)
        0.60  # 3 out of 5 actual in top-20 → 3/5 = 0.60
    """
    actual_set = set(actual_positions)
    predicted_set = set(predicted_positions[:k])

    matches = len(actual_set & predicted_set)
    recall = matches / len(actual_positions)

    return recall


def compute_precision_at_k(
    actual_positions: List[int],
    predicted_positions: List[int],
    k: int
) -> float:
    """
    Compute precision@k for a single event.

    Precision@k = (number of actual positions in top-k) / k

    This measures how many of the top-k predictions are correct.

    Args:
        actual_positions: List of actual quantum positions
        predicted_positions: List of predicted positions (ranked)
        k: Number of top predictions to consider

    Returns:
        Precision score between 0.0 and 1.0

    Example:
        >>> compute_precision_at_k([1, 15, 22, 30, 39], [1, 2, 15, ...], k=20)
        0.15  # 3 out of 20 predictions correct → 3/20 = 0.15
    """
    actual_set = set(actual_positions)
    predicted_set = set(predicted_positions[:k])

    matches = len(actual_set & predicted_set)
    precision = matches / k

    return precision
