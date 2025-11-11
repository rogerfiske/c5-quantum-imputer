"""
Lightweight Ising Re-Ranker (Repo-Style)
========================================
Path suggestion in repo: src/modeling/ensembles/ising_reranker.py

Learns symmetric pairwise couplings (J) and unary fields (h) from historical
labels via empirical co-occurrence statistics. Uses these to augment base
per-position scores with interaction-aware synergy and re-rank the Top-20.

Advantages:
    - No heavy optimization loops
    - Closed-form, stable estimates from counts
    - Works as a *drop-in* score augmenter on top of calibrated probabilities

Expected data:
    - Scores frame: ["event_id"] + ["p1".. "p39"] (prefer calibrated)
    - Labels frame: ["event_id"] + ["y1".. "y39"] (training history)

Output:
    - Re-ranked Top-20 indices parquet (top20_1..top20_20)
    - Optional augmented probabilities parquet (p1..p39)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class IsingConfig:
    """
    Configuration for lightweight Ising-style re-ranking.

    Args:
        lambda_pair: Weight for pairwise synergy term added to base scores.
        clip_j: Absolute clip for J_ij magnitude to ensure stability.
        eps: Smoothing constant for probability estimates.
    """
    lambda_pair: float = 0.4
    clip_j: float = 2.0
    eps: float = 1e-6


class LightweightIsingReRanker:
    """
    Lightweight Ising-style re-ranker using empirical co-occurrence statistics.

    Methods:
        fit(labels): Estimate unary fields (h) and pairwise couplings (J).
        augment_scores(base_scores): Add synergy term to base scores.
        topk_indices(p_aug, k): Return Top-k indices per event.
    """

    def __init__(self, config: Optional[IsingConfig] = None) -> None:
        self.config = config or IsingConfig()
        self.h: Optional[np.ndarray] = None  # shape (39,)
        self.J: Optional[np.ndarray] = None  # shape (39, 39)
        self._fitted: bool = False

    @staticmethod
    def _empirical_stats(Y: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute empirical unary and pairwise probabilities with smoothing.

        Args:
            Y: Binary matrix (n_events, 39).
            eps: Smoothing constant.

        Returns:
            (p1, p2) where
                p1[i] ~ P(x_i = 1),
                p2[i, j] ~ P(x_i = 1, x_j = 1), i != j, symmetric.
        """
        n = Y.shape[0]
        p1 = (Y.sum(axis=0) + eps) / (n + 2 * eps)
        # Pairwise co-occurrence counts
        C = (Y.T @ Y).astype(float)  # shape (39, 39)
        np.fill_diagonal(C, 0.0)
        p2 = (C + eps) / (n + 4 * eps)
        return p1, p2

    @staticmethod
    def _log_odds(x: float) -> float:
        """Safe logit (log-odds)."""
        x = np.clip(x, 1e-12, 1.0 - 1e-12)
        return float(np.log(x) - np.log(1.0 - x))

    def fit(self, labels: np.ndarray) -> "LightweightIsingReRanker":
        """
        Estimate fields and couplings via smoothed empirical log-odds ratios.

        Args:
            labels: Binary labels (n_events, 39).

        Returns:
            Self.
        """
        eps = self.config.eps
        p1, p2 = self._empirical_stats(labels, eps=eps)

        # Unary fields: h_i = logit(P(x_i=1))
        h = np.vectorize(self._log_odds)(p1)

        # Pairwise couplings: J_ij ~ log ( P(i,j) / (P(i)P(j)) )
        # Symmetric, zero diagonal, clipped.
        outer = np.outer(p1, p1)
        ratio = np.clip(p2 / np.clip(outer, 1e-12, None), 1e-12, 1e12)
        J = np.log(ratio)
        np.fill_diagonal(J, 0.0)
        J = np.clip(J, -self.config.clip_j, self.config.clip_j)

        # Center J by row/col means to reduce global bias (optional).
        J = J - J.mean(axis=0, keepdims=True) - J.mean(axis=1, keepdims=True) + J.mean()

        self.h = h
        self.J = (J + J.T) / 2.0  # enforce symmetry
        self._fitted = True
        return self

    def augment_scores(self, base_scores: np.ndarray) -> np.ndarray:
        """
        Augment base per-position scores with pairwise synergy term.

        Args:
            base_scores: Array (n_events, 39). Should be probabilities or positive scores.

        Returns:
            Augmented scores with same shape. For probabilities, we renormalize rows.
        """
        assert self._fitted, "LightweightIsingReRanker is not fitted."
        assert self.J is not None

        # Synergy for each position i: sum_j J_ij * p_j (expected activation of neighbors).
        synergy = base_scores @ self.J.T  # shape (n_events, 39)
        s_aug = base_scores + self.config.lambda_pair * synergy

        # If base_scores look like probabilities, renormalize for safety.
        row_sum = np.sum(np.maximum(s_aug, 1e-12), axis=1, keepdims=True)
        s_aug = s_aug / row_sum
        return s_aug

    @staticmethod
    def topk_indices(p: np.ndarray, k: int = 20) -> np.ndarray:
        """
        Return Top-k indices per event (1..39).

        Args:
            p: Scores/probabilities (n_events, 39).
            k: Top-k size.

        Returns:
            Integer array (n_events, k) with 1-based indices.
        """
        idx = np.argsort(-p, axis=1)[:, :k]
        return idx + 1


# ----------------
# Helper functions
# ----------------

def recall_at_20(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Recall@20 = average hits / 5, expressed as percentage.

    Args:
        y_true: Binary labels (n_events, 39).
        p: Probabilities (n_events, 39).

    Returns:
        Recall@20 as percentage.
    """
    idx = np.argsort(-p, axis=1)[:, :20]
    rows = np.arange(p.shape[0])[:, None]
    hits = np.sum(y_true[rows, idx], axis=1)
    return float(np.mean(hits) / 5.0 * 100.0)


# ---
# CLI
# ---

def _cli() -> None:
    """
    Command-line interface.

    Modes:
        - fit_apply: Fit J/h on labels and re-rank given scores.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, required=True, help="Path to scores.parquet (prefer calibrated)")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels.parquet (training history)")
    parser.add_argument("--lambda-pair", type=float, default=0.4, help="Weight for pairwise synergy term")
    parser.add_argument("--clip-j", type=float, default=2.0, help="Absolute clip for J magnitude")
    parser.add_argument("--out-topk", type=str, required=True, help="Output Top-20 parquet")
    parser.add_argument("--out-aug", type=str, required=False, help="Optional augmented probabilities parquet")
    args = parser.parse_args()

    scores_df = pd.read_parquet(args.scores)
    labels_df = pd.read_parquet(args.labels)

    score_cols = [f"p{i}" for i in range(1, 40)]
    label_cols = [f"y{i}" for i in range(1, 40)]

    P = scores_df[score_cols].to_numpy(dtype=float)
    Y = labels_df[label_cols].to_numpy(dtype=int)

    reranker = LightweightIsingReRanker(
        IsingConfig(lambda_pair=args.lambda_pair, clip_j=args.clip_j)
    )
    reranker.fit(Y)
    P_aug = reranker.augment_scores(P)

    idx = LightweightIsingReRanker.topk_indices(P_aug, k=20)
    out_df = pd.DataFrame({"event_id": scores_df["event_id"]})
    for k in range(20):
        out_df[f"top20_{k+1}"] = idx[:, k]
    out_df.to_parquet(args.out_topk, index=False)

    if args.out_aug:
        aug_df = pd.DataFrame({"event_id": scores_df["event_id"]})
        for i in range(39):
            aug_df[f"p{i+1}"] = P_aug[:, i]
        aug_df.to_parquet(args.out_aug, index=False)

    # Quick report if labels present
    r20 = recall_at_20(Y, P_aug)
    print(f"Recall@20 (after Ising re-rank): {r20:.2f}%")


if __name__ == "__main__":
    _cli()

# -------------------------------
# requirements.txt (used versions)
# -------------------------------
# numpy==2.1.3
# pandas==2.2.3
# pyarrow==17.0.0
